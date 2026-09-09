// Shared prologue for the PAW CUDA ops. Split from paw.cu; each
// paw-<op>.cu includes this header. The 15 shared host helpers below
// are defined once here and used by several op files.
#pragma once
#include "common.cuh"
#include "paw.cuh"
#include "cp-async.cuh"
#include <cstring>
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>

#include "common.cuh"
#include "paw.cuh"
#include "cp-async.cuh"
#include <cstring>
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>

// CUDA port of the 7 PAW codec ops. Numeric source of truth is the CPU
// implementation (ggml-cpu/ops.cpp); the GPU decomposition mirrors the Vulkan
// shaders (ggml-vulkan/vulkan-shaders/paw_*.comp) 1:1. The build compiles
// with -use_fast_math, so the documented two-rounding sites are pinned with
// _rn intrinsics (embed_rows q*step, exp_basis B-loop, exp_walk V2 fp16 round,
// every FWHT final scale); dot accumulations stay contractible on purpose.
//
// Prefill additionally has dense-materialize paths (mirroring the Mach-1
// engine's expert_forward_prefill_dense / NeCodesLinear.dense split): instead
// of re-decoding the trellis per (pair, token), the walk stage is replaced by
// one decode of hatWr into a transient fp16 bank plus a plain fp32 apply.
// Because the Hadamards here act on ACTIVATIONS (exp_u/exp_out/rt_u/rt_out),
// the bank holds ONLY the fp16-rounded LUT values — no FWHT on weights. The
// bank values are the exact halves the fused walk multiplies (V2 pins the
// fp16 round; V8/rt tluts are pre-rounded F16), so the dense paths differ
// from the fused ones only in contractible fp32 summation order (and, for
// V8, in the association of the wave-gamma product — also contractible).
//   GGML_PAW_DENSE_MIN     (default 1024): EXP_MM pairs (n_used*n_tok)
//                            at/above which the dense path runs.
//   GGML_PAW_DENSE_MIN_TOK (default 4): RT_MM token count at/above which
//                            the dense path runs.
// Both are read once and cached; set =1 to force the dense paths in tests.

#define GGML_CUDA_PAW_DEM_FLAG 0x40000000u

// dense-path thresholds, read once (host)
static int paw_env_int(const char * name, const int def) {
    const char * s = getenv(name);
    if (s == nullptr && strncmp(name, "GGML_PAW_", 9) == 0) {
        // back-compat: these knobs shipped as GGML_MACH1_* before the rename.
        // Without this an old script silently loses the optimizations it asks
        // for, which looks like a performance regression with no error.
        char legacy[128];
        snprintf(legacy, sizeof(legacy), "GGML_MACH1_%s", name + 9);
        s = getenv(legacy);
    }
    return s != nullptr ? atoi(s) : def;
}

// paw kernels are launched with plain stream launches, NOT via
// ggml_cuda_kernel_launch: the multi-stage paw ops have true RAW
// dependencies through global pool scratch between consecutive kernels, and
// under programmatic dependent launch (PDL) on Hopper (H100) large prefills
// (e.g. llama-bench -p 1024) crash within seconds, while GGML_CUDA_PDL=0 is
// stable. Plain stream launches are always correctly ordered, so paw opts
// out of PDL entirely. The ggml_cuda_pdl_sync()/ggml_cuda_pdl_lc() calls in
// the kernels are documented no-ops for kernels launched without the PDL
// launch attribute (grid dependencies are already satisfied at launch), so
// they are left in place.
template <typename Kernel, typename... Args>
static void paw_launch(Kernel kernel, const ggml_cuda_kernel_launch_params & p, Args &&... args) {
    kernel<<<p.block_nums, p.block_dims, p.shmem, p.stream>>>(std::forward<Args>(args)...);
    CUDA_CHECK(cudaGetLastError());
}

// --- measurement-only instrumentation (default off, no numeric effect) ---
// GGML_PAW_TIME=1: wall-clock every paw kernel launch (stream sync before
// and after) keyed by stage + shape; totals dump to stderr at process exit.
// Do NOT combine with CUDA graph capture (set GGML_CUDA_DISABLE_GRAPHS=1).
// GGML_PAW_DEBUG=1: one-time diagnostics (p4 repack status).
#include <chrono>
#include <map>
#include <vector>

static bool paw_time_on() {
    static const bool on = paw_env_int("GGML_PAW_TIME", 0) != 0;
    return on;
}

static bool paw_debug_on() {
    static const bool on = paw_env_int("GGML_PAW_DEBUG", 0) != 0;
    return on;
}

static bool debug_diff_on() {
    static const bool on = paw_env_int("GGML_PAW_RT_WALK_QTIP_DEBUG", 0) != 0;
    return on;
}

struct paw_time_table {
    std::mutex mtx;
    std::map<std::string, std::pair<long long, double>> acc;   // key -> (calls, total us)
    ~paw_time_table() {
        double tot = 0.0;
        for (const auto & kv : acc) {
            tot += kv.second.second;
        }
        fprintf(stderr, "paw-time: TOTAL %.1f ms across timed paw kernels\n", tot/1000.0);
        for (const auto & kv : acc) {
            fprintf(stderr, "paw-time: %-64s calls=%7lld total=%10.1f us avg=%9.2f us\n",
                    kv.first.c_str(), kv.second.first, kv.second.second,
                    kv.second.second/(double) kv.second.first);
        }
    }
};

static paw_time_table & paw_times() {
    static paw_time_table t;
    return t;
}

// secondary stream for decoding expert slabs concurrently with the cuBLAS
// GEMMs of the previous slab (created once, never captured by graphs)
static cudaStream_t paw_aux_stream() {
    static cudaStream_t s = nullptr;
    if (!s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }
    return s;
}

template <typename F>
static void paw_timed(cudaStream_t stream, const std::string & key, F && launch) {
    if (!paw_time_on()) {
        launch();
        return;
    }
    // timing syncs are illegal inside a graph capture (the dense path
    // captures where the 35B blas path did not); time only the replay-free
    // eager launches and let captured work run unprofiled
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        launch();
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto t0 = std::chrono::steady_clock::now();
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
    paw_time_table & tt = paw_times();
    std::lock_guard<std::mutex> lock(tt.mtx);
    auto & e = tt.acc[key];
    e.first  += 1;
    e.second += us;
}

// block size for the four FWHT-stage kernels (bit-exact for any value; the
// A/B knob GGML_PAW_FWHT_WG selects 256 or 512)
static bool paw_fwht_wg512() {
    static const bool wg512 = paw_env_int("GGML_PAW_FWHT_WG", 512) == 512;
    return wg512;
}

// orthonormal Walsh-Hadamard butterflies over a shared-memory vector of
// power-of-two length d (Sylvester order). The caller loads sh, issues one
// __syncthreads(), calls this, then applies the single final
// __fdiv_rn(sh[i], __fsqrt_rn((float) d)) — the reference op order.
//
// The low stages (span 1..16, warp-local when wg is a multiple of 32 and
// every thread owns sh[tid]) run via __shfl_xor_sync instead of shared-memory
// read-modify-write + __syncthreads — the butterfly math is identical (same
// pair decomposition, same add/subtract order), so the result is bit-identical
// to the all-shared version. Only the barrier count drops (11-13 -> 6-9 for
// d = 2048-8192), which dominates the latency of these tiny per-token kernels.
// A/B knob GGML_PAW_FWHT_SHMEM=1 forces the old all-shared-memory butterfly
// (no warp shuffles), for correctness A/B against the shfl path. Read once on
// the host into a __device__ flag (the fwht helper runs in-kernel).
static __device__ int paw_fwht_mode_dev = 0;

static void paw_fwht_set_mode() {
    static bool init = false;
    if (!init) {
        const int mode = paw_env_int("GGML_PAW_FWHT_SHMEM", 0) != 0 ? 1 : 0;
        CUDA_CHECK(cudaMemcpyToSymbol(paw_fwht_mode_dev, &mode, sizeof(mode)));
        init = true;
    }
}

static __device__ __forceinline__ void paw_fwht_block(float * sh, const int d, const int tid, const int wg) {
    const bool shfl_ok = paw_fwht_mode_dev == 0 && (wg % 32 == 0) && (d >= wg) && (d % 32 == 0);
    if (shfl_ok) {
        // stage 1..16 via warp shuffle, per element slot: thread tid owns
        // elements {tid + j*wg}. Each slot's butterflies stay within the slot
        // (span < 32 < wg), so shfl_xor over the lane id is exact.
        const int nslots = d / wg;
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = j < nslots ? sh[tid + j*wg] : 0.0f;
        }
#pragma unroll
        for (int span = 1; span < 32 && span < d; span <<= 1) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (j < nslots) {
                    const float o = __shfl_xor_sync(0xffffffffu, v[j], span);
                    v[j] = (tid & span) ? o - v[j] : v[j] + o;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < nslots) {
                sh[tid + j*wg] = v[j];
            }
        }
        __syncthreads();
        for (int span = 32; span < d; span <<= 1) {
            for (int b = tid; b < d/2; b += wg) {
                const int base = (b / span)*(span << 1) + (b % span);
                const float a0 = sh[base];
                const float a1 = sh[base + span];
                sh[base]        = a0 + a1;
                sh[base + span] = a0 - a1;
            }
            __syncthreads();
        }
    } else {
        for (int span = 1; span < d; span <<= 1) {
            for (int b = tid; b < d/2; b += wg) {
                const int base = (b / span)*(span << 1) + (b % span);
                const float a0 = sh[base];
                const float a1 = sh[base + span];
                sh[base]        = a0 + a1;
                sh[base + span] = a0 - a1;
            }
            __syncthreads();
        }
    }
}

// v2: fully register-resident FWHT. Each warp owns one contiguous 512-element
// chunk (so this requires wg == 32*(d/512) == d/16 threads). The 9 butterflies
// of each 512-chunk run entirely in registers: spans 1..16 via __shfl_xor_sync,
// spans 32..256 via intra-lane register pairs -- zero shared traffic and zero
// barriers. The cross-chunk spans (512, 1024, ...) go through shared once the
// register chunks are flushed, needing only log2(d/512) shared stages instead
// of the full log2(d) barrier-per-stage shared version. This is the mature
// llama.cpp fwht_cuda pattern (register-only, shfl) extended past N=512 by
// decomposing into one-512-chunk-per-warp. Same butterfly math as
// paw_fwht_block, so results agree to within FP rounding (summation order of
// the cross-chunk stages differs).
static __device__ __forceinline__ void paw_fwht_block_v2(float * sh, const int d, const int tid, const int wg) {
    constexpr int chunk = 512;
    const int wid  = tid / 32;
    const int lane = tid & 31;
    float reg[16];
    const int base = wid*chunk + lane;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        reg[k] = sh[base + 32*k];
    }
#pragma unroll
    for (int h = 1; h <= 16; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float o = __shfl_xor_sync(0xffffffffu, reg[k], h);
            reg[k] = (lane & h) ? o - reg[k] : reg[k] + o;
        }
    }
    for (int h = 32; h < chunk; h <<= 1) {
        const int step = h / 32;
        for (int j = 0; j < 16; j += 2*step) {
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        sh[base + 32*k] = reg[k];
    }
    __syncthreads();
    for (int h = chunk; h < d; h <<= 1) {
        for (int i = tid; i < d/2; i += wg) {
            const int b = (i / h)*(h << 1) + (i % h);
            const float a0 = sh[b];
            const float a1 = sh[b + h];
            sh[b]        = a0 + a1;
            sh[b + h]    = a0 - a1;
        }
        __syncthreads();
    }
}

// host gate + geometry for the register-chunk FWHT (GGML_PAW_FWHT_V2=1).
static bool paw_fwht_v2_on() {
    static const bool on = paw_env_int("GGML_PAW_FWHT_V2", 1) != 0;
    return on;
}
static __host__ __device__ __forceinline__ bool paw_fwht_v2_ok(int d) {
    return d >= 512 && (d & 511) == 0;
}

// run f with the compile-time WG matching wg (v2 needs wg == d/16; this also
// keeps the wg512/256 A/B path working).
template <typename F>
static void paw_fwht_for_wg(int wg, F && f) {
    switch (wg) {
        case 32:  f(std::integral_constant<int, 32>{});  break;
        case 64:  f(std::integral_constant<int, 64>{});  break;
        case 128: f(std::integral_constant<int, 128>{}); break;
        case 256: f(std::integral_constant<int, 256>{}); break;
        case 512: f(std::integral_constant<int, 512>{}); break;
    }
}

// store 16 contiguous halves as two 16-byte transactions (the value bits are
// exactly the input halves — packing only changes the store width). dst must
// be 16-byte aligned; every bank row chunk is (offsets are multiples of 16
// halves = 32 bytes).
static __device__ __forceinline__ void paw_store_half16(half * dst, const half * v) {
    uint4 a, b;
    a.x = (uint32_t) __half_as_ushort(v[0])  | ((uint32_t) __half_as_ushort(v[1])  << 16);
    a.y = (uint32_t) __half_as_ushort(v[2])  | ((uint32_t) __half_as_ushort(v[3])  << 16);
    a.z = (uint32_t) __half_as_ushort(v[4])  | ((uint32_t) __half_as_ushort(v[5])  << 16);
    a.w = (uint32_t) __half_as_ushort(v[6])  | ((uint32_t) __half_as_ushort(v[7])  << 16);
    b.x = (uint32_t) __half_as_ushort(v[8])  | ((uint32_t) __half_as_ushort(v[9])  << 16);
    b.y = (uint32_t) __half_as_ushort(v[10]) | ((uint32_t) __half_as_ushort(v[11]) << 16);
    b.z = (uint32_t) __half_as_ushort(v[12]) | ((uint32_t) __half_as_ushort(v[13]) << 16);
    b.w = (uint32_t) __half_as_ushort(v[14]) | ((uint32_t) __half_as_ushort(v[15]) << 16);
    ((uint4 *) dst)[0] = a;
    ((uint4 *) dst)[1] = b;
}

// storage-expert group of flat pair rank p (see paw_exp_group.comp)
static __device__ __forceinline__ uint32_t paw_group_of(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const int p, const int n_used, const int n_kept,
        const int ids_s0, const int ids_s1) {
    const int s = p % n_used;
    const int t = p / n_used;
    const uint32_t id = (uint32_t) ids[s*ids_s0 + t*ids_s1];
    const uint32_t rm = (uint32_t) remap[id];
    return (rm & GGML_CUDA_PAW_DEM_FLAG) != 0u ? (uint32_t) n_kept + (rm & ~GGML_CUDA_PAW_DEM_FLAG) : rm;
}

//
// EMBED_GATHER — one thread per output element (paw_embed_gather.comp)
//

static __global__ void paw_embed_gather_kernel(
        const uint8_t  * GGML_CUDA_RESTRICT codes,
        const uint16_t * GGML_CUDA_RESTRICT lut,    // bf16 bit patterns
        const int32_t  * GGML_CUDA_RESTRICT ids,
        float          * GGML_CUDA_RESTRICT dst,
        const int n_embd, const int gsh, const int64_t total) {
    const int64_t gid = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }
    ggml_cuda_pdl_sync();
    const int64_t tok = gid / n_embd;
    const int     j   = (int)(gid % n_embd);
    const int64_t r   = ids[tok];
    const int     ng  = n_embd >> gsh;

    const uint32_t by = codes[r*(n_embd/2) + (j >> 1)];
    const uint32_t q  = (j & 1) ? (by >> 4) : (by & 0x0Fu);
    const uint32_t bits = (uint32_t) lut[(r*ng + (j >> gsh))*16 + q] << 16;
    dst[gid] = __uint_as_float(bits);
}

