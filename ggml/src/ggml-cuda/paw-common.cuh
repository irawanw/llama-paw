// Shared PAW CUDA helpers. Split from paw.cu; each paw-<op>.cu
// includes this header. Functions below are each defined once and
// reused by several ops (see docs/paw/README.md).
#pragma once
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


// shared: paw_embed_rows_kernel (defined once, reused across ops)
static __global__ void paw_embed_rows_kernel(
        const uint8_t * GGML_CUDA_RESTRICT q,
        const half    * GGML_CUDA_RESTRICT mn,
        const half    * GGML_CUDA_RESTRICT mx,
        const int32_t * GGML_CUDA_RESTRICT ids,
        float         * GGML_CUDA_RESTRICT dst,
        const int n_embd, const int ng, const int grp, const int n_tokens, const int row_bytes) {
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= n_tokens*ng) {
        return;
    }
    ggml_cuda_pdl_sync();
    const int     tok = gid / ng;
    const int     g   = gid % ng;
    const int64_t r   = ids[tok];

    const float mnf  = __half2float(mn[r*ng + g]);
    const float d    = __half2float(mx[r*ng + g]) - mnf;
    // reference: step = max(mx - mn, 1e-8) / 7, all fp32 (pinned rounding)
    const float step = __fdiv_rn(d > 1e-8f ? d : 1e-8f, 7.0f);

    int64_t  pbi   = r*row_bytes + (g*grp*3)/8;   // groups are byte-aligned (grp*3 % 8 == 0)
    uint32_t acc   = 0;
    int      nbits = 0;
    const int64_t obase = (int64_t) tok*n_embd + (int64_t) g*grp;
    for (int t = 0; t < grp; ++t) {
        while (nbits < 3) {
            acc = (acc << 8) | q[pbi++];
            nbits += 8;
        }
        nbits -= 3;
        const uint32_t qv = (acc >> nbits) & 0x7u;
        // reference rounds q*step to fp32 BEFORE the add — pinned, no fma
        const float prod = __fmul_rn((float) qv, step);
        dst[obase + t] = __fadd_rn(mnf, prod);
    }
}

// shared: paw_bank_fp8_on (defined once, reused across ops)
static bool paw_bank_fp8_on() {
    static const bool on = paw_env_int("GGML_PAW_BANK_FP8", 0) != 0;
    return on;
}

// shared: paw_rt_bank_fp8_on (defined once, reused across ops)
static bool paw_rt_bank_fp8_on() {
    static const bool on = paw_env_int("GGML_PAW_RT_BANK_FP8", paw_bank_fp8_on() ? 1 : 0) != 0;
    return on;
}

// shared: paw_rt_bank_idx_on (defined once, reused across ops)
static bool paw_rt_bank_idx_on() {
    static const bool on = paw_env_int("GGML_PAW_RT_BANK_IDX", 0) != 0;
    return on;
}

// shared: paw_e5m2_to_f32 (defined once, reused across ops)
__device__ __forceinline__ float paw_e5m2_to_f32(uint8_t b) {
    const uint32_t s = ((uint32_t) b & 0x80u) << 24;
    const uint32_t e = ((uint32_t) b >> 2) & 0x1Fu;
    const uint32_t m = (uint32_t) b & 0x3u;
    uint32_t bits;
    if (e == 0) {
        bits = s | (111u << 23) | (m << 21);   // m * 2^-16
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 21);
    } else {
        bits = s | ((e + 112u) << 23) | (m << 21);
    }
    return __uint_as_float(bits);
}

// shared: paw_f32_to_e5m2 (defined once, reused across ops)
__device__ __forceinline__ uint8_t paw_f32_to_e5m2(float f) {
    const uint32_t bits = __float_as_uint(f);
    const uint32_t s = (bits >> 31) & 1u;
    const uint32_t e = (bits >> 23) & 0xFFu;
    const uint32_t m = bits & 0x7FFFFFu;
    if (e == 0xFF) {
        return (uint8_t) ((s << 7) | 0x7Cu | (m ? 1u : 0u));
    }
    if (e > 142) {                       // |f| >= 2^15 -> overflow
        return (uint8_t) ((s << 7) | 0x7Cu);
    }
    if (e <= 112) {                      // |f| < 2^-15 -> subnormal or zero
        const int shift = 16 - (int) e;  // mantissa bits to drop
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) r++;   // round to nearest even
        if (r > 3) {
            return (uint8_t) ((s << 7) | (1u << 2));        // rounds to 2^-14
        }
        return (uint8_t) ((s << 7) | r);
    }
    uint32_t e5 = e - 127 + 15;
    uint32_t m2 = (m + 0x100000u) >> 21;   // round to 2 mantissa bits
    if (m2 == 4) { m2 = 0; e5++; }
    if (e5 > 30) {
        return (uint8_t) ((s << 7) | 0x7Cu);
    }
    return (uint8_t) ((s << 7) | (e5 << 2) | m2);
}

// shared: paw_rt_bank_gemv_fp8 (defined once, reused across ops)
static __global__ void paw_rt_bank_gemv_fp8(
        const uint8_t * GGML_CUDA_RESTRICT bank,
        const float   * GGML_CUDA_RESTRICT scr_u,
        float         * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n, const int nt) {
    __shared__ float lut[256];
    const int tid = threadIdx.x;
    if (tid < 256) {
        lut[tid] = paw_e5m2_to_f32((uint8_t) tid);
    }
    __syncthreads();

    const int row  = blockIdx.x*8 + (threadIdx.x >> 5);
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;

    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();

    const float   * u = scr_u + (int64_t) t*n;
    const uint8_t * W = bank   + (int64_t) row*n;

    float acc = 0.0f;
    if (n % 4 == 0) {
        const uint32_t * W4 = (const uint32_t *) W;   // 4 e5m2 bytes / lane, lanes adjacent
        const float4   * u4 = (const float4 *) u;     // 16 bytes, 16-byte aligned (n mult of 4)
        const int n4 = n/4;
        for (int i = lane; i < n4; i += 128) {
            const uint32_t w = __ldcs(W4 + i);   // streamed once, evict-first
            const uint8_t * wb = (const uint8_t *) &w;
            const float4 x = u4[i];
            acc += lut[wb[0]]*x.x + lut[wb[1]]*x.y + lut[wb[2]]*x.z + lut[wb[3]]*x.w;
            const int j = i + 32;
            if (j < n4) {
                const uint32_t wj = __ldcs(W4 + j);
                const uint8_t * wjb = (const uint8_t *) &wj;
                const float4 xj = u4[j];
                acc += lut[wjb[0]]*xj.x + lut[wjb[1]]*xj.y + lut[wjb[2]]*xj.z + lut[wjb[3]]*xj.w;
            }
            const int k = i + 64;
            if (k < n4) {
                const uint32_t wk = __ldcs(W4 + k);
                const uint8_t * wkb = (const uint8_t *) &wk;
                const float4 xk = u4[k];
                acc += lut[wkb[0]]*xk.x + lut[wkb[1]]*xk.y + lut[wkb[2]]*xk.z + lut[wkb[3]]*xk.w;
            }
            const int l = i + 96;
            if (l < n4) {
                const uint32_t wl = __ldcs(W4 + l);
                const uint8_t * wlb = (const uint8_t *) &wl;
                const float4 xl = u4[l];
                acc += lut[wlb[0]]*xl.x + lut[wlb[1]]*xl.y + lut[wlb[2]]*xl.z + lut[wlb[3]]*xl.w;
            }
        }
    } else {
        for (int i = lane; i < n; i += 32) {
            acc += lut[W[i]] * u[i];
        }
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}

// shared: paw_rt_bank_gemv_fp8_v3 (defined once, reused across ops)
static __global__ void paw_rt_bank_gemv_fp8_v3(
        const uint8_t * GGML_CUDA_RESTRICT bank,  // [m, n] row-major e5m2
        const float   * GGML_CUDA_RESTRICT scr_u, // [nt, n] row-major
        float         * GGML_CUDA_RESTRICT scr_v, // [nt, m] row-major
        const int m, const int n, const int nt) {
    constexpr int WARPS = 8;
    __shared__ float lut[256];
    __shared__ float u_sh[4096];

    const int tid = threadIdx.x;
    if (tid < 256) {
        lut[tid] = paw_e5m2_to_f32((uint8_t) tid);
    }
    const int t    = blockIdx.z;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    ggml_cuda_pdl_sync();
    const float * u = scr_u + (int64_t) t*n;
    for (int i = tid; i < n; i += 256) {
        u_sh[i] = u[i];
    }
    __syncthreads();

    const int n4 = n/4;
    const float4 * u4 = (const float4 *) u_sh;

    for (int r = wid; r < WARPS*2; r += WARPS) {
        const int row = blockIdx.x*WARPS*2 + r;
        if (row < m) {
            const uint32_t * W4 = (const uint32_t *) (bank + (int64_t) row*n);
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            int i = lane;
            for (; i + 32 < n4; i += 64) {
                const uint32_t w0 = __ldcs(W4 + i);
                const uint32_t w1 = __ldcs(W4 + i + 32);
                const uint8_t * wb0 = (const uint8_t *) &w0;
                const uint8_t * wb1 = (const uint8_t *) &w1;
                const float4 x0 = u4[i];
                const float4 x1 = u4[i + 32];
                acc0 += lut[wb0[0]]*x0.x + lut[wb0[1]]*x0.y + lut[wb0[2]]*x0.z + lut[wb0[3]]*x0.w;
                acc1 += lut[wb1[0]]*x1.x + lut[wb1[1]]*x1.y + lut[wb1[2]]*x1.z + lut[wb1[3]]*x1.w;
            }
            for (; i < n4; i += 32) {
                const uint32_t w0 = __ldcs(W4 + i);
                const uint8_t * wb0 = (const uint8_t *) &w0;
                const float4 x0 = u4[i];
                acc0 += lut[wb0[0]]*x0.x + lut[wb0[1]]*x0.y + lut[wb0[2]]*x0.z + lut[wb0[3]]*x0.w;
            }
            float acc = acc0 + acc1;
            acc = warp_reduce_sum<32>(acc);
            if (lane == 0) {
                scr_v[(int64_t) t*m + row] = acc;
            }
        }
    }
}

// shared: paw_idx80_get (defined once, reused across ops)
static __device__ __forceinline__ uint16_t paw_idx80_get(const uint32_t * W, const int i) {
    const int bit = 10*(i & 15);
    const int wi = bit >> 5;
    const int off = bit & 31;
    const uint32_t * p = W + (i >> 4)*5;
    uint32_t code = p[wi] >> off;
    if (off > 22) {
        code |= p[wi + 1] << (32 - off);
    }
    return (uint16_t) (code & 0x3FFu);
}

// shared: paw_rt_bank_gemv_idx80 (defined once, reused across ops)
static __global__ void paw_rt_bank_gemv_idx80(
        const uint16_t * GGML_CUDA_RESTRICT bank,
        const half     * GGML_CUDA_RESTRICT tlut,
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n, const int nt) {
    __shared__ half2 slut[512];
    const int tid = threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        slut[i] = ((const half2 *) tlut)[i];
    }
    __syncthreads();

    const int row  = blockIdx.x*16 + (tid >> 5);
    const int t    = blockIdx.z;
    const int lane = tid & 31;
    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();

    const float    * u = scr_u + (int64_t) t*n;
    const uint32_t * W = (const uint32_t *) bank + (int64_t) row*(n/32)*5;
    const float2   * u2 = (const float2 *) u;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const uint16_t code = paw_idx80_get(W, i);
        const int lr = code & 511;
        float2 w = __half22float2(slut[lr]);
        if (code & 512) {
            w.x = -w.x;
        }
        const float2 x = u2[i];
        acc += w.x*x.x + w.y*x.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}

// shared: paw_ne_mm_kernel (defined once, reused across ops)
static __global__ void paw_ne_mm_kernel(
        const uint8_t * GGML_CUDA_RESTRICT packed,
        const half    * GGML_CUDA_RESTRICT gscale,
        const half    * GGML_CUDA_RESTRICT lut,
        const float   * GGML_CUDA_RESTRICT x,
        float         * GGML_CUDA_RESTRICT dst,
        const int B, const int T, const int k, const int ng,
        const int rows_per_chunk, const int n_tokens) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= B) {
        return;
    }
    ggml_cuda_pdl_sync();
    const int jt0 = blockIdx.y*8;
    const int njt = min(8, n_tokens - jt0);

    const int64_t  row_bytes = (int64_t) T*k/8;
    const int64_t  lut_off   = (int64_t)(r / rows_per_chunk)*4096;
    const uint32_t kmask     = (1u << k) - 1;

    float accs[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accs[j] = 0.0f;
    }

    const uint8_t * pb = packed + (int64_t) r*row_bytes;
    uint32_t acc   = 0;
    uint32_t state = 0;
    int      nbits = 0;
    for (int g = 0; g < ng; ++g) {
        const float gsc = __half2float(gscale[(int64_t) r*ng + g]);
        for (int t = 0; t < 128; ++t) {
            while (nbits < k) {
                acc = (acc << 8) | *pb++;
                nbits += 8;
            }
            nbits -= k;
            state = ((state << k) | ((acc >> nbits) & kmask)) & 0xFFFu;
            // reference decode: f32(lut) * f32(gscale), one fp32 product
            const float w = __half2float(lut[lut_off + state]) * gsc;
            const int idx = g*128 + t;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < njt) {
                    accs[j] += w * x[(int64_t)(jt0 + j)*T + idx];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < njt) {
            dst[(int64_t)(jt0 + j)*B + r] = accs[j];
        }
    }
}

// shared: paw_rt_bank_gemv (defined once, reused across ops)
static __global__ void paw_rt_bank_gemv(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT scr_u,  // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    const int row  = blockIdx.x*8 + (threadIdx.x >> 5);
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;

    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();

    const float  * u = scr_u + (int64_t) t*n;
    const half   * W = bank   + (int64_t) row*n;
    const half2  * W2 = (const half2 *) W;
    const float2 * u2 = (const float2 *) u;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const float2 w = __half22float2(W2[i]);
        const float2 x = u2[i];
        acc += w.x*x.x + w.y*x.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}

// shared: paw_exp_basis_kernel (defined once, reused across ops)
static __global__ void paw_exp_basis_kernel(
        const half    * GGML_CUDA_RESTRICT a,
        const half    * GGML_CUDA_RESTRICT b,
        const half    * GGML_CUDA_RESTRICT c,
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT x,
        const float   * GGML_CUDA_RESTRICT acc_in,   // nullptr when has_acc == 0
        float         * GGML_CUDA_RESTRICT dst,
        const int n, const int r, const int m, const int n_used, const int xne1,
        const int ids_s0, const int ids_s1, const int has_acc) {
    constexpr int WG = 256;
    __shared__ float tv[256];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const uint32_t id = (uint32_t) ids[s*ids_s0 + t*ids_s1];
    const uint32_t rm = (uint32_t) remap[id];

    const int64_t obase = (int64_t) t*n_used*m + (int64_t) s*m;
    if ((rm & GGML_CUDA_PAW_DEM_FLAG) == 0u) {   // kept slot: block-uniform
        for (int i = tid; i < m; i += WG) {
            dst[obase + i] = has_acc ? acc_in[obase + i] : 0.0f;
        }
        return;
    }
    const int di = (int)(rm & ~GGML_CUDA_PAW_DEM_FLAG);
    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;

    // t_j = c_j * (A_j . x)
    for (int j = tid; j < r; j += WG) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) {
            acc += __half2float(a[(int64_t) j*n + i]) * x[xbase + i];
        }
        // one rounding: t_j = fl(c_j * v_j)
        tv[j] = __half2float(c[(int64_t) di*r + j]) * acc;
    }
    __syncthreads();
    for (int i = tid; i < m; i += WG) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            // reference rounds B_ij*t_j before the add — pinned, no fma
            const float pr = __fmul_rn(__half2float(b[(int64_t) i*r + j]), tv[j]);
            acc = __fadd_rn(acc, pr);
        }
        dst[obase + i] = has_acc ? acc_in[obase + i] + acc : acc;
    }
}

// shared: paw_v_reorder_kernel (defined once, reused across ops)
static __global__ void paw_v_reorder_kernel(
        const float * GGML_CUDA_RESTRICT y,   // [M, T], rows y_stride floats apart
        float       * GGML_CUDA_RESTRICT dst, // [M, T] packed
        const int M, const int T, const int y_stride,
        const int seg_off, const int hd, const int K, const int r) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) M*T;
    if (idx >= total) {
        return;
    }
    const int row = (int)(idx % M);
    const int t   = (int)(idx / M);
    int src_row = row;
    const int j = row - seg_off;
    if (j >= 0 && j < hd*K*r) {
        const int v   = j / (K*hd);
        const int rem = j % (K*hd);
        const int k   = rem / hd;
        const int d   = rem % hd;
        src_row = seg_off + (k*r + v)*hd + d;
    }
    dst[idx] = y[(int64_t) t*y_stride + src_row];
}

// shared: paw_moe_reduce_kernel (defined once, reused across ops)
static __global__ void paw_moe_reduce_kernel(
        const float * GGML_CUDA_RESTRICT experts,   // [n_embd, n_used, n_tok]
        const float * GGML_CUDA_RESTRICT weights,   // [1,      n_used, n_tok]
        float       * GGML_CUDA_RESTRICT dst,       // [n_embd, n_tok]
        const int n_embd, const int n_used) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n_embd) {
        return;
    }
    ggml_cuda_pdl_sync();

    const float * ebase = experts + (int64_t) t*n_used*n_embd + i;
    const float * wbase = weights + (int64_t) t*n_used;

    float acc = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        acc = __fadd_rn(acc, __fmul_rn(ebase[(int64_t) s*n_embd], wbase[s]));
    }
    dst[(int64_t) t*n_embd + i] = acc;
}
