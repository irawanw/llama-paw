#include "common.cuh"
#include "paw.cuh"
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

void ggml_cuda_op_paw_embed_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * lut   = dst->src[1];
    const ggml_tensor * ids   = dst->src[2];

    GGML_ASSERT(codes->type == GGML_TYPE_I8);
    GGML_ASSERT(lut->type   == GGML_TYPE_BF16);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(codes));
    GGML_ASSERT(ggml_is_contiguous(lut));
    GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int     n_embd = (int) dst->ne[0];
    const int64_t nt     = ids->ne[0];
    const int64_t total  = nt*n_embd;

    // group size comes from the LUT shape (64 on every shipped 35B payload,
    // 256 on paw-dense); it is a power of two, so index by shift
    const int64_t ng = lut->ne[1] / codes->ne[1];
    const int64_t group = n_embd / ng;
    GGML_ASSERT(group*ng == n_embd && (group & (group - 1)) == 0);
    int gsh = 0;
    while ((1 << gsh) < group) {
        ++gsh;
    }

    const dim3 grid((unsigned)((total + 255)/256), 1, 1);
    paw_launch(paw_embed_gather_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(256, 1, 1), 0, ctx.stream()),
        (const uint8_t  *) codes->data,
        (const uint16_t *) lut->data,
        (const int32_t  *) ids->data,
        (float          *) dst->data,
        n_embd, gsh, total);
}

//
// EMBED_ROWS — one thread per (token, group), serial 3-bit unpack
// (paw_embed_rows.comp)
//

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

void ggml_cuda_op_paw_embed_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * mn  = dst->src[1];
    const ggml_tensor * mx  = dst->src[2];
    const ggml_tensor * ids = dst->src[3];

    GGML_ASSERT(q->type   == GGML_TYPE_I8);
    GGML_ASSERT(mn->type  == GGML_TYPE_F16);
    GGML_ASSERT(mx->type  == GGML_TYPE_F16);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(mn));
    GGML_ASSERT(ggml_is_contiguous(mx));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_embd    = (int) dst->ne[0];
    const int ng        = (int) mn->ne[0];
    const int grp       = n_embd / ng;
    const int n_tokens  = (int) ids->ne[0];
    const int row_bytes = (int) q->ne[0];

    const dim3 grid((unsigned)((n_tokens*ng + 255)/256), 1, 1);
    paw_launch(paw_embed_rows_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(256, 1, 1), 0, ctx.stream()),
        (const uint8_t *) q->data,
        (const half    *) mn->data,
        (const half    *) mx->data,
        (const int32_t *) ids->data,
        (float         *) dst->data,
        n_embd, ng, grp, n_tokens, row_bytes);
}

//
// HEAD_MM — one thread per vocab row x 8-token tile (paw_head_mm.comp)
//

// head_mm with 4-way split-K per vocab row: each thread decodes a quarter of
// the row's 5-bit codes, then a shared reduction sums the partials. Same
// little-endian 5-bit decode (code j at bits [5j, 5j+5), gscale per 64-code
// group), same output layout. The split changes only the fp32 summation order
// (contractible per the file header). Grid (vocab/64, nt/8), 256 threads
// (64 rows x 4 splits). Enabled via GGML_PAW_HEAD_SPLITK=1.
static __global__ void paw_head_mm_splitk_kernel(
        const uint8_t * GGML_CUDA_RESTRICT qp,
        const half    * GGML_CUDA_RESTRICT gscale,
        const float   * GGML_CUDA_RESTRICT x,
        float         * GGML_CUDA_RESTRICT dst,
        const int n, const int vocab, const int n_tokens) {
    constexpr int SPLIT = 4;
    __shared__ float red[64][8][SPLIT];

    const int r  = blockIdx.x*64 + threadIdx.x / SPLIT;
    const int sk = threadIdx.x % SPLIT;
    const int jt0 = blockIdx.y*8;
    const int njt = min(8, n_tokens - jt0);

    if (r >= vocab) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;
    const int     per       = n / SPLIT;

    float accs[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accs[j] = 0.0f;
    }

    const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);
    const int b4_start = (sk*per) / 32;   // per is a multiple of 32 (n % SPLIT == 0)
    for (int b4i = b4_start; b4i < b4_start + per/32; ++b4i) {
        uint32_t w5[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w5[q] = pw[b4i*5 + q];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int bit0 = 5*i;
            const int wi   = bit0 >> 5;
            const int o    = bit0 & 31;
            uint32_t qv = w5[wi] >> o;
            if (o > 27) {
                qv |= w5[wi + 1] << (32 - o);
            }
            qv &= 31u;
            const int j = b4i*32 + i;
            const float w = (float)((int) qv - 16) * __half2float(gscale[(int64_t) r*ng + (j >> 6)]);
#pragma unroll
            for (int jt = 0; jt < 8; ++jt) {
                if (jt < njt) {
                    accs[jt] += w * x[(int64_t)(jt0 + jt)*n + j];
                }
            }
        }
    }

#pragma unroll
    for (int jt = 0; jt < 8; ++jt) {
        red[threadIdx.x / SPLIT][jt][sk] = accs[jt];
    }
    __syncthreads();
    if (sk == 0) {
#pragma unroll
        for (int jt = 0; jt < 8; ++jt) {
            if (jt < njt) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < SPLIT; ++k) {
                    s += red[threadIdx.x / SPLIT][jt][k];
                }
                dst[(int64_t)(jt0 + jt)*vocab + r] = s;
            }
        }
    }
}

static __global__ void paw_head_mm_kernel(
        const uint8_t * GGML_CUDA_RESTRICT qp,
        const half    * GGML_CUDA_RESTRICT gscale,
        const float   * GGML_CUDA_RESTRICT x,
        float         * GGML_CUDA_RESTRICT dst,
        const int n, const int vocab, const int n_tokens) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= vocab) {
        return;
    }
    ggml_cuda_pdl_sync();
    const int jt0 = blockIdx.y*8;
    const int njt = min(8, n_tokens - jt0);

    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;

    float accs[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accs[j] = 0.0f;
    }

    // the row is a little-endian 5-bit code stream (code j at bits [5j, 5j+5));
    // process 4 blocks (32 codes, 20 bytes) per iteration with 5 aligned
    // 32-bit loads instead of byte loads. Rows are 4-byte aligned: row_bytes =
    // n/8*5 and n % 64 == 0 (gscale groups), so row_bytes % 4 == 0. The
    // extracted code bits — and therefore all arithmetic — are identical to
    // the byte-wise unpack.
    const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);
    for (int b4i = 0; b4i < n/32; ++b4i) {
        uint32_t w5[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w5[q] = pw[b4i*5 + q];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int bit0 = 5*i;
            const int wi   = bit0 >> 5;
            const int o    = bit0 & 31;
            uint32_t qv = w5[wi] >> o;
            if (o > 27) {
                qv |= w5[wi + 1] << (32 - o);   // straddles a word boundary
            }
            qv &= 31u;
            const int j = b4i*32 + i;
            // reference decode: q_f32 * gscale_f32, one fp32 rounding
            const float w = (float)((int) qv - 16) * __half2float(gscale[(int64_t) r*ng + (j >> 6)]);
#pragma unroll
            for (int jt = 0; jt < 8; ++jt) {
                if (jt < njt) {
                    accs[jt] += w * x[(int64_t)(jt0 + jt)*n + j];
                }
            }
        }
    }
#pragma unroll
    for (int jt = 0; jt < 8; ++jt) {
        if (jt < njt) {
            dst[(int64_t)(jt0 + jt)*vocab + r] = accs[jt];
        }
    }
}

// --- fp8 (e5m2) runtime banks, GGML_PAW_BANK_FP8=1 ------------------------
// The stored weights (int16 trellis / int5 head) are untouched; only the
// runtime decode output is stored as e5m2 (1 byte/weight) instead of fp16,
// halving the bank bytes the per-token gemvs read. Software conversion for
// sm_86 (no native fp8). Authorized separately (stage3 report).
static bool paw_bank_fp8_on() {
    static const bool on = paw_env_int("GGML_PAW_BANK_FP8", 0) != 0;
    return on;
}

static bool paw_rt_bank_fp8_on() {
    static const bool on = paw_env_int("GGML_PAW_RT_BANK_FP8", paw_bank_fp8_on() ? 1 : 0) != 0;
    return on;
}

static bool paw_head_bank_fp8_on() {
    static const bool on = paw_env_int("GGML_PAW_HEAD_BANK_FP8", 1) != 0;
    return on;
}

static bool paw_rt_bank_idx_on() {
    static const bool on = paw_env_int("GGML_PAW_RT_BANK_IDX", 0) != 0;
    return on;
}

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

// --- head bank cache (same decode-once-forever pattern as paw_rt_bank_gemv,
// GGML_PAW_HEAD_CACHE=1) ---
//
// Unlike the trellis/expert codecs, int5g64_packed is a plain per-row
// affine quant (5-bit code + one fp16 gscale per 64-code group) -- no
// codebook/LUT, no routing/sparsity. The head matrix is dense and fully
// used every token, exactly the pattern paw_rt_bank_gemv already proved
// out (52.0 vs 47.6 tok/s single-stream, 2026-08-18 session): decode once,
// then a plain bandwidth GEMV. Reuses paw_rt_bank_gemv unchanged for the
// apply step -- x is already [nt, n] row-major and dst [nt, vocab] row-
// major, the exact layout that kernel expects with m=vocab. Bank size:
// vocab * n * 2 bytes (~1 GB for reason8192's ~248K vocab, n=2048).
static std::mutex paw_head_bank_mutex;
static std::unordered_map<const void *, const void *> paw_head_banks;

// defined later (RT_MM section); reused here unchanged for the head bank's
// apply step -- same [m,n]/[nt,n]/[nt,m] row-major GEMV shape.
static __global__ void paw_rt_bank_gemv(
        const half * GGML_CUDA_RESTRICT bank, const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v, const int m, const int n, const int nt);
static __global__ void paw_rt_bank_gemv_fp8(
        const uint8_t * GGML_CUDA_RESTRICT bank, const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v, const int m, const int n, const int nt);
static __global__ void paw_rt_bank_gemv_idx80(
        const uint16_t * GGML_CUDA_RESTRICT bank, const half * GGML_CUDA_RESTRICT tlut,
        const float * GGML_CUDA_RESTRICT scr_u, float * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n, const int nt);
// fp8 twin of paw_rt_bank_gemv: bank is e5m2 (1 byte/weight). Shared LUT
// built once per block for the 256 conversions. uint4 loads (4 bytes/weight
// per lane/iter) keep the kernel bandwidth-bound like the fp16 path.
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

// v3-pattern twin of paw_rt_bank_gemv_fp8: u staged in shared once per block
// (kills the 2-row-per-warp redundant global re-reads), two rows per warp
// with independent accumulators for FMA-chain ILP, uint4-width bank loads.
// Grid (m/16, 1, nt), 256 threads. Requires n % 4 == 0.
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

static bool paw_head_packed_on() {
    // measured 2.4x faster than the I16-code gemv on RTX 3060 decode
    static const bool on = paw_env_int("GGML_PAW_HEAD_PACKED", 1) != 0;
    return on;
}

static __device__ __forceinline__ uint32_t paw_head_code(const uint32_t * W, const int j) {
    const int bit = 5*j;
    const int wi = bit >> 5;
    const int off = bit & 31;
    uint32_t code = W[wi] >> off;
    if (off > 27) {
        code |= W[wi + 1] << (32 - off);
    }
    return code & 31u;
}

static __global__ void paw_head_packed_gemv_kernel(
        const uint8_t * GGML_CUDA_RESTRICT qp,
        const half    * GGML_CUDA_RESTRICT gscale,
        const float   * GGML_CUDA_RESTRICT x,
        float         * GGML_CUDA_RESTRICT dst,
        const int n, const int vocab, const int nt) {
    const int tid  = threadIdx.x;
    const int row  = blockIdx.x*8 + (tid >> 5);
    const int lane = tid & 31;
    const int t    = blockIdx.z;
    if (row >= vocab) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t row_words = (int64_t) n/32*5;
    const int ng = n/64;
    const uint32_t * W = (const uint32_t *) qp + (int64_t) row*row_words;
    const half * S = gscale + (int64_t) row*ng;
    const float2 * x2 = (const float2 *) (x + (int64_t) t*n);
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const float sf = __half2float(S[i >> 5]);
        const float w0 = __half2float(__float2half_rn(((int) paw_head_code(W, 2*i + 0) - 16)*sf));
        const float w1 = __half2float(__float2half_rn(((int) paw_head_code(W, 2*i + 1) - 16)*sf));
        const float2 xx = x2[i];
        acc += w0*xx.x + w1*xx.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        dst[(int64_t) t*vocab + row] = acc;
    }
}

// Multi-token twin of paw_head_packed_gemv_kernel. The original uses
// grid.z = nt, so every token re-reads AND re-unpacks the whole int5 head
// (~318 MB at vocab=248320, n=2048). The decoded weight does not depend on
// the token, so at nt>1 that work is repeated for nothing -- measured 0.954
// ms/call at nt=1 versus 7.98 ms/call at nt=8 (nsys, graphs off). Here each
// block unpacks the row once and accumulates NT token columns from it.
// Per-token accumulation order (lane stride, expression, warp reduction) is
// unchanged, so the logits are bit-identical to the single-token kernel.
// x is only nt*n floats (64 KB at nt=8), so the added activation reads stay
// L2-resident. GGML_PAW_HEAD_PACKED_MT=1 selects it.
template <int NT>
static __global__ void paw_head_packed_gemv_mt_kernel(
        const uint8_t * GGML_CUDA_RESTRICT qp,
        const half    * GGML_CUDA_RESTRICT gscale,
        const float   * GGML_CUDA_RESTRICT x,
        float         * GGML_CUDA_RESTRICT dst,
        const int n, const int vocab, const int nt) {
    const int tid  = threadIdx.x;
    const int row  = blockIdx.x*8 + (tid >> 5);
    const int lane = tid & 31;
    const int t0   = blockIdx.z*NT;
    if (row >= vocab) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t row_words = (int64_t) n/32*5;
    const int ng = n/64;
    const uint32_t * W = (const uint32_t *) qp + (int64_t) row*row_words;
    const half * S = gscale + (int64_t) row*ng;
    const int n2 = n/2;

    float acc[NT];
#pragma unroll
    for (int u = 0; u < NT; ++u) {
        acc[u] = 0.0f;
    }

    for (int i = lane; i < n2; i += 32) {
        const float sf = __half2float(S[i >> 5]);
        const float w0 = __half2float(__float2half_rn(((int) paw_head_code(W, 2*i + 0) - 16)*sf));
        const float w1 = __half2float(__float2half_rn(((int) paw_head_code(W, 2*i + 1) - 16)*sf));
#pragma unroll
        for (int u = 0; u < NT; ++u) {
            const int t = t0 + u;
            if (NT == 1 || t < nt) {
                const float2 xx = ((const float2 *) (x + (int64_t) t*n))[i];
                acc[u] += w0*xx.x + w1*xx.y;
            }
        }
    }

#pragma unroll
    for (int u = 0; u < NT; ++u) {
        const float s = warp_reduce_sum<32>(acc[u]);
        const int t = t0 + u;
        if (lane == 0 && (NT == 1 || t < nt)) {
            dst[(int64_t) t*vocab + row] = s;
        }
    }
}

static bool paw_head_packed_mt_on() {
    static const bool on = paw_env_int("GGML_PAW_HEAD_PACKED_MT", 1) != 0;
    return on;
}

static bool paw_head_cache_on() {
    static const bool on = paw_env_int("GGML_PAW_HEAD_CACHE", 0) != 0;
    return on;
}

// one thread per 32-code group (matches the packed layout's 4x uint32 =
// 160-bit = 32 x 5-bit chunks); bit-unpack identical to paw_head_mm_kernel.
static __global__ void paw_head_bank_decode_kernel(
        const uint8_t * GGML_CUDA_RESTRICT qp,
        const half    * GGML_CUDA_RESTRICT gscale,
        half          * GGML_CUDA_RESTRICT bank,   // [vocab, n]
        const int n, const int vocab) {
    // vocab (~248K) exceeds CUDA's 65535 grid.y/z limit -- must be grid.x.
    const int r   = blockIdx.x;
    const int b4i = blockIdx.y*blockDim.x + threadIdx.x;
    if (r >= vocab || b4i*32 >= n) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;
    const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);

    uint32_t w5[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w5[q] = pw[b4i*5 + q];
    }
    const float gsf = __half2float(gscale[(int64_t) r*ng + ((b4i << 5) >> 6)]);
    half * dst = bank + (int64_t) r*n + b4i*32;

#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int bit0 = 5*i;
        const int wi   = bit0 >> 5;
        const int o    = bit0 & 31;
        uint32_t qv = w5[wi] >> o;
        if (o > 27) {
            qv |= w5[wi + 1] << (32 - o);
        }
        qv &= 31u;
        const float w = (float)((int) qv - 16) * gsf;
        dst[i] = __float2half_rn(w);
    }
}

// fp8 twin of paw_head_bank_decode_kernel: output stored as e5m2 (1 byte)
// instead of fp16.
static __global__ void paw_head_bank_decode_kernel_fp8(
        const uint8_t * GGML_CUDA_RESTRICT qp,
        const half    * GGML_CUDA_RESTRICT gscale,
        uint8_t       * GGML_CUDA_RESTRICT bank,   // [vocab, n]
        const int n, const int vocab) {
    const int r   = blockIdx.x;
    const int b4i = blockIdx.y*blockDim.x + threadIdx.x;
    if (r >= vocab || b4i*32 >= n) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t row_bytes = (int64_t) n/8*5;
    const int     ng        = n/64;
    const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);

    uint32_t w5[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w5[q] = pw[b4i*5 + q];
    }
    const float gsf = __half2float(gscale[(int64_t) r*ng + ((b4i << 5) >> 6)]);
    uint8_t * dst = bank + (int64_t) r*n + b4i*32;

#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int bit0 = 5*i;
        const int wi   = bit0 >> 5;
        const int o    = bit0 & 31;
        uint32_t qv = w5[wi] >> o;
        if (o > 27) {
            qv |= w5[wi + 1] << (32 - o);
        }
        qv &= 31u;
        const float w = (float)((int) qv - 16) * gsf;
        dst[i] = paw_f32_to_e5m2(w);
    }
}

static const void * paw_head_bank_get(
        const void * qp, const void * gscale, const int n, const int vocab, cudaStream_t stream) {
    const bool fp8 = paw_head_bank_fp8_on();
    {
        std::lock_guard<std::mutex> lock(paw_head_bank_mutex);
        auto it = paw_head_banks.find(qp);
        if (it != paw_head_banks.end()) {
            return it->second;
        }
    }
    void * bank = nullptr;
    CUDA_CHECK(cudaMalloc(&bank, (size_t) vocab*n*(fp8 ? 1 : (int) sizeof(half))));
    const int b4_per_row = (n + 31) / 32;
    if (fp8) {
        paw_launch(paw_head_bank_decode_kernel_fp8,
            ggml_cuda_kernel_launch_params(dim3(vocab, (b4_per_row + 127)/128, 1), dim3(128, 1, 1), 0, stream),
            (const uint8_t *) qp, (const half *) gscale, (uint8_t *) bank, n, vocab);
    } else {
        paw_launch(paw_head_bank_decode_kernel,
            ggml_cuda_kernel_launch_params(dim3(vocab, (b4_per_row + 127)/128, 1), dim3(128, 1, 1), 0, stream),
            (const uint8_t *) qp, (const half *) gscale, (half *) bank, n, vocab);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        std::lock_guard<std::mutex> lock(paw_head_bank_mutex);
        auto it = paw_head_banks.find(qp);
        if (it != paw_head_banks.end()) {
            cudaFree(bank);
            return it->second;
        }
        paw_head_banks.emplace(qp, bank);
    }
    return bank;
}

void ggml_cuda_op_paw_head_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * qp     = dst->src[0];
    const ggml_tensor * gscale = dst->src[1];
    const ggml_tensor * x      = dst->src[2];

    GGML_ASSERT(qp->type     == GGML_TYPE_I8);
    GGML_ASSERT(gscale->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(qp));
    GGML_ASSERT(ggml_is_contiguous(gscale));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n     = (int) x->ne[0];
    const int vocab = (int) qp->ne[1];
    const int nt    = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    char shp[64];
    snprintf(shp, sizeof(shp), " n=%d vocab=%d nt=%d", n, vocab, nt);

    if (paw_head_packed_on()) {
        if (paw_head_packed_mt_on() && nt > 1) {
            const int NT = nt >= 8 ? 8 : (nt >= 4 ? 4 : 2);
            paw_timed(ctx.stream(), std::string("head_packed_gemv_mt") + shp, [&]() {
            const dim3 g((vocab + 7)/8, 1, (unsigned)((nt + NT - 1)/NT));
            const dim3 b(256, 1, 1);
            if (NT == 8) {
                paw_launch(paw_head_packed_gemv_mt_kernel<8>,
                    ggml_cuda_kernel_launch_params(g, b, 0, ctx.stream()),
                    (const uint8_t *) qp->data, (const half *) gscale->data,
                    (const float *) x->data, (float *) dst->data, n, vocab, nt);
            } else if (NT == 4) {
                paw_launch(paw_head_packed_gemv_mt_kernel<4>,
                    ggml_cuda_kernel_launch_params(g, b, 0, ctx.stream()),
                    (const uint8_t *) qp->data, (const half *) gscale->data,
                    (const float *) x->data, (float *) dst->data, n, vocab, nt);
            } else {
                paw_launch(paw_head_packed_gemv_mt_kernel<2>,
                    ggml_cuda_kernel_launch_params(g, b, 0, ctx.stream()),
                    (const uint8_t *) qp->data, (const half *) gscale->data,
                    (const float *) x->data, (float *) dst->data, n, vocab, nt);
            }
            });
            return;
        }
        paw_timed(ctx.stream(), std::string("head_packed_gemv") + shp, [&]() {
        paw_launch(paw_head_packed_gemv_kernel,
            ggml_cuda_kernel_launch_params(dim3((vocab + 7)/8, 1, nt), dim3(256, 1, 1), 0, ctx.stream()),
            (const uint8_t *) qp->data, (const half *) gscale->data,
            (const float *) x->data, (float *) dst->data, n, vocab, nt);
        });
        return;
    }

    if (paw_head_cache_on()) {
        const void * bank = paw_head_bank_get(qp->data, gscale->data, n, vocab, ctx.stream());
        paw_timed(ctx.stream(), std::string("head_bank_gemv") + shp, [&]() {
        if (paw_head_bank_fp8_on()) {
            paw_launch(paw_rt_bank_gemv_fp8,
                ggml_cuda_kernel_launch_params(dim3((vocab + 7)/8, 1, nt), dim3(256, 1, 1), 0, ctx.stream()),
                (const uint8_t *) bank, (const float *) x->data, (float *) dst->data, vocab, n, nt);
        } else {
            paw_launch(paw_rt_bank_gemv,
                ggml_cuda_kernel_launch_params(dim3((vocab + 7)/8, 1, nt), dim3(256, 1, 1), 0, ctx.stream()),
                (const half *) bank, (const float *) x->data, (float *) dst->data, vocab, n, nt);
        }
        });
        return;
    }

    static const bool head_splitk = paw_env_int("GGML_PAW_HEAD_SPLITK", 1) != 0;
    const dim3 grid((unsigned)((vocab + 127)/128), (unsigned)((nt + 7)/8), 1);
    paw_timed(ctx.stream(), std::string("head_mm") + shp, [&]() {
    if (head_splitk && n % 4 == 0 && vocab % 64 == 0) {
        paw_launch(paw_head_mm_splitk_kernel,
            ggml_cuda_kernel_launch_params(dim3(vocab/64, (nt + 7)/8, 1), dim3(256, 1, 1), 0, ctx.stream()),
            (const uint8_t *) qp->data,
            (const half    *) gscale->data,
            (const float   *) x->data,
            (float         *) dst->data,
            n, vocab, nt);
    } else {
        paw_launch(paw_head_mm_kernel,
            ggml_cuda_kernel_launch_params(grid, dim3(128, 1, 1), 0, ctx.stream()),
            (const uint8_t *) qp->data,
            (const half    *) gscale->data,
            (const float   *) x->data,
            (float         *) dst->data,
            n, vocab, nt);
    }
    });
}

//
// NE_MM — one thread per output row x 8-token tile, serial L=12 walk
// (paw_ne_mm.comp)
//

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

void ggml_cuda_op_paw_ne_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * packed = dst->src[0];
    const ggml_tensor * gscale = dst->src[1];
    const ggml_tensor * lut    = dst->src[2];
    const ggml_tensor * x      = dst->src[3];

    GGML_ASSERT(packed->type == GGML_TYPE_I8);
    GGML_ASSERT(gscale->type == GGML_TYPE_F16);
    GGML_ASSERT(lut->type    == GGML_TYPE_F16);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(packed));
    GGML_ASSERT(ggml_is_contiguous(gscale));
    GGML_ASSERT(ggml_is_contiguous(lut));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int T   = (int) x->ne[0];
    const int B   = (int) packed->ne[1];
    const int k   = (int)(packed->ne[0]*8 / T);
    const int ng  = (int) gscale->ne[0];
    const int rpc = (int)(B / lut->ne[1]);
    const int nt  = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    const dim3 grid((unsigned)((B + 127)/128), (unsigned)((nt + 7)/8), 1);
    char shp[64];
    snprintf(shp, sizeof(shp), " B=%d T=%d k=%d nt=%d", B, T, k, nt);
    paw_timed(ctx.stream(), std::string("ne_mm") + shp, [&]() {
    paw_launch(paw_ne_mm_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(128, 1, 1), 0, ctx.stream()),
        (const uint8_t *) packed->data,
        (const half    *) gscale->data,
        (const half    *) lut->data,
        (const float   *) x->data,
        (float         *) dst->data,
        B, T, k, ng, rpc, nt);
    });
}

//
// RT_MM — 3 kernels: u = H(su ⊙ x) per token -> K4 V2 trellis walk ->
// y = sv ⊙ H(v) (paw_rt_u/rt_walk/rt_out.comp). Scratch: u [nt, n] then
// v [nt, m] in one pool allocation.
//

// WG is a template knob (GGML_PAW_FWHT_WG): every loop strides by WG and
// each FWHT pass is elementwise, so the results are bit-identical for any WG.
template <int WG>
static __global__ void paw_rt_u_kernel(
        const float * GGML_CUDA_RESTRICT su,
        const float * GGML_CUDA_RESTRICT x,
        float       * GGML_CUDA_RESTRICT scr_u,
        const int n,
        const int blk) {
    // blk == n is the legacy single-Hadamard path: the loop runs once with
    // off == 0 and the arithmetic is identical to before blocking existed.
    // blockIdx.y selects the rotation chunk so chunks run on separate SMs
    // instead of serializing inside one block (per-chunk math is unchanged).
    __shared__ float sh[4096];

    const int t   = blockIdx.x;
    const int off = blockIdx.y * blk;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float sc     = __fsqrt_rn((float) blk);
    const float inv_sc = __frcp_rn(sc);
    {
        for (int i = tid; i < blk; i += WG) {
            sh[i] = su[off + i] * x[(int64_t) t*n + off + i];
        }
        __syncthreads();
        if (WG == blk/16 && paw_fwht_v2_ok(blk)) {
            paw_fwht_block_v2(sh, blk, tid, WG);
        } else {
            paw_fwht_block(sh, blk, tid, WG);
        }
        for (int i = tid; i < blk; i += WG) {
            scr_u[(int64_t) t*n + off + i] = sh[i] * inv_sc;
        }
    }
}

// WG covers one thread per column tile (tid < tiles_y walk); WG=128 avoids
// launching idle warps when tiles_y <= 128 (n <= 2048). The dropped warps
// only ever contributed exact zeros to the cross-warp reduction.
// WORDS is the trellis rate in int16 words per 16x16 tile: bits-per-tile is
// WORDS*16 over 128 states, so step = WORDS/8 fresh bits per state and the rate
// is K = step/V = WORDS/16 at V=2. 64 -> K=4 (the shipped 35B payload),
// 48 -> K=3, 32 -> K=2, 24 -> K=1.5, 16 -> K=1.
//
// Only WORDS=64 gives byte-aligned state windows, and only 16/32/64 give
// word-aligned tile rows, so the window is extracted by global bit offset
// rather than from a per-row register cache. At WORDS=64 that folds back to
// exactly the hand-unrolled arithmetic this replaced.
template <int WG, int WORDS>
static __global__ void paw_rt_walk_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][8];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float slut[1024];   // full F16 [2,512] tlut, staged as fp32

    const int t   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    static_assert(WORDS % 8 == 0 && WORDS >= 16 && WORDS <= 64, "unsupported trellis rate");
    constexpr int step  = WORDS/8;   // K*V fresh bits per state, V = 2
    constexpr int words = WORDS;     // int16 words per 16x16 tile
    const int tiles_y = n / 16;

    ggml_cuda_pdl_sync();

    // 2 KB table: gather from shared instead of global (same fp32 values —
    // the __half2float just moves ahead of the gather)
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = __half2float(tlut[i]);
    }
    __syncthreads();

    float partial[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        partial[i] = 0.0f;
    }

    // One thread per tile column, strided so n is not capped at WG*16. This
    // used to be a bare `if (tid < tiles_y)`, which silently dropped every
    // column past 4096 -- invisible until now because K=4 takes the bank path
    // and the walk only ran on the narrow GDN projections. For tiles_y <= WG
    // the loop runs exactly once at c == tid, so the arithmetic is unchanged.
    for (int c = tid; c < tiles_y; c += WG) {
        const int64_t tw = ((int64_t) tr*tiles_y + c)*words;
        const float * ub = scr_u + (int64_t) t*n + c*16;

        // the tile column's 16 u values are shared by all 16 tile rows
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }

        for (int ri = 0; ri < 16; ++ri) {
            // 8 states per tile row, each a 16-bit window starting at bit
            // step*(8*ri + j) of the tile's MSB-first stream, wrapping at the end
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int b   = step*(8*ri + j);
                const int wi  = (b >> 4) % words;
                const int off = b & 15;
                const uint32_t w0 = trellis[tw + wi];
                const uint32_t st = off == 0 ? w0
                    : (((w0 << off) | (trellis[tw + (wi + 1) % words] >> (16 - off)))
                       & 0xFFFFu);
                ph[j] = st*(st + 1u);
            }
            // same accumulation order as the serial walk: per state, w0 then w1
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                float a0 = slut[2*row + 0];
                const float a1 = slut[2*row + 1];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;                             // exact in fp16 (sign bit)
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[ri] += acc;
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s = warp_reduce_sum<warp_size>(partial[i]);
        if (lane == 0) {
            red[i][wid] = s;
        }
    }
    __syncthreads();
    if (tid < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < n_warps; ++wj) {
            sum += red[tid][wj];
        }
        scr_v[(int64_t) t*m + tr*16 + tid] = sum;
    }
}

// --- QTIP-style tensor-core walk: decode-from-compressed-trellis fused
// with mma.sync accumulate, GGML_PAW_RT_WALK_QTIP=1, nt==1 only ---
//
// rt_bank_gemv (the current default at nt<4) reads a pre-decoded fp16 bank
// (2 bytes/weight) -- confirmed memory-bandwidth-bound (~590 GB/s
// effective). The earlier rt_apply_kernel_mma attempts (both reverted,
// see "Already tried") tried to speed up compute on top of THAT same
// bandwidth-bound bank read -- Amdahl's law says that can't win: if a
// kernel is bandwidth-bound, faster compute doesn't help, you still stall
// on the same DRAM traffic. That's the likely reason both attempts came
// back negative regardless of design.
//
// This is a different bet: read the COMPRESSED trellis directly (K4 =
// 4 bits/weight = 0.5 bytes -- 4x less than the fp16 bank) and decode
// on the fly, fused with the matvec via raw-PTX mma.sync (not wmma --
// wmma has no m16n8k16 fp16 shape, confirmed by an earlier failed build
// attempt; QTIP's own reference kernel uses this exact raw-PTX shape for
// the same reason). Even wasting 7/8 of the N=8 MMA tile (nt=1 padded)
// should still win on bandwidth alone, since tensor-core throughput has
// enormous headroom over what's needed here -- the goal is fewer DRAM
// bytes, not faster FMA.
//
// The A/B (m16n8k16) fragment-to-lane mapping below was verified against
// a CPU reference in an isolated standalone test before being written
// here (not just recalled from the PTX ISA docs from memory) -- see
// mma_layout_test.cu in the session scratch dir. Decode math (per-row
// state walk, tlut lookup, sign flip) is copied verbatim from
// paw_rt_walk_kernel above, just reorganized: one lane decodes one full
// row (16 columns at once, same j=0..7 loop) into shared memory, then all
// 32 lanes read out the specific (row,col) pairs mma.sync needs for their
// fragment slot -- necessary because decode naturally produces "one row,
// all columns" per step, while the MMA fragment layout needs "one lane,
// two arbitrary (row,col) pairs" -- these are different axes, so a
// shared-memory handoff (not a redistribution I can avoid) sits between
// decode and mma. One block = one warp = one 16-row output tile; grid.x =
// m/16 (matches rt_bank_gemv's block count, avoiding the v1 occupancy
// mistake from the rt_apply attempts).
// rate-templated twin of paw_rt_walk_qtip_kernel: same mma.sync epilogue,
// but the per-row state windows come from the generic bit-stream math
// (paw_rt_walk_kernel) instead of the K4-only (4*ri+q)&63 wrap, so the
// dense payload's K1/K1.5 rates get the register-walk fast path too.
template <int WORDS>
static __global__ void paw_rt_walk_qtip_rate_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16, [2,512]
        const float    * GGML_CUDA_RESTRICT scr_u,    // [nt=1, n]
        float          * GGML_CUDA_RESTRICT scr_v,    // [nt=1, m]
        const int m, const int n) {
    // stride-3 padding: a packed [512][2] float table makes every warp's
    // random row gather land on even banks only (2+ way conflicts); with a
    // stride of 3 the bank index (3*row)%32 covers all 32 banks uniformly
    __shared__ float  slut[512*3];
    // per-warp decode scratch: each warp owns an independent output tile
    __shared__ half   tile16[8][16][16];
    __shared__ half   bcol[8][16];   // this K-step's 16 activation values (col 0 real, rest 0)

    constexpr int step  = WORDS/8;   // fresh bits per state, V = 2
    // one BLOCK per output tile; four warps SPLIT the tile-column range so
    // small-m matrices (down proj: only m/16 tiles) still fill the machine
    // -- the walk is latency-bound, memory-level parallelism is everything
    const int wid  = threadIdx.x >> 5;
    const int tr   = blockIdx.x;
    const int lane = threadIdx.x & 31;

    const int tiles_y = n / 16;

    ggml_cuda_pdl_sync();

    for (int i = threadIdx.x; i < 512; i += 256) {
        slut[3*i + 0] = __half2float(tlut[2*i + 0]);
        slut[3*i + 1] = __half2float(tlut[2*i + 1]);
    }
    __syncthreads();   // all four warps share this table

    const int groupID = lane >> 2;
    const int tid4     = lane & 3;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // the per-lane state windows (bit offsets into the tile's stream) do not
    // depend on ct -- only the tile base word moves -- so the trellis loads
    // are software-pipelined one ct iteration ahead: issue iteration ct+8's
    // loads before consuming iteration ct's, giving them a full MMA-plus-
    // decode of latency slack instead of stalling the loop every step
    const int ri = lane >> 1;
    const int jb = (lane & 1)*4;
    int      wi0[4], wi1[4], offv[4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        const int b = step*(8*ri + jb + jj);
        wi0[jj]  = (b >> 4) % WORDS;
        offv[jj] = b & 15;
        wi1[jj]  = (wi0[jj] + 1) % WORDS;
    }
    uint32_t c0[4], c1[4];
    float    ucur = 0.f;
    {
        const int64_t tw = ((int64_t) tr*tiles_y + wid)*WORDS;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            c0[jj] = trellis[tw + wi0[jj]];
            c1[jj] = offv[jj] != 0 ? trellis[tw + wi1[jj]] : 0u;
        }
        if (lane < 16) {
            ucur = scr_u[wid*16 + lane];
        }
    }

    // gridDim.y > 1: several blocks SHARE one output tile, each walking a
    // strided subset of the tile columns (the serial chain per warp shrinks
    // by gridDim.y); partials are atomic-added into scr_v
    for (int ct = wid + 8*blockIdx.y; ct < tiles_y; ct += 8*gridDim.y) {
        // --- prefetch this lane's windows for the NEXT ct this warp owns ---
        const int ctn = ct + 8*gridDim.y;
        uint32_t n0[4], n1[4];
        float    unext = 0.f;
        if (ctn < tiles_y) {
            const int64_t twn = ((int64_t) tr*tiles_y + ctn)*WORDS;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                n0[jj] = trellis[twn + wi0[jj]];
                n1[jj] = offv[jj] != 0 ? trellis[twn + wi1[jj]] : 0u;
            }
            if (lane < 16) {
                unext = scr_u[ctn*16 + lane];
            }
        }
        // --- decode this (tr, ct) 16x16 tile: lanes 0-15 each decode one
        // full row (16 columns), generic rate-aware window math ---
        {
            // all 32 lanes decode: two lanes per row, four states each --
            // halves the serial chain vs one lane doing the whole row
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j   = jb + jj;
                const int off = offv[jj];
                const uint32_t w0 = c0[jj];
                const uint32_t st = off == 0 ? w0
                    : (((w0 << off) | (c1[jj] >> (16 - off)))
                       & 0xFFFFu);
                const uint32_t ph = st*(st + 1u);
                const uint32_t row = (ph >> 6) & 511u;
                float a0 = slut[3*row + 0];
                const float a1 = slut[3*row + 1];
                if (ph & 0x8000u) {
                    a0 = -a0;
                }
                tile16[wid][ri][2*j + 0] = __float2half_rn(a0);
                tile16[wid][ri][2*j + 1] = __float2half_rn(a1);
            }
        }
        // --- stage this K-step's 16 activation values (nt==1: col 0 real) ---
        if (lane < 16) {
            bcol[wid][lane] = __float2half_rn(ucur);   // nt==1: one shared u vector
        }
        __syncwarp();

        // --- build A/B fragments per the validated m16n8k16 layout ---
        auto Aat = [&](int r, int c) -> half { return tile16[wid][r][c]; };
        half2 a01 = __halves2half2(Aat(groupID,   tid4*2+0), Aat(groupID,   tid4*2+1));
        half2 a23 = __halves2half2(Aat(groupID+8, tid4*2+0), Aat(groupID+8, tid4*2+1));
        half2 a45 = __halves2half2(Aat(groupID,   tid4*2+8), Aat(groupID,   tid4*2+9));
        half2 a67 = __halves2half2(Aat(groupID+8, tid4*2+8), Aat(groupID+8, tid4*2+9));
        uint32_t ra0 = *(uint32_t*)&a01;
        uint32_t ra1 = *(uint32_t*)&a23;
        uint32_t ra2 = *(uint32_t*)&a45;
        uint32_t ra3 = *(uint32_t*)&a67;

        // B is [16,1] logically (nt=1, this K-step's 16 real activation
        // values), broadcast to all 8 MMA columns by reading the SAME bcol
        // regardless of groupID -- since B doesn't vary by column, every
        // output column ends up numerically identical, so any one of them
        // (tid4==0 below) is a valid read of the real result. Wasteful
        // (computes the same dot product 8x redundantly across tid4) but
        // not incorrect, and simpler than truly zero-padding 7 columns.
        half2 b01 = __halves2half2(bcol[wid][tid4*2+0], bcol[wid][tid4*2+1]);
        half2 b23 = __halves2half2(bcol[wid][tid4*2+8], bcol[wid][tid4*2+9]);
        uint32_t rb0 = *(uint32_t*)&b01;
        uint32_t rb1 = *(uint32_t*)&b23;

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1)
        );
        __syncwarp();   // tile16/bcol reused by the next ct

        // slide the prefetched windows down for the next iteration
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            c0[jj] = n0[jj];
            c1[jj] = n1[jj];
        }
        ucur = unext;
    }

    // only tid4==0 (groupID's col 0 slot) is the real token; d0 -> row
    // groupID, d2 -> row groupID+8 (per the validated D-fragment layout).
    // eight warps covered disjoint ct subsets -- reduce through shared
    __shared__ float xred[8][16];
    if (tid4 == 0) {
        xred[wid][groupID]     = d0;
        xred[wid][groupID + 8] = d2;
    }
    __syncthreads();
    if (wid == 0 && tid4 == 0) {
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float sum = xred[0][r] + xred[1][r] + xred[2][r] + xred[3][r]
                            + xred[4][r] + xred[5][r] + xred[6][r] + xred[7][r];
            if (gridDim.y == 1) {
                scr_v[(int64_t) blockIdx.z*m + tr*16 + r] = sum;
            } else {
                atomicAdd(&scr_v[(int64_t) blockIdx.z*m + tr*16 + r], sum);
            }
        }
    }
}


// fragment-direct qtip walk: each lane decodes EXACTLY its own mma.sync
// fragment elements straight out of the trellis stream -- the four states a
// lane needs are fixed by the m16n8k16 layout ((row=g,g+8) x (j=tid4,tid4+4)),
// so no shared-memory staging, no syncwarp in the loop, no cross-lane
// exchange at all. Verified bit-exact against paw_rt_walk_qtip_rate_kernel
// in an isolated probe before landing here.
template <int WORDS>
static __global__ void paw_rt_walk_qtip_frag_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n) {
    constexpr int step = WORDS/8;
    const int wid  = threadIdx.x >> 5;
    const int tr   = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int tiles_y = n / 16;
    const int groupID = lane >> 2;
    const int tid4    = lane & 3;

    const int rrows[2] = {groupID, groupID + 8};
    const int rjs[2]   = {tid4, tid4 + 4};
    int wi0[2][2], wi1[2][2], offv[2][2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int b = step*(8*rrows[k] + rjs[q]);
            wi0[k][q]  = (b >> 4) % WORDS;
            offv[k][q] = b & 15;
            wi1[k][q]  = (wi0[k][q] + 1) % WORDS;
        }

    // B fragment: this K-step's (== this tile column's) 16 activation
    // values; the lane needs k in {tid4*2,+1, tid4*2+8,+9}
    auto load_b = [&](int ct, half2& h0, half2& h1) {
        const float * ub = scr_u + ct*16;
        h0 = __floats2half2_rn(ub[tid4*2 + 0], ub[tid4*2 + 1]);
        h1 = __floats2half2_rn(ub[tid4*2 + 8], ub[tid4*2 + 9]);
    };
    half2 hb0 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
    half2 hb1 = hb0;
    if (wid < tiles_y) load_b(wid, hb0, hb1);
    const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;

    // WORDS==16 packed path: a lane's four states live in just two unit
    // pairs ((ri,ri+1) for ri = g and g+8), so each pair is fetched once
    // per column -- one aligned u32 load when the pair does not wrap --
    // instead of eight separate u16 gathers.
    auto fetch_pair = [&](int64_t tw, int u, uint32_t& lo_u, uint32_t& hi_u) {
        const int v = (u + 1) % WORDS;
        if ((u & 1) == 0 && v == u + 1) {
            const uint32_t W = *(const uint32_t*)(const void*)(trellis + tw + u);
            lo_u = W & 0xFFFFu;
            hi_u = W >> 16;
        } else {
            lo_u = __ldg(trellis + tw + u);
            hi_u = __ldg(trellis + tw + v);
        }
    };
    auto dec_state = [&](uint32_t lo, uint32_t hi, int off) {
        const uint32_t st = off == 0 ? lo
            : (((lo << off) | (hi >> (16 - off))) & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const uint32_t pair = *(const uint32_t*)(const void*)(tlut + 2*row);
        return (ph & 0x8000u) ? (pair ^ 0x00008000u) : pair;
    };

    auto load_at = [&](int64_t tw, uint32_t (&a)[2][2], uint32_t (&b)[2][2]) {
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                a[k][q] = __ldg(trellis + tw + wi0[k][q]);
                b[k][q] = offv[k][q] ? __ldg(trellis + tw + wi1[k][q]) : 0u;
            }
    };
    // gridDim.y > 1: several blocks share an output tile, each walking a
    // strided subset of its columns (small-m matrices otherwise leave the
    // machine idle); partials fold through the atomicAdd epilogue
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const int ct_first = wid + 8*(int) blockIdx.y;

    if constexpr (WORDS == 16) {
        uint32_t L0, H0, L8, H8, nL0, nH0, nL8, nH8;
        L0 = H0 = L8 = H8 = nL0 = nH0 = nL8 = nH8 = 0;
        const bool has = wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + 7)/8);
        if (has) {
            const int64_t tw = (int64_t) tr*tiles_y*WORDS + ct_first*WORDS;
            fetch_pair(tw, groupID, L0, H0);
            fetch_pair(tw, (groupID + 8) % WORDS, L8, H8);
        }
        for (int ct = ct_first; ct < tiles_y; ct += 8*gridDim.y) {
            const int ctn = ct + 8*gridDim.y;
            if (ctn < tiles_y) {
                const int64_t twn = (int64_t) tr*tiles_y*WORDS + ctn*WORDS;
                fetch_pair(twn, groupID, nL0, nH0);
                fetch_pair(twn, (groupID + 8) % WORDS, nL8, nH8);
            }
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            // ra order: [k + 2*q]; k=row-group (g,g+8), q=j-pair (t,t+4)
            const uint32_t ra0 = dec_state(L0, H0, offv[0][0]);
            const uint32_t ra1 = dec_state(L8, H8, offv[1][0]);
            const uint32_t ra2 = dec_state(L0, H0, offv[0][1]);
            const uint32_t ra3 = dec_state(L8, H8, offv[1][1]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
            if (ctn < tiles_y) { L0=nL0; H0=nH0; L8=nL8; H8=nH8; }
            hb0 = nb0; hb1 = nb1;
        }
    } else if constexpr (WORDS == 24) {
        // w24: state (ri, j=t) starts at bit 24*ri + 3t; the j=t+4 state
        // starts 12 bits later -- same unit pair unless that crosses into
        // the next unit (then one extra unit covers it). Two fetch_pairs
        // plus occasional single units replace eight per-state gathers.
        uint32_t L0, H0, X0, L8, H8, X8, nL0, nH0, nX0, nL8, nH8, nX8;
        uint32_t o0[2], o2[2]; int u0[2], u2[2];
        L0=H0=X0=L8=H8=X8=nL0=nH0=nX0=nL8=nH8=nX8=0;
        const bool has = wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + 7)/8);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int bk = 3*(8*rrows[k] + tid4);
            u0[k]  = (bk >> 4) % WORDS;
            o0[k]  = bk & 15u;
            const int cross = (o0[k] + 12) >> 4;   // 0 or 1
            u2[k]  = (u0[k] + cross) % WORDS;
            o2[k]  = (o0[k] + 12) & 15u;
        }
        if (has) {
            const int64_t tw = (int64_t) tr*tiles_y*WORDS + ct_first*WORDS;
            fetch_pair(tw, u0[0], L0, H0);
            fetch_pair(tw, u0[1], L8, H8);
            X0 = __ldg(trellis + tw + ((u0[0] + 2) % WORDS));
            X8 = __ldg(trellis + tw + ((u0[1] + 2) % WORDS));
            (void) u2; (void) o2;
        }
        for (int ct = ct_first; ct < tiles_y; ct += 8*gridDim.y) {
            const int ctn = ct + 8*gridDim.y;
            if (ctn < tiles_y) {
                const int64_t twn = (int64_t) tr*tiles_y*WORDS + ctn*WORDS;
                fetch_pair(twn, u0[0], nL0, nH0);
                fetch_pair(twn, u0[1], nL8, nH8);
                nX0 = __ldg(trellis + twn + ((u0[0] + 2) % WORDS));
                nX8 = __ldg(trellis + twn + ((u0[1] + 2) % WORDS));
            }
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t lo = k ? L8 : L0, hi = k ? H8 : H0;
                const uint32_t xx = k ? X8 : X0;
                // j=t state: units (lo,hi) at offset o0[k]
                ra[k + 2*0] = dec_state(lo, hi, o0[k]);
                // j=t+4 state: starts 12 bits later
                ra[k + 2*1] = ((o0[k] + 12) >> 4)
                    ? dec_state(hi, xx, o2[k])     // crossed into next unit
                    : dec_state(lo, hi, o2[k]);    // still inside the pair
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
            if (ctn < tiles_y) {
                L0=nL0; H0=nH0; X0=nX0; L8=nL8; H8=nH8; X8=nX8;
            }
            hb0 = nb0; hb1 = nb1;
        }
    } else if constexpr (WORDS == 32) {
        // w32: state (ri, j) starts at bit 32*ri + 4*j -> unit
        // 2*ri + j/4, offset 4*(j&3). The j=t+4 state lives one unit
        // later at the SAME offset, so each row-group needs units
        // {u, u+1, u+2}: one fetch_pair plus a single extra unit.
        uint32_t L0, H0, X0, L8, H8, X8, nL0, nH0, nX0, nL8, nH8, nX8;
        uint32_t ov[2]; int uv[2];
        L0=H0=X0=L8=H8=X8=nL0=nH0=nX0=nL8=nH8=nX8=0;
        const bool has = wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + 7)/8);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int b = 4*(8*rrows[k] + tid4);
            uv[k] = (b >> 4) % WORDS;
            ov[k] = b & 15u;
        }
        if (has) {
            const int64_t tw = (int64_t) tr*tiles_y*WORDS + ct_first*WORDS;
            fetch_pair(tw, uv[0], L0, H0);
            fetch_pair(tw, uv[1], L8, H8);
            X0 = __ldg(trellis + tw + ((uv[0] + 2) % WORDS));
            X8 = __ldg(trellis + tw + ((uv[1] + 2) % WORDS));
        }
        for (int ct = ct_first; ct < tiles_y; ct += 8*gridDim.y) {
            const int ctn = ct + 8*gridDim.y;
            if (ctn < tiles_y) {
                const int64_t twn = (int64_t) tr*tiles_y*WORDS + ctn*WORDS;
                fetch_pair(twn, uv[0], nL0, nH0);
                fetch_pair(twn, uv[1], nL8, nH8);
                nX0 = __ldg(trellis + twn + ((uv[0] + 2) % WORDS));
                nX8 = __ldg(trellis + twn + ((uv[1] + 2) % WORDS));
            }
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t lo = k ? L8 : L0, hi = k ? H8 : H0;
                const uint32_t xx = k ? X8 : X0;
                // j=t state from units (lo,hi); j=t+4 from (hi,x) -- same offset
                ra[k + 2*0] = dec_state(lo, hi, ov[k]);
                ra[k + 2*1] = dec_state(hi, xx, ov[k]);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
            if (ctn < tiles_y) {
                L0=nL0; H0=nH0; X0=nX0; L8=nL8; H8=nH8; X8=nX8;
            }
            hb0 = nb0; hb1 = nb1;
        }
    } else {
        uint32_t c0[2][2], c1[2][2];
        if (wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + 7)/8))
            load_at((int64_t) tr*tiles_y*WORDS + ct_first*WORDS, c0, c1);

        for (int ct = ct_first; ct < tiles_y; ct += 8*gridDim.y) {
            const int ctn = ct + 8*gridDim.y;
            uint32_t n0[2][2], n1[2][2];
            if (ctn < tiles_y) load_at((int64_t) tr*tiles_y*WORDS + ctn*WORDS, n0, n1);
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k)
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int off = offv[k][q];
                    const uint32_t w0 = c0[k][q];
                    const uint32_t st = off == 0 ? w0
                        : (((w0 << off) | (c1[k][q] >> (16 - off))) & 0xFFFFu);
                    const uint32_t ph  = st*(st + 1u);
                    const uint32_t row = (ph >> 6) & 511u;
                        // packed u32 tlut access: lo half = a0 (sign-flippable by
                        // xor on the fp16 sign bit), hi half = a1
                    const uint32_t pair = *(const uint32_t*)(const void*)(tlut + 2*row);
                    const uint32_t a0h  = (ph & 0x8000u) ? (pair ^ 0x00008000u) : pair;
                    half2 h = __halves2half2(__ushort_as_half((unsigned short) a0h),
                                             __ushort_as_half((unsigned short)(pair >> 16)));
                    ra[k + 2*q] = *(uint32_t*)&h;   // (row-group, j-pair): q is the outer axis
                }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
#pragma unroll
            for (int k = 0; k < 2; ++k)
#pragma unroll
                for (int q = 0; q < 2; ++q) { c0[k][q] = n0[k][q]; c1[k][q] = n1[k][q]; }
            hb0 = nb0; hb1 = nb1;
        }
    }
    if (tid4 == 0) {
        atomicAdd(&scr_v[(size_t) tr*16 + groupID],      d0);
        atomicAdd(&scr_v[(size_t) tr*16 + groupID + 8],  d2);
    }
}

// dense prefill: decode the K4/V2 trellis ONCE into an fp16 bank [m, n] —
// one thread per (tile, tile-row), same window/state math as rt_walk, same
// pre-rounded tlut halves (sign flip is exact), then apply the bank with a
// plain fp32 GEMM-ish kernel. Replaces only the walk stage; rt_u/rt_out are
// untouched.

// rate-templated twin of paw_rt_dense_decode_kernel: same bank layout,
// generic per-row state windows (paw_rt_walk_kernel math) so the dense
// payload's K1/K1.5 rates can materialize fp16 banks for batched apply.
template <int WORDS>
static __global__ void paw_rt_dense_decode_rate_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        half           * GGML_CUDA_RESTRICT bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    constexpr int step = WORDS/8;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: a warp writes 16 consecutive 16-half row chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*WORDS;
    half * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    half tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int b   = step*(8*rr + j);
        const int wi  = (b >> 4) % WORDS;
        const int off = b & 15;
        const uint32_t w0 = trellis[tw + wi];
        const uint32_t st = off == 0 ? w0
            : (((w0 << off) | (trellis[tw + (wi + 1) % WORDS] >> (16 - off)))
               & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        // negate-then-round-trip is bit-exact for fp16 values (sign bit)
        tmp[2*j + 0] = __float2half_rn((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = slut[2*row + 1];
    }
    paw_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}

static __global__ void paw_rt_dense_decode_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        half           * GGML_CUDA_RESTRICT bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: a warp writes 16 consecutive 16-half row chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*64;   // K = 4: 64 words/tile
    uint32_t w[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w[q] = trellis[tw + ((4*rr + q) & 63)];   // only q=4,rr=15 wraps
    }
    half * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    half tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t hi = w[j >> 1];
        const uint32_t st = (j & 1) == 0 ? hi
            : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        // negate-then-round-trip is bit-exact for fp16 values (sign bit)
        tmp[2*j + 0] = __float2half_rn((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = slut[2*row + 1];
    }
    paw_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}

// fp8 twin of paw_rt_dense_decode_kernel: same decode, output stored as
// e5m2 (1 byte/weight) instead of fp16.
static __global__ void paw_rt_dense_decode_kernel_fp8(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        uint8_t        * GGML_CUDA_RESTRICT bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {
        return;
    }
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*64;
    uint32_t w[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w[q] = trellis[tw + ((4*rr + q) & 63)];
    }
    uint8_t * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    uint8_t tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t hi = w[j >> 1];
        const uint32_t st = (j & 1) == 0 ? hi
            : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        const float a1 = __half2float(slut[2*row + 1]);
        tmp[2*j + 0] = paw_f32_to_e5m2((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = paw_f32_to_e5m2(a1);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i] = tmp[i];
    }
}

static __global__ void paw_rt_dense_decode_kernel_idx80(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        uint16_t       * GGML_CUDA_RESTRICT bank,
        const int m, const int n) {
    constexpr int WG = 256;
    const int tid = threadIdx.x;
    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;
    const int64_t tw = ((int64_t) tr*tiles_y + tc)*64;

    uint32_t w[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w[q] = trellis[tw + ((4*rr + q) & 63)];
    }

    uint16_t packed[5] = {0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t hi = w[j >> 1];
        const uint32_t st = (j & 1) == 0 ? hi
            : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
        const uint32_t code = (st*(st + 1u) >> 6) & 0x3FFu;
        const int bit = 10*j;
        const int wi = bit >> 4;
        const int off = bit & 15;
        packed[wi] |= (uint16_t) (code << off);
        if (off > 6) {
            packed[wi + 1] |= (uint16_t) (code >> (16 - off));
        }
    }

    uint16_t * dst = bank + ((int64_t)(tr*16 + rr)*tiles_y + tc)*5;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        dst[i] = packed[i];
    }
}

// grid (m/16, 1, ceil(nt/TC)): each block owns 16 output rows and a TC-token
// chunk; per column-stride iteration the 16-row W slice is held in registers
// and reused across the chunk's tokens, so bank traffic is amortized TC-fold
// (per-chunk L2 re-reads keep grid-level parallelism, which a single token
// loop over grid (m/16) blocks would not). TC is an A/B knob
// (GGML_PAW_RT_TC, 4 or 8); numerics are TC-independent (per-token order
// unchanged, tail clamps are write-guarded).
template <int TC>
static __global__ void paw_rt_apply_kernel(
        const half  * GGML_CUDA_RESTRICT bank,     // [m, n]
        const float * GGML_CUDA_RESTRICT scr_u,    // [nt, n]
        float       * GGML_CUDA_RESTRICT scr_v,    // [nt, m]
        const int m, const int n, const int nt) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[TC][16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)

    const int tr  = blockIdx.x;
    const int t0  = blockIdx.z*TC;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    const half * W = bank + (int64_t) tr*16*n;

    float acc[TC][16];
#pragma unroll
    for (int t = 0; t < TC; ++t) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc[t][i] = 0.0f;
        }
    }

    // half2/float2 loads: each thread covers column pair (c0, c0+1) — halves
    // the load instruction count. The two columns' terms are accumulated into
    // the same per-thread partial instead of two lanes' partials, a
    // contractible fp32 reassociation (the apply path's summation order is
    // already declared contractible).
    for (int c0 = 2*tid; c0 < n; c0 += 2*WG) {
        float2 wc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            wc[i] = __half22float2(*(const half2 *)(W + (int64_t) i*n + c0));
        }
#pragma unroll
        for (int t = 0; t < TC; ++t) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            const int tt = min(t0 + t, nt - 1);
            const float2 uc = *(const float2 *)(scr_u + (int64_t) tt*n + c0);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[t][i] += wc[i].x*uc.x + wc[i].y*uc.y;
            }
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int t = 0; t < TC; ++t) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(acc[t][i]);
            if (lane == 0) {
                red[t][i][wid] = s;
            }
        }
    }
    __syncthreads();
    if (tid < TC*16) {
        const int t = tid >> 4;
        const int i = tid & 15;
        if (t0 + t < nt) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[t][i][wj];
            }
            scr_v[(int64_t)(t0 + t)*m + tr*16 + i] = sum;
        }
    }
}










// MODE 5: weight-stationary across tokens AND rows. The first WS cut kept
// one 16-row strip per block, so every row-strip block still re-read the
// whole nt x n activation slab (~8 GB per call on the shared-expert shape).
// Staging BM=64 rows per block cuts that by 4x; activations are cast once
// per K-chunk and reused by all four A tiles held live in registers.
static __global__ void paw_rt_apply_kernel_mma(
        const half  * GGML_CUDA_RESTRICT bank,
        const float * GGML_CUDA_RESTRICT scr_u,
        float       * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n, const int nt) {
    using namespace nvcuda;

    constexpr int n_warps = 4;
    constexpr int bm_t    = 4;                  // 16-row tiles -> BM = 64
    constexpr int tpb     = 2;                  // token subtiles -> BN = 128
    constexpr int bn      = n_warps * tpb * 16;
    constexpr int bk      = 16;                 // == the wmma K tile

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row0 = blockIdx.x * (bm_t * 16);  // first output row
    const int tok0 = blockIdx.z * bn;

    __shared__ half Xsh[n_warps][tpb][16][bk];         // 4 KB, warp-private
    __shared__ float out_sh[n_warps][16*16];           // 4 KB

    ggml_cuda_pdl_sync();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[bm_t][tpb];
#pragma unroll
    for (int r = 0; r < bm_t; ++r) {
#pragma unroll
        for (int t = 0; t < tpb; ++t) {
            wmma::fill_fragment(acc[r][t], 0.0f);
        }
    }

    for (int k0 = 0; k0 < n; k0 += bk) {
#pragma unroll
        for (int t = 0; t < tpb; ++t) {
            const int tt0 = tok0 + (warp_id*tpb + t)*16;
            for (int idx = lane; idx < 16*bk; idx += 32) {
                const int tt = idx / bk;
                const int kk = idx % bk;
                const int tok = tt0 + tt;
                const int kg = k0 + kk;
                Xsh[warp_id][t][tt][kk] =
                    (tok < nt && kg < n) ? __float2half(scr_u[(int64_t) tok*n + kg])
                                         : __float2half(0.0f);
            }
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < bm_t; ++r) {
            wmma::load_matrix_sync(a_frag,
                bank + (int64_t)(row0 + r*16)*n + k0, n);
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                wmma::load_matrix_sync(b_frag, &Xsh[warp_id][t][0][0], bk);
                wmma::mma_sync(acc[r][t], a_frag, b_frag, acc[r][t]);
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int r = 0; r < bm_t; ++r) {
#pragma unroll
        for (int t = 0; t < tpb; ++t) {
            wmma::store_matrix_sync(&out_sh[warp_id][0], acc[r][t], 16, wmma::mem_row_major);
            __syncwarp();
            const int tt0 = tok0 + (warp_id*tpb + t)*16;
            for (int idx = lane; idx < 16*16; idx += 32) {
                const int row = idx / 16;
                const int tt  = idx % 16;
                const int tok = tt0 + tt;
                if (tok < nt) {
                    scr_v[(int64_t) tok*m + row0 + r*16 + row] = out_sh[warp_id][idx];
                }
            }
            __syncwarp();
        }
    }
}

// host-side mode dispatch shared by every rt_apply mma call site so the
// bisect modes stay consistent across the cached-bank, fresh-bank and
// batched paths.

// fp32 -> fp16 activation cast feeding the cuBLAS dense apply
static __global__ void paw_cast_f32_f16_kernel(
        const float * GGML_CUDA_RESTRICT x, half * GGML_CUDA_RESTRICT h, const int k) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < k) {
        h[i] = __float2half(x[i]);
    }
}

// host dispatch shared by every rt_apply call site. The weight-stationary
// tensor-core kernel needs m tiled by 64; anything else falls back to the
// scalar dense apply.
static void paw_launch_rt_apply_mma(ggml_backend_cuda_context & ctx,
        cudaStream_t stream, const half * bank,
        const float * scr_u, float * scr_v, const int m, const int n, const int nt,
        const int voff = 0) {
    // voff: first output ROW of this chunk inside the full-m strided dst
    // (the two-half decode/apply overlap writes scr_v[m0..m) from a bank
    // slice while keeping the full-row leading dimension)
    static const int blas_min_tok = paw_env_int("GGML_PAW_RT_BLAS_MIN_TOK", 128);
    // the host count/cast staging is incompatible with graph capture
    cudaStreamCaptureStatus rcst = cudaStreamCaptureStatusNone;
    const bool rt_capturing =
        cudaStreamIsCapturing(stream, &rcst) == cudaSuccess &&
        rcst == cudaStreamCaptureStatusActive;
    if (!rt_capturing && m % 16 == 0 && n % 8 == 0 && nt >= blas_min_tok) {
        // dense apply against a materialized bank is an ordinary GEMM; hand
        // large batches to the tensor-core BLAS. Activations are cast once
        // to fp16 -- the same conversion the custom kernels stage anyway.
        ggml_cuda_pool_alloc<half> u_h_alloc(ctx.pool());
        half * u_h = u_h_alloc.alloc((size_t) nt*n);
        paw_launch(paw_cast_f32_f16_kernel,
            ggml_cuda_kernel_launch_params(
                dim3(((size_t) nt*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            scr_u, u_h, (int)((size_t) nt*n));
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
        CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, nt, n,
                &alpha,
                bank, CUDA_R_16F, n,
                u_h,  CUDA_R_16F, n,
                &beta,
                scr_v + voff, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return;
    }
    GGML_ASSERT(voff == 0 && "row-offset apply only supported on the cublas path");
    if (m % 64 == 0) {
        paw_launch(paw_rt_apply_kernel_mma,
            ggml_cuda_kernel_launch_params(dim3(m/64, 1, (nt + 127)/128), dim3(128, 1, 1), 0, stream),
            bank, scr_u, scr_v, m, n, nt);
    } else {
        paw_launch(paw_rt_apply_kernel<8>,
            ggml_cuda_kernel_launch_params(dim3(m/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
            bank, (const float *) scr_u, scr_v, m, n, nt);
    }
}

// --- experimental: tensor-core walk, opt-in via GGML_PAW_RT_WALK_MMA=1 ---
//
// paw_rt_walk_kernel above is a scalar matvec: per (token, 16-row tile) the
// trellis is decoded once per column-tile and dotted against the token's u
// values with FMA + warp reduce. This variant keeps the exact same decode
// math (the fp16 values match paw_rt_dense_decode_kernel, the dense-path
// reference) but stages each decoded 16x16 tile in shared memory and
// multiplies it onto the tokens' activations with m16n8k16 tensor-core MMA
// instead of scalar FMA. The K dimension (n columns) is split across the
// block's 4 warps, each accumulating a partial 16x8 tile; a small cross-warp
// reduction finishes the 16 output rows. Activations are cast fp32->fp16 for
// the B operand (standard for tensor-core GEMM; the one deliberate numeric
// difference the correctness gate checks). The walk still materializes only
// the nt x m output -- no full m x n bank -- which is the point of the walk
// over the dense path at small nt.
static __global__ void paw_rt_walk_kernel_mma(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n, const int nt) {
    using namespace nvcuda;
    constexpr int warp_size = 32;
    constexpr int n_warps   = 128 / warp_size;

    __shared__ half  slut[1024];
    __shared__ __align__(16) half shW[n_warps][16*16];   // decoded tile, row-major
    __shared__ __align__(16) half shU[n_warps][16*16];   // [16 tokens][16 K], col-major B
    __shared__ __align__(16) float out_sh[n_warps][16*16];

    const int t      = blockIdx.z*16;   // first token of this block's 16-tile
    const int tr     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane    = tid % warp_size;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += 128) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int64_t tw0 = (int64_t) tr*tiles_y;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // K split across warps: warp w owns K-chunks {w, w+4, w+8, ...}
    for (int kt = warp_id; kt < tiles_y; kt += n_warps) {
        const int64_t tw = (tw0 + kt)*64;
        // decode this 16x16 tile into this warp's shared region, row-major.
        // lane (2r) covers states 0..3 of row r, lane (2r+1) states 4..7.
        const int r  = lane >> 1;
        const int st0 = (lane & 1) * 4;
        uint32_t w[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w[q] = trellis[tw + ((4*r + q) & 63)];
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int kk  = st0 + s;
            const uint32_t hi = w[kk >> 1];
            const uint32_t st = (kk & 1) == 0 ? hi
                : (((hi << 8) | (w[(kk >> 1) + 1] >> 8)) & 0xFFFFu);
            const uint32_t ph  = st*(st + 1u);
            const uint32_t row = (ph >> 6) & 511u;
            const float a0 = __half2float(slut[2*row + 0]);
            shW[warp_id][r*16 + 2*kk]     = __float2half_rn((ph & 0x8000u) ? -a0 : a0);
            shW[warp_id][r*16 + 2*kk + 1] = slut[2*row + 1];
        }
        // stage 16 token columns' 16 activations into this warp's shared,
        // col-major B (element (k, tt) at shU[tt*16 + k]); columns past nt
        // are zero-padded (block z-tile is 16 tokens).
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int e   = lane*8 + q;   // 0..255
            const int tt  = e >> 4;
            const int kk  = e & 15;
            float v = 0.0f;
            if (t + tt < nt) {
                v = scr_u[(int64_t)(t + tt)*n + kt*16 + kk];
            }
            shU[warp_id][tt*16 + kk] = __float2half(v);
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, shW[warp_id], 16);
        wmma::load_matrix_sync(b_frag, shU[warp_id], 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(&out_sh[warp_id][0], acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < 16*16; idx += 128) {
        const int row = idx >> 4;
        const int tt  = idx & 15;
        const int tok = t + tt;
        if (tok < nt) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += out_sh[wj][idx];
            }
            scr_v[(int64_t) tok*m + tr*16 + row] = sum;
        }
    }
}

// --- pre-decoded bank cache (Marlin-style: decode static weights once, then
// GEMM fast) ---
//
// The trellis weights are static, so the decoded fp16 bank for each RT matrix
// is identical across steps; the dense path has been re-decoding it every
// step. Caching it here turns the per-step walk/decode into a plain
// bandwidth-bound fp16 GEMV/GEMM over the cached bank. Keyed by the trellis
// data pointer *and shape*; entries are intentionally leaked (one per matrix,
// matching the weight allocation's lifetime). GGML_PAW_BANK_CACHE=0 falls back
// to the original per-step walk/decode.
//
// The shape belongs in the key because a pointer alone is only unique for the
// lifetime of the allocation behind it. Model weights live forever, so the old
// pointer-only key was safe in a server; test-backend-ops recycles buffers
// between cases, so a later case with a different shape used to be handed the
// previous case's bank -- wrong numbers, and an out-of-bounds read as soon as
// the new shape was larger.
struct paw_bank_key {
    const void * trellis;
    int m;
    int n;
    bool operator==(const paw_bank_key & o) const {
        return trellis == o.trellis && m == o.m && n == o.n;
    }
};
struct paw_bank_key_hash {
    size_t operator()(const paw_bank_key & k) const {
        return std::hash<const void *>()(k.trellis) ^ (std::hash<int>()(k.m) << 1)
                                                    ^ (std::hash<int>()(k.n) << 2);
    }
};
static std::mutex paw_rt_bank_mutex;
static std::unordered_map<paw_bank_key, const void *, paw_bank_key_hash> paw_rt_banks;

// rate-dispatch wrapper: any supported trellis rate through the templated
// tensor-core walk (K4 included -- the generic window math is bit-identical)
template <int WORDS>
static void paw_rt_walk_qtip_frag_dispatch(const uint16_t * trellis,
        const half * tlut, const float * scr_u, float * scr_v,
        const int m, const int n, cudaStream_t stream) {
    // enough blocks that no warp walks a long serial chain of tile columns
    // (measured neutral on PAW-27B -- small-m attention projections are not
    // on the critical path; kept for narrow-m payloads)
    static const int split_min_blocks =
        paw_env_int("GGML_PAW_WALK_SPLIT_BLOCKS", 0);
    static const int split_max   = paw_env_int("GGML_PAW_WALK_SPLIT_MAX", 8);
    int S = 1;
    if (split_min_blocks > 0 && m/16 < split_min_blocks && n >= 256) {
        S = (split_min_blocks + m/16 - 1)/(m/16);
        if (S > split_max) S = split_max;
        if (S < 1)         S = 1;
    }
    if (S > 1) {
        CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) m*sizeof(float), stream));
    }
    paw_launch(paw_rt_walk_qtip_frag_kernel<WORDS>,
        ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
        trellis, tlut, scr_u, scr_v, m, n);
}

static void paw_rt_walk_qtip_rate_launch(const uint16_t * trellis,
        const half * tlut, const float * scr_u, float * scr_v,
        const int m, const int n, const int words, cudaStream_t stream) {
    // column-split factor: enough blocks in flight that no single warp walks
    // a long serial chain of tile columns (the walk is latency-bound); the
    // machine holds ~42k threads = ~168 of these 256-thread blocks, so any
    // matrix with fewer than ~1500 output-tile blocks gets split
    static const int split_min_blocks =
        paw_env_int("GGML_PAW_WALK_SPLIT_BLOCKS", 1500);
    static const int split_max   = paw_env_int("GGML_PAW_WALK_SPLIT_MAX", 8);
    int S = 1;
    if (split_min_blocks > 0 && m/16 < split_min_blocks && n >= 256) {
        S = (split_min_blocks + m/16 - 1)/(m/16);
        if (S > split_max)  S = split_max;
        if (S*16 > n/16*8)  S = std::max(1, (n/16*8)/16);   // keep every warp busy
        if (S < 1)          S = 1;
    }
    if (S > 1) {
        CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) m*sizeof(float), stream));
    }
    switch (words) {
        case 16: paw_launch(paw_rt_walk_qtip_rate_kernel<16>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 24: paw_launch(paw_rt_walk_qtip_rate_kernel<24>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 32: paw_launch(paw_rt_walk_qtip_rate_kernel<32>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 40: paw_launch(paw_rt_walk_qtip_rate_kernel<40>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 56: paw_launch(paw_rt_walk_qtip_rate_kernel<56>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 64: paw_launch(paw_rt_walk_qtip_rate_kernel<64>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        default: GGML_ABORT("paw: unsupported trellis rate for qtip walk");
    }
}

static void paw_rt_dense_decode_rate_launch(const uint16_t * trellis,
        const half * tlut, half * bank, const int m, const int n,
        const int words, cudaStream_t stream) {
    switch (words) {
        case 16: paw_launch(paw_rt_dense_decode_rate_kernel<16>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 24: paw_launch(paw_rt_dense_decode_rate_kernel<24>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 32: paw_launch(paw_rt_dense_decode_rate_kernel<32>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 40: paw_launch(paw_rt_dense_decode_rate_kernel<40>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 56: paw_launch(paw_rt_dense_decode_rate_kernel<56>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 64: paw_launch(paw_rt_dense_decode_rate_kernel<64>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        default: GGML_ABORT("paw: unsupported trellis rate for dense decode");
    }
}


static bool paw_bank_cache_on() {
    static const bool on = paw_env_int("GGML_PAW_BANK_CACHE", 1) != 0;
    return on;
}

static size_t paw_rt_idx_bytes(const int m, const int n) {
    return (size_t) m*(n/16)*5*sizeof(uint16_t);
}

// decode-once and return the cached bank [m, n] for this trellis (fp16 or
// e5m2 fp8). Caller must not hold the mutex; the first call per matrix syncs
// the stream to make the decode visible before the dependent GEMM launches.
static const void * paw_rt_bank_get(
        const void * trellis, const void * tlut, const int m, const int n, cudaStream_t stream) {
    const bool idx = paw_rt_bank_idx_on();
    const bool fp8 = !idx && paw_rt_bank_fp8_on();
    const paw_bank_key key{trellis, m, n};
    {
        std::lock_guard<std::mutex> lock(paw_rt_bank_mutex);
        auto it = paw_rt_banks.find(key);
        if (it != paw_rt_banks.end()) {
            return it->second;
        }
    }
    void * bank = nullptr;
    const size_t idx_bytes = paw_rt_idx_bytes(m, n);
    const size_t bank_bytes = idx ? idx_bytes + (size_t) m*n*sizeof(half) : (size_t) m*n*(fp8 ? 1 : (int) sizeof(half));
    CUDA_CHECK(cudaMalloc(&bank, bank_bytes));
    if (idx) {
        paw_launch(paw_rt_dense_decode_kernel_idx80,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis, (uint16_t *) bank, m, n);
        paw_launch(paw_rt_dense_decode_kernel,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis, (const half *) tlut, (half *) ((uint8_t *) bank + idx_bytes), m, n);
    } else if (fp8) {
        paw_launch(paw_rt_dense_decode_kernel_fp8,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis, (const half *) tlut, (uint8_t *) bank, m, n);
    } else {
        paw_launch(paw_rt_dense_decode_kernel,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis, (const half *) tlut, (half *) bank, m, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        std::lock_guard<std::mutex> lock(paw_rt_bank_mutex);
        auto it = paw_rt_banks.find(key);
        if (it != paw_rt_banks.end()) {
            cudaFree(bank);
            return it->second;
        }
        paw_rt_banks.emplace(key, bank);
    }
    return bank;
}

static const half * paw_rt_idx_fp16_bank(const void * bank, const int m, const int n) {
    return (const half *) ((const uint8_t *) bank + paw_rt_idx_bytes(m, n));
}

// bandwidth-bound fp16 GEMV/GEMM over a pre-decoded bank:
// scr_v[t, m] = scr_u[t, n] @ bank[m, n]^T. One warp per (output row, token);
// lanes read the row's half2s stride-32 (coalesced) and a warp-shuffle reduce
// finishes the row. Grid (m/8, 1, nt). This is the walk path's replacement at
// small nt: the decode work is amortized to one time per matrix instead of
// every step.
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

// float4 variant of paw_rt_bank_gemv: wider loads (16 B/lane/iter instead of
// 4) to cut load-issue pressure, and u is staged in shared once per block so
// the 8 row-warps do not re-issue the same L2 loads. Same math, same output.
// Grid (m/16, 1, nt), 256 threads (8 warps, each handles 2 rows via a 4-row
// stride). GGML_PAW_RT_GEMV2=1 opt-in.
static __global__ void paw_rt_bank_gemv_v2(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT scr_u,  // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    constexpr int WARPS = 8;
    __shared__ float u_sh[4096];

    const int blk  = blockIdx.x*WARPS*2;    // first row of this block
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    ggml_cuda_pdl_sync();
    const float * u = scr_u + (int64_t) t*n;
    for (int i = tid; i < n; i += 256) {
        u_sh[i] = u[i];
    }
    __syncthreads();

    const int n4 = n/4;
    for (int r = wid; r < WARPS*2; r += WARPS) {   // 2 rows per warp
        const int row = blk + r;
        if (row < m) {
            const float4 * u4 = (const float4 *) u_sh;
            const uint2  * W4 = (const uint2 *)(bank + (int64_t) row*n);   // 4 halves per uint2
            float acc = 0.0f;
            for (int i = lane; i < n4; i += 32) {
                const half2 w0 = *((const half2 *) &W4[i].x);
                const half2 w1 = *((const half2 *) &W4[i].y);
                const float2 wf0 = __half22float2(w0);
                const float2 wf1 = __half22float2(w1);
                const float4 uu = u4[i];
                acc += wf0.x*uu.x + wf0.y*uu.y + wf1.x*uu.z + wf1.y*uu.w;
            }
            acc = warp_reduce_sum<32>(acc);
            if (lane == 0) {
                scr_v[(int64_t) t*m + row] = acc;
            }
        }
    }
}

// u-fused GEMV: computes u = FWHT(su .* x) internally (redundantly in every
// block, from the same su/x the rt_u kernel reads) instead of consuming the
// separate rt_u kernel's scr_u. This removes the rt_u stage's launch and
// serialization from every rt matrix's critical path -- the rt_u cost
// (~2.8 ms/token timed) becomes a few us of concurrent redundant FWHT work
// inside the gemv kernels. scr_v is the only output. The scale is applied
// inside the kernel so the gemv reads the exact normalized u. Grid
// (m/8, 1, nt), 256 threads (one row per warp). Requires n >= 256 and
// n % 2 == 0 (FWHT wg=256 + half2 gemv). GGML_PAW_RT_GEMVU=1 opt-in.
static __global__ void paw_rt_gemv_u_kernel(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT su,     // [n]
        const float * GGML_CUDA_RESTRICT x,      // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    __shared__ float sh[4096];

    const int row  = blockIdx.x*8 + (threadIdx.x >> 5);
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();
    for (int i = tid; i < n; i += 256) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    paw_fwht_block(sh, n, tid, 256);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += 256) {
        sh[i] = __fdiv_rn(sh[i], sc);
    }
    __syncthreads();

    const half2  * W2 = (const half2 *) (bank + (int64_t) row*n);
    const float2 * u2 = (const float2 *) sh;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const float2 w = __half22float2(W2[i]);
        const float2 uu = u2[i];
        acc += w.x*uu.x + w.y*uu.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}
// fused rt_u + fp8-bank GEMV: the e5m2 twin of paw_rt_gemv_u_kernel. The
// half-reading variant must not run against an fp8 cached bank, so when
// GGML_PAW_RT_BANK_FP8=1 this kernel handles the fused path instead.
static __global__ void paw_rt_gemv_u_fp8_kernel(
        const uint8_t * GGML_CUDA_RESTRICT bank,  // [m, n] row-major e5m2
        const float   * GGML_CUDA_RESTRICT su,    // [n]
        const float   * GGML_CUDA_RESTRICT x,     // [nt, n] row-major
        float         * GGML_CUDA_RESTRICT scr_v, // [nt, m] row-major
        const int m, const int n, const int nt) {
    __shared__ float sh[4096];
    __shared__ float lut[256];

    const int tid  = threadIdx.x;
    if (tid < 256) {
        lut[tid] = paw_e5m2_to_f32((uint8_t) tid);
    }
    __syncthreads();

    const int row  = blockIdx.x*8 + (tid >> 5);
    const int t    = blockIdx.z;
    const int lane = tid & 31;

    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();
    for (int i = tid; i < n; i += 256) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    paw_fwht_block(sh, n, tid, 256);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += 256) {
        sh[i] = __fdiv_rn(sh[i], sc);
    }
    __syncthreads();

    const uint8_t * W = bank + (int64_t) row*n;
    const float4 * u4 = (const float4 *) sh;

    float acc = 0.0f;
    if (n % 4 == 0) {
        const uint32_t * W4 = (const uint32_t *) W;
        const int n4 = n/4;
        for (int i = lane; i < n4; i += 32) {
            const uint32_t w = W4[i];
            const uint8_t * wb = (const uint8_t *) &w;
            const float4 uu = u4[i];
            acc += lut[wb[0]]*uu.x + lut[wb[1]]*uu.y + lut[wb[2]]*uu.z + lut[wb[3]]*uu.w;
        }
    } else {
        for (int i = lane; i < n; i += 32) {
            acc += lut[W[i]]*sh[i];
        }
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}
// bank GEMV. (1) u is staged in shared once per block, killing the 8x
// redundant per-warp u re-reads that were inflating DRAM traffic past the
// 32 MB bank. (2) the bank is read with __ldcs (evict-first streaming): it is
// touched exactly once per step, so polluting L2 with it only evicts the hot
// activations/scratch. (3) two independent accumulators give the FMA chain
// ILP. Same math, same output. Grid (m/16, 1, nt), 256 threads (2 rows/warp).
// GGML_PAW_RT_GEMV3=1 opt-in.
static __global__ void paw_rt_bank_gemv_v3(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT scr_u,  // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    constexpr int WARPS = 8;
    __shared__ float u_sh[4096];

    const int blk  = blockIdx.x*WARPS*2;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    ggml_cuda_pdl_sync();
    const float * u = scr_u + (int64_t) t*n;
    for (int i = tid; i < n; i += 256) {
        u_sh[i] = u[i];
    }
    __syncthreads();

    const int     n2 = n/2;
    const half2 * W2 = (const half2 *) bank;
    const float2 * u2 = (const float2 *) u_sh;

    for (int r = wid; r < WARPS*2; r += WARPS) {
        const int row = blk + r;
        if (row < m) {
            const half2 * Wr = W2 + (int64_t) row*n2;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            int i = lane;
            for (; i + 32 < n2; i += 64) {
                const float2 w0 = __half22float2(__ldcs(Wr + i));
                const float2 w1 = __half22float2(__ldcs(Wr + i + 32));
                const float2 x0 = u2[i];
                const float2 x1 = u2[i + 32];
                acc0 += w0.x*x0.x + w0.y*x0.y;
                acc1 += w1.x*x1.x + w1.y*x1.y;
            }
            for (; i < n2; i += 32) {
                const float2 w0 = __half22float2(__ldcs(Wr + i));
                const float2 x0 = u2[i];
                acc0 += w0.x*x0.x + w0.y*x0.y;
            }
            float acc = acc0 + acc1;
            acc = warp_reduce_sum<32>(acc);
            if (lane == 0) {
                scr_v[(int64_t) t*m + row] = acc;
            }
        }
    }
}

// --- fused RT_MM (cooperative): rt_u + bank GEMV + rt_out in ONE launch ---
//
// The RT op was three launches per matrix (rt_u, bank GEMV, rt_out), each
// tiny and mostly launch-latency-bound at nt=1. This variant runs all three
// stages in a single cooperative launch: block 0 does the u-FWHT, grid.sync,
// all blocks do the fp16 GEMV over the cached bank, grid.sync, block 0 does
// the v-FWHT + scale. The u/v scratch goes through the same global pool
// buffers (L2-hot, only 8-32KB per stage). Grid must be co-resident, so it is
// capped at nsm blocks with each block looping over its share of rows.
// GGML_PAW_RT_FUSED=1 opt-in.
static __global__ void paw_rt_fused_kernel(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] pre-decoded fp16
        const float * GGML_CUDA_RESTRICT su,     // [n]
        const float * GGML_CUDA_RESTRICT sv,     // [m]
        const float * GGML_CUDA_RESTRICT x,      // [nt, n]
        float       * GGML_CUDA_RESTRICT scr_u,  // [nt, n] scratch
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] scratch
        float       * GGML_CUDA_RESTRICT dst,    // [nt, m]
        const int m, const int n, const int nt) {
    namespace cg = cooperative_groups;
    const cg::grid_group grid = cg::this_grid();

    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    __shared__ float sh[8192];   // reused for u (n <= 4096) and v (m <= 8192)

    // Phase 1: rt_u on block 0 only
    if (blockIdx.x == 0) {
        for (int i = tid; i < n; i += 256) {
            sh[i] = su[i] * x[(int64_t) t*n + i];
        }
        __syncthreads();
        paw_fwht_block(sh, n, tid, 256);
        const float sc = __fsqrt_rn((float) n);
        for (int i = tid; i < n; i += 256) {
            scr_u[(int64_t) t*n + i] = __fdiv_rn(sh[i], sc);
        }
    }
    grid.sync();

    // Phase 2: fp16 GEMV over the cached bank, rows striped across blocks
    {
        const float  * u  = scr_u + (int64_t) t*n;
        const float2 * u2 = (const float2 *) u;
        const half2  * b2 = (const half2 *) bank;
        const int n2 = n/2;
        for (int r0 = blockIdx.x*8; r0 < m; r0 += gridDim.x*8) {
            const int row = r0 + (tid >> 5);
            if (row < m) {
                const half2 * W2 = b2 + (int64_t) row*n2;
                float acc = 0.0f;
                for (int i = lane; i < n2; i += 32) {
                    const float2 w  = __half22float2(W2[i]);
                    const float2 uu = u2[i];
                    acc += w.x*uu.x + w.y*uu.y;
                }
                acc = warp_reduce_sum<32>(acc);
                if (lane == 0) {
                    scr_v[(int64_t) t*m + row] = acc;
                }
            }
        }
    }
    grid.sync();

    // Phase 3: rt_out on block 0 only
    if (blockIdx.x == 0) {
        for (int i = tid; i < m; i += 256) {
            sh[i] = scr_v[(int64_t) t*m + i];
        }
        __syncthreads();
        paw_fwht_block(sh, m, tid, 256);
        const float sc = __fsqrt_rn((float) m);
        for (int i = tid; i < m; i += 256) {
            dst[(int64_t) t*m + i] = __fdiv_rn(sh[i], sc) * sv[i];
        }
    }
}

template <int WG, bool EPILOGUE>
static __global__ void paw_rt_out_kernel(
        const float * GGML_CUDA_RESTRICT sv,
        const float * GGML_CUDA_RESTRICT scr_v,
        float       * GGML_CUDA_RESTRICT dst,
        const float * GGML_CUDA_RESTRICT gate,
        const float * GGML_CUDA_RESTRICT acc,
        const int m,
        const int blk) {
    // blk == m reproduces the pre-blocking kernel exactly (one iteration).
    // blockIdx.y selects the output chunk so chunks run on separate SMs
    // instead of serializing inside one block (per-chunk math is unchanged).
    __shared__ float sh[8192];

    const int t   = blockIdx.x;
    const int off = blockIdx.y * blk;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float sc     = __fsqrt_rn((float) blk);
    const float inv_sc = __frcp_rn(sc);
    {
        for (int i = tid; i < blk; i += WG) {
            __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) t*m + off + i], sizeof(float));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        if (WG == blk/16 && paw_fwht_v2_ok(blk)) {
            paw_fwht_block_v2(sh, blk, tid, WG);
        } else {
            paw_fwht_block(sh, blk, tid, WG);
        }
        for (int i = tid; i < blk; i += WG) {
            const int64_t oi = (int64_t) t*m + off + i;
            const float y = sh[i] * inv_sc * sv[off + i];
            if constexpr (EPILOGUE) {
                const float sigmoid = 1.0f / (1.0f + expf(-gate[t]));
                const float gated = __fmul_rn(y, sigmoid);
                dst[oi] = __fadd_rn(acc[oi], gated);
            } else {
                dst[oi] = y;
            }
        }
        __syncthreads();
    }
}

template <int WG>
static __global__ void paw_rt_out_epilogue_dot_kernel(
        const float * GGML_CUDA_RESTRICT sv,
        const float * GGML_CUDA_RESTRICT scr_v,
        float       * GGML_CUDA_RESTRICT dst,
        const float * GGML_CUDA_RESTRICT gate_w,
        const float * GGML_CUDA_RESTRICT gate_x,
        const float * GGML_CUDA_RESTRICT acc,
        const int m, const int gate_n, const int blk) {
    __shared__ float sh[8192];
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    float gate_sum = 0.0f;
    for (int i = tid; i < gate_n; i += WG) {
        gate_sum = fmaf(gate_w[i], gate_x[(int64_t) t*gate_n + i], gate_sum);
    }
    sh[tid] = gate_sum;
    __syncthreads();
    for (int stride = WG/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sh[0]));
    __syncthreads();   // sh is about to be reused as the FWHT staging buffer

    const float inv_sc = __frcp_rn(__fsqrt_rn((float) blk));
    for (int off = 0; off < m; off += blk) {
        for (int i = tid; i < blk; i += WG) {
            __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) t*m + off + i], sizeof(float));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        if (WG == blk/16 && paw_fwht_v2_ok(blk)) {
            paw_fwht_block_v2(sh, blk, tid, WG);
        } else {
            paw_fwht_block(sh, blk, tid, WG);
        }
        for (int i = tid; i < blk; i += WG) {
            const int64_t oi = (int64_t) t*m + off + i;
            const float y = sh[i] * inv_sc * sv[off + i];
            dst[oi] = __fadd_rn(acc[oi], __fmul_rn(y, sigmoid));
        }
        __syncthreads();
    }
}

void ggml_cuda_op_paw_rt_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * su      = dst->src[1];
    const ggml_tensor * sv      = dst->src[2];
    const ggml_tensor * tlut    = dst->src[3];
    const ggml_tensor * x       = dst->src[4];
    const int epilogue_mode       = dst->op_params[0];
    const bool epilogue           = epilogue_mode != 0;
    const bool epilogue_dot       = epilogue_mode == 2;
    const ggml_tensor * gate      = epilogue ? dst->src[5] : nullptr;
    const ggml_tensor * gate_x    = epilogue_dot ? dst->src[6] : nullptr;
    const ggml_tensor * acc       = epilogue_dot ? dst->src[7] : epilogue ? dst->src[6] : nullptr;

    GGML_ASSERT(trellis->type == GGML_TYPE_I16);
    GGML_ASSERT(su->type   == GGML_TYPE_F32);
    GGML_ASSERT(sv->type   == GGML_TYPE_F32);
    GGML_ASSERT(tlut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!epilogue || (gate && acc && gate->type == GGML_TYPE_F32 && acc->type == GGML_TYPE_F32));
    GGML_ASSERT(!epilogue_dot || (gate_x && gate_x->type == GGML_TYPE_F32));
    GGML_ASSERT(ggml_is_contiguous(trellis));
    GGML_ASSERT(ggml_is_contiguous(su));
    GGML_ASSERT(ggml_is_contiguous(sv));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // Trellis rate from the payload: words-per-tile = 16*K at V = 2.
    // 64 -> K=4 (the shipped 35B NE spine), 32 -> K=2, 16 -> K=1.
    const int rt_words = (int) trellis->ne[0];
    const int rt_K     = rt_words / 16;   // == 4 only for the shipped payload
    GGML_ASSERT(rt_words % 8 == 0 && rt_words >= 16 && rt_words <= 64);

    const int n  = (int) su->ne[0];
    const int m  = (int) sv->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);
    // Rotation block size. 0 means one Hadamard over the whole dimension --
    // the shipped power-of-two payloads. A blocked payload rotates within
    // blk-wide groups, which is what lets non-power-of-two dense shapes
    // (5120, 17408, ...) run at all: only one block is ever staged in shared.
    const int rht_blk = dst->op_params[GGML_PAW_RHT_BLK_SLOT];
    const int bn = rht_blk ? rht_blk : n;
    const int bm = rht_blk ? rht_blk : m;
    const bool blocked = (bn != n) || (bm != m);
    // The bank decode, the fused cooperative kernel, the gemv variants and the
    // qtip walk all bake in the K=4 stream layout. Only the generic walk below
    // is rate-aware, so every other rate is routed through it.
    const bool k4 = rt_words == 64;
    GGML_ASSERT(bn <= 4096);          // rt_u shared bound, now per block
    GGML_ASSERT(bm <= 8192);          // rt_out shared bound, now per block
    GGML_ASSERT(n % bn == 0 && m % bm == 0);

    ggml_cuda_pool_alloc<float> scr(ctx.pool(), (size_t) nt*n + (size_t) nt*m);
    float * scr_u = scr.get();
    float * scr_v = scr_u + (size_t) nt*n;

    cudaStream_t stream = ctx.stream();

    char shp[64];
    snprintf(shp, sizeof(shp), " m=%d n=%d nt=%d", m, n, nt);

    paw_fwht_set_mode();

    static const bool rt_fused = paw_env_int("GGML_PAW_RT_FUSED", 0) != 0;
    static const int  dense_min_tok = paw_env_int("GGML_PAW_DENSE_MIN_TOK", 4);
    // above this many tokens, a per-pass fp16 bank + batched apply beats the
    // generic per-token walk on non-K4 payloads
    static const int  rt_blas_min_nt = paw_env_int("GGML_PAW_RT_BLAS_MIN_NT", 8);
    if (k4 && !blocked && rt_fused && !paw_rt_bank_fp8_on() && !paw_rt_bank_idx_on() && nt < dense_min_tok && paw_bank_cache_on()) {
        // cooperative single-launch RT_MM (rt_u + bank GEMV + rt_out)
        const int id = ggml_cuda_get_device();
        const bool coop = ggml_cuda_info().devices[id].supports_cooperative_launch;
        if (coop) {
            const half * bank = (const half *) paw_rt_bank_get(trellis->data, tlut->data, m, n, stream);
            const float * su_p = (const float *) su->data;
            const float * sv_p = (const float *) sv->data;
            const float * x_p  = (const float *) x->data;
            float * dst_p      = (float *) dst->data;
            const dim3 block_nums(ggml_cuda_info().devices[id].nsm, 1, nt);
            const dim3 block_dims(256, 1, 1);
            void * args[] = { (void *) &bank, (void *) &su_p,
                                    (void *) &sv_p, (void *) &x_p,
                                    (void *) &scr_u, (void *) &scr_v,
                                    (void *) &dst_p, (void *) &m,
                                    (void *) &n, (void *) &nt };
            paw_timed(stream, std::string("rt_fused") + shp, [&]() {
                CUDA_CHECK(cudaLaunchCooperativeKernel((void *) paw_rt_fused_kernel,
                        block_nums, block_dims, args, 0, stream));
            });
            return;
        }
    }

    static const bool rt_walk_qtip = paw_env_int("GGML_PAW_RT_WALK_QTIP", 0) != 0;
    const bool bank_cache = paw_bank_cache_on();

    // u-fused gemv: skip the separate rt_u launch entirely; the gemv kernel
    // computes u internally. Valid when the gemv geometry matches the kernel's
    // assumptions (n >= 256, n % 2 == 0, m % 8 == 0).
    static const bool rt_gemvu = paw_env_int("GGML_PAW_RT_GEMVU", 0) != 0;
    const bool gemvu = k4 && !blocked && rt_gemvu && bank_cache && !paw_rt_bank_idx_on() &&
                       nt < dense_min_tok &&
                       n >= 256 && n % 2 == 0 && m % 8 == 0;

    static const bool uout_noop = paw_env_int("GGML_PAW_UOUT_NOOP", 0) != 0;
    if (!gemvu && !uout_noop) {
    paw_timed(stream, std::string("rt_u") + shp, [&]() {
    if (!blocked && paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
        paw_fwht_for_wg(n/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_rt_u_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(nt, n/bn, 1), dim3(wg, 1, 1), 0, stream),
                (const float *) su->data, (const float *) x->data, scr_u, n, bn);
        });
    } else if (!blocked && paw_fwht_wg512()) {
        paw_launch(paw_rt_u_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(nt, n/bn, 1), dim3(512, 1, 1), 0, stream),
            (const float *) su->data, (const float *) x->data, scr_u, n, bn);
    } else {
        paw_launch(paw_rt_u_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(nt, n/bn, 1), dim3(256, 1, 1), 0, stream),
            (const float *) su->data, (const float *) x->data, scr_u, n, bn);
    }
    });
    }

    // the rate-templated walk is default-ON for non-K4 payloads: they have
    // no cached-bank alternative at nt==1 (banking the whole dense FFN would
    // not fit VRAM), and the generic walk is ~10x slower per call
    static const bool rt_walk_qtip_dense = paw_env_int("GGML_PAW_RT_WALK_QTIP_DENSE", 1) != 0;
    const bool use_qtip = nt == 1 && !debug_diff_on() &&
                          (k4 ? rt_walk_qtip : rt_walk_qtip_dense);
    if (use_qtip) {
        static int debug_calls = 0;
        static const bool debug_diff = paw_env_int("GGML_PAW_RT_WALK_QTIP_DEBUG", 0) != 0;
        if (debug_diff && debug_calls < 6) {
            ggml_cuda_pool_alloc<float> ref_v_alloc(ctx.pool(), (size_t) m);
            float * ref_v = ref_v_alloc.get();
            if (n/16 <= 128) {
                paw_launch((paw_rt_walk_kernel<128, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(128, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n);
            } else {
                paw_launch((paw_rt_walk_kernel<256, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n);
            }
            paw_rt_walk_qtip_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, (const float *) scr_u, scr_v,
                m, n, rt_words, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> hv(m), hr(m);
            CUDA_CHECK(cudaMemcpy(hv.data(), scr_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hr.data(), ref_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            float maxdiff = 0.0f; int maxi = -1;
            for (int i = 0; i < m; ++i) {
                float d = fabsf(hv[i] - hr[i]);
                if (d > maxdiff) { maxdiff = d; maxi = i; }
            }
            fprintf(stderr, "[rt_walk_qtip DEBUG] m=%d n=%d maxdiff=%g at i=%d qtip=%g ref=%g  first5: qtip=[%g,%g,%g,%g,%g] ref=[%g,%g,%g,%g,%g]\n",
                m, n, maxdiff, maxi, maxi>=0?hv[maxi]:0.0, maxi>=0?hr[maxi]:0.0,
                hv[0], hv[1], hv[2], hv[3], hv[4], hr[0], hr[1], hr[2], hr[3], hr[4]);
            // use the reference output so generation stays coherent while debugging
            CUDA_CHECK(cudaMemcpyAsync(scr_v, ref_v, m*sizeof(float), cudaMemcpyDeviceToDevice, stream));
            debug_calls++;
        } else if (debug_diff) {
            // past the debug window: use the known-good reference path
            if (n/16 <= 128) {
                paw_launch((paw_rt_walk_kernel<128, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(128, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            } else {
                paw_launch((paw_rt_walk_kernel<256, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            }
        } else {
        // bypasses the bank cache entirely -- reads the compressed trellis
        // directly instead of a pre-decoded fp16 bank. See
        // paw_rt_walk_qtip_kernel's header comment. Falls through to the
        // rt_out call below (same as every other branch here) -- does NOT
        // return early.
        static const bool walk_noop = paw_env_int("GGML_PAW_WALK_NOOP", 0) != 0;
        // fragment-direct variant (default ON): each lane decodes exactly its
        // own mma.sync fragment elements straight from the trellis -- no
        // shared staging, no syncwarp in the loop; scr_v is accumulated with
        // atomics so it must be zeroed first
        static const bool walk_frag = paw_env_int("GGML_PAW_WALK_FRAG", 1) != 0;
        static const bool frag_dbg_on = paw_env_int("GGML_PAW_WALK_FRAG_DEBUG", 0) != 0;
        static int frag_dbg_calls = 0;
        if (!walk_noop && walk_frag && frag_dbg_on &&
            frag_dbg_calls < paw_env_int("GGML_PAW_WALK_FRAG_DEBUG_MAX", 4) &&
            (rt_words == 16 || rt_words == 24 || rt_words == 32)) {
            // correctness probe: frag variant vs the generic reference walk
            ggml_cuda_pool_alloc<float> ref_v_alloc(ctx.pool(), (size_t) m);
            float * ref_v = ref_v_alloc.get();
            CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) m*sizeof(float), stream));
            switch (rt_words) {
                case 24: paw_rt_walk_qtip_frag_dispatch<24>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                case 32: paw_rt_walk_qtip_frag_dispatch<32>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                default: paw_rt_walk_qtip_frag_dispatch<16>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
            }
            #define PAW_FRAG_DBG(WG) \
                switch (rt_words) { \
                    case 16: paw_launch((paw_rt_walk_kernel<WG, 16>),\
                        ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(WG, 1, 1), 0, stream),\
                        (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n); break;\
                    case 24: paw_launch((paw_rt_walk_kernel<WG, 24>),\
                        ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(WG, 1, 1), 0, stream),\
                        (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n); break;\
                    case 32: paw_launch((paw_rt_walk_kernel<WG, 32>),\
                        ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(WG, 1, 1), 0, stream),\
                        (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n); break;\
                    default: GGML_ABORT("frag-dbg: rate");\
                }
            if (n/16 <= 128) { PAW_FRAG_DBG(128) } else { PAW_FRAG_DBG(256) }
            #undef PAW_FRAG_DBG
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> hv(m), hr(m);
            CUDA_CHECK(cudaMemcpy(hv.data(), scr_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hr.data(), ref_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            double md=0, sc=0; int maxi=-1;
            for (int i = 0; i < m; ++i) { double d=fabs(hv[i]-hr[i]); if(d>md){md=d;maxi=i;} sc=fmax(sc,fabs(hr[i])); }
            fprintf(stderr, "[frag-dbg] m=%d n=%d words=%d maxdiff=%g scale=%g rel=%.2e at %d | frag[0..2]=%g,%g,%g ref=%g,%g,%g\n",
                m,n,rt_words,md,sc,md/(sc+1e-30),maxi,hv[0],hv[1],hv[2],hr[0],hr[1],hr[2]);
            frag_dbg_calls++;
            return;   // skip normal path this call; v already computed by frag
        }
        // timing-only attribution knob: skip walks whose m is in the list
        // (wrong results; use with GGML_PAW_WALK_NOOP-style runs only)
        static const bool walk_skip_on = paw_env_int("GGML_PAW_WALK_SKIP_M_ON", 0) != 0;
        static const std::string walk_skip_csv = std::string(",") +
            (getenv("GGML_PAW_WALK_SKIP_M") ? getenv("GGML_PAW_WALK_SKIP_M") : "") + ",";
        const bool walk_skip = walk_skip_on &&
            walk_skip_csv.find("," + std::to_string(m) + ",") != std::string::npos;
        if (walk_skip_on) {
            static std::string seen;
            const std::string key = "," + std::to_string(m) + "x" + std::to_string(n) + "w" + std::to_string(rt_words);
            if (seen.find(key) == std::string::npos) { seen += key; fprintf(stderr, "[walk-shape]%s\n", key.c_str()); }
        }
        if (!walk_noop && !walk_skip && walk_frag) {
            CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) m*sizeof(float), stream));
            switch (rt_words) {
                case 16: paw_rt_walk_qtip_frag_dispatch<16>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                case 24: paw_rt_walk_qtip_frag_dispatch<24>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                case 32: paw_rt_walk_qtip_frag_dispatch<32>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                case 40: paw_rt_walk_qtip_frag_dispatch<40>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                case 56: paw_rt_walk_qtip_frag_dispatch<56>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                case 64: paw_rt_walk_qtip_frag_dispatch<64>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, m, n, stream); break;
                default: GGML_ABORT("paw: unsupported trellis rate for frag walk");
            }
        } else if (!walk_noop && !walk_skip) paw_timed(stream, std::string("rt_walk_qtip") + shp, [&]() {
        paw_rt_walk_qtip_rate_launch((const uint16_t *) trellis->data,
            (const half *) tlut->data, (const float *) scr_u, scr_v,
            m, n, rt_words, stream);
        });
        }
    } else if (k4 && bank_cache) {
        // Marlin-style: the bank is decoded once and cached; the per-step
        // decode is skipped and both paths run a plain GEMM/GEMV over it.
        const void * bank = paw_rt_bank_get(trellis->data, tlut->data, m, n, stream);
        const bool idx = paw_rt_bank_idx_on();
        const bool fp8 = !idx && paw_rt_bank_fp8_on();
        const half * dense_bank = idx ? paw_rt_idx_fp16_bank(bank, m, n) : (const half *) bank;
        static const bool rt_tc4     = paw_env_int("GGML_PAW_RT_TC", 8) == 4;
        static const bool rt_apply_mma = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
        if (gemvu) {
            // fused rt_u + bank GEMV; the half-reading variant must not run
            // against an fp8 cached bank, so pick the twin by bank format
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            if (fp8) {
                paw_launch(paw_rt_gemv_u_fp8_kernel,
                    ggml_cuda_kernel_launch_params(dim3(m/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) dense_bank, (const float *) su->data,
                    (const float *) x->data, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_gemv_u_kernel,
                    ggml_cuda_kernel_launch_params(dim3(m/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    dense_bank, (const float *) su->data, (const float *) x->data, scr_v, m, n, nt);
            }
            });
        } else if (idx && nt < dense_min_tok) {
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            paw_launch(paw_rt_bank_gemv_idx80,
                ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(512, 1, 1), 0, stream),
                (const uint16_t *) bank, (const half *) tlut->data,
                (const float *) scr_u, scr_v, m, n, nt);
            });
        } else if (fp8 && nt < dense_min_tok) {
            // fp8 bank gemv: decode-side only. Large batches go through the
            // dense-apply pipeline below, which re-decodes from trellis and
            // never touches the stored bank, so the bank format is free to
            // differ per batch size.
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            if (n % 4 == 0) {
                paw_launch(paw_rt_bank_gemv_fp8_v3,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_bank_gemv_fp8,
                    ggml_cuda_kernel_launch_params(dim3((m + 7)/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) bank, (const float *) scr_u, scr_v, m, n, nt);
            }
            });
        } else if (nt < dense_min_tok) {
            static const bool rt_gemv2 = paw_env_int("GGML_PAW_RT_GEMV2", 0) != 0;
            // float4+shared variant: measured ~3.5 t/s gen gain on RTX 3060
            static const bool rt_gemv3 = paw_env_int("GGML_PAW_RT_GEMV3", 1) != 0;
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            if (rt_gemv3 && n % 2 == 0) {
                paw_launch(paw_rt_bank_gemv_v3,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    dense_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else if (rt_gemv2 && n % 4 == 0) {
                paw_launch(paw_rt_bank_gemv_v2,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    dense_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
            paw_launch(paw_rt_bank_gemv,
                ggml_cuda_kernel_launch_params(dim3((m + 7)/8, 1, nt), dim3(256, 1, 1), 0, stream),
                dense_bank, (const float *) scr_u, scr_v, m, n, nt);
            }
            });
        } else {
            // an fp8 cached bank cannot feed the fp16 GEMM: decode a fresh
            // fp16 bank from trellis into scratch for this pass
            ggml_cuda_pool_alloc<half> fp16_bank_alloc(ctx.pool());
            const half * apply_bank = dense_bank;
            if (fp8) {
                apply_bank = fp16_bank_alloc.alloc((size_t) m*n);
                paw_launch(paw_rt_dense_decode_kernel,
                    ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data,
                    (half *) apply_bank, m, n);
            }
            paw_timed(stream, std::string("rt_apply") + shp, [&]() {
            if (rt_apply_mma) {
                paw_launch_rt_apply_mma(ctx, stream, apply_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else if (rt_tc4) {
                paw_launch(paw_rt_apply_kernel<4>,
                    ggml_cuda_kernel_launch_params(
                        dim3(m/16, 1, (nt + 3)/4), dim3(128, 1, 1), 0, stream),
                    apply_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_apply_kernel<8>,
                    ggml_cuda_kernel_launch_params(
                        dim3(m/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                    apply_bank, (const float *) scr_u, scr_v, m, n, nt);
            }
            });
        }
    } else if (!k4 && nt >= rt_blas_min_nt) {
        // dense payload prefill: materialize an fp16 bank ONCE per pass
        // (rate-templated decode) and run the same batched apply the K4
        // cached-bank branch uses. The generic walk this replaces re-decodes
        // every weight per token -- measured 7.4 t/s prompt at nt~1000.
        static const bool rt_apply_mma_d = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
        // split the matrix into two row halves and decode half 2 on the aux
        // stream while half 1 applies on the main stream -- hides roughly
        // half the decode traffic behind apply traffic. Same event pattern
        // as the expert blas pipeline; capture-unsafe, so graphs-off only.
        cudaStreamCaptureStatus dcst = cudaStreamCaptureStatusNone;
        const bool dcap = cudaStreamIsCapturing(stream, &dcst) == cudaSuccess &&
                          dcst == cudaStreamCaptureStatusActive;
        // default OFF: the two-half overlap produces corrupted logits after a
        // long prefill (generation degenerates to repeated '?'); bisected to
        // this commit. Re-enable only after the race is found.
        static const bool rt_ovl = paw_env_int("GGML_PAW_RT_DECODE_OVL", 0) != 0;
        // voff applies only on the cublas path, which needs nt >= its own
        // threshold; below that keep the serial single-bank flow
        const bool ovl = rt_ovl && !dcap && m >= 2048 &&
                         nt >= paw_env_int("GGML_PAW_RT_BLAS_MIN_TOK", 128);
        ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());
        half * apply_bank = bank_alloc.alloc((size_t) m*n);
        auto apply_fn = [&](const half * bk, int rows, int voff) {
            if (rt_apply_mma_d) {
                paw_launch_rt_apply_mma(ctx, stream, bk,
                    (const float *) scr_u, scr_v, rows, n, nt, voff);
            } else {
                paw_launch(paw_rt_apply_kernel<8>,
                    ggml_cuda_kernel_launch_params(
                        dim3(rows/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                    bk, (const float *) scr_u, scr_v, rows, n, nt);
            }
        };
        if (!ovl) {
            paw_rt_dense_decode_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, apply_bank, m, n, rt_words, stream);
            paw_timed(stream, std::string("rt_apply_dense") + shp, [&]() {
                apply_fn(apply_bank, m, 0);
            });
        } else {
            const int m0 = m/2;   // multiples of 16 guaranteed (m >= 2048)
            cudaEvent_t ev0, ev1;
            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));
            cudaStream_t ds = paw_aux_stream();
            paw_rt_dense_decode_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, apply_bank, m0, n, rt_words, stream);
            CUDA_CHECK(cudaEventRecord(ev0, stream));
            paw_rt_dense_decode_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, apply_bank + (size_t) m0*n,
                m - m0, n, rt_words, ds);
            CUDA_CHECK(cudaEventRecord(ev1, ds));
            paw_timed(stream, std::string("rt_apply_dense") + shp, [&]() {
                apply_fn(apply_bank, m0, 0);
                CUDA_CHECK(cudaStreamWaitEvent(stream, ev1));
                apply_fn(apply_bank + (size_t) m0*n, m - m0, m0);
            });
            CUDA_CHECK(cudaEventDestroy(ev0));
            CUDA_CHECK(cudaEventDestroy(ev1));
        }
    } else {
    ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());
    // paw_rt_dense_decode_kernel bakes in the K=4 stream layout: 64 words per
    // tile and the byte-aligned state window that only WORDS=64 gives. On a
    // lower-rate payload (paw-dense runs K=1/1.5/2) it reads past the end of
    // the trellis -- an illegal access on the first prefill, since nt >= 4
    // there. Only the walk below is rate-aware, so every other rate takes it.
    // TODO: templating the decode kernel on WORDS the way the walk is would
    // restore the prefill fast path for dense payloads; it needs the generic
    // bit-offset window extraction and a numeric check against the walk.
    if (k4 && nt >= dense_min_tok) {
        half * bank = bank_alloc.alloc((size_t) m*n);
        paw_timed(stream, std::string("rt_dense_decode") + shp, [&]() {
        paw_launch(paw_rt_dense_decode_kernel,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis->data, (const half *) tlut->data, bank, m, n);
        });
        static const bool rt_tc4     = paw_env_int("GGML_PAW_RT_TC", 8) == 4;
        static const bool rt_apply_mma = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
        paw_timed(stream, std::string("rt_apply") + shp, [&]() {
        if (rt_apply_mma) {
            // experimental tensor-core path -- see paw_rt_apply_kernel_mma
            // comment above. TC fixed at 16 (wmma's fp16 tile shape). m must
            // be a multiple of 16 (same assumption the scalar path already
            // makes via m/16 grid dims).
            paw_launch_rt_apply_mma(ctx, stream, (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        } else if (rt_tc4) {
            paw_launch(paw_rt_apply_kernel<4>,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 3)/4), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        } else {
            paw_launch(paw_rt_apply_kernel<8>,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        }
        });
    } else {
        static const bool rt_walk_mma = paw_env_int("GGML_PAW_RT_WALK_MMA", 0) != 0;
        paw_timed(stream, std::string("rt_walk") + shp, [&]() {
        if (rt_walk_mma) {
            paw_launch(paw_rt_walk_kernel_mma,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 15)/16), dim3(128, 1, 1), 0, stream),
                (const uint16_t *) trellis->data, (const half *) tlut->data,
                scr_u, scr_v, m, n, nt);
        } else {
            const int wg = n/16 <= 128 ? 128 : 256;
            auto go = [&](auto WGC, auto KC) {
                paw_launch((paw_rt_walk_kernel<decltype(WGC)::value, decltype(KC)::value>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(wg, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data,
                    scr_u, scr_v, m, n);
            };
            #define PAW_WALK_DISPATCH(WGV)                                        \
                switch (rt_words) {                                               \
                    case 16: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 16>{}); break;        \
                    case 24: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 24>{}); break;        \
                    case 32: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 32>{}); break;        \
                    case 40: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 40>{}); break;        \
                    case 48: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 48>{}); break;        \
                    case 56: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 56>{}); break;        \
                    case 64: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 64>{}); break;        \
                    default: GGML_ABORT("paw: unsupported trellis rate, "          \
                                        "%d words per tile", rt_words);           \
                }
            if (wg == 128) { PAW_WALK_DISPATCH(128) } else { PAW_WALK_DISPATCH(256) }
            #undef PAW_WALK_DISPATCH
        }
        });
    }
    }

    if (!uout_noop) paw_timed(stream, std::string("rt_out") + shp, [&]() {
    if (epilogue_dot) {
        GGML_ASSERT(!blocked && m == 2048 && paw_fwht_v2_on() && paw_fwht_v2_ok(m));
        paw_launch((paw_rt_out_epilogue_dot_kernel<128>),
            ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(128, 1, 1), 0, stream),
            (const float *) sv->data, scr_v, (float *) dst->data,
            (const float *) gate->data, (const float *) gate_x->data,
            (const float *) acc->data, m, (int) gate->ne[0], bm);
    } else
    if (!blocked && paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
        paw_fwht_for_wg(m/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            if (epilogue) {
                paw_launch((paw_rt_out_kernel<wg, true>),
                    ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(wg, 1, 1), 0, stream),
                    (const float *) sv->data, scr_v, (float *) dst->data,
                    (const float *) gate->data, (const float *) acc->data, m, bm);
            } else {
                paw_launch((paw_rt_out_kernel<wg, false>),
                    ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(wg, 1, 1), 0, stream),
                    (const float *) sv->data, scr_v, (float *) dst->data, nullptr, nullptr, m, bm);
            }
        });
    } else if (!blocked && paw_fwht_wg512()) {
        if (epilogue) {
            paw_launch((paw_rt_out_kernel<512, true>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(512, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data,
                (const float *) gate->data, (const float *) acc->data, m, bm);
        } else {
            paw_launch((paw_rt_out_kernel<512, false>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(512, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data, nullptr, nullptr, m, bm);
        }
    } else {
        if (epilogue) {
            paw_launch((paw_rt_out_kernel<256, true>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(256, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data,
                (const float *) gate->data, (const float *) acc->data, m, bm);
        } else {
            paw_launch((paw_rt_out_kernel<256, false>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(256, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data, nullptr, nullptr, m, bm);
        }
    }
    });
}

// ---------------------------------------------------------------------------
// batched RT_MM (GGML_OP_PAW_RT_MM_BATCH, GGML_PAW_RT_BATCH=1)
//
// K matrices sharing one input x run through three phase kernels (u, gemv,
// out) launched ONCE for the whole group instead of once per matrix. At
// nt=1 each per-matrix launch is latency/serialization-bound (~10-20 us for
// a few us of work); packing K independent matrices into one launch lets the
// blocks fill the GPU concurrently (measured 2.3x on a 3-matrix m=512 probe:
// 15.2 us serialized -> 6.6 us batched). The graph builder emits this op for
// adjacent same-x groups like (wq,wk,wv) and (wqkv,wqkv_gate); the op's
// output is the K outputs concatenated row-wise and sliced with views, so
// the batch is self-contained (flush-safe: consumers read the group's output
// only after this one op completes).
struct paw_rt_batch_desc {
    const void  * bank;   // cached bank [m, n]: fp16, or e5m2 fp8 (GGML_PAW_BANK_FP8=1)
    const half  * dense_bank;
    const half  * tlut;
    const float * su;     // [n]
    const float * sv;     // [m]
    float       * dst;    // output region (dst + row_off)
    int m, n;
    int u_off;            // offset into the batched scr_u
    int v_off;            // offset into the batched scr_v
    int row_off;          // output row offset
    int map_off;          // first row of an optional grouped-to-tiled segment
    int map_hd, map_k, map_r;
};

// u-phase: one block per (matrix, token); computes u = FWHT(su .* x) / sqrt(n)
// WG is a template knob (matches paw_rt_u_kernel's convention): n is the
// same for every matrix in a batch group (constructor-asserted), so unlike
// the out-kernel there's no per-matrix mismatch to guard against here --
// the call site can pick WG == n/16 for the whole launch whenever v2 applies.
template <int WG>
static __global__ void paw_rt_batch_u_kernel(
        paw_rt_batch_desc * GGML_CUDA_RESTRICT ddesc,
        const paw_rt_batch_desc d0, const paw_rt_batch_desc d1,
        const paw_rt_batch_desc d2, const paw_rt_batch_desc d3,
        const float * GGML_CUDA_RESTRICT x,
        float * GGML_CUDA_RESTRICT scr_u,
        const int nt) {
    const int mat = blockIdx.x;
    const int t   = blockIdx.z;
    const int tid = threadIdx.x;
    const paw_rt_batch_desc desc = mat == 0 ? d0 : mat == 1 ? d1 : mat == 2 ? d2 : d3;
    const int n = desc.n;
    __shared__ float sh[4096];

    if (t == 0 && tid == 0) {
        ddesc[mat] = desc;
    }
    ggml_cuda_pdl_sync();
    const float * su = desc.su;
    for (int i = tid; i < n; i += WG) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) n);
    const float inv_sc = __frcp_rn(sc);
    float * u = scr_u + desc.u_off + (int64_t) t*n;
    for (int i = tid; i < n; i += WG) {
        u[i] = sh[i] * inv_sc;
    }
}

// gemv phase: one block per 8 rows across the whole group; each block finds
// its matrix by scanning the (small) descriptor array.
static __global__ void paw_rt_batch_gemv_kernel(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v,
        const int nt, const int n_matrices) {
    const int b    = blockIdx.x;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;

    int mat = 0, base = 0;
    while (mat + 1 < n_matrices && b >= base + desc[mat].m/32) {
        base += desc[mat].m/32;
        mat++;
    }
    const int m = desc[mat].m;
    const int n = desc[mat].n;
    const int row = (b - base)*8 + (threadIdx.x >> 5);
    if (row >= m) {
        return;
    }

    const float  * u  = scr_u + desc[mat].u_off + (int64_t) t*n;
    const half2  * W2 = (const half2 *) (desc[mat].dense_bank + (int64_t) row*n);
    const float2 * u2 = (const float2 *) u;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const float2 w = __half22float2(W2[i]);
        const float2 uu = u2[i];
        acc += w.x*uu.x + w.y*uu.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[desc[mat].v_off + (int64_t) t*m + row] = acc;
    }
}

static __global__ void paw_rt_batch_gemv_kernel_idx80(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v,
        const int nt, const int n_matrices) {
    const int b    = blockIdx.x;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    int mat = 0, base = 0;
    while (mat + 1 < n_matrices && b >= base + desc[mat].m/16) {
        base += desc[mat].m/16;
        mat++;
    }

    __shared__ half2 slut[512];
    for (int i = tid; i < 512; i += blockDim.x) {
        slut[i] = ((const half2 *) desc[mat].tlut)[i];
    }
    __syncthreads();

    const int m = desc[mat].m;
    const int n = desc[mat].n;
    const int row = (b - base)*16 + (tid >> 5);
    if (row >= m) {
        return;
    }

    const float    * u = scr_u + desc[mat].u_off + (int64_t) t*n;
    const uint32_t * W = (const uint32_t *) desc[mat].bank + (int64_t) row*(n/32)*5;
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
        scr_v[desc[mat].v_off + (int64_t) t*m + row] = acc;
    }
}

// fp8 twin of paw_rt_batch_gemv_kernel: bank is e5m2 (1 byte/weight), same
// per-block software LUT + uint32/float4 wide loads as paw_rt_bank_gemv_fp8.
static __global__ void paw_rt_batch_gemv_kernel_fp8(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v,
        const int nt, const int n_matrices) {
    __shared__ float lut[256];
    const int tid0 = threadIdx.x;
    if (tid0 < 256) {
        lut[tid0] = paw_e5m2_to_f32((uint8_t) tid0);
    }
    __syncthreads();

    const int b    = blockIdx.x;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;

    int mat = 0, base = 0;
    while (mat + 1 < n_matrices && b >= base + desc[mat].m/8) {
        base += desc[mat].m/8;
        mat++;
    }
    const int m = desc[mat].m;
    const int n = desc[mat].n;
    const int row = (b - base)*8 + (threadIdx.x >> 5);
    if (row >= m) {
        return;
    }

    const float   * u = scr_u + desc[mat].u_off + (int64_t) t*n;
    const uint8_t * W = (const uint8_t *) desc[mat].bank + (int64_t) row*n;

    float acc = 0.0f;
    if (n % 4 == 0) {
        const uint32_t * W4 = (const uint32_t *) W;
        const float4   * u4 = (const float4 *) u;
        const int n4 = n/4;
        for (int i = lane; i < n4; i += 128) {
            const uint32_t w = __ldcs(W4 + i);
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
        scr_v[desc[mat].v_off + (int64_t) t*m + row] = acc;
    }
}

// out phase: one block per (matrix, token); FWHT over m, scale by sv
static __global__ void paw_rt_batch_out_kernel(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_v,
        float * GGML_CUDA_RESTRICT dst,
        const int nt, const int m_sum) {
    const int mat = blockIdx.x;
    const int t   = blockIdx.z;
    const int tid = threadIdx.x;
    const int m   = desc[mat].m;
    const int wg  = blockDim.x;   // == m/16, so fwht_block sees nslots <= 16
    __shared__ float sh[8192];

    ggml_cuda_pdl_sync();
    const float * v = scr_v + desc[mat].v_off + (int64_t) t*m;
    for (int i = tid; i < m; i += wg) {
        __pipeline_memcpy_async(&sh[i], &v[i], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // wg is sized for the group's largest matrix (blockDim.x == max_m/16), so
    // v2's wg==m/16 precondition only holds when this matrix's own m equals
    // that max -- must check per-matrix, not assume it like a single-matrix
    // launch could (paw_fwht_block itself is wg-agnostic either way, so v1
    // here is always safe, just not maximally fast).
    if (wg == m/16 && paw_fwht_v2_ok(m)) {
        paw_fwht_block_v2(sh, m, tid, wg);
    } else {
        paw_fwht_block(sh, m, tid, wg);
    }
    const float sc     = __fsqrt_rn((float) m);
    const float inv_sc = __frcp_rn(sc);
    const float * sv = desc[mat].sv;
    // dst is the single [m_sum, T] concatenated output tensor (ggml.c
    // allocates it contiguous with ne[0]=m_sum), so its per-token stride is
    // m_sum, NOT this matrix's own m -- using m here aliased every token
    // after the first for any group with >1 matrix (invisible at nt==1,
    // corrupting every prompt-eval pass since nt there is the prompt length).
    float * y = dst + desc[mat].row_off + (int64_t) t*m_sum;
    for (int i = tid; i < m; i += wg) {
        int oi = i;
        if (i >= desc[mat].map_off && desc[mat].map_hd != 0) {
            const int j = i - desc[mat].map_off;
            const int d = j % desc[mat].map_hd;
            const int kr = j / desc[mat].map_hd;
            const int k = kr / desc[mat].map_r;
            const int vhead = kr % desc[mat].map_r;
            oi = desc[mat].map_off + (vhead*desc[mat].map_k + k)*desc[mat].map_hd + d;
        }
        y[oi] = sh[i] * inv_sc * sv[i];
    }
}

void ggml_cuda_op_paw_rt_mm_batch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!paw_bank_cache_on()) {
        GGML_ASSERT(false);   // batch op requires the bank cache (default on)
    }

    const int n_matrices = dst->op_params[0];
    GGML_ASSERT(n_matrices >= 2 && n_matrices <= 4);
    GGML_ASSERT(dst->op == GGML_OP_PAW_RT_MM_BATCH);
    // supports_op already refuses these, so reaching here means a caller built
    // the op directly; fail loudly rather than rotate with the wrong basis.
    GGML_ASSERT(dst->op_params[GGML_PAW_RHT_BLK_SLOT] == 0 &&
                "paw: the batched RT path has no blocked-rotation kernel yet");

    // tlut is src[3K], x is src[3K+1]
    const ggml_tensor * tlut = dst->src[3*n_matrices];
    const ggml_tensor * x    = dst->src[3*n_matrices + 1];
    const int n = (int) x->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(tlut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    // scratch: u[n_sum] + v[m_sum], per-matrix contiguous
    int n_sum = 0, m_sum = 0;
    for (int i = 0; i < n_matrices; ++i) {
        n_sum += (int) dst->src[3*i + 1]->ne[0];
        m_sum += (int) dst->src[3*i + 2]->ne[0];
    }

    // Persistent device descriptors are published by the u-phase kernel from
    // by-value arguments captured in the graph node.
    static paw_rt_batch_desc * ddesc = nullptr;
    if (ddesc == nullptr) {
        CUDA_CHECK(cudaMalloc(&ddesc, sizeof(paw_rt_batch_desc)*4));
    }

    ggml_cuda_pool_alloc<float> scr(ctx.pool(), (size_t) nt*n_sum + (size_t) nt*m_sum);
    float * scr_u = scr.get();
    float * scr_v = scr_u + (size_t) nt*n_sum;

    // build the descriptor array on the host (static, so the address is
    // stable; the desc-writer kernel copies it by value every call).
    static paw_rt_batch_desc hdesc[4];
    int n_off = 0, m_off = 0, row_off = 0;
    int nblk = 0, nblk16 = 0;
    for (int i = 0; i < n_matrices; ++i) {
        const ggml_tensor * trellis = dst->src[3*i + 0];
        const ggml_tensor * su      = dst->src[3*i + 1];
        const ggml_tensor * sv      = dst->src[3*i + 2];
        const int mi = (int) sv->ne[0];
        const int ni = (int) su->ne[0];
        GGML_ASSERT(ni == n);
        GGML_ASSERT(ni >= 256 && ni % 2 == 0 && mi % 8 == 0);
        hdesc[i].bank    = paw_rt_bank_get(trellis->data, tlut->data, mi, ni, stream);
        hdesc[i].dense_bank = paw_rt_bank_idx_on() ? paw_rt_idx_fp16_bank(hdesc[i].bank, mi, ni) : (const half *) hdesc[i].bank;
        hdesc[i].tlut    = (const half *) tlut->data;
        hdesc[i].su      = (const float *) su->data;
        hdesc[i].sv      = (const float *) sv->data;
        hdesc[i].dst     = (float *) dst->data;
        hdesc[i].m       = mi;
        hdesc[i].n       = ni;
        hdesc[i].u_off   = n_off;
        hdesc[i].v_off   = m_off;
        hdesc[i].row_off = row_off;
        if (dst->op_params[8] != 0) {
            hdesc[i].map_hd  = dst->op_params[9];
            hdesc[i].map_k   = dst->op_params[10];
            hdesc[i].map_r   = dst->op_params[11];
            hdesc[i].map_off = i == 0 ? 2*hdesc[i].map_hd*hdesc[i].map_k : 0;
            GGML_ASSERT(hdesc[i].map_off + hdesc[i].map_hd*hdesc[i].map_k*hdesc[i].map_r == mi);
        } else {
            hdesc[i].map_off = mi;
            hdesc[i].map_hd  = 0;
            hdesc[i].map_k   = 0;
            hdesc[i].map_r   = 0;
        }
        // NOTE: the scratch is [nt, n_sum] / [nt, m_sum]; the kernels index it
        // as [u_off + t*n + i], so per-matrix offsets must stride by nt*n /
        // nt*m (NOT n / m -- that aliases matrix regions when nt > 1).
        n_off  += nt*ni;
        m_off  += nt*mi;
        row_off += mi;
        nblk   += mi/8;
        nblk16 += mi/16;
    }

    char shp[64];
    snprintf(shp, sizeof(shp), " batch=%d nt=%d n=%d", n_matrices, nt, n);

    paw_timed(stream, std::string("rt_batch_u") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
        paw_fwht_for_wg(n/16, [&](auto WGC) {
            constexpr int wg = decltype(WGC)::value;
            paw_launch(paw_rt_batch_u_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(n_matrices, 1, nt), dim3(wg, 1, 1), 0, stream),
                ddesc, hdesc[0], hdesc[1], hdesc[2], hdesc[3], (const float *) x->data, scr_u, nt);
        });
    } else {
        paw_launch(paw_rt_batch_u_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_matrices, 1, nt), dim3(256, 1, 1), 0, stream),
            ddesc, hdesc[0], hdesc[1], hdesc[2], hdesc[3], (const float *) x->data, scr_u, nt);
    }
    });

    static const int  dense_min_tok = paw_env_int("GGML_PAW_DENSE_MIN_TOK", 4);
    static const bool rt_tc4        = paw_env_int("GGML_PAW_RT_TC", 8) == 4;
    static const bool rt_apply_mma  = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
    const bool idx = paw_rt_bank_idx_on();
    const bool fp8 = !idx && paw_rt_bank_fp8_on();
    if (idx && nt < dense_min_tok) {
        paw_timed(stream, std::string("rt_batch_gemv") + shp, [&]() {
        paw_launch(paw_rt_batch_gemv_kernel_idx80,
            ggml_cuda_kernel_launch_params(dim3(nblk16, 1, nt), dim3(512, 1, 1), 0, stream),
            ddesc, scr_u, scr_v, nt, n_matrices);
        });
    } else if (fp8) {
        // fp8 bank: the dense-apply kernels (paw_rt_apply_kernel*) read the
        // wrong element size, so one gemv kernel serves every nt (decode and
        // prefill alike), same simplification as the non-batched fp8 path.
        paw_timed(stream, std::string("rt_batch_gemv") + shp, [&]() {
        paw_launch(paw_rt_batch_gemv_kernel_fp8,
            ggml_cuda_kernel_launch_params(dim3(nblk, 1, nt), dim3(256, 1, 1), 0, stream),
            ddesc, scr_u, scr_v, nt, n_matrices);
        });
    } else if (nt >= dense_min_tok) {
        // prompt eval / batched decode: use the same dense apply kernels as
        // the per-matrix path (the gemv path below is a different summation
        // order; switching nt>=4 to it changes prompt-eval rounding and
        // diverges generation).
        for (int i = 0; i < n_matrices; ++i) {
            const int mi = hdesc[i].m;
            const int ni = hdesc[i].n;
            const float * u_i = scr_u + hdesc[i].u_off;
            float * v_i       = scr_v + hdesc[i].v_off;
            paw_timed(stream, std::string("rt_apply") + shp, [&]() {
            if (rt_apply_mma) {
                paw_launch_rt_apply_mma(ctx, stream, hdesc[i].dense_bank, u_i, v_i, mi, ni, nt);
            } else if (rt_tc4) {
                paw_launch(paw_rt_apply_kernel<4>,
                    ggml_cuda_kernel_launch_params(
                        dim3(mi/16, 1, (nt + 3)/4), dim3(128, 1, 1), 0, stream),
                    hdesc[i].dense_bank, u_i, v_i, mi, ni, nt);
            } else {
                paw_launch(paw_rt_apply_kernel<8>,
                    ggml_cuda_kernel_launch_params(
                        dim3(mi/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                    hdesc[i].dense_bank, u_i, v_i, mi, ni, nt);
            }
            });
        }

    } else {
        paw_timed(stream, std::string("rt_batch_gemv") + shp, [&]() {
        paw_launch(paw_rt_batch_gemv_kernel,
            ggml_cuda_kernel_launch_params(dim3(nblk, 1, nt), dim3(256, 1, 1), 0, stream),
            ddesc, scr_u, scr_v, nt, n_matrices);
        });
    }

    // out phase: FWHT(m) + sv-scale + write into dst. Must run regardless of
    // which branch above filled scr_v -- the nt>=dense_min_tok branch used to
    // fall straight through to the function's end without this, leaving dst
    // as stale pool memory for every prompt-eval pass (nt = prompt length,
    // virtually always >= dense_min_tok=4). That corrupted the hidden state
    // from the very first token of every request while remaining invisible
    // to nt==1 decode-only correctness checks, which never took this branch.
    paw_timed(stream, std::string("rt_batch_out") + shp, [&]() {
    int max_m = 0;
    for (int i = 0; i < n_matrices; ++i) {
        max_m = std::max(max_m, hdesc[i].m);
    }
    const int out_wg = max_m/16;
    paw_launch(paw_rt_batch_out_kernel,
        ggml_cuda_kernel_launch_params(dim3(n_matrices, 1, nt), dim3(out_wg, 1, 1), 0, stream),
        ddesc, scr_v, (float *) dst->data, nt, m_sum);
    });
}


//
// EXP_BASIS — one block per (slot, token) pair (paw_exp_basis.comp)
//

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

void ggml_cuda_op_paw_exp_basis(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * b     = dst->src[1];
    const ggml_tensor * c     = dst->src[2];
    const ggml_tensor * remap = dst->src[3];
    const ggml_tensor * ids   = dst->src[4];
    const ggml_tensor * x     = dst->src[5];
    const ggml_tensor * accs  = dst->src[6];

    GGML_ASSERT(a->type     == GGML_TYPE_F16);
    GGML_ASSERT(b->type     == GGML_TYPE_F16);
    GGML_ASSERT(c->type     == GGML_TYPE_F16);
    GGML_ASSERT(remap->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(x->type     == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(c));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    if (accs != nullptr) {
        GGML_ASSERT(accs->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(accs));
    }

    const int n      = (int) a->ne[0];
    const int r      = (int) a->ne[1];
    const int m      = (int) b->ne[1];
    const int n_used = (int) ids->ne[0];
    const int n_tok  = (int) ids->ne[1];
    const int xne1   = (int) x->ne[1];
    const int ids_s0 = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1 = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(r <= 256);   // tv shared bound

    paw_launch(paw_exp_basis_kernel,
        ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, ctx.stream()),
        (const half    *) a->data,
        (const half    *) b->data,
        (const half    *) c->data,
        (const int32_t *) remap->data,
        (const int32_t *) ids->data,
        (const float   *) x->data,
        accs != nullptr ? (const float *) accs->data : nullptr,
        (float         *) dst->data,
        n, r, m, n_used, xne1, ids_s0, ids_s1, accs != nullptr ? 1 : 0);
}

//
// EXP_MM — 4 kernels: group pairs by storage expert -> u = H(su_e ⊙ x) per
// pair -> trellis walk/dot per group -> y = sv_e ⊙ H(v) per pair
// (paw_exp_group/exp_u/exp_walk/exp_out.comp). Scratch: int32
// [4*n_groups + P] (cnt | off | orig | cursor(unused) | pairs) and float
// [P*n + P*m] (u then v).
//

static __global__ void paw_exp_group_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t       * GGML_CUDA_RESTRICT scr,
        const int n_used, const int n_tok, const int n_kept, const int n_groups,
        const int ids_s0, const int ids_s1) {
    constexpr int WG = 256;
    __shared__ int sh_cnt[512];    // MAX_GROUPS = 512, enforced by supports_op
    __shared__ int sh_cur[512];
    __shared__ int sh_orig[512];

    const int tid = threadIdx.x;
    const int P   = n_used*n_tok;

    ggml_cuda_pdl_sync();
    for (int g = tid; g < n_groups; g += WG) {
        sh_cnt[g]  = 0;
        sh_orig[g] = 0;
    }
    __syncthreads();
    for (int p = tid; p < P; p += WG) {
        const int s = p % n_used;
        const int t = p / n_used;
        const int g = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
        atomicAdd(&sh_cnt[g], 1);
        atomicExch(&sh_orig[g], ids[s*ids_s0 + t*ids_s1]);   // orig id (same value per group)
    }
    __syncthreads();
    if (tid == 0) {
        int off = 0;
        for (int g = 0; g < n_groups; ++g) {
            const int c = sh_cnt[g];
            sh_cur[g] = off;
            scr[g]              = c;            // cnt
            scr[n_groups + g]   = off;          // off
            scr[2*n_groups + g] = sh_orig[g];   // orig id
            off += c;
        }
    }
    __syncthreads();
    for (int p = tid; p < P; p += WG) {
        const int g   = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
        const int pos = atomicAdd(&sh_cur[g], 1);
        scr[4*n_groups + pos] = p;              // flat pair rank p = t*n_used + s
    }
}

template <int WG>
static __global__ void paw_exp_u_kernel(
        const half    * GGML_CUDA_RESTRICT su,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT x,
        float         * GGML_CUDA_RESTRICT scr_u,
        const int n, const int n_used, const int xne1,
        const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;
    for (int i = tid; i < n; i += WG) {
        sh[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) p*n + i] = __fdiv_rn(sh[i], sc);
    }
}

// --- exp_group + exp_u launch fusion, GGML_PAW_EXP_GROUP_FUSE=1 ---
//
// exp_group_kernel is a single, tiny block (~10.8us average per call per
// GGML_PAW_TIME -- almost entirely kernel-launch overhead, the real work
// is a couple hundred int ops) that's otherwise independent of exp_u_kernel
// (exp_u reads only su/ids/x, never scr_i -- no data dependency between
// them). This shares one launch: blockIdx.x==0 && blockIdx.y==0 does
// exp_group's unchanged body; every other block does exp_u's unchanged
// body with s = blockIdx.x - 1. Grid grows from (n_used, n_tok, 1) to
// (n_used+1, n_tok, 1); the extra (0, y>0) blocks when n_tok>1 are
// harmless no-ops (exp_group's result doesn't depend on t). Both bodies
// are copied verbatim -- no numeric changes, purely a launch-count cut.
template <int WG>
static __global__ void paw_exp_group_u_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT scr,
        const half     * GGML_CUDA_RESTRICT su,
        const float    * GGML_CUDA_RESTRICT x,
        float          * GGML_CUDA_RESTRICT scr_u,
        const int n, const int n_used, const int n_tok, const int n_kept, const int n_groups,
        const int xne1, const int ids_s0, const int ids_s1) {
    if (blockIdx.x == 0) {
        if (blockIdx.y != 0) {
            return;   // exp_group's result is global, only run it once
        }
        __shared__ int sh_cnt[512];
        __shared__ int sh_cur[512];
        __shared__ int sh_orig[512];

        const int tid = threadIdx.x;
        const int P   = n_used*n_tok;

        ggml_cuda_pdl_sync();
        for (int g = tid; g < n_groups; g += WG) {
            sh_cnt[g]  = 0;
            sh_orig[g] = 0;
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int s = p % n_used;
            const int t = p / n_used;
            const int g = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            atomicAdd(&sh_cnt[g], 1);
            atomicExch(&sh_orig[g], ids[s*ids_s0 + t*ids_s1]);
        }
        __syncthreads();
        if (tid == 0) {
            int off = 0;
            for (int g = 0; g < n_groups; ++g) {
                const int c = sh_cnt[g];
                sh_cur[g] = off;
                scr[g]              = c;
                scr[n_groups + g]   = off;
                scr[2*n_groups + g] = sh_orig[g];
                off += c;
            }
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int g   = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            const int pos = atomicAdd(&sh_cur[g], 1);
            scr[4*n_groups + pos] = p;
        }
        return;
    }

    __shared__ float sh[2048];

    const int s   = blockIdx.x - 1;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;
    for (int i = tid; i < n; i += WG) {
        sh[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) p*n + i] = __fdiv_rn(sh[i], sc);
    }
}

// One-time host-side repack of the V8 tlut (F16 [8, 32768], 512 KB) into the
// Mach-1 engine's p4 form (PAW_V8_LUT=p4): the V8 alphabet is <= 16 distinct
// fp16 levels, so a codeword row becomes one nibble-packed u32 (128 KB table)
// resolved through a 16-entry level table. The levels are stored as the exact
// fp32 values of the distinct F16 entries, so the resolved value is
// bit-identical to __half2float of the original entry by construction (no
// step multiply at runtime; +0/-0 collapse to one level like np.unique in
// the reference). Falls back to the fp16-gather path (packed == nullptr)
// when the alphabet has more than 16 distinct values.
//
// The cache is keyed by the tlut's device pointer and is NEVER freed: one
// entry is 128 KB + 64 B per distinct tlut (one per model) and the key's
// lifetime matches the weight allocation — intentionally leaked.
struct paw_p4_table {
    const uint32_t * packed;   // device [32768], 8 x 4-bit level indices per row
    const float    * levels;   // device [16], exact fp32 of the fp16 alphabet
};

static paw_p4_table paw_exp_p4_table(const void * tlut_data, cudaStream_t stream) {
    struct entry { const uint32_t * packed; const float * levels; };
    static std::mutex paw_p4_mutex;
    static std::unordered_map<const void *, entry> paw_p4_cache;

    std::lock_guard<std::mutex> lock(paw_p4_mutex);
    const auto it = paw_p4_cache.find(tlut_data);
    if (it != paw_p4_cache.end()) {
        return {it->second.packed, it->second.levels};
    }

    constexpr int ROWS = 32768;
    std::vector<uint16_t> host(8*ROWS);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tlut_data, host.size()*sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float levels[16];
    int   n_levels = 0;
    bool  ok = true;
    for (size_t i = 0; i < host.size(); ++i) {
        const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[i]);
        if (v != v) {   // NaN can never match a level slot
            ok = false;
            break;
        }
        int j = 0;
        while (j < n_levels && levels[j] != v) {
            j++;
        }
        if (j == n_levels) {
            if (n_levels == 16) {
                ok = false;
                break;
            }
            levels[n_levels++] = v;
        }
    }

    entry e = {nullptr, nullptr};
    if (ok) {
        std::sort(levels, levels + n_levels);   // deterministic nibble order
        std::vector<uint32_t> packed(ROWS);
        for (int r = 0; r < ROWS; ++r) {
            uint32_t pk = 0;
            for (int c = 0; c < 8; ++c) {
                const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[8*r + c]);
                uint32_t idx = 0;
                while (levels[idx] != v) {
                    idx++;
                }
                pk |= idx << (4*c);
            }
            packed[r] = pk;
        }
        float lv16[16] = {0.0f};
        for (int i = 0; i < n_levels; ++i) {
            lv16[i] = levels[i];
        }
        void * buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, ROWS*sizeof(uint32_t) + 16*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(buf, packed.data(), ROWS*sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) buf + ROWS*sizeof(uint32_t), lv16,
                                   16*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        e.packed = (const uint32_t *) buf;
        e.levels = (const float *)((const char *) buf + ROWS*sizeof(uint32_t));
    }
    paw_p4_cache.emplace(tlut_data, e);
    if (paw_debug_on() || paw_time_on()) {
        fprintf(stderr, "paw: p4 tlut repack %s (n_levels=%d, tlut=%p)\n",
                ok ? "ENGAGED" : "FALLBACK (fp16 gather)", n_levels, tlut_data);
    }
    return {e.packed, e.levels};
}

// V2 serial walk over register-held trellis words. STEPB/WORDS are template
// constants so the fully-unrolled window indices are compile-time and wl
// stays in registers (kept: <4,32>, demoted: <2,16>). Numerics identical to
// the previous in-loop decode.
template <int STEPB, int WORDS>
static __device__ __forceinline__ void paw_exp_walk_v2_steps(
        const uint16_t * GGML_CUDA_RESTRICT wl,      // [WORDS] registers
        const float    * GGML_CUDA_RESTRICT tlut,    // F32 [512, 2]
        const float    * GGML_CUDA_RESTRICT ub,
        float          * GGML_CUDA_RESTRICT partial) {
#pragma unroll
    for (int i = 0; i < 128; ++i) {
        // direct-window state: 16 bits at stream bit STEPB*i (wrapping)
        const int      bb  = STEPB*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < WORDS ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) wl[wi] << 16) | (uint32_t) wl[wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        float v0 = tlut[2*row + 0];
        const float v1raw = tlut[2*row + 1];
        if (ph & 0x8000u) {
            v0 = -v0;                         // exact either side of the round
        }
        // hatWr is defined at fp16 precision — pinned round-trip
        const float w0 = __half2float(__float2half_rn(v0));
        const float w1 = __half2float(__float2half_rn(v1raw));
        const int ri = (2*i) >> 4;
        const int ci = (2*i) & 15;
        partial[ri] += w0*ub[ci] + w1*ub[ci + 1];
    }
}

// V8 = false: payload v2, V=2, K=2 kept (32 words) / K=1 demoted (16 words),
//             tlut F32 [512,2], row = (p>>6)&511, per-value fp16 round here.
// V8 = true:  payload v3, V=8, K=1.5 (24 words, 12 fresh bits per step), tlut
//             pre-rounded F16 [32768,8], row = p&0x7FFF, per-tile wave gamma
//             applied after the round, no demoted tier.
// P4 (V8 only): gather one nibble-packed u32 per state from the repacked
//             table and resolve levels from shared — bit-identical values.
// --- grid compaction for exp_walk at nt==1, GGML_PAW_EXP_WALK_COMPACT=1 ---
//
// consolidated P4 tables in a single allocation so an L2 persisting window
// can pin them: the fused kernel's gathers were going to DRAM whenever the
// live attention/KV traffic evicted the tables between launches
struct paw_l2_tables {
    uint32_t * packed = nullptr;
    float    * levels = nullptr;
    void     * base   = nullptr;
};
static paw_l2_tables g_paw_l2tab;

static const paw_l2_tables & paw_l2_tables_get(cudaStream_t stream,
        const paw_p4_table & t) {
    if (!g_paw_l2tab.base && t.packed) {
        constexpr size_t packed_b = (size_t) 32768*4;
        constexpr size_t levels_b = 16*4;
        void * base = nullptr;
        if (cudaMalloc(&base, packed_b + levels_b) == cudaSuccess) {
            CUDA_CHECK(cudaMemcpyAsync(base, t.packed, packed_b,
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync((char *) base + packed_b, t.levels,
                levels_b, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr = base;
            attr.accessPolicyWindow.num_bytes = packed_b + levels_b;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream,
                cudaStreamAttributeAccessPolicyWindow, &attr);
            size_t carve = 0;
            cudaDeviceGetLimit(&carve, cudaLimitPersistingL2CacheSize);
            if (carve < packed_b + levels_b) {
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                    packed_b + levels_b);
            }
            g_paw_l2tab.packed = (uint32_t *) base;
            g_paw_l2tab.levels = (float *)((char *) base + packed_b);
            g_paw_l2tab.base = base;
        }
    }
    return g_paw_l2tab;
}


// The walk kernels below launch grid.z=n_groups (up to 256) but at nt==1
// only n_used (~8) are ever active -- every other block reads scr_i[g]==0
// and exits immediately. This writes the n_used active group ids directly
// (no scan, no early-exit blocks needed) so the dispatcher can launch
// grid.z=n_used instead. Unlike the (reverted) per-slot bank cache, this
// changes NOTHING about what gets decoded or how -- same fused decode+
// accumulate, same scr_i-indexed arrays, just fewer blocks that would have
// scanned-and-skipped. Safe at nt==1 specifically because top-k routing
// guarantees no expert appears twice in one token's own routing list, so
// there's no cnt>1 sharing case this bypasses.
static __global__ void paw_exp_active_groups_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT active_g,
        const int n_used, const int n_kept, const int ids_s0) {
    const int s = threadIdx.x;
    if (s >= n_used) {
        return;
    }
    active_g[s] = (int32_t) paw_group_of(remap, ids, s, n_used, n_kept, ids_s0, 0);
}


// The tile's trellis words and (for V8) the whole per-step state stream are
// pair-invariant, so they are hoisted out of the pair loop (for V2 the full
// 128-step state precompute would cost ~64+ registers, so only the words are
// hoisted there).
template <bool V8, bool P4>
static __global__ void paw_exp_walk_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const uint16_t * GGML_CUDA_RESTRICT dem,     // placeholder (= kept) when absent
        const void     * GGML_CUDA_RESTRICT tlut,    // float (V2) or half (V8)
        const uint32_t * GGML_CUDA_RESTRICT p4,      // P4 only, nullptr otherwise
        const float    * GGML_CUDA_RESTRICT p4lv,    // P4 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const half     * GGML_CUDA_RESTRICT gamma,   // V8 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT active_g, // compacted group ids, nullptr = scan all n_groups
        const int m, const int n, const int n_kept, const int n_groups) {
    static_assert(V8 || !P4, "P4 is a V8-tlut repack");
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float lv[16];       // P4 level table

    (void) dem;
    (void) gamma;
    (void) p4;
    (void) p4lv;

    const int g   = active_g != nullptr ? active_g[blockIdx.z] : blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    int              words;
    int              stepb;        // fresh bits per step
    const uint16_t * trd;
    int64_t          tbase;
    float            gsc = 0.0f;
    if constexpr (V8) {
        words = 24;
        stepb = 12;
        trd   = kept;
        tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;
        // per-tile wave gamma: last anti-diagonal wavefront writing tile (tr, tid)
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        if (have) {
            const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                     : Mb + tiles_y - 2 - (tr + tid);
            gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
    } else {
        const bool is_dem = g >= n_kept;
        const int  ei     = is_dem ? g - n_kept : g;
        words = is_dem ? 16 : 32;
        stepb = is_dem ? 2 : 4;
        trd   = is_dem ? dem : kept;
        tbase = ((int64_t) ei*ntiles + (int64_t) tr*tiles_y)*words;
    }
    (void) stepb;

    // V2 stages the tile words in registers (pair-invariant hoist); V8's
    // window state is recomputed per-qq instead of cached (see exp_walk2's
    // matching change -- caching phv[32] only pays off when cnt is large,
    // which it typically isn't at decode; w8[24] is the actual global read
    // so that alone stays resident).
    uint16_t w8[V8 ? 24 : 1];
    uint16_t wl[V8 ? 1 : 32];
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
        if constexpr (V8) {
#pragma unroll
            for (int q = 0; q < 24; ++q) {
                w8[q] = trd[tw + q];
            }
        } else {
#pragma unroll
            for (int q = 0; q < 32; ++q) {   // predicated: no read past 16-word dem tiles
                wl[q] = q < words ? trd[tw + q] : (uint16_t) 0;
            }
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    // one pair per iteration on purpose — see the MEASURED note in
    // paw_exp_walk.comp: blocking pairs 2-at-a-time regressed 3x.
    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + tid*16;
            if constexpr (V8) {
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int      bb  = 12*i;
                    const int      wi  = bb >> 4;
                    const int      o   = bb & 15;
                    const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                    const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                    const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
                    const uint32_t ph  = reg*(reg + 1u);
                    const uint32_t row = ph & 0x7FFFu;
                    const int      ri  = (8*i) >> 4;
                    const int      ci  = (8*i) & 15;
                    float dotp = 0.0f;
                    if constexpr (P4) {
                        const uint32_t pk = p4[row];
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            // exact fp32 of the fp16 entry — bit-identical to
                            // the __half2float gather; gamma order unchanged
                            float vv = lv[(pk >> (4*c)) & 0xFu];
                            if (c == 0 && (ph & 0x8000u)) {
                                vv = -vv;                 // exact (sign bit)
                            }
                            dotp += (vv * gsc) * ub[ci + c];
                        }
                    } else {
                        const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            // tlut is pre-rounded fp16; wave gamma then
                            // multiplies in fp32 (reference op order)
                            float vv = __half2float(tl[c]);
                            if (c == 0 && (ph & 0x8000u)) {
                                vv = -vv;                 // exact in fp16 (sign bit)
                            }
                            dotp += (vv * gsc) * ub[ci + c];
                        }
                    }
                    partial[ri] += dotp;
                }
            } else {
                // stepb is block-uniform (kept vs demoted group); branch so
                // the unrolled walk sees compile-time window indices
                if (words == 32) {
                    paw_exp_walk_v2_steps<4, 32>(wl, (const float *) tlut, ub, partial);
                } else {
                    paw_exp_walk_v2_steps<2, 16>(wl, (const float *) tlut, ub, partial);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(partial[i]);
            if (lane == 0) {
                red[i][wid] = s;
            }
        }
        __syncthreads();
        if (tid < 16) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[tid][wj];
            }
            scr_v[(int64_t) p*m + tr*16 + tid] = sum;
        }
        if (qq + 1 < cnt) __syncthreads();   // red is reused by the next pair -- skip on last iter, nothing left to protect
    }
}

// V8 fused walk for narrow inputs (tiles_y <= warp_size): one WARP per row
// tile, lanes = column tiles, 4 row tiles per block — the block-per-row-tile
// kernel above leaves (WG - tiles_y) of its threads idle in the walk, which
// wastes 3/4 of the block for the real model's down projection (n = 512).
// Per-(row tile, column tile) state math, gathers, wave gamma and per-pair
// accumulation order are identical to paw_exp_walk_kernel<true, P4>; the
// warp reduction is the same shuffle tree, and the cross-warp shared step it
// replaces only ever added exact zeros there.
template <bool P4>
static __global__ void paw_exp_walk_v8_warp_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,    // half [32768, 8]
        const uint32_t * GGML_CUDA_RESTRICT p4,      // P4 only, nullptr otherwise
        const float    * GGML_CUDA_RESTRICT p4lv,    // P4 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const half     * GGML_CUDA_RESTRICT gamma,
        const int32_t  * GGML_CUDA_RESTRICT remap,
        const int32_t  * GGML_CUDA_RESTRICT ids,
        const int m, const int n, const int n_groups,
        const int n_used, const int n_kept, const int ids_s0) {
    constexpr int warp_size = 32;   // caller guards tiles_y <= 32
    __shared__ float lv[16];

    (void) p4;
    (void) p4lv;

    const int g   = remap != nullptr ? (int) paw_group_of(remap, ids, blockIdx.z,
                                                            n_used, n_kept, ids_s0, 0)
                                     : blockIdx.z;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const int  lane    = tid % warp_size;
    const int  tr      = blockIdx.x*4 + tid / warp_size;   // 4 warps = 4 row tiles
    const bool have    = lane < tiles_y;                    // tr < m/16 by grid

    constexpr int words = 24;   // V = 8, K = 1.5
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    // per-tile wave gamma: same closed form as the block kernel
    const int e_orig = scr_i[2*n_groups + g];
    const int Mb     = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + lane <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + lane)
                                                  : Mb + tiles_y - 2 - (tr + lane);
        gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
    }

    // w8 (the real global read) stays resident; phv is recomputed per-qq
    // instead of cached (see exp_walk2's matching change -- caching all 32
    // step states only pays off when cnt is large, which it typically
    // isn't at decode).
    uint16_t w8[24];
    if (have) {
        const int64_t tw = tbase + (int64_t) lane*words;
#pragma unroll
        for (int q = 0; q < 24; ++q) {
            w8[q] = kept[tw + q];
        }
    }

    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + lane*16;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
                const uint32_t ph  = reg*(reg + 1u);
                const uint32_t row = ph & 0x7FFFu;
                const int      ri  = (8*i) >> 4;
                const int      ci  = (8*i) & 15;
                float dotp = 0.0f;
                if constexpr (P4) {
                    const uint32_t pk = p4[row];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = lv[(pk >> (4*c)) & 0xFu];
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;                 // exact (sign bit)
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                } else {
                    const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = __half2float(tl[c]);
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;                 // exact in fp16 (sign bit)
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                }
                partial[ri] += dotp;
            }
        }
        float sums[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sums[i] = warp_reduce_sum<warp_size>(partial[i]);
        }
        if (lane == 0) {
            float * vo = scr_v + (int64_t) p*m + tr*16;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                vo[i] = sums[i];
            }
        }
        // no __syncthreads(): warps share nothing across pairs
    }
}

// dense prefill: when P = n_used*n_tok is large, most pairs share a storage
// group and the fused walk re-decodes each group's trellis once per pair.
// Instead decode every ACTIVE group (cnt > 0) once into an fp16 bank
// [n_groups, m, n] and apply it with plain fp32 dots. The bank holds the
// exact hatWr halves of the fused path (V2 pins the fp16 round; V8 stores
// the pre-rounded tlut entries UNSCALED — wave gamma is applied in the
// apply kernel, folded into u per column tile, so no extra fp16 rounding
// enters the chain). exp_group/exp_u/exp_out are untouched.

template <bool V8, bool P4>
static __global__ void __launch_bounds__(256, 8) paw_exp_dense_decode_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const uint16_t * GGML_CUDA_RESTRICT dem,     // placeholder (= kept) when absent
        const void     * GGML_CUDA_RESTRICT tlut,    // float (V2) or half (V8)
        const uint32_t * GGML_CUDA_RESTRICT p4,      // P4 only, nullptr otherwise
        const float    * GGML_CUDA_RESTRICT p4lv,    // P4 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        half           * GGML_CUDA_RESTRICT bank,    // [n_groups, m, n]
        const int m, const int n, const int n_kept, const int n_groups,
        const int g0,                                 // first group of this slab
        const half    * GGML_CUDA_RESTRICT wgamma) { // V8 wave gamma or nullptr
    static_assert(V8 || !P4, "P4 is a V8-tlut repack");
    constexpr int WG = 256;
    __shared__ float lv[16];

    (void) dem;
    (void) p4;
    (void) p4lv;
    (void) n_groups;

    const int g   = g0 + (int) blockIdx.z;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (scr_i[g] == 0) {           // block-uniform: skip inactive groups
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: adjacent threads write adjacent 16-half chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;
    const int tile   = tr*tiles_y + tc;
    const int ntiles = (m/16)*tiles_y;

    int              words;
    int              stepb;
    const uint16_t * trd;
    int64_t          tbase;
    if constexpr (V8) {
        words = 24;
        stepb = 12;
        trd   = kept;
        tbase = ((int64_t) g*ntiles + tile)*words;
    } else {
        const bool is_dem = g >= n_kept;
        const int  ei     = is_dem ? g - n_kept : g;
        words = is_dem ? 16 : 32;
        stepb = is_dem ? 2 : 4;
        trd   = is_dem ? dem : kept;
        tbase = ((int64_t) ei*ntiles + tile)*words;
    }

    half * dst = bank + ((int64_t)(g - g0)*m + tr*16 + rr)*n + tc*16;
    half tmp[16];

    if constexpr (V8) {
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {     // tile row rr = states 2*rr, 2*rr+1
            const int      i   = 2*rr + jj;
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) trd[tbase + wi] << 16) | (uint32_t) trd[tbase + wn];
            const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            if constexpr (P4) {
                const uint32_t pk = p4[row];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = lv[(pk >> (4*c)) & 0xFu];
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;
                    }
                    // vv is an exact fp16 value: the round-trip stores its bits
                    tmp[8*jj + c] = __float2half_rn(vv);
                }
            } else {
                const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = __half2float(tl[c]);
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                         // exact in fp16 (sign bit)
                    }
                    tmp[8*jj + c] = __float2half_rn(vv);
                }
            }
        }
    } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {        // tile row rr = steps 8*rr .. 8*rr+8
            const int      i   = 8*rr + s;
            const int      bb  = stepb*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < words ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) trd[tbase + wi] << 16) | (uint32_t) trd[tbase + wn];
            const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = (ph >> 6) & 511u;
            const float * tl = (const float *) tlut;
            float v0 = tl[2*row + 0];
            const float v1 = tl[2*row + 1];
            if (ph & 0x8000u) {
                v0 = -v0;                                 // exact either side of the round
            }
            // hatWr is defined at fp16 precision — the fused path's pinned
            // round IS this store
            tmp[2*s + 0] = __float2half_rn(v0);
            tmp[2*s + 1] = __float2half_rn(v1);
        }
    }
    if (V8 && wgamma != nullptr) {
        // fold the wave gamma into the weights so the apply runs as a plain
        // GEMM; product terms match folding it into the activations up to a
        // contractible fp16 rounding site
        const int e_ori = scr_i[2*n_groups + g];
        const int Mb      = m/16;
        const int tiles_y = n/16;
        const int wv = (tr + tc <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tc)
                                                : Mb + tiles_y - 2 - (tr + tc);
        const float gv = __half2float(wgamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            tmp[c] = __float2half_rn(gv*__half2float(tmp[c]));
        }
    }
    paw_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}


// warp-per-tile variant of paw_exp_dense_decode_kernel (V8 layout): the 16
// rows of a tile all live in the same 24 stream words, so the warp stages
// them in shared once instead of every lane re-reading them from global;
// lanes 0..15 then decode one row each off shared. Removes the redundant
// global traffic and the per-row serial load chain that starved the old
// mapping (measured ~3.4 GB/s effective on RTX 3060).
template <bool P4>
static __global__ void paw_exp_dense_decode_kernel_v2(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        half           * GGML_CUDA_RESTRICT bank,
        const int m, const int n, const int n_groups,
        const int g0,
        const half    * GGML_CUDA_RESTRICT wgamma) {
    constexpr int WARPS = 8;
    constexpr int WG    = 24;                    // words per tile
    __shared__ float    lv[16];
    __shared__ uint16_t wsh[WARPS][WG];

    const int g    = g0 + (int) blockIdx.z;
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;

    ggml_cuda_pdl_sync();
    if (scr_i[g] == 0) {                         // block-uniform
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int ntiles  = (m / 16)*tiles_y;
    const int tile    = blockIdx.x*WARPS + wid;
    if (tile >= ntiles) {
        return;
    }

    // stage this tile's stream words once per warp
    if (lane < WG) {
        wsh[wid][lane] = kept[((int64_t) g*ntiles + tile)*WG + lane];
    }
    __syncwarp();

    half tmp16[16];

    const int tr   = tile / tiles_y;
    const int tc   = tile % tiles_y;
    const int rr   = lane & 15;
    half * dst = bank + ((int64_t)(g - g0)*m + tr*16 + rr)*n + tc*16;

#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int      i   = 2*rr + jj;
        const int      bb  = 12*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < WG ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) wsh[wid][wi] << 16) | (uint32_t) wsh[wid][wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t row = ph & 0x7FFFu;
        if constexpr (P4) {
            const uint32_t pk = p4[row];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = lv[(pk >> (4*c)) & 0xFu];
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp16[jj*8 + c] = __float2half_rn(vv);
            }
        } else {
            const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = __half2float(tl[c]);
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp16[jj*8 + c] = __float2half_rn(vv);
            }
        }
    }
    if (wgamma != nullptr) {
        const int e_ori = scr_i[2*n_groups + g];
        const int Mb    = m/16;
        const int wv    = (tr + tc <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tc)
                                                   : Mb + tiles_y - 2 - (tr + tc);
        const float gv = __half2float(wgamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            tmp16[c] = __float2half_rn(gv*__half2float(tmp16[c]));
        }
    }
    paw_store_half16(dst, tmp16);
}




// decode one 16-row weight strip of one k-tile straight into shared memory
template <bool P4>
static __device__ __forceinline__ void paw_fused_decode_strip(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * lv,
        const half     * GGML_CUDA_RESTRICT gamma,
        const int g, const int tr_r, const int tc,
        const int tiles_y, const int ntiles, const int m, const int n,
        const int n_groups, const int e_ori,
        uint16_t * wsh, half (* wout)[16], const int lane) {
    const int tile  = tr_r*tiles_y + tc;
    const int tbase = ((int64_t) g*ntiles + tile)*24;
    if (lane < 24) {
        wsh[lane] = kept[tbase + lane];
    }
    __syncwarp();
    const int rr  = lane & 15;
    const int Mb  = m/16;
    const int wv  = (tr_r + tc <= tiles_y - 1)
                        ? Mb + tiles_y - 1 - (tr_r + tc)
                        : Mb + tiles_y - 2 - (tr_r + tc);
    const float gv =
        __half2float(gamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
    if (lane >= 16) {
        return;
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int      i   = 2*rr + jj;
        const int      bb  = 12*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) wsh[wi] << 16) | (uint32_t) wsh[wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t lrow = ph & 0x7FFFu;
        if constexpr (P4) {
            const uint32_t pk = p4[lrow];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = lv[(pk >> (4*c)) & 0xFu];
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                wout[rr][8*jj + c] =
                    __float2half_rn(gv*__half2float(__float2half_rn(vv)));
            }
        } else {
            const half * tl = (const half *) tlut + 8*(int64_t) lrow;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = __half2float(tl[c]);
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                wout[rr][8*jj + c] =
                    __float2half_rn(gv*__half2float(__float2half_rn(vv)));
            }
        }
    }
}

// Fully-fused expert apply: the WS apply skeleton, but the weight tiles are
// decoded from the QTIP stream directly into shared memory right before the
// wmma load -- no fp16 bank materialization at all. Kills the ~2 GB/pass
// bank write+readback of the decode->GEMM pipeline and overlaps the LUT
// gather latency with tensor-core work.
template <bool P4, int BMT>
static __global__ void __launch_bounds__(128, 4) paw_exp_apply_kernel_fused(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const half     * GGML_CUDA_RESTRICT bank,   // debug: pre-decoded weights
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const half     * GGML_CUDA_RESTRICT xg,      // [P, n] grouped fp16
        float          * GGML_CUDA_RESTRICT scr_v,   // [P, m]
        const half     * GGML_CUDA_RESTRICT gamma,   // V8 wave gamma
        float          * GGML_CUDA_RESTRICT dbg,     // debug dump or nullptr
        const int m, const int n, const int n_groups) {
    using namespace nvcuda;

    constexpr int n_warps  = 4;
    constexpr int bk       = 16;                  // == the wmma K tile
    constexpr int bn       = 64;                  // token chunk, whole block
    constexpr int n_tt     = bn / 16;             // token subtiles per chunk

    const int g    = blockIdx.y;
    const int row0 = blockIdx.x*(n_warps*16);
    const int tid  = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int tiles_y = n/16;
    const int ntiles  = (m/16)*tiles_y;

    __shared__ half     Xsh[bn][bk];              // staged activation tile
    // double-buffered per-warp weight strips: tile k+1 decodes while tile k
    // is still being multiplied, hiding LUT gather latency behind tensor math
    __shared__ half     Wsh[2][n_warps][16][bk];
    __shared__ float    out_sh[n_warps][16*16];
    __shared__ int      pidx_sh[bn];
    __shared__ uint16_t wsh[n_warps][24];
    __shared__ float    lv[16];                   // staged P4 levels

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {
        return;
    }
    const int off   = scr_i[n_groups + g];
    const int e_ori = scr_i[2*n_groups + g];
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[n_tt];

    for (int q0 = 0; q0 < cnt; q0 += bn) {
#pragma unroll
        for (int t = 0; t < n_tt; ++t) {
            wmma::fill_fragment(acc[t], 0.0f);
        }
        for (int i = tid; i < bn; i += 128) {
            pidx_sh[i] = scr_i[4*n_groups + off + min(q0 + i, cnt - 1)];
        }
        __syncthreads();

        // prologue: decode strip k=0 into buffer 0 before the loop
        int cur = 0;
        paw_fused_decode_strip<P4>(kept, tlut, p4, lv, gamma,
            g, row0/16 + warp_id, 0, tiles_y, ntiles, m, n, n_groups, e_ori,
            wsh[warp_id], Wsh[0][warp_id], lane);

        for (int k0 = 0; k0 < n; k0 += bk, cur ^= 1) {
            // kick off the next strip into the other buffer; its LUT
            // gathers then overlap with this tile's mma chain
            if (k0 + bk < n) {
                paw_fused_decode_strip<P4>(kept, tlut, p4, lv, gamma,
                    g, row0/16 + warp_id, (k0 + bk)/16, tiles_y, ntiles, m, n,
                    n_groups, e_ori, wsh[warp_id], Wsh[cur ^ 1][warp_id], lane);
            }

            __syncwarp();
            wmma::load_matrix_sync(a_frag, &Wsh[cur][warp_id][0][0], bk);

            if (dbg && g == 0 && blockIdx.x == 0 && q0 == 0 && k0 == 0) {
                for (int idx = lane; idx < 16*bk; idx += 32) {
                    const int rrr = idx / bk;
                    const int kkk = idx % bk;
                    dbg[(warp_id*16 + rrr)*bk + kkk] =
                        __half2float(Wsh[cur][warp_id][rrr][kkk]);
                }
                __syncwarp();
            }

            // vectorized activation staging: 16B per lane, tail rows clamp
            // onto the last valid pair (store guards discard those slots)
            {
                const int last = off + cnt - 1;
                uint4 * xd = (uint4 *) &Xsh[0][0];
                for (int idx = tid; idx < bn*(bk/8); idx += 128) {
                    const int tt = idx / (bk/8);
                    const int vq = off + q0 + tt;
                    const uint4 * xs = (const uint4 *)
                        (xg + (int64_t) min(vq, last)*n + k0);
                    xd[idx] = xs[idx % (bk/8)];
                }
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < n_tt; ++t) {
                wmma::load_matrix_sync(b_frag, &Xsh[t*16][0], bk);
                wmma::mma_sync(acc[t], a_frag, b_frag, acc[t]);
            }
        }

        // store this warp's strip for every token subtile
#pragma unroll
        for (int t = 0; t < n_tt; ++t) {
            wmma::store_matrix_sync(&out_sh[warp_id][0], acc[t], 16,
                                    wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16*16; idx += 32) {
                const int row = idx / 16;
                const int tt  = idx % 16;
                if (q0 + t*16 + tt < cnt) {
                    scr_v[(int64_t) pidx_sh[t*16 + tt]*m + row0 + warp_id*16 + row] =
                        out_sh[warp_id][idx];
                }
            }
            __syncwarp();
        }
        // block-wide barrier before the next chunk refills pidx_sh: the
        // store loop above reads it under a mere per-warp sync
        __syncthreads();
    }
}

// ---- decode-kernel microbench (dev harness, not used by inference) -------
// Launches the v1 and v2 dense-decode kernels on synthetic data at real
// shapes, checks bit-exact agreement of the banks, and prints timings.


extern "C" void paw_decode_bench(int m, int n, int n_groups, int iters) {
    const size_t words = (size_t) n_groups*(m/16)*(n/16)*24;
    const size_t tlut_n = 32768*8;
    uint16_t * kept;  CUDA_CHECK(cudaMalloc(&kept,  words*2));
    half     * tlut;  CUDA_CHECK(cudaMalloc(&tlut,  tlut_n*2));
    uint32_t * p4t;   CUDA_CHECK(cudaMalloc(&p4t,   32768*4));
    float    * p4l;   CUDA_CHECK(cudaMalloc(&p4l,   16*4));
    int32_t  * scr;   CUDA_CHECK(cudaMalloc(&scr,   (size_t) 3*n_groups*4));
    half     * b1;    CUDA_CHECK(cudaMalloc(&b1,   (size_t) n_groups*m*n*2));
    half     * b2;    CUDA_CHECK(cudaMalloc(&b2,   (size_t) n_groups*m*n*2));
    half     * gam;   CUDA_CHECK(cudaMalloc(&gam,   (size_t) 256*(m/16 + n/16)*2));

    // deterministic pseudo-random fill
    std::vector<uint16_t> hk(words);
    unsigned rng = 12345;
    for (auto & w : hk) { rng = rng*1103515245u + 12345u; w = (uint16_t)(rng >> 9); }
    CUDA_CHECK(cudaMemcpy(kept, hk.data(), words*2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(tlut, 0, tlut_n*2));
    CUDA_CHECK(cudaMemset(p4t, 0, 32768*4));
    std::vector<float> hl(16, 0.5f);
    CUDA_CHECK(cudaMemcpy(p4l, hl.data(), 64, cudaMemcpyHostToDevice));
    std::vector<int32_t> hs(3*(size_t) n_groups, 8);   // all groups active
    CUDA_CHECK(cudaMemcpy(scr, hs.data(), hs.size()*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gam, 0, (size_t) 256*(m/16 + n/16)*2));

    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    half * gnullptr_h = nullptr;

    auto launch_v1 = [&](half * bank) {
        if (m % 16 == 0 && n % 16 == 0) {
            paw_launch(paw_exp_dense_decode_kernel<true, true>,
                ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups), dim3(256,1,1), 0, st),
                kept, kept, (const void *) tlut, p4t,
                p4l, scr, bank, m, n, n_groups, n_groups, 0,
                (const half *) gam);
        }
    };
    auto launch_v2 = [&](half * bank) {
        paw_launch(paw_exp_dense_decode_kernel_v2<true>,
            ggml_cuda_kernel_launch_params(dim3(((m/16)*(n/16) + 7)/8, 1, n_groups), dim3(256,1,1), 0, st),
            kept, (const void *) tlut, p4t,
            p4l, scr, bank, m, n, n_groups, 0,
            (const half *) gam);
    };

    // correctness: one pass each, compare banks
    CUDA_CHECK(cudaMemset(b1, 0xaa, (size_t) n_groups*m*n*2));
    CUDA_CHECK(cudaMemset(b2, 0xbb, (size_t) n_groups*m*n*2));
    launch_v1(b1); launch_v2(b2);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<half> h1((size_t) n_groups*m*n), h2((size_t) n_groups*m*n);
    CUDA_CHECK(cudaMemcpy(h1.data(), b1, h1.size()*2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), b2, h2.size()*2, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < h1.size(); ++i) {
        if (((uint16_t*)h1.data())[i] != ((uint16_t*)h2.data())[i]) ++diff;
    }

    // timing
    float ms1 = -1.f, ms2 = -1.f;
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int i = 0; i < iters; ++i) { launch_v1(b1); }
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms1, e0, e1));
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int i = 0; i < iters; ++i) { launch_v2(b2); }
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms2, e0, e1));
    (void) gnullptr_h;
    printf("decode-bench m=%d n=%d groups=%d | v1 %.3f ms | v2 %.3f ms | speedup %.2fx | bitdiff %zu / %zu\n",
           m, n, n_groups, ms1/iters, ms2/iters, ms1/ms2, diff, h1.size());
    fflush(stdout);

    cudaStreamDestroy(st);
    cudaFree(kept); cudaFree(tlut); cudaFree(p4t); cudaFree(p4l);
    cudaFree(scr); cudaFree(b1); cudaFree(b2); cudaFree(gam);
}

// grid (m/16, n_groups): each block owns 16 output rows of one group's bank
// tile and loops the group's pair list (group-ordered, so the 16 x n slice
// stays hot in L2 across pairs). V8 folds wave gamma into u per column tile
// (gamma is constant within a tile and W*(g*u) == (g*W)*u up to contractible
// fp32 association).
// Pairs are processed PC at a time: the 16-row W column slice is loaded into
// registers once per column iteration and reused across the chunk's pairs,
// amortizing the dominant bank traffic PC-fold (same trick as
// paw_rt_apply_kernel's token chunk). Per-pair accumulation order over the
// columns is unchanged (same WG stride, same warp reduction), so numerics are
// identical to the unchunked kernel; the tail chunk clamps to the last pair
// and write-guards (duplicate compute, no duplicate store). PC=1 reproduces
// the unchunked kernel exactly; GGML_PAW_EXP_PC=1 selects it (A/B knob).
template <bool V8, int PC>
static __global__ void paw_exp_apply_kernel(
        const half    * GGML_CUDA_RESTRICT bank,     // [n_groups, m, n]
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const float   * GGML_CUDA_RESTRICT scr_u,    // [P, n]
        float         * GGML_CUDA_RESTRICT scr_v,    // [P, m]
        const half    * GGML_CUDA_RESTRICT gamma,    // V8 only, nullptr otherwise
        const int m, const int n, const int n_groups) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[PC][16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float gsh[128];         // V8: per-column-tile wave gamma (tiles_y <= 128)

    (void) gamma;

    const int g   = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    const int off     = scr_i[n_groups + g];
    const int tiles_y = n / 16;

    if constexpr (V8) {
        // same closed-form wave index as the fused walk, per column tile
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        for (int i = tid; i < tiles_y; i += WG) {
            const int wv = (tr + i <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + i)
                                                   : Mb + tiles_y - 2 - (tr + i);
            gsh[i] = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
        __syncthreads();
    }

    const half * W = bank + ((int64_t) g*m + tr*16)*n;

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    for (int q0 = 0; q0 < cnt; q0 += PC) {
        int pidx[PC];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            pidx[t] = scr_i[4*n_groups + off + min(q0 + t, cnt - 1)];
        }

        float acc[PC][16];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[t][i] = 0.0f;
            }
        }
        // half2/float2 loads over column pairs (c0 even, so both columns are
        // in the same 16-wide gamma tile). Merging a pair's two terms into
        // one partial is a contractible fp32 reassociation, like the fused
        // walk's own ordering.
        for (int c0 = 2*tid; c0 < n; c0 += 2*WG) {
            float2 wc[16];
#pragma unroll
            for (int ri = 0; ri < 16; ++ri) {
                wc[ri] = __half22float2(*(const half2 *)(W + (int64_t) ri*n + c0));
            }
#pragma unroll
            for (int t = 0; t < PC; ++t) {
                float2 uc = *(const float2 *)(scr_u + (int64_t) pidx[t]*n + c0);
                if constexpr (V8) {
                    const float gv = gsh[c0 >> 4];
                    uc.x *= gv;
                    uc.y *= gv;
                }
#pragma unroll
                for (int ri = 0; ri < 16; ++ri) {
                    acc[t][ri] += wc[ri].x*uc.x + wc[ri].y*uc.y;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float s = warp_reduce_sum<warp_size>(acc[t][i]);
                if (lane == 0) {
                    red[t][i][wid] = s;
                }
            }
        }
        __syncthreads();
        if (tid < PC*16) {
            const int t = tid >> 4;
            const int i = tid & 15;
            if (q0 + t < cnt) {
                float sum = 0.0f;
#pragma unroll
                for (int wj = 0; wj < n_warps; ++wj) {
                    sum += red[t][i][wj];
                }
                scr_v[(int64_t) pidx[t]*m + tr*16 + i] = sum;
            }
        }
        if (q0 + PC < cnt) __syncthreads();   // red is reused by the next chunk -- skip on last iter, nothing left to protect
    }
}



// Groups the routed-pair activation slab by expert: xg[(off_g + j)*n + k] =
// x[plist[off_g + j]*n + k]. Turns the apply kernels' random-row gathers
// into sequential reads; costs one extra pass over P*n floats.
static __global__ void paw_exp_permute_x_kernel(
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const float   * GGML_CUDA_RESTRICT x,
        half          * GGML_CUDA_RESTRICT xg,
        const int n, const int n_groups) {
    const int g  = blockIdx.y;
    const int kg = blockIdx.x*blockDim.x + threadIdx.x;
    if (kg >= n) {
        return;
    }
    const int cnt = scr_i[g];
    const int off = scr_i[n_groups + g];
    const int32_t * plist = scr_i + 4*n_groups;
    // halves on purpose: the ws apply staged these exact conversions anyway,
    // and the slab must fit next to the weights on small cards
    for (int j = 0; j < cnt; ++j) {
        xg[(int64_t)(off + j)*n + kg] = __float2half(x[(int64_t) plist[off + j]*n + kg]);
    }
}


// In-place wave-gamma scale of a decoded group bank: W'[r,k] = W[r,k] * g,
// with the closed-form diagonal wave index of the fused walk. Folding the
// gamma into the weights (instead of the activations) lets the apply run
// as a plain GEMM; the product terms are unchanged up to one fp16 rounding
// site, the same contractible class the dense paths already accept.
static __global__ void paw_exp_scale_bank_gamma_kernel(
        half          * GGML_CUDA_RESTRICT bank,
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const half    * GGML_CUDA_RESTRICT gamma,
        const int m, const int n, const int n_groups) {
    const int g = blockIdx.y;
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m*n) {
        return;
    }
    const int tr_r = (i / n) >> 4;
    const int kt   = (i % n) >> 4;
    const int e_ori = scr_i[2*n_groups + g];
    const int Mb      = m/16;
    const int tiles_y = n/16;
    const int wv = (tr_r + kt <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr_r + kt)
                                              : Mb + tiles_y - 2 - (tr_r + kt);
    const size_t o = (size_t) g*m*n + i;
    bank[o] = __float2half(__half2float(bank[o]) *
                           __half2float(gamma[(int64_t) e_ori*(Mb + tiles_y) + wv]));
}

// inverse of the grouped slab: scr_v[pidx] = yg[off + j]
static __global__ void paw_exp_unpermute_y_kernel(
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const half    * GGML_CUDA_RESTRICT yg,
        float         * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n_groups) {
    const int g   = blockIdx.y;
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col >= m) {
        return;
    }
    const int cnt = scr_i[g];
    const int off = scr_i[n_groups + g];
    const int32_t * plist = scr_i + 4*n_groups;
    for (int j = 0; j < cnt; ++j) {
        scr_v[(int64_t) plist[off + j]*m + col] =
            __half2float(yg[(int64_t)(off + j)*m + col]);
    }
}

// Weight-stationary tensor-core twin of paw_exp_apply_kernel for the dense
// prefill path (V8 and legacy payloads). One persistent block per (group,
// 64-row tile) sweeps that group's routed pairs in chunks of bn pairs; a
// full-K accumulation finishes each chunk before the next starts, so a
// group with cnt <= bn reads its bank strip exactly once instead of
// ceil(cnt/PC) times. The V8 wave gamma is row-strip dependent, so instead
// of folding it into the shared activation slab the B tile is rescaled per
// A-tile into a small scratch buffer -- the association is contractible in
// the same sense the fused walk documents. Requires m % 64 == 0.
template <bool V8, int BMT, bool GX>
static __global__ void paw_exp_apply_kernel_ws(
        const half    * GGML_CUDA_RESTRICT bank,     // [n_groups, m, n]
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const float   * GGML_CUDA_RESTRICT scr_u,    // [P, n]
        const half    * GGML_CUDA_RESTRICT xg,       // [P, n] grouped fp16 (GX)
        float         * GGML_CUDA_RESTRICT scr_v,    // [P, m]
        const half    * GGML_CUDA_RESTRICT gamma,    // V8 only, else nullptr
        const int m, const int n, const int n_groups) {
    using namespace nvcuda;

    constexpr int n_warps = 4;
    constexpr int bm_t    = BMT;                  // narrow tiles for small m
    constexpr int tpb     = 2;                    // BN = 128 pairs per chunk
    constexpr int bn      = n_warps * tpb * 16;
    constexpr int bk      = 16;                   // == the wmma K tile

    const int g    = blockIdx.y;
    const int row0 = blockIdx.x * (bm_t * 16);
    const int tid  = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int tiles_y = n / 16;
    const int Mb      = m / 16;

    __shared__ half Xsh[n_warps][tpb][16][bk];    // raw gathered activations
    __shared__ half Bsh[n_warps][16][bk];         // gamma-scaled B tile
    __shared__ float out_sh[n_warps][16*16];
    __shared__ int pidx_sh[bn];
    __shared__ int scnt, soff, e_ori;

    ggml_cuda_pdl_sync();
    if (tid == 0) {
        scnt   = scr_i[g];
        soff   = scr_i[n_groups + g];
        e_ori  = V8 ? scr_i[2*n_groups + g] : 0;
    }
    __syncthreads();
    const int cnt = scnt;
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    const int off = soff;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[bm_t][tpb];

    for (int q0 = 0; q0 < cnt; q0 += bn) {
#pragma unroll
        for (int r = 0; r < bm_t; ++r) {
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                wmma::fill_fragment(acc[r][t], 0.0f);
            }
        }
        for (int i = tid; i < bn; i += 128) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            pidx_sh[i] = scr_i[4*n_groups + off + min(q0 + i, cnt - 1)];
        }
        __syncthreads();

        for (int k0 = 0; k0 < n; k0 += bk) {
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                const int tt0 = (warp_id*tpb + t)*16;
                for (int idx = lane; idx < 16*bk; idx += 32) {
                    const int tt = idx / bk;
                    const int kk = idx % bk;
                    const int kg = k0 + kk;
                    float v = 0.0f;
                    if (kg < n && q0 + tt0 + tt < cnt) {
                        // GX: activations arrive pre-grouped (and pre-cast),
                        // rows are read sequentially instead of gathered
                        const int64_t row = GX ? (int64_t)(off + q0 + tt0 + tt)
                                               : (int64_t) pidx_sh[tt0 + tt];
                        v = GX ? __half2float(xg[row*n + kg])
                               : scr_u[(int64_t) pidx_sh[tt0 + tt]*n + kg];
                    }
                    Xsh[warp_id][t][tt][kk] = __float2half(v);
                }
            }
            __syncwarp();
#pragma unroll
            for (int r = 0; r < bm_t; ++r) {
                wmma::load_matrix_sync(a_frag,
                    bank + ((int64_t) g*m + row0 + r*16)*n + k0, n);
                float gv = 1.0f;
                if (V8) {
                    // closed-form wave index of the fused walk, per column
                    // tile, for this block's r-th output strip
                    const int tr_r = row0/16 + r;
                    const int wv = (tr_r + k0/16 <= tiles_y - 1)
                        ? Mb + tiles_y - 1 - (tr_r + k0/16)
                        : Mb + tiles_y - 2 - (tr_r + k0/16);
                    gv = __half2float(gamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
                }
#pragma unroll
                for (int t = 0; t < tpb; ++t) {
                    if (V8) {
                        for (int idx = lane; idx < 16*bk; idx += 32) {
                            const int tt = idx / bk;
                            const int kk = idx % bk;
                            Bsh[warp_id][tt][kk] = __float2half(
                                gv*__half2float(Xsh[warp_id][t][tt][kk]));
                        }
                        __syncwarp();
                        wmma::load_matrix_sync(b_frag, &Bsh[warp_id][0][0], bk);
                    } else {
                        wmma::load_matrix_sync(b_frag, &Xsh[warp_id][t][0][0], bk);
                    }
                    wmma::mma_sync(acc[r][t], a_frag, b_frag, acc[r][t]);
                }
            }
            __syncwarp();
        }

#pragma unroll
        for (int r = 0; r < bm_t; ++r) {
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                wmma::store_matrix_sync(&out_sh[warp_id][0], acc[r][t], 16, wmma::mem_row_major);
                __syncwarp();
                const int tt0 = (warp_id*tpb + t)*16;
                for (int idx = lane; idx < 16*16; idx += 32) {
                    const int row = idx / 16;
                    const int tt  = idx % 16;
                    if (q0 + tt0 + tt < cnt) {
                        scr_v[(int64_t) pidx_sh[tt0 + tt]*m + row0 + r*16 + row] =
                            out_sh[warp_id][idx];
                    }
                }
                __syncwarp();
            }
        }
    }
}

extern "C" int paw_fused_bench(int m, int n, int n_groups, int P) {
    const size_t words = (size_t) n_groups*(m/16)*(n/16)*24;
    uint16_t * kept;  CUDA_CHECK(cudaMalloc(&kept, words*2));
    half     * tlut;  CUDA_CHECK(cudaMalloc(&tlut, (size_t)32768*8*2));
    uint32_t * p4t;   CUDA_CHECK(cudaMalloc(&p4t,  32768*4));
    float    * p4l;   CUDA_CHECK(cudaMalloc(&p4l,  16*4));
    int32_t  * scr;   CUDA_CHECK(cudaMalloc(&scr,  (size_t)(5*n_groups + P)*4));
    half     * xg;    CUDA_CHECK(cudaMalloc(&xg,   (size_t) P*n*2));
    float    * v;     CUDA_CHECK(cudaMalloc(&v,    (size_t) P*m*4));
    half     * gam;   CUDA_CHECK(cudaMalloc(&gam,  (size_t) 256*(m/16 + n/16)*2));

    std::vector<uint16_t> hk(words);
    {
        unsigned rng = 9973;
        for (size_t i = 0; i < words; ++i) {
            rng = rng*1103515245u + 12345u;
            hk[i] = (uint16_t)((rng >> 8) | 1u);   // nonzero streams
        }
    }
    CUDA_CHECK(cudaMemcpy(kept, hk.data(), words*2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(tlut, 0, (size_t)32768*16));
    CUDA_CHECK(cudaMemset(p4t, 0, 32768*4));
    std::vector<float> hl(16, 0.5f);
    CUDA_CHECK(cudaMemcpy(p4l, hl.data(), 64, cudaMemcpyHostToDevice));
    // group g active with cnt=8, offsets staggered
    const int CNT = getenv("FUSED_CNT") ? atoi(getenv("FUSED_CNT")) : 8;
    std::vector<int32_t> hs(5*(size_t)n_groups + (size_t)(CNT>0?0:P), 0);
    // pair-list region sized by total routed pairs
    hs.resize(5*(size_t)n_groups + (size_t) n_groups*CNT);
    int32_t * pairs = hs.data() + 4*n_groups;
    for (int g = 0; g < n_groups; ++g) {
        hs[g] = CNT;
        hs[n_groups + g] = CNT*g;            // off
        hs[2*n_groups + g] = g % 256;        // e_ori
        for (int q = 0; q < CNT; ++q) {
            pairs[CNT*g + q] = (CNT*g + q) % P;         // pair list
        }
    }
    CUDA_CHECK(cudaMemcpy(scr, hs.data(), hs.size()*4, cudaMemcpyHostToDevice));
    {
        std::vector<half> hx((size_t)P*n);
        unsigned rng = 31;
        for (size_t i = 0; i < hx.size(); ++i) {
            rng = rng*1103515245u + 12345u;
            hx[i] = __float2half(((int)(rng >> 16 & 0xFF) - 128) / 512.0f);
        }
        CUDA_CHECK(cudaMemcpy(xg, hx.data(), hx.size()*2, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(v, 0, (size_t)P*m*4));
    std::vector<half> hg((size_t)256*(m/16 + n/16), __float2half(1.0f));
    CUDA_CHECK(cudaMemcpy(gam, hg.data(), hg.size()*2, cudaMemcpyHostToDevice));

    float * dbgf; CUDA_CHECK(cudaMalloc(&dbgf, 64*16*4));
    CUDA_CHECK(cudaMemset(dbgf, 0, 64*16*4));
    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    paw_launch(paw_exp_apply_kernel_fused<true, 4>,
        ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128,1,1), 0, st),
        kept, (const half *) nullptr, (const void *) tlut, (const uint32_t *) p4t,
        (const float *) p4l, (const int32_t *) scr, xg, v, (const half *) gam,
        (float *) nullptr, m, n, n_groups);
    cudaError_t le = cudaGetLastError();

    // structural twin: same loops, bank-fed weights
    half * bank2; CUDA_CHECK(cudaMalloc(&bank2, (size_t) n_groups*m*n*2));
    float * vf;   CUDA_CHECK(cudaMalloc(&vf, (size_t) P*m*4));
    CUDA_CHECK(cudaMemset(vf, 0, (size_t) P*m*4));
    {
        std::vector<uint16_t> hkz(words, 0x1555u);
        CUDA_CHECK(cudaMemcpy(kept, hkz.data(), words*2, cudaMemcpyHostToDevice));
    }
    paw_launch(paw_exp_dense_decode_kernel<true, true>,
        ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups), dim3(256,1,1), 0, st),
        kept, kept, (const void *) tlut, p4t, p4l,
        (const int32_t *) scr, bank2, m, n, n_groups, n_groups, 0,
        (const half *) gam);
    {
        std::vector<uint16_t> hkr(words);
        unsigned rng = 9973;
        for (size_t i = 0; i < words; ++i) {
            rng = rng*1103515245u + 12345u;
            hkr[i] = (uint16_t)((rng >> 8) | 1u);
        }
        CUDA_CHECK(cudaMemcpy(kept, hkr.data(), words*2, cudaMemcpyHostToDevice));
    }
    paw_launch(paw_exp_apply_kernel_fused<true, 4>,
        ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128,1,1), 0, st),
        kept, (const half *) bank2, (const void *) tlut, p4t, p4l,
        (const int32_t *) scr, xg, vf, (const half *) gam,
        dbgf, m, n, n_groups);

    // reference: decode into a bank, then WS apply over it
    half * bank; CUDA_CHECK(cudaMalloc(&bank, (size_t) n_groups*m*n*2));
    float * vref; CUDA_CHECK(cudaMalloc(&vref, (size_t) P*m*4));
    CUDA_CHECK(cudaMemset(vref, 0, (size_t) P*m*4));
    paw_launch(paw_exp_dense_decode_kernel<true, true>,
        ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups), dim3(256,1,1), 0, st),
        kept, kept, (const void *) tlut, p4t, p4l,
        (const int32_t *) scr, bank, m, n, n_groups, n_groups, 0,
        (const half *) gam);
    paw_launch(paw_exp_apply_kernel_ws<true, 4, true>,
        ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128,1,1), 0, st),
        (const half *) bank, (const int32_t *) scr, nullptr, xg, vref,
        (const half *) gam, m, n, n_groups);

    cudaError_t se = cudaStreamSynchronize(st);
    printf("fused-bench m=%d n=%d groups=%d P=%d | launch=%s sync=%s\n",
           m, n, n_groups, P, cudaGetErrorString(le), cudaGetErrorString(se));
    std::vector<float> hv((size_t)P*m), hr((size_t)P*m), hf((size_t)P*m);
    CUDA_CHECK(cudaMemcpy(hv.data(), v, hv.size()*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hr.data(), vref, hr.size()*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hf.data(), vf, hf.size()*4, cudaMemcpyDeviceToHost));
    size_t fdiff = 0;
    for (size_t i = 0; i < hf.size(); ++i) {
        if (fabs((double)hf[i] - (double)hr[i]) > 0) ++fdiff;
    }
    printf("bankfed-fused diff %zu / %zu\n", fdiff, hf.size());
    {
        std::vector<float> hd(64*16);
        CUDA_CHECK(cudaMemcpy(hd.data(), dbgf, hd.size()*4, cudaMemcpyDeviceToHost));
        std::vector<half> hb((size_t) n_groups*m*n);
        CUDA_CHECK(cudaMemcpy(hb.data(), bank2, hb.size()*2, cudaMemcpyDeviceToHost));
        int shown = 0;
        for (int w = 0; w < 4 && shown < 8; ++w) {
            for (int e = 0; e < 4 && shown < 8; ++e) {
                float got = hd[(w*16 + 0)*16 + e];      // row 0 of strip w, k=e
                float want = __half2float(hb[((size_t)0*m + (0*16 + w*16))*n + e]);
                if (fabs(got - want) > 1e-3) {
                    printf("  dbg warp%d k%d: Wsh=%.4f bank=%.4f\n", w, e, got, want);
                    ++shown;
                }
            }
        }
        if (!shown) printf("  dbg: all sampled W tiles match bank\n");
    }

    // ---- host ground truth over group 0's pairs only (fast enough) ----
    {
        std::vector<half> hb((size_t) n_groups*m*n);
        CUDA_CHECK(cudaMemcpy(hb.data(), bank2, hb.size()*2, cudaMemcpyDeviceToHost));
        std::vector<half> hx((size_t)P*n);
        CUDA_CHECK(cudaMemcpy(hx.data(), xg, hx.size()*2, cudaMemcpyDeviceToHost));
        std::vector<int32_t> hsc(5*(size_t)n_groups + (size_t)n_groups*CNT);
        CUDA_CHECK(cudaMemcpy(hsc.data(), scr, hsc.size()*4, cudaMemcpyDeviceToHost));
        double worst_f = 0, worst_r = 0;
        long long bad_f = 0, bad_r = 0, bad_f_row[4] = {0,0,0,0};
        const int GCHK = n_groups < 4 ? n_groups : 4;
        for (int g = 0; g < GCHK; ++g) {
            const int cg = hsc[g];
            const int og = hsc[n_groups + g];
            const int32_t * pp = hsc.data() + 4*n_groups + og;
            for (int i = 0; i < cg; ++i) {
                const int qp = pp[i];
                for (int rr = 0; rr < 64; rr += 16) {   // strips via blocks; check first block rows 0..63
                    for (int r16 = 0; r16 < 16; ++r16) {
                        const int row = rr + r16;
                        const half * wr = hb.data() + ((size_t) g*m + row)*n;
                        const half * xr = hx.data() + (size_t) qp*n;
                        double acc = 0;
                        for (int k = 0; k < n; ++k) {
                            acc += (double)__half2float(wr[k]) * (double)__half2float(xr[k]);
                        }
                        const size_t idx = (size_t) qp*m + row;
                        double df = fabs(acc - (double)hf[idx]);
                        double dr = fabs(acc - (double)hr[idx]);
                        if (df > worst_f) worst_f = df;
                        if (dr > worst_r) worst_r = dr;
                        if (df > 0.05) { ++bad_f; ++bad_f_row[rr/16]; }
                        if (dr > 0.05) ++bad_r;
                    }
                }
            }
        }
        printf("host-truth: fused bad=%lld worst=%.4f | ref bad=%lld worst=%.4f\n",
               bad_f, worst_f, bad_r, worst_r);
        printf("  fused bad by strip: %lld %lld %lld %lld\n",
               bad_f_row[0], bad_f_row[1], bad_f_row[2], bad_f_row[3]);
    }
    size_t diff = 0; double maxd = 0;
    int shown = 0;
    long long by_strip[8] = {0,0,0,0,0,0,0,0}, by_qmod[8] = {0,0,0,0,0,0,0,0};
    for (size_t i = 0; i < hv.size(); ++i) {
        double d = fabs((double)hv[i] - (double)hr[i]);
        if (d > 0) ++diff;
        if (d > maxd) maxd = d;
        if (d > 0 && shown < 5 && hr[i] != 0.0f) {
            size_t qq = i / m, rr = i % m;
            printf("  mismatch q=%zu row=%zu fused=%.4f ref=%.4f\n",
                   qq, rr, (double)hv[i], (double)hr[i]);
            ++shown;
        }
        if (d > 0) {
            by_strip[(i % m)/16 & 7]++;
            by_qmod[(i / m) & 7]++;
        }
    }
    printf("  by_strip:");
    for (int k2 = 0; k2 < 8; ++k2) printf(" %lld", by_strip[k2]);
    printf("\n  by_qmod: ");
    for (int k2 = 0; k2 < 8; ++k2) printf(" %lld", by_qmod[k2]);
    printf("\n");
    printf("out diff %zu / %zu max=%.6f\n", diff, hv.size(), maxd);
    {
        long long by_strip[8] = {0}, by_qmod[8] = {0};
        for (size_t i = 0; i < hv.size(); ++i) {
            if (((uint32_t)0)) break;
        }
        (void)by_strip; (void)by_qmod;
    }
    fflush(stdout);
    cudaFree(bank); cudaFree(vref);
    cudaStreamDestroy(st);
    cudaFree(kept); cudaFree(tlut); cudaFree(p4t); cudaFree(p4l);
    cudaFree(scr); cudaFree(xg); cudaFree(v); cudaFree(gam);
    return (int)(le != cudaSuccess || se != cudaSuccess);
}

// --- per-routing-slot expert bank cache (nt==1 only), GGML_PAW_EXP_CACHE=1 ---
//
// The dense decode/apply path above already only touches active groups
// (scr_i[g]==0 skips); the actual inefficiency is upstream of it: at nt=1,
// P=n_used (~8) is always below GGML_PAW_DENSE_MIN (default 1024), so the
// walk path runs instead, which fuses decode+apply and throws the decoded
// values away every single token even though routing is sticky (the same
// expert is often reused a few tokens in a row). This caches the decoded
// bank per ROUTING-RANK SLOT s (0..n_used-1), persistent across tokens: slot
// s holds whichever group last occupied rank s, tagged by that group's id.
// If rank s routes to the same group again next token, decode is skipped.
//
// Scoped to v8+p4 (reason8192's runtime config, confirmed ng=256=n_expert
// via GGML_PAW_TIME output -- no expert-group clustering, so group id ==
// remapped expert id, and per-rank caching cannot mis-attribute a pair to
// the wrong group). Falls back to the existing walk/dense path otherwise.
// If a future checkpoint DOES use clustering (n_groups < n_expert), two
// ranks sharing a group would just decode it twice into two slots --
// redundant, not incorrect, since each slot is independently tagged.
//
// Decode math below is a verbatim copy of paw_exp_dense_decode_kernel's
// V8/P4 branch (see its own numeric-identity comments) -- only the outer
// indexing (persistent slot instead of scanned n_groups block, tag-gated
// instead of scr_i-cnt-gated) differs. The tag write is a SEPARATE kernel
// launched after decode (stream-ordered, so it can never observe a
// partially-written bank) rather than done inline, to avoid one block
// racing another block's still-in-flight decode of the same slot.
struct paw_exp_slots {
    half    * banks;     // [n_used, m, n]
    int32_t * tags_dev;  // [n_used], -1 = empty
};
static std::mutex paw_exp_slots_mutex;
static std::unordered_map<const void *, paw_exp_slots> paw_exp_slot_map;

static bool paw_exp_cache_on() {
    static const bool on = paw_env_int("GGML_PAW_EXP_CACHE", 0) != 0;
    return on;
}

static paw_exp_slots paw_exp_slots_get(
        const void * kept, const int m, const int n, const int n_used, cudaStream_t stream) {
    {
        std::lock_guard<std::mutex> lock(paw_exp_slots_mutex);
        auto it = paw_exp_slot_map.find(kept);
        if (it != paw_exp_slot_map.end()) {
            return it->second;
        }
    }
    paw_exp_slots s;
    CUDA_CHECK(cudaMalloc(&s.banks, (size_t) n_used*m*n*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s.tags_dev, (size_t) n_used*sizeof(int32_t)));
    std::vector<int32_t> init_tags(n_used, -1);
    CUDA_CHECK(cudaMemcpyAsync(s.tags_dev, init_tags.data(), n_used*sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        std::lock_guard<std::mutex> lock(paw_exp_slots_mutex);
        auto it = paw_exp_slot_map.find(kept);
        if (it != paw_exp_slot_map.end()) {
            cudaFree(s.banks);
            cudaFree(s.tags_dev);
            return it->second;
        }
        paw_exp_slot_map.emplace(kept, s);
    }
    return s;
}

template <bool P4>
static __global__ void paw_exp_slot_decode_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT remap,
        const int32_t  * GGML_CUDA_RESTRICT ids,
        const int32_t  * GGML_CUDA_RESTRICT tags,     // [n_used]
        half           * GGML_CUDA_RESTRICT banks,    // [n_used, m, n]
        const int m, const int n, const int ids_s0) {
    constexpr int WG = 256;
    __shared__ float lv[16];

    (void) p4; (void) p4lv;

    const int sslot = blockIdx.z;
    const int tid   = threadIdx.x;

    ggml_cuda_pdl_sync();

    const uint32_t id = (uint32_t) ids[sslot*ids_s0];
    const int      g  = (int) (uint32_t) remap[id];   // no dem flag: reason8192 never demotes

    if (tags[sslot] == g) {   // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {
        return;
    }
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;
    const int tile    = tr*tiles_y + tc;
    const int ntiles  = (m/16)*tiles_y;

    const int64_t tbase = ((int64_t) g*ntiles + tile)*24;

    half * dst = banks + ((int64_t) sslot*m + tr*16 + rr)*n + tc*16;
    half tmp[16];

#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int      i   = 2*rr + jj;
        const int      bb  = 12*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t row = ph & 0x7FFFu;
        if constexpr (P4) {
            const uint32_t pk = p4[row];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = lv[(pk >> (4*c)) & 0xFu];
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp[8*jj + c] = __float2half_rn(vv);
            }
        } else {
            const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = __half2float(tl[c]);
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp[8*jj + c] = __float2half_rn(vv);
            }
        }
    }
    paw_store_half16(dst, tmp);
}

// stream-ordered after the decode launch above, before the apply launch
// below: safe to observe a bank as soon as its tag matches, since decode is
// fully complete (same stream) by the time this or apply runs.
static __global__ void paw_exp_slot_tag_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT tags,
        const int n_used, const int ids_s0) {
    const int s = threadIdx.x;
    if (s >= n_used) {
        return;
    }
    const uint32_t id = (uint32_t) ids[s*ids_s0];
    tags[s] = (int32_t) (uint32_t) remap[id];
}

// builds a trivial scr_i (cnt=1, off=s, orig_id=ids[s], pidx=s per slot) so
// the existing, unmodified paw_exp_apply_kernel<true,PC> can run over the
// slot banks as if n_groups==n_used and every "group" has exactly one pair.
static __global__ void paw_exp_slot_scri_kernel(
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT scr_i,
        const int n_used, const int ids_s0) {
    const int s = threadIdx.x;
    if (s >= n_used) {
        return;
    }
    scr_i[s]             = 1;
    scr_i[n_used + s]    = s;
    scr_i[2*n_used + s]  = ids[s*ids_s0];
    scr_i[4*n_used + s]  = s;
}

template <int WG>
static __global__ void paw_exp_out_kernel(
        const half    * GGML_CUDA_RESTRICT sv,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT scr_v,
        float         * GGML_CUDA_RESTRICT dst,
        const int m, const int n_used, const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    for (int i = tid; i < m; i += WG) {
        __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) p*m + i], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    if (WG == m/16 && paw_fwht_v2_ok(m)) {
        paw_fwht_block_v2(sh, m, tid, WG);
    } else {
        paw_fwht_block(sh, m, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) m);
    const float inv_sc = __frcp_rn(sc);
    const int64_t obase = (int64_t) t*n_used*m + (int64_t) s*m;
    for (int i = tid; i < m; i += WG) {
        dst[obase + i] = sh[i] * inv_sc * __half2float(sv[e*m + i]);
    }
}

// ---------------------------------------------------------------------------
// batched EXP_MM for the gate+up pair (GGML_OP_PAW_EXP_MM_BATCH2,
// GGML_PAW_EXP_BATCH2=1). gate_exps and up_exps share one input (xexp),
// one routing decision (remap/ids), and one model-global tlut/p4 codebook --
// only the per-expert trellis (kept), wave_gamma, and su/sv scale vectors
// differ. Scoped by the caller (paw.cpp) to this checkpoint's actual
// runtime shape: V8+P4, no demotion, no low-rank basis correction,
// decode-only (n_tok==1) -- anything else falls back to two separate
// ggml_paw_exp_mm calls. group (routing) is shared and runs once; u/walk/
// out each run once for the whole pair via an extra grid dimension selecting
// matrix 0 (gate) vs 1 (up). Kernel bodies below are copied verbatim from
// the single-matrix versions per matrix -- see those kernels' comments for
// the math; only the per-matrix pointer/index selection is new, the same
// pattern as paw_exp_group_u_kernel's existing group+u fusion.

template <int WG>
static __global__ void paw_exp_group_u2_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT scr,
        int32_t        * GGML_CUDA_RESTRICT active_g,
        const half     * GGML_CUDA_RESTRICT su0,
        const half     * GGML_CUDA_RESTRICT su1,
        const float    * GGML_CUDA_RESTRICT x,
        float          * GGML_CUDA_RESTRICT scr_u0,
        float          * GGML_CUDA_RESTRICT scr_u1,
        const int n, const int n_used, const int n_tok, const int n_kept, const int n_groups,
        const int xne1, const int ids_s0, const int ids_s1) {
    if (blockIdx.x == 0) {
        if (blockIdx.y != 0) {
            return;   // exp_group's result is global, only run it once
        }
        __shared__ int sh_cnt[512];
        __shared__ int sh_cur[512];
        __shared__ int sh_orig[512];

        const int tid = threadIdx.x;
        const int P   = n_used*n_tok;

        ggml_cuda_pdl_sync();
        for (int g = tid; g < n_groups; g += WG) {
            sh_cnt[g]  = 0;
            sh_orig[g] = 0;
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int s = p % n_used;
            const int t = p / n_used;
            const int g = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            atomicAdd(&sh_cnt[g], 1);
            atomicExch(&sh_orig[g], ids[s*ids_s0 + t*ids_s1]);
        }
        __syncthreads();
        if (tid == 0) {
            int off = 0;
            for (int g = 0; g < n_groups; ++g) {
                const int c = sh_cnt[g];
                sh_cur[g] = off;
                scr[g]              = c;
                scr[n_groups + g]   = off;
                scr[2*n_groups + g] = sh_orig[g];
                off += c;
            }
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int g   = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            const int pos = atomicAdd(&sh_cur[g], 1);
            scr[4*n_groups + pos] = p;
        }
        if (active_g != nullptr && tid < n_used) {
            active_g[tid] = (int32_t) paw_group_of(remap, ids, tid, n_used, n_kept, ids_s0, 0);
        }
        return;
    }

    __shared__ float sh[2048];

    const int which = blockIdx.x > n_used ? 1 : 0;
    const int s      = which == 0 ? blockIdx.x - 1 : blockIdx.x - 1 - n_used;
    const int t      = blockIdx.y;
    const int p      = t*n_used + s;
    const int tid    = threadIdx.x;
    const half  * su    = which == 0 ? su0    : su1;
    float       * scr_u = which == 0 ? scr_u0 : scr_u1;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;
    for (int i = tid; i < n; i += WG) {
        sh[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) n);
    const float inv_sc = __frcp_rn(sc);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) p*n + i] = sh[i] * inv_sc;
    }
}

// V8+P4 only (this batched path never sees the V2 codec or the narrow-input
// warp variant -- gate/up's n=n_embd is always > 512 in this checkpoint).
// blockIdx.y selects gate (0) / up (1); routing (scr_i, active_g) and grid
// geometry are shared since gate/up share one selected_experts/remap.
static __global__ void paw_exp_walk2_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept0,
        const uint16_t * GGML_CUDA_RESTRICT kept1,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const float    * GGML_CUDA_RESTRICT scr_u0,
        const float    * GGML_CUDA_RESTRICT scr_u1,
        float          * GGML_CUDA_RESTRICT scr_v0,
        float          * GGML_CUDA_RESTRICT scr_v1,
        const half     * GGML_CUDA_RESTRICT gamma0,
        const half     * GGML_CUDA_RESTRICT gamma1,
        const int32_t  * GGML_CUDA_RESTRICT active_g,
        const int m, const int n, const int n_groups) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][4];
    __shared__ float lv[16];

    const bool which = blockIdx.y != 0;
    const uint16_t * kept  = which ? kept1  : kept0;
    const float     * scr_u = which ? scr_u1 : scr_u0;
    float            * scr_v = which ? scr_v1 : scr_v0;
    const half       * gamma = which ? gamma1 : gamma0;

    const int g   = active_g != nullptr ? active_g[blockIdx.z] : blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    const bool p4_on = p4 != nullptr;
    if (p4_on) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    const int words = 24;
    const uint16_t * trd = kept;
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;
    const int e_orig = scr_i[2*n_groups + g];
    const int Mb     = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                 : Mb + tiles_y - 2 - (tr + tid);
        gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
    }

    uint16_t w8[24];
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
#pragma unroll
        for (int q = 0; q < 24; ++q) {
            w8[q] = trd[tw + q];
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + tid*16;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
                const uint32_t ph  = reg*(reg + 1u);
                const uint32_t row = ph & 0x7FFFu;
                const int      ri  = (8*i) >> 4;
                const int      ci  = (8*i) & 15;
                float dotp = 0.0f;
                if (p4_on) {
                    const uint32_t pk = p4[row];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = lv[(pk >> (4*c)) & 0xFu];
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;
                        }
                        dotp = fmaf(vv, ub[ci + c], dotp);
                    }
                } else {
                    const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = __half2float(tl[c]);
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;
                        }
                        dotp = fmaf(vv, ub[ci + c], dotp);
                    }
                }
                // gsc is loop-invariant across every i/c/qq for this thread
                // (depends only on the thread's fixed row-tile); deferring
                // the scale to here instead of every inner c-iteration cuts
                // 256 redundant multiplies/token/thread down to 16.
                partial[ri] += dotp;
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] *= gsc;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(partial[i]);
            if (lane == 0) {
                red[i][wid] = s;
            }
        }
        __syncthreads();
        if (tid < 16) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[tid][wj];
            }
            scr_v[(int64_t) p*m + tr*16 + tid] = sum;
        }
        if (qq + 1 < cnt) __syncthreads();   // red is reused by the next pair -- skip on last iter, nothing left to protect
    }
}

// blockIdx.z selects gate (0) / up (1). dst is [m, n_used, 2, n_tok], so
// each complete projection is contiguous and directly viewable.
template <int WG>
static __global__ void paw_exp_out2_kernel(
        const half    * GGML_CUDA_RESTRICT sv0,
        const half    * GGML_CUDA_RESTRICT sv1,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT scr_v0,
        const float   * GGML_CUDA_RESTRICT scr_v1,
        float         * GGML_CUDA_RESTRICT dst,
        const int m, const int n_used, const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s     = blockIdx.x;
    const int t     = blockIdx.y;
    const int which = blockIdx.z;
    const int p     = t*n_used + s;
    const int tid   = threadIdx.x;

    const half  * sv    = which == 0 ? sv0    : sv1;
    const float * scr_v = which == 0 ? scr_v0 : scr_v1;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    for (int i = tid; i < m; i += WG) {
        __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) p*m + i], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    if (WG == m/16 && paw_fwht_v2_ok(m)) {
        paw_fwht_block_v2(sh, m, tid, WG);
    } else {
        paw_fwht_block(sh, m, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) m);
    const float inv_sc = __frcp_rn(sc);
    const int64_t obase = ((int64_t) t*2 + which)*n_used*m + (int64_t) s*m;
    for (int i = tid; i < m; i += WG) {
        dst[obase + i] = sh[i] * inv_sc * __half2float(sv[e*m + i]);
    }
}

void ggml_cuda_op_paw_exp_mm_batch2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kept0  = dst->src[0];
    const ggml_tensor * su0    = dst->src[1];
    const ggml_tensor * sv0    = dst->src[2];
    const ggml_tensor * gamma0 = dst->src[3];
    const ggml_tensor * kept1  = dst->src[4];
    const ggml_tensor * su1    = dst->src[5];
    const ggml_tensor * sv1    = dst->src[6];
    const ggml_tensor * gamma1 = dst->src[7];
    const ggml_tensor * tlut   = dst->src[8];
    const ggml_tensor * remap  = dst->src[9];
    const ggml_tensor * ids    = dst->src[10];
    const ggml_tensor * x      = dst->src[11];

    GGML_ASSERT(kept0->type  == GGML_TYPE_I16 && kept1->type  == GGML_TYPE_I16);
    GGML_ASSERT(su0->type    == GGML_TYPE_F16 && su1->type    == GGML_TYPE_F16);
    GGML_ASSERT(sv0->type    == GGML_TYPE_F16 && sv1->type    == GGML_TYPE_F16);
    GGML_ASSERT(gamma0->type == GGML_TYPE_F16 && gamma1->type == GGML_TYPE_F16);
    GGML_ASSERT(tlut->type   == GGML_TYPE_F16);
    GGML_ASSERT(remap->type  == GGML_TYPE_I32);
    GGML_ASSERT(ids->type    == GGML_TYPE_I32);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kept0)  && ggml_is_contiguous(kept1));
    GGML_ASSERT(ggml_is_contiguous(su0)    && ggml_is_contiguous(su1));
    GGML_ASSERT(ggml_is_contiguous(sv0)    && ggml_is_contiguous(sv1));
    GGML_ASSERT(ggml_is_contiguous(gamma0) && ggml_is_contiguous(gamma1));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(kept0->ne[0] == 24 && kept1->ne[0] == 24);   // V8 walk rate
    GGML_ASSERT(tlut->ne[0] == 8);

    const int n        = (int) su0->ne[0];
    const int m        = (int) sv0->ne[0];
    const int n_kept   = (int) kept0->ne[2];
    const int n_groups = n_kept;             // batched path requires no demotion
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int P        = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int ids_s0   = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1   = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(n <= 2048 && m <= 2048);
    GGML_ASSERT(n_groups <= 512);
    GGML_ASSERT(n_tok == 1);   // caller-scoped: decode-only

    static const bool walk_compact = paw_env_int("GGML_PAW_EXP_WALK_COMPACT", 1) != 0;
    ggml_cuda_pool_alloc<int32_t> scr_i_alloc(ctx.pool(), (size_t) 4*n_groups + P + (walk_compact ? n_used : 0));
    ggml_cuda_pool_alloc<float>   scr_f_alloc(ctx.pool(), 2*((size_t) P*n + (size_t) P*m));
    int32_t * scr_i  = scr_i_alloc.get();
    int32_t * active_g = walk_compact ? scr_i + 4*n_groups + P : nullptr;
    float   * scr_u0 = scr_f_alloc.get();
    float   * scr_v0 = scr_u0 + (size_t) P*n;
    float   * scr_u1 = scr_v0 + (size_t) P*m;
    float   * scr_v1 = scr_u1 + (size_t) P*n;

    cudaStream_t stream = ctx.stream();

    paw_p4_table p4t = paw_exp_p4_table(tlut->data, stream);
    const bool p4 = p4t.packed != nullptr;

    char shp[96];
    snprintf(shp, sizeof(shp), " m=%d n=%d n_used=%d n_tok=%d ng=%d p4=%d",
             m, n, n_used, n_tok, n_groups, (int) p4);

    paw_fwht_set_mode();

    paw_timed(stream, std::string("exp_group_u2") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
        paw_fwht_for_wg(n/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_exp_group_u2_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(2*n_used + 1, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i, active_g,
                (const half *) su0->data, (const half *) su1->data,
                (const float *) x->data, scr_u0, scr_u1,
                n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
        });
    } else if (paw_fwht_wg512()) {
        paw_launch(paw_exp_group_u2_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(2*n_used + 1, n_tok, 1), dim3(512, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i, active_g,
            (const half *) su0->data, (const half *) su1->data,
            (const float *) x->data, scr_u0, scr_u1,
            n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
    } else {
        paw_launch(paw_exp_group_u2_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(2*n_used + 1, n_tok, 1), dim3(256, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i, active_g,
            (const half *) su0->data, (const half *) su1->data,
            (const float *) x->data, scr_u0, scr_u1,
            n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
    }
    });

    const int grid_z = walk_compact ? n_used : n_groups;

    paw_timed(stream, std::string("exp_walk2") + shp, [&]() {
    paw_launch(paw_exp_walk2_kernel,
        ggml_cuda_kernel_launch_params(dim3(m/16, 2, grid_z), dim3(128, 1, 1), 0, stream),
        (const uint16_t *) kept0->data, (const uint16_t *) kept1->data,
        (const void *) tlut->data, p4t.packed, p4t.levels,
        (const int32_t *) scr_i, scr_u0, scr_u1, scr_v0, scr_v1,
        (const half *) gamma0->data, (const half *) gamma1->data,
        active_g, m, n, n_groups);
    });

    paw_timed(stream, std::string("exp_out2") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
        paw_fwht_for_wg(m/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_exp_out2_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 2), dim3(wg, 1, 1), 0, stream),
                (const half *) sv0->data, (const half *) sv1->data, (const int32_t *) ids->data,
                scr_v0, scr_v1, (float *) dst->data, m, n_used, ids_s0, ids_s1);
        });
    } else if (paw_fwht_wg512()) {
        paw_launch(paw_exp_out2_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 2), dim3(512, 1, 1), 0, stream),
            (const half *) sv0->data, (const half *) sv1->data, (const int32_t *) ids->data,
            scr_v0, scr_v1, (float *) dst->data, m, n_used, ids_s0, ids_s1);
    } else {
        paw_launch(paw_exp_out2_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 2), dim3(256, 1, 1), 0, stream),
            (const half *) sv0->data, (const half *) sv1->data, (const int32_t *) ids->data,
            scr_v0, scr_v1, (float *) dst->data, m, n_used, ids_s0, ids_s1);
    }
    });
}

void ggml_cuda_op_paw_exp_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kept  = dst->src[0];
    const ggml_tensor * dem   = dst->src[1];
    const ggml_tensor * su    = dst->src[2];
    const ggml_tensor * sv    = dst->src[3];
    const ggml_tensor * tlut  = dst->src[4];
    const ggml_tensor * remap = dst->src[5];
    const ggml_tensor * ids   = dst->src[6];
    const ggml_tensor * x     = dst->src[7];
    const ggml_tensor * gamma = dst->src[8];   // payload v3 (V8 walk) only

    const bool v8 = tlut->ne[0] == 8;
    GGML_ASSERT(v8 == (gamma != nullptr));

    GGML_ASSERT(kept->type  == GGML_TYPE_I16);
    GGML_ASSERT(su->type    == GGML_TYPE_F16);
    GGML_ASSERT(sv->type    == GGML_TYPE_F16);
    GGML_ASSERT(tlut->type  == (v8 ? GGML_TYPE_F16 : GGML_TYPE_F32));
    GGML_ASSERT(remap->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(x->type     == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kept));
    GGML_ASSERT(ggml_is_contiguous(su));
    GGML_ASSERT(ggml_is_contiguous(sv));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // the walk kernel hardcodes the rates like the Vulkan shader
    GGML_ASSERT(kept->ne[0] == (v8 ? 24 : 32));
    if (dem != nullptr) {
        GGML_ASSERT(dem->type == GGML_TYPE_I16);
        GGML_ASSERT(ggml_is_contiguous(dem));
        GGML_ASSERT(dem->ne[0] == 16);
    }
    if (gamma != nullptr) {
        GGML_ASSERT(gamma->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(gamma));
    }

    const int n        = (int) su->ne[0];
    const int m        = (int) sv->ne[0];
    const int n_kept   = (int) kept->ne[2];
    const int n_groups = n_kept + (int)(dem ? dem->ne[2] : 0);
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int P        = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int ids_s0   = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1   = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(n <= 2048 && m <= 2048);   // exp_u/exp_out shared bounds
    GGML_ASSERT(n_groups <= 512);          // exp_group shared bound

    ggml_cuda_pool_alloc<int32_t> scr_i_alloc(ctx.pool(), (size_t) 4*n_groups + P);
    ggml_cuda_pool_alloc<float>   scr_f_alloc(ctx.pool(), (size_t) P*n + (size_t) P*m);
    int32_t * scr_i = scr_i_alloc.get();
    float   * scr_u = scr_f_alloc.get();
    float   * scr_v = scr_u + (size_t) P*n;

    cudaStream_t stream = ctx.stream();

    // p4 repack of the V8 tlut (one-time, cached); nullptr => fp16 gathers
    paw_p4_table p4t = {nullptr, nullptr};
    if (v8) {
        GGML_ASSERT(tlut->ne[1] == 32768);
        p4t = paw_exp_p4_table(tlut->data, stream);
    }
    const bool p4 = p4t.packed != nullptr;

    char shp[96];
    snprintf(shp, sizeof(shp), " m=%d n=%d n_used=%d n_tok=%d ng=%d v8=%d p4=%d",
             m, n, n_used, n_tok, n_groups, (int) v8, (int) p4);

    paw_fwht_set_mode();

    static const bool group_fuse = paw_env_int("GGML_PAW_EXP_GROUP_FUSE", 0) != 0;
    if (group_fuse) {
        paw_timed(stream, std::string("exp_group_u") + shp, [&]() {
        if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
            paw_fwht_for_wg(n/16, [&](auto WG) {
                constexpr int wg = decltype(WG)::value;
                paw_launch(paw_exp_group_u_kernel<wg>,
                    ggml_cuda_kernel_launch_params(dim3(n_used + 1, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                    (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
                    (const half *) su->data, (const float *) x->data, scr_u,
                    n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
            });
        } else if (paw_fwht_wg512()) {
            paw_launch(paw_exp_group_u_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(n_used + 1, n_tok, 1), dim3(512, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
                (const half *) su->data, (const float *) x->data, scr_u,
                n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
        } else {
            paw_launch(paw_exp_group_u_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(n_used + 1, n_tok, 1), dim3(256, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
                (const half *) su->data, (const float *) x->data, scr_u,
                n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
        }
        });
    } else {
        paw_timed(stream, std::string("exp_group") + shp, [&]() {
        paw_launch(paw_exp_group_kernel,
            ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(256, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
            n_used, n_tok, n_kept, n_groups, ids_s0, ids_s1);
        });

        paw_timed(stream, std::string("exp_u") + shp, [&]() {
        if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
            paw_fwht_for_wg(n/16, [&](auto WG) {
                constexpr int wg = decltype(WG)::value;
                paw_launch(paw_exp_u_kernel<wg>,
                    ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                    (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
                    scr_u, n, n_used, xne1, ids_s0, ids_s1);
            });
        } else if (paw_fwht_wg512()) {
            paw_launch(paw_exp_u_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
                (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
                scr_u, n, n_used, xne1, ids_s0, ids_s1);
        } else {
            paw_launch(paw_exp_u_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
                (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
                scr_u, n, n_used, xne1, ids_s0, ids_s1);
        }
        });
    }

    const uint16_t * kept_d = (const uint16_t *) kept->data;
    const uint16_t * dem_d  = dem != nullptr ? (const uint16_t *) dem->data : kept_d;

    if (paw_exp_cache_on() && v8 && p4 && n_tok == 1 && dem == nullptr) {
        // single-token decode, the case the walk path never amortizes.
        // Scoped to v8+p4+no-demotion (reason8192's actual runtime config)
        // -- see paw_exp_slot_decode_kernel's header comment.
        paw_exp_slots slots = paw_exp_slots_get(kept_d, m, n, n_used, stream);
        ggml_cuda_pool_alloc<int32_t> scr_i_slot_alloc(ctx.pool(), (size_t) 5*n_used);
        int32_t * scr_i_slot = scr_i_slot_alloc.get();

        const ggml_cuda_kernel_launch_params dec_params =
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_used),
                                           dim3(256, 1, 1), 0, stream);
        paw_timed(stream, std::string("exp_slot_decode") + shp, [&]() {
        paw_launch(paw_exp_slot_decode_kernel<true>, dec_params,
            kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
            (const int32_t *) remap->data, (const int32_t *) ids->data,
            slots.tags_dev, slots.banks, m, n, ids_s0);
        });
        paw_launch(paw_exp_slot_tag_kernel,
            ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(n_used, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, slots.tags_dev, n_used, ids_s0);
        paw_launch(paw_exp_slot_scri_kernel,
            ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(n_used, 1, 1), 0, stream),
            (const int32_t *) ids->data, scr_i_slot, n_used, ids_s0);

        static const int exp_pc = paw_env_int("GGML_PAW_EXP_PC", 8);
        paw_timed(stream, std::string("exp_slot_apply") + shp, [&]() {
        if (exp_pc == 1) {
            paw_launch(paw_exp_apply_kernel<true, 1>,
                ggml_cuda_kernel_launch_params(dim3(m/16, n_used, 1), dim3(128, 1, 1), 0, stream),
                (const half *) slots.banks, scr_i_slot, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_used);
        } else if (exp_pc == 4) {
            paw_launch(paw_exp_apply_kernel<true, 4>,
                ggml_cuda_kernel_launch_params(dim3(m/16, n_used, 1), dim3(128, 1, 1), 0, stream),
                (const half *) slots.banks, scr_i_slot, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_used);
        } else {
            paw_launch(paw_exp_apply_kernel<true, 8>,
                ggml_cuda_kernel_launch_params(dim3(m/16, n_used, 1), dim3(128, 1, 1), 0, stream),
                (const half *) slots.banks, scr_i_slot, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_used);
        }
        });

        paw_timed(stream, std::string("exp_out") + shp, [&]() {
        if (paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
            paw_fwht_for_wg(m/16, [&](auto WG) {
                constexpr int wg = decltype(WG)::value;
                paw_launch(paw_exp_out_kernel<wg>,
                    ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                    (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                    m, n_used, ids_s0, ids_s1);
            });
        } else if (paw_fwht_wg512()) {
            paw_launch(paw_exp_out_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        } else {
            paw_launch(paw_exp_out_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        }
        });
        return;
    }

    static const int dense_min = paw_env_int("GGML_PAW_DENSE_MIN", 1024);
    ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());
    if (P >= dense_min) {
        // dense prefill: decode active groups once, then apply
        static const bool exp_ws  = paw_env_int("GGML_PAW_EXP_APPLY_WS", 1) != 0;
        static const bool exp_gp  = paw_env_int("GGML_PAW_EXP_XGROUP", 1) != 0;
        static const int exp_pc   = paw_env_int("GGML_PAW_EXP_PC", 8);
        static const bool exp_blas = paw_env_int("GGML_PAW_EXP_BLAS", 1) != 0;
        // slab width for the chunked blas pipeline; bounds staging VRAM
        static const int exp_chunk = paw_env_int("GGML_PAW_EXP_BLAS_CHUNK_G", 16);
        // pool is strict LIFO: bank must be allocated before xg so the
        // destructor order (xg first, bank last) unwinds it correctly
        // the blas branch pulls group counts to host (uncaptured stream);
        // under graph capture fall back to the fully device-driven path
        cudaStreamCaptureStatus cst = cudaStreamCaptureStatusNone;
        const bool capturing = cudaStreamIsCapturing(stream, &cst) == cudaSuccess &&
                               cst == cudaStreamCaptureStatusActive;
        // single-token passes go through the fused WS kernel instead: no
        // staging round-trip and no host sync (measured +3 t/s generation)
        // below this, micro-batches skip the cuBLAS pipeline (host sync
        // per pass dominates); measured neutral-to-better at 29k ctx
        static const int exp_blas_min_nt = paw_env_int("GGML_PAW_EXP_BLAS_MIN_NT", 128);
        // fully-fused trellis->wmma apply: no bank materialization, no host
        // sync; supersedes the chunked cuBLAS pipeline where it applies
        // correct and graph-safe, but on RTX 3060 the cuBLAS split still
        // beats it at large nt (see paw_fused_bench); flip when the kernel
        // closes the activation-restage gap
        static const bool exp_fused = paw_env_int("GGML_PAW_EXP_FUSED", 0) != 0;
        const bool use_fused = v8 && exp_fused && exp_ws && exp_gp &&
                               m % 16 == 0 && n_tok >= exp_blas_min_nt;
        const bool use_blas = v8 && exp_blas && exp_ws && exp_gp &&
                              m % 16 == 0 && n_tok >= exp_blas_min_nt &&
                              !use_fused && !capturing;
        // overlap decode of slab i+1 with GEMMs of slab i on a second stream
        // off by default: measured no gain (decode and GEMM contend for the same
// bandwidth on this part); kept for parts where that does not hold
        static const bool exp_ovl = paw_env_int("GGML_PAW_EXP_OVERLAP", 0) != 0;
        const bool do_ovl = use_blas && exp_ovl && n_groups > exp_chunk;
        const int n_slabs = use_blas ? (n_groups + exp_chunk - 1)/exp_chunk : 1;
        const int bank_bufs = do_ovl ? 2 : 1;
        const int bank_g = use_blas ? (exp_chunk < n_groups ? exp_chunk : n_groups)
                                    : n_groups;
        half * bank = use_fused ? nullptr
                                : bank_alloc.alloc((size_t) bank_g*bank_bufs*m*n);
        ggml_cuda_pool_alloc<half> xg_alloc(ctx.pool());
        half * xg = nullptr;
        if (v8 && exp_ws && exp_gp) {
            xg = xg_alloc.alloc((size_t) P*n);
            paw_timed(stream, std::string("exp_xgroup") + shp, [&]() {
            paw_launch(paw_exp_permute_x_kernel,
                ggml_cuda_kernel_launch_params(
                    dim3((n + 255)/256, n_groups, 1), dim3(256, 1, 1), 0, stream),
                (const int32_t *) scr_i, (const float *) scr_u, xg, n, n_groups);
            });
        }
        if (use_fused) {
            const ggml_cuda_kernel_launch_params fparams =
                ggml_cuda_kernel_launch_params(
                    dim3(m/64, n_groups, 1), dim3(128, 1, 1), 0, stream);
            paw_timed(stream, std::string("exp_fused") + shp, [&]() {
                if (p4) {
                    const paw_l2_tables & lt = paw_l2_tables_get(stream, p4t);
                    paw_launch(paw_exp_apply_kernel_fused<true, 4>, fparams,
                        kept_d, (const half *) nullptr, (const void *) tlut->data,
                        lt.packed ? lt.packed : p4t.packed,
                        lt.levels ? lt.levels : p4t.levels,
                        (const int32_t *) scr_i, xg, scr_v,
                        (const half *) gamma->data,
                        (float *) nullptr, m, n, n_groups);
                } else {
                    paw_launch(paw_exp_apply_kernel_fused<false, 4>, fparams,
                        kept_d, (const half *) nullptr, (const void *) tlut->data,
                        (const uint32_t *) nullptr, (const float *) nullptr,
                        (const int32_t *) scr_i, xg, scr_v,
                        (const half *) gamma->data,
                        (float *) nullptr, m, n, n_groups);
                }
            });
        } else if (!use_blas) {
        static const bool dec_v2 = paw_env_int("GGML_PAW_EXP_DECODE_V2", 1) != 0;
        const ggml_cuda_kernel_launch_params dec_params =
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups),
                                           dim3(256, 1, 1), 0, stream);
        paw_timed(stream, std::string("exp_dense_decode") + shp, [&]() {
        if (v8) {
            // warp-per-tile decode: same bits out, fewer passes over the
            // stream words
            const ggml_cuda_kernel_launch_params v2_params =
                ggml_cuda_kernel_launch_params(
                    dim3(((m/16)*(n/16) + 7)/8, 1, n_groups), dim3(256, 1, 1), 0, stream);
            if (p4 && dec_v2) {
                paw_launch(paw_exp_dense_decode_kernel_v2<true>, v2_params,
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, bank, m, n, n_groups, 0,
                    nullptr);
            } else if (dec_v2) {
                paw_launch(paw_exp_dense_decode_kernel_v2<false>, v2_params,
                    kept_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, bank, m, n, n_groups, 0,
                    nullptr);
            }
        } else {
            paw_launch(paw_exp_dense_decode_kernel<false, false>, dec_params,
                kept_d, dem_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const int32_t *) scr_i, bank, m, n, n_kept, n_groups, 0,
                nullptr);
        }
        });
        }
        const ggml_cuda_kernel_launch_params app_params =
            ggml_cuda_kernel_launch_params(dim3(m/16, n_groups, 1), dim3(128, 1, 1), 0, stream);
        if (use_blas) {
            // tall-skinny shape: per-expert dense GEMMs over the grouped
            // slab beat every custom kernel here; the wave gamma is folded
            // into the decoded banks first
            // fp16 output keeps the padded slab small enough for 12GB cards;
            // the unpermute pass widens back to fp32 for downstream ops
            ggml_cuda_pool_alloc<half> ygf_alloc(ctx.pool());
            half * ygf = ygf_alloc.alloc((size_t) P*m);
            // the per-group GEMM dims live on device; pull the count table
            // once (a few hundred bytes). Requires uncaptured stream, so the
            // caller keeps this branch behind GGML_CUDA_DISABLE_GRAPHS=1.
            std::vector<int32_t> hscr(2*n_groups);
            CUDA_CHECK(cudaMemcpyAsync(hscr.data(), scr_i,
                (size_t) 2*n_groups*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            // diagnostic: GGML_PAW_EXP_BLAS_SKIP_DECODE=1 times GEMMs alone
            // (output is garbage; perf signal only)
            static const bool blas_skip_dec =
                paw_env_int("GGML_PAW_EXP_BLAS_SKIP_DECODE", 0) != 0;
            paw_timed(stream, std::string("exp_blas") + shp, [&]() {
            std::vector<cudaEvent_t> ev(do_ovl ? 2*n_slabs + 1 : 0);
            if (do_ovl) {
                for (auto & e : ev) {
                    CUDA_CHECK(cudaEventCreate(&e));
                }
                CUDA_CHECK(cudaEventRecord(ev[0], stream));
            }
            cudaStream_t ds = do_ovl ? paw_aux_stream() : stream;
            for (int si = 0; si < n_slabs; ++si) {
                const int g0 = si*exp_chunk;
                const int gc = exp_chunk < n_groups - g0 ? exp_chunk : n_groups - g0;
                half * buf = bank + (size_t)(do_ovl ? (si % 2) : 0)*bank_g*m*n;
                const ggml_cuda_kernel_launch_params dp =
                    ggml_cuda_kernel_launch_params(
                        dim3((m/16*n + 255)/256, 1, gc), dim3(256, 1, 1), 0, ds);
                if (!blas_skip_dec) {
                    if (do_ovl) {
                        // gate: previous slab's GEMMs released this buffer
                        CUDA_CHECK(cudaStreamWaitEvent(ds, ev[2*si]));
                    }
                    static const bool dec_v2 = paw_env_int("GGML_PAW_EXP_DECODE_V2", 1) != 0;
                    const ggml_cuda_kernel_launch_params v2p =
                        ggml_cuda_kernel_launch_params(
                            dim3(((m/16)*(n/16) + 7)/8, 1, gc), dim3(256, 1, 1), 0, ds);
                    if (p4 && dec_v2) {
                        paw_launch(paw_exp_dense_decode_kernel_v2<true>, v2p,
                            kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                            (const int32_t *) scr_i, buf, m, n, n_groups, g0,
                            (const half *) gamma->data);
                    } else if (dec_v2) {
                        paw_launch(paw_exp_dense_decode_kernel_v2<false>, v2p,
                            kept_d, (const void *) tlut->data,
                            (const uint32_t *) nullptr, (const float *) nullptr,
                            (const int32_t *) scr_i, buf, m, n, n_groups, g0,
                            (const half *) gamma->data);
                    }
                    if (do_ovl) {
                        CUDA_CHECK(cudaEventRecord(ev[2*si + 1], ds));
                        CUDA_CHECK(cudaStreamWaitEvent(stream, ev[2*si + 1]));
                    }
                }
                for (int gi = 0; gi < gc; ++gi) {
                    const int cnt = hscr[g0 + gi];
                    if (cnt == 0) {
                        continue;
                    }
                    const int off = hscr[n_groups + g0 + gi];
                    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(),
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            m, cnt, n,
                            &alpha,
                            buf + (size_t) gi*m*n,                  CUDA_R_16F, n,
                            xg + (int64_t) off*n,                   CUDA_R_16F, n,
                            &beta,
                            ygf + (int64_t) off*m,                  CUDA_R_16F, m,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
                if (do_ovl && si + 1 < n_slabs) {
                    CUDA_CHECK(cudaEventRecord(ev[2*si + 2], stream));
                }
            }
            if (do_ovl) {
                // pool frees ride on the main stream: pin aux work to it too
                CUDA_CHECK(cudaEventRecord(ev[2*n_slabs], ds));
                CUDA_CHECK(cudaStreamWaitEvent(stream, ev[2*n_slabs]));
                CUDA_CHECK(cudaStreamWaitEvent(ds, ev[2*n_slabs]));
                for (auto & e : ev) {
                    CUDA_CHECK(cudaEventDestroy(e));
                }
            }
            });
            paw_timed(stream, std::string("exp_unperm") + shp, [&]() {
            paw_launch(paw_exp_unpermute_y_kernel,
                ggml_cuda_kernel_launch_params(
                    dim3((m + 255)/256, n_groups, 1), dim3(256, 1, 1), 0, stream),
                (const int32_t *) scr_i, ygf, scr_v, m, n_groups);
            });
        } else if (!use_fused) {
        // fused handled everything above; this branch applies over a
        // materialized bank and must not run when no bank exists
        paw_timed(stream, std::string("exp_apply") + shp, [&]() {
        static const int ws_min_m = paw_env_int("GGML_PAW_EXP_WS_MIN_M", 512);
        if (v8 && exp_ws && xg && m % 64 == 0 && m >= ws_min_m) {
            paw_launch(paw_exp_apply_kernel_ws<true, 4, true>,
                ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const int32_t *) scr_i, nullptr, xg, scr_v,
                (const half *) gamma->data, m, n, n_groups);
        } else if (v8) {
        // m < 1024 keeps the scalar PC path: measured slower than WS there
        // (the pair-gather latency chain dominates the short row dimension)
            if (exp_pc == 1) {
                paw_launch(paw_exp_apply_kernel<true, 1>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else if (exp_pc == 4) {
                paw_launch(paw_exp_apply_kernel<true, 4>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else {
                paw_launch(paw_exp_apply_kernel<true, 8>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            }
        } else {
            if (exp_pc == 1) {
                paw_launch(paw_exp_apply_kernel<false, 1>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            } else if (exp_pc == 4) {
                paw_launch(paw_exp_apply_kernel<false, 4>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            } else {
                paw_launch(paw_exp_apply_kernel<false, 8>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            }
        }
        });
        }
    } else {
        static const bool walk_compact = paw_env_int("GGML_PAW_EXP_WALK_COMPACT", 1) != 0;
        const bool do_compact = walk_compact && n_tok == 1;
        const bool walk_warp = v8 && n/16 <= 32 && m % 64 == 0;
        const bool fuse_warp_group = do_compact && walk_warp;
        ggml_cuda_pool_alloc<int32_t> active_g_alloc(ctx.pool());
        const int32_t * active_g = nullptr;
        const int grid_z = do_compact ? n_used : n_groups;
        if (do_compact && !fuse_warp_group) {
            int32_t * ag = active_g_alloc.alloc(n_used);
            paw_launch(paw_exp_active_groups_kernel,
                ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(n_used, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, ag, n_used, n_kept, ids_s0);
            active_g = ag;
        }

        const ggml_cuda_kernel_launch_params walk_params =
            ggml_cuda_kernel_launch_params(dim3(m/16, 1, grid_z), dim3(128, 1, 1), 0, stream);
        // narrow inputs: warp-per-row-tile variant keeps whole blocks busy
        const ggml_cuda_kernel_launch_params warp_params =
            ggml_cuda_kernel_launch_params(dim3(m/64, 1, grid_z), dim3(128, 1, 1), 0, stream);
        paw_timed(stream, std::string("exp_walk") + shp, [&]() {
        if (walk_warp) {
            if (p4) {
                paw_launch(paw_exp_walk_v8_warp_kernel<true>, warp_params,
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data,
                    fuse_warp_group ? (const int32_t *) remap->data : nullptr,
                    fuse_warp_group ? (const int32_t *) ids->data : nullptr,
                    m, n, n_groups, n_used, n_kept, ids_s0);
            } else {
                paw_launch(paw_exp_walk_v8_warp_kernel<false>, warp_params,
                    kept_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data,
                    fuse_warp_group ? (const int32_t *) remap->data : nullptr,
                    fuse_warp_group ? (const int32_t *) ids->data : nullptr,
                    m, n, n_groups, n_used, n_kept, ids_s0);
            }
        } else if (v8) {
            if (p4) {
                paw_launch(paw_exp_walk_kernel<true, true>, walk_params,
                    kept_d, dem_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, active_g, m, n, n_kept, n_groups);
            } else {
                paw_launch(paw_exp_walk_kernel<true, false>, walk_params,
                    kept_d, dem_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, active_g, m, n, n_kept, n_groups);
            }
        } else {
            paw_launch(paw_exp_walk_kernel<false, false>, walk_params,
                kept_d, dem_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const int32_t *) scr_i, scr_u, scr_v,
                (const half *) nullptr, active_g, m, n, n_kept, n_groups);
        }
        });
    }

    paw_timed(stream, std::string("exp_out") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
        paw_fwht_for_wg(m/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_exp_out_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        });
    } else if (paw_fwht_wg512()) {
        paw_launch(paw_exp_out_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
            (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
            m, n_used, ids_s0, ids_s1);
    } else {
        paw_launch(paw_exp_out_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
            (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
            m, n_used, ids_s0, ids_s1);
    }
    });
}

//
// supports_op — mirrors the Vulkan predicate (ggml-vulkan.cpp)
//

// --- PAW_V_REORDER --------------------------------------------------------
//
// Row permutation for the v3 mach1 codec. Within the segment starting at
// seg_off (seg_rows = hd*K*r rows): out[(v*K + k)*hd + d] =
// in[(k*r + v)*hd + d]. All other rows copy through. Replaces the
// cont/permute/cont + concat chains the graph used per SSM layer.
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

void ggml_cuda_op_paw_v_reorder(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * y = dst->src[0];
    GGML_ASSERT(y->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int seg_off = dst->op_params[0];
    const int hd      = dst->op_params[1];
    const int K       = dst->op_params[2];
    const int r       = dst->op_params[3];

    const int M     = (int) y->ne[0];
    const int T     = (int) y->ne[1];
    const int y_str = (int)(y->nb[1] / sizeof(float));
    const int64_t total = (int64_t) M*T;
    const int blocks = (int)((total + 255)/256);
    paw_launch(paw_v_reorder_kernel,
        ggml_cuda_kernel_launch_params(dim3(blocks, 1, 1), dim3(256, 1, 1), 0, ctx.stream()),
        (const float *) y->data, (float *) dst->data, M, T, y_str, seg_off, hd, K, r);
}

// --- PAW_DUAL_MM ----------------------------------------------------------
//
// Two small [n_embd -> R] GEMVs sharing one input, fused into a single
// launch (the mamba ssm_alpha/ssm_beta projections at decode: two launches
// of ~10 us each for a few KB of arithmetic, 30 layers). One warp per
// output row; x is staged in shared once per block.
template <int WARPS>
static __global__ void paw_dual_mm_kernel(
        const float * GGML_CUDA_RESTRICT w0,  // [K, R]
        const float * GGML_CUDA_RESTRICT w1,  // [K, R]
        const float * GGML_CUDA_RESTRICT x,   // [K, T]
        float       * GGML_CUDA_RESTRICT dst, // [2R, T]
        const int K, const int R, const int T) {
    __shared__ float sh[4096];
    const int t    = blockIdx.z;
    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (int i = threadIdx.x; i < K; i += WARPS*32) {
        sh[i] = x[(int64_t) t*K + i];
    }
    __syncthreads();

    for (int r = wid; r < WARPS*4; r += WARPS) {
        const int row  = blockIdx.x*WARPS*4 + r;
        if (row >= 2*R) {
            return;
        }
        const float * w = (row < R ? w0 : w1) + (int64_t)(row % R)*K;
        float acc = 0.0f;
        for (int i = lane; i < K; i += 32) {
            acc += w[i]*sh[i];
        }
        acc = warp_reduce_sum<32>(acc);
        if (lane == 0) {
            dst[(int64_t) t*(2*R) + row] = acc;
        }
    }
}

void ggml_cuda_op_paw_dual_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w0 = dst->src[0];
    const ggml_tensor * w1 = dst->src[1];
    const ggml_tensor * x  = dst->src[2];
    GGML_ASSERT(x->ne[0] <= 4096);

    const int K = (int) w0->ne[0];
    const int R = (int) w0->ne[1];
    const int T = (int) x->ne[1];
    constexpr int WARPS = 8;
    paw_launch(paw_dual_mm_kernel<WARPS>,
        ggml_cuda_kernel_launch_params(dim3((2*R + WARPS*4 - 1)/(WARPS*4), 1, T),
                                       dim3(WARPS*32, 1, 1), 0, ctx.stream()),
        (const float *) w0->data, (const float *) w1->data, (const float *) x->data,
        (float *) dst->data, K, R, T);
}

bool ggml_cuda_paw_supported(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_PAW_EXP_MM:
            // walk/out kernels stage u and H(v) in fixed shared arrays of 2048
            // floats; the group kernel keeps per-group tables in shared arrays
            // of 512; V=8 (payload v3) requires wave_gamma and vice versa. The
            // two walk variants also fix the tlut storage type: V=8 reads the
            // pre-rounded F16 table, V=2 the F32 one.
            return op->src[2]->ne[0] <= 2048 && op->src[3]->ne[0] <= 2048 &&
                   op->src[0]->ne[2] + (op->src[1] ? op->src[1]->ne[2] : 0) <= 512 &&
                   (op->src[4]->ne[0] == 2) == (op->src[8] == nullptr) &&
                   op->src[4]->type == (op->src[4]->ne[0] == 2 ? GGML_TYPE_F32
                                                               : GGML_TYPE_F16);
        case GGML_OP_PAW_RT_MM: {
            // rt_u stages one rotation block of n (<= 4096) and rt_out one of
            // m (<= 8192 = 32 KB static shared). Unblocked payloads take the
            // whole dimension as the block, which is the old bound verbatim.
            const int64_t n   = op->src[1]->ne[0];
            const int64_t m   = op->src[2]->ne[0];
            const int64_t blk = op->op_params[GGML_PAW_RHT_BLK_SLOT];
            const int64_t bn  = blk ? blk : n;
            const int64_t bm  = blk ? blk : m;
            const int64_t words = op->src[0]->ne[0];
            return bn <= 4096 && bm <= 8192 && n % bn == 0 && m % bm == 0 &&
                   words % 8 == 0 && words >= 16 && words <= 64 &&
                   op->src[3]->type == GGML_TYPE_F16;
        }
        case GGML_OP_PAW_RT_MM_BATCH:
            // per-matrix bounds same as rt_mm; tlut is src[3K]. The batched
            // kernels have not been taught blocked rotations yet, so a blocked
            // payload falls back to the per-matrix path.
            return op->op_params[GGML_PAW_RHT_BLK_SLOT] == 0 &&
                   op->src[3*op->op_params[0]]->type == GGML_TYPE_F16 &&
                   op->src[3*op->op_params[0] + 1]->type == GGML_TYPE_F32;
        case GGML_OP_PAW_NE_MM:
        case GGML_OP_PAW_EMBED_ROWS:
        case GGML_OP_PAW_EXP_BASIS:
        case GGML_OP_PAW_HEAD_MM:
        case GGML_OP_PAW_EMBED_GATHER:
        case GGML_OP_PAW_EXP_MM_BATCH2:
        case GGML_OP_PAW_MOE_REDUCE:
        case GGML_OP_PAW_V_REORDER:
        case GGML_OP_PAW_DUAL_MM:
            // shapes/types are enforced by the ggml builders
            return true;
        default:
            return false;
    }
}

// --- PAW_MOE_REDUCE -------------------------------------------------------
//
// dst[:,t] = sum_s experts[:,s,t] * weights[0,s,t]
//
// Replaces the MoE aggregation's ggml_mul + (n_used-1) ggml_add chain: at
// decode that was 8 elementwise launches per layer (320/token over 40 layers)
// moving a few KB each -- essentially pure launch overhead.
//
// Bit-exactness: the old chain rounded each product to fp32 (ggml_mul wrote it
// to memory) and then summed slot-by-slot in increasing s. __fmul_rn/__fadd_rn
// reproduce exactly that -- plain `acc += e*w` would let nvcc contract into an
// FMA, skipping the intermediate rounding and changing the result.
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

void ggml_cuda_op_paw_moe_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * experts = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_embd = (int) experts->ne[0];
    const int n_used = (int) experts->ne[1];
    const int n_tok  = (int) experts->ne[2];

    constexpr int WG = 256;
    const dim3 grid((unsigned)((n_embd + WG - 1)/WG), (unsigned) n_tok, 1);
    paw_launch(paw_moe_reduce_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(WG, 1, 1), 0, ctx.stream()),
        (const float *) experts->data,
        (const float *) weights->data,
        (float *) dst->data,
        n_embd, n_used);
}
