// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"

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

