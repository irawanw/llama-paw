// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"

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
        case GGML_OP_PAW_X3_MM:
            // mul1-v1 sq kernel: K in {1,2,3,4}, fp16 side scales; nt == 1 runs
            // the fused GEMV directly, nt > 1 loops it per token
            return (op->src[0]->ne[0] == 16 || op->src[0]->ne[0] == 32 ||
                    op->src[0]->ne[0] == 48 || op->src[0]->ne[0] == 64) &&
                   op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16;
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

