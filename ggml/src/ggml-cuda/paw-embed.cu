// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"

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

