// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"


//
// EMBED_ROWS — one thread per (token, group), serial 3-bit unpack
// (paw_embed_rows.comp)
//



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

