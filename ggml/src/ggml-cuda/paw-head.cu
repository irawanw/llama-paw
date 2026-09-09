// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"

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

