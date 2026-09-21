#include "softcap.cuh"

// scale -> unary -> optional post-scale, in one kernel.
//
// The graph shape "scale, unary, scale" already existed here for tanh
// (Gemma-style softcapping). qwen4exp builds the same shape twice per layer in
// its hyper-connection blocks, on tensors of 4 and 320 elements, where the
// per-node cost is entirely launch overhead - so the fusion matters far more
// there than the arithmetic does.
template <int KIND>
static __global__ void scale_unary_f32(const float * x, float * dst, const float scale, const float post, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    const float v = scale * x[i];
    float u;
    if (KIND == GGML_UNARY_OP_TANH) {
        u = tanhf(v);
    } else if (KIND == GGML_UNARY_OP_SIGMOID) {
        u = 1.0f / (1.0f + expf(-v));
    } else { // GGML_UNARY_OP_SILU
        u = v / (1.0f + expf(-v));
    }
    dst[i] = u * post;
}

template <int KIND>
static void scale_unary_f32_cuda(const float * x, float * dst, const float scale, const float post, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_unary_f32<KIND>, launch_params, x, dst, scale, post, k);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_{TANH,SIGMOID} + GGML_OP_SCALE
// src is the first scale node, dst the second one.
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src, ggml_unary_op unary_op) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float post;
    memcpy(&scale, (float *) src->op_params + 0, sizeof(float));
    memcpy(&post,  (float *) dst->op_params + 0, sizeof(float));

    const int k = ggml_nelements(src0);
    switch (unary_op) {
        case GGML_UNARY_OP_TANH:
            scale_unary_f32_cuda<GGML_UNARY_OP_TANH>(src0_d, dst_d, scale, post, k, stream);
            break;
        case GGML_UNARY_OP_SIGMOID:
            scale_unary_f32_cuda<GGML_UNARY_OP_SIGMOID>(src0_d, dst_d, scale, post, k, stream);
            break;
        default:
            GGML_ABORT("unsupported unary op in scale/unary/scale fusion");
    }
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_SILU (no trailing scale)
// src is the scale node, dst the unary node.
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, (float *) src->op_params + 0, sizeof(float));

    scale_unary_f32_cuda<GGML_UNARY_OP_SILU>(src0_d, dst_d, scale, 1.0f, ggml_nelements(src0), ctx.stream());
}
