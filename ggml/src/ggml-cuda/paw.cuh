#include "common.cuh"

void ggml_cuda_op_paw_ne_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_embed_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_exp_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_exp_basis(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_rt_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_rt_mm_batch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_x3_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_exp_mm_batch2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_head_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_embed_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_moe_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_v_reorder(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_paw_dual_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_paw_supported(const ggml_tensor * op);
