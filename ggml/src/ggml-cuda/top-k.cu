#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE


// ---------------------------------------------------------------------------
// Small-k top-k for wide rows.
//
// The CUB DeviceTopK path needs CCCL >= 3.2; below that ggml falls back to a
// full argsort of the row. For a DFlash2 selector that is a 248320-wide radix
// sort per drafted position just to read 16 values -- measured at 7.7 ms per
// speculative round on an RTX 3090, ~15% of the whole round. This kernel reads
// each row once instead: every thread keeps its own top-k in shared memory,
// then K block-wide argmax rounds drain the winners in order.
//
// Ordering matches a stable descending sort (ties broken by lower column), so
// it is a drop-in replacement for the argsort+truncate path.
// ---------------------------------------------------------------------------

#define TOPK_SMALL_NT 128

template <int K>
static __global__ void k_top_k_small(const float * __restrict__ x, int * __restrict__ dst, const int ncols) {
    extern __shared__ unsigned char smem_raw[];
    float * sv = (float *) smem_raw;                     // [NT*K] per-thread candidates, descending
    int   * si = (int *)   (sv + TOPK_SMALL_NT * K);     // [NT*K] their column indices
    float * rv = (float *) (si + TOPK_SMALL_NT * K);     // [NT]   reduction: head value
    int   * rc = (int *)   (rv + TOPK_SMALL_NT);         // [NT]   reduction: head column
    int   * ro = (int *)   (rc + TOPK_SMALL_NT);         // [NT]   reduction: owning thread

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const float * xr = x   + (size_t) row * ncols;
    int         * dr = dst + (size_t) row * K;

    float * mv = sv + tid * K;
    int   * mi = si + tid * K;

#pragma unroll
    for (int i = 0; i < K; ++i) {
        mv[i] = -INFINITY;
        mi[i] = INT_MAX;
    }

    // Single streaming pass. The insert branch is taken only when an element
    // beats this thread's current K-th best, which after the first few hundred
    // columns is rare, so the loop stays a plain coalesced read.
    for (int c = tid; c < ncols; c += TOPK_SMALL_NT) {
        const float v = xr[c];
        if (v > mv[K - 1] || (v == mv[K - 1] && c < mi[K - 1])) {
            int j = K - 1;
            while (j > 0 && (mv[j - 1] < v || (mv[j - 1] == v && mi[j - 1] > c))) {
                mv[j] = mv[j - 1];
                mi[j] = mi[j - 1];
                --j;
            }
            mv[j] = v;
            mi[j] = c;
        }
    }

    int ptr = 0;  // how many of this thread's candidates have been drained
    __syncthreads();

    for (int out = 0; out < K; ++out) {
        const bool live = ptr < K;
        rv[tid] = live ? mv[ptr] : -INFINITY;
        rc[tid] = live ? mi[ptr] : INT_MAX;
        ro[tid] = tid;
        __syncthreads();

        for (int s = TOPK_SMALL_NT / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const int o = tid + s;
                // strictly greater wins; equal values fall to the lower column
                const bool take = (rv[o] > rv[tid]) || (rv[o] == rv[tid] && rc[o] < rc[tid]);
                if (take) {
                    rv[tid] = rv[o];
                    rc[tid] = rc[o];
                    ro[tid] = ro[o];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            dr[out] = rc[0];
        }
        if (tid == ro[0]) {
            ptr++;
        }
        __syncthreads();
    }
}

static bool top_k_small_supported(int64_t ncols, int64_t k) {
    return (k == 8 || k == 16 || k == 32) && ncols > 4096 && ncols <= INT_MAX;
}

template <int K>
static void top_k_small_launch(const float * src, int * dst, int ncols, int nrows, cudaStream_t stream) {
    const size_t smem = (size_t) TOPK_SMALL_NT * K * (sizeof(float) + sizeof(int))
                      + (size_t) TOPK_SMALL_NT * (2 * sizeof(int) + sizeof(float));
    k_top_k_small<K><<<nrows, TOPK_SMALL_NT, smem, stream>>>(src, dst, ncols);
}

static void top_k_small(const float * src, int * dst, int64_t ncols, int64_t nrows, int64_t k, cudaStream_t stream) {
    switch (k) {
        case 8:  top_k_small_launch< 8>(src, dst, (int) ncols, (int) nrows, stream); break;
        case 16: top_k_small_launch<16>(src, dst, (int) ncols, (int) nrows, stream); break;
        case 32: top_k_small_launch<32>(src, dst, (int) ncols, (int) nrows, stream); break;
        default: GGML_ABORT("top_k_small: unsupported k");
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
    (void) pool;

    // GGML_CUDA_NO_TOPK_SMALL=1 restores the previous behaviour for A/B.
    static const bool topk_small_off = [] {
        const char * e = getenv("GGML_CUDA_NO_TOPK_SMALL");
        return e && atoi(e);
    }();

    if (!topk_small_off && top_k_small_supported(ncols, k)) {
        top_k_small(src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
