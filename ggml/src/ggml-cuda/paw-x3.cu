// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"



// ---------------------------------------------------------------------------
// paw_x3 — EXL3-compatible fused decode GEMV (mul1-v1 codec), Phase 4
// integration. Device code ported verbatim from the Phase 3 standalone port
// (scripts/exl3_parity/port/paw_int8_gemv.cu), which is itself a verbatim
// port of MiaAI-Lab/exllamav3 commit 63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed:
//   exllamav3_ext/quant/exl3_gemv_int8_kernel.cuh (sq kernel + units + epilogue)
//   exllamav3_ext/quant/exl3_gemv_int8.cu         (host launch + workspace)
// Scope: per-slice-scale ("sq") path, m == 1, plain INT8 mode 2, K in {2, 3}.
// Differences from the reference host side: the workspace comes from the ggml
// CUDA pool (graph-safe fixed layout, counters memset per launch), the plan
// is cached per (bits, k, n), and the activation row is cast F32 -> F16
// (the reference runtime feeds fp16 activations) before the kernel launch.
// ---------------------------------------------------------------------------

#include <tuple>

namespace paw_x3 {

#define NUM_THREADS 256
#define GEMV_STAGE_D 4
#define SQ_KSPLIT_CAP 64
#define SQ_MINROWS 16
#define SQ_ROWS_MAX 512
#define SQ_COUNTERS_CAP 4096
// qsums holds 4 floats per (slice, row): 4*ksplit*M with ksplit <=
// SQ_KSPLIT_CAP and M <= 8, i.e. 4*64*8 ints. The old *4 (1024 ints)
// overflowed into partials whenever ksplit*M > 256 (e.g. ksplit=34 at
// M=8 gives 272), corrupting slice-0/row-0 outputs of the call.
#define SQ_WS_RESERVED (SQ_COUNTERS_CAP + 4 * SQ_KSPLIT_CAP * 8)

// ---------------------------------------------------------------------------------------------------------
// ptx.cuh primitives

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c)
{
    int d;
    asm ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
    // exl3_dq.cuh fshift: plain 64-bit merge+shift (shift may exceed 32, up to
    // bits*7+16; __funnelshift_r would clamp at 32 and corrupt K=3 windows)
    uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}

#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// ---------------------------------------------------------------------------------------------------------
// hadamard_inner.cuh (128-point, natural order, 1/sqrt(128) per side)

struct half4 { half2 x, y; };

__device__ inline void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        uint64_t h01 =  (uint64_t) i0 | (((uint64_t) i1) << 32);
        uint64_t h23 =  (uint64_t) i2 | (((uint64_t) i3) << 32);
        uint64_t ph01 = __shfl_xor_sync(0xffffffff, h01, i);
        uint64_t ph23 = __shfl_xor_sync(0xffffffff, h23, i);
        float ph0 = __uint_as_float((uint32_t) (ph01 & 0xffffffff));
        float ph1 = __uint_as_float((uint32_t) (ph01 >> 32));
        float ph2 = __uint_as_float((uint32_t) (ph23 & 0xffffffff));
        float ph3 = __uint_as_float((uint32_t) (ph23 >> 32));
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

__device__ inline void shuffle_had_f2x32(float& v, float& w, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint64_t vw = ((uint64_t) __float_as_uint(v)) | (((uint64_t) __float_as_uint(w)) << 32);
        uint64_t pvw = __shfl_xor_sync(0xffffffff, vw, i);
        float pv = __uint_as_float((uint32_t) (pvw & 0xffffffff));
        float pw = __uint_as_float((uint32_t) (pvw >> 32));
        uint32_t vi = __float_as_uint(v);
        uint32_t wi = __float_as_uint(w);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        vi ^= (sfm & 0x80000000);
        wi ^= (sfm & 0x80000000);
        v = __uint_as_float(vi) + pv;
        w = __uint_as_float(wi) + pw;
    }
}

// Half vector, half scales (input transform: pre_scale = suh)
template <bool pre_scale, bool post_scale>
inline __device__
void had_hf_r_128_inner(const half* __restrict__ input_ptr, half* __restrict__ output_ptr,
                        const half* __restrict__ scale, const float r_scale)
{
    int t = threadIdx.x & 31;
    half4 v = ((half4*) input_ptr)[t];
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    ((half4*) output_ptr)[t] = v;
}

// Fused f32-input variant of had_hf_r_128_inner for the x3v/x3_sq input prologue:
// each element is converted with round-to-nearest (__floats2half2_rn matches the
// scalar __float2half used by x3_cast_f32_f16_kernel), then the body is identical
// (fp16 pre-scale, fp32 Hadamard, rn pack-back). Output is bit-identical to
// cast-then-transform, so the separate cast pass can be skipped.
template <bool pre_scale, bool post_scale>
inline __device__
void had_hf_r_128_inner_f32(const float* __restrict__ input_ptr, half* __restrict__ output_ptr,
                        const half* __restrict__ scale, const float r_scale)
{
    int t = threadIdx.x & 31;
    float4 f = ((float4*) input_ptr)[t];
    half4 v;
    v.x = __floats2half2_rn(f.x, f.y);
    v.y = __floats2half2_rn(f.z, f.w);
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    ((half4*) output_ptr)[t] = v;
}
// for completeness with the reference epilogue's c_fp32 branch)
template <bool pre_scale, bool post_scale>
inline __device__
void had_ff_r_128_inner(const float* __restrict__ input_ptr, float* __restrict__ output_ptr,
                        const half* __restrict__ scale, const float r_scale)
{
    int t = threadIdx.x & 31;
    float4 v = ((float4*) input_ptr)[t];
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }
    float v0 = v.x, v1 = v.y, v2 = v.z, v3 = v.w;
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }
    ((float4*) output_ptr)[t] = v;
}

// Float vector, half scales, half output (epilogue transform: post_scale = svh)
template <bool pre_scale, bool post_scale>
inline __device__
void had_fh_r_128_inner(const float* __restrict__ input_ptr, half* __restrict__ output_ptr,
                        const half* __restrict__ scale, const float r_scale)
{
    int t = threadIdx.x & 31;
    float4 v = ((float4*) input_ptr)[t];
    if constexpr (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }
    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;
    half4 o;
    o.x = __floats2half2_rn(v.x, v.y);
    o.y = __floats2half2_rn(v.z, v.w);
    if constexpr (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        o.x = __hmul2(o.x, scales.x);
        o.y = __hmul2(o.y, scales.y);
    }
    ((half4*) output_ptr)[t] = o;
}

// ---------------------------------------------------------------------------------------------------------
// Window extraction (exl3_dq.cuh / kernel ext8w specializations for K = 2, 3)

template <int bits>
__device__ __forceinline__ int wrap_idx(int i)
{
    constexpr int words = bits * 256 / 32;
    return i >= words ? i - words : i;
}

template <int bits>
__device__ __forceinline__ void ext8w
(
    const uint32_t* ptr, int t0,
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3,
    uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7
)
{
    if constexpr (bits == 1)
    {
        uint32_t i1 = t0 >> 5;
        uint32_t i0 = (i1 + 7) & 7;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 24));
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 1);
        BFE16_IMM(w5, b, 2);
        BFE16_IMM(w4, b, 3);
        BFE16_IMM(w3, b, 4);
        BFE16_IMM(w2, b, 5);
        BFE16_IMM(w1, b, 6);
        BFE16_IMM(w0, b, 7);
    }
    else if constexpr (bits == 2)
    {
        uint32_t i1 = t0 >> 4;
        uint32_t i0 = (i1 + 15) & 15;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 8) << 1);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 2);
        BFE16_IMM(w5, b, 4);
        BFE16_IMM(w4, b, 6);
        BFE16_IMM(w3, b, 8);
        BFE16_IMM(w2, b, 10);
        BFE16_IMM(w1, b, 12);
        BFE16_IMM(w0, b, 14);
    }
    else if constexpr (bits == 3)
    {
        int b1 = (t0 + 257) * bits;
        int b0 = b1 - 16;
        int b2 = b1 + bits * 7;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[wrap_idx<bits>(i0)];
        uint32_t b = ptr[wrap_idx<bits>(i2)];
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
        w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
        w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
    }
    else if constexpr (bits == 4)
    {
        uint32_t i1 = t0 >> 3;
        uint32_t i0 = (i1 + 31) & 31;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        uint32_t s;
        FSHF_IMM(s, b, a, 20);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 4);
        BFE16_IMM(w5, b, 8);
        BFE16_IMM(w4, b, 12);
        BFE16_IMM(w3, b, 16);
        w2 = s & 0xffff;
        BFE16_IMM(w1, s, 4);
        BFE16_IMM(w0, s, 8);
    }
}

// ---------------------------------------------------------------------------------------------------------
// sq kernel building blocks (exl3_gemv_int8_kernel.cuh), M = 1, no residual

#define NUM_THREADS 256
#define GEMV_STAGE_D 4
#define SQ_KSPLIT_CAP 64
#define SQ_MINROWS 16
#define SQ_ROWS_MAX 512
#define SQ_COUNTERS_CAP 4096
// qsums holds 4 floats per (slice, row): 4*ksplit*M with ksplit <=
// SQ_KSPLIT_CAP and M <= 8, i.e. 4*64*8 ints. The old *4 (1024 ints)
// overflowed into partials whenever ksplit*M > 256 (e.g. ksplit=34 at
// M=8 gives 272), corrupting slice-0/row-0 outputs of the call.
#define SQ_WS_RESERVED (SQ_COUNTERS_CAP + 4 * SQ_KSPLIT_CAP * 8)

__host__ __device__ constexpr bool gemv_int8_stage_smem(int bits)
{
    return bits == 3 || bits == 5 || bits == 7;
}

__host__ __device__ constexpr int gemv_int8_sq_rows_max(int M, bool residual)
{
    int cap = (80 * 1024) / (32 + 64 * M * (residual ? 2 : 1));
    cap &= ~7;
    return cap < SQ_ROWS_MAX ? cap : SQ_ROWS_MAX;
}

// One k-row of an adjacent block pair, generic K: extract + dp4a for both blocks
template <int bits, int M, bool residual>
__device__ __forceinline__ void gemv_int8_pair_row
(
    const uint32_t* blockA, const uint32_t* blockB,
    const uint32_t* as_kb,
    int slice_stride, int c2, int t0,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    ext8w<bits>(blockA, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ia0[r] = dp4a_us(w0, as01.x, ia0[r]);
        ia0[r] = dp4a_us(w1, as01.y, ia0[r]);
        ia0[r] = dp4a_us(w2, as89.x, ia0[r]);
        ia0[r] = dp4a_us(w3, as89.y, ia0[r]);
        ia1[r] = dp4a_us(w4, as01.x, ia1[r]);
        ia1[r] = dp4a_us(w5, as01.y, ia1[r]);
        ia1[r] = dp4a_us(w6, as89.x, ia1[r]);
        ia1[r] = dp4a_us(w7, as89.y, ia1[r]);
    }

    ext8w<bits>(blockB, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ib0[r] = dp4a_us(w0, as01.x, ib0[r]);
        ib0[r] = dp4a_us(w1, as01.y, ib0[r]);
        ib0[r] = dp4a_us(w2, as89.x, ib0[r]);
        ib0[r] = dp4a_us(w3, as89.y, ib0[r]);
        ib1[r] = dp4a_us(w4, as01.x, ib1[r]);
        ib1[r] = dp4a_us(w5, as01.y, ib1[r]);
        ib1[r] = dp4a_us(w6, as89.x, ib1[r]);
        ib1[r] = dp4a_us(w7, as89.y, ib1[r]);
    }
}

// Shared reduction tail: four lanes share each n; exclusive plain stores for the
// per-slice partials of the sq kernel (atomic = false)
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_pair_tail
(
    int* __restrict__ accs, size_t acc_stride, int nbp, int lane, int size_n,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        #pragma unroll
        for (int o = 1; o < 4; o <<= 1)
        {
            ia0[r] += __shfl_xor_sync(0xffffffff, ia0[r], o);
            ia1[r] += __shfl_xor_sync(0xffffffff, ia1[r], o);
            ib0[r] += __shfl_xor_sync(0xffffffff, ib0[r], o);
            ib1[r] += __shfl_xor_sync(0xffffffff, ib1[r], o);
        }
    }
    if ((lane & 3) == 0)
    {
        int nA = (nbp * 2) * 16 + (lane >> 2);
        int nB = nA + 16;
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic)
            {
                atomicAdd(acc + nA, ia0[r]);
                atomicAdd(acc + nA + 8, ia1[r]);
                atomicAdd(acc + nB, ib0[r]);
                atomicAdd(acc + nB + 8, ib1[r]);
            }
            else
            {
                acc[nA] = ia0[r];
                acc[nA + 8] = ia1[r];
                acc[nB] = ib0[r];
                acc[nB + 8] = ib1[r];
            }
        }
    }
}

// Narrow generic unit (any K): warp per adjacent block pair, pointer-based
// extraction straight from global memory
template <int bits, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_narrow
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n * bits / 2;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (bits * 16);
    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb)
    {
        const uint32_t* blockA = bp + (size_t) kb * row_stride;
        gemv_int8_pair_row<bits, M, residual>(blockA, blockA + 8 * bits,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// Smem-staged generic unit: warp-private cp.async staging (K = 3, 5, 7)
template <int bits, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_smem
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    uint32_t* __restrict__ sh_b,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    constexpr int D = GEMV_STAGE_D;
    constexpr int pairwords = 16 * bits;
    constexpr int chunks = pairwords / 4;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n * bits / 2;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t* sb = sh_b + warp * (D * pairwords);

    auto stage_row = [&] (int kb)
    {
        if (kb < nrows && lane < chunks)
            cp_async(sb + (kb % D) * pairwords + lane * 4, bp + (size_t) kb * row_stride + lane * 4);
        cp_async_fence();
    };
    #pragma unroll
    for (int r = 0; r < D - 1; ++r) stage_row(r);

    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb)
    {
        cp_async_wait<D - 2>();
        __syncwarp();
        stage_row(kb + D - 1);

        const uint32_t* blockA = sb + (kb % D) * pairwords;
        gemv_int8_pair_row<bits, M, residual>(blockA, blockA + 8 * bits,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// Stage one slice for M activation rows: Hadamard from A into sh_ah, slice max
// -> q_s, splats + exact sums per row. Bit-identical in every block.
template <int M, bool residual>
__device__ __forceinline__ void gemv_int8_stage_slice
(
    const half* __restrict__ A,
    int size_m,
    int size_k,
    const half* __restrict__ suh,
    float* __restrict__ qs,
    half* __restrict__ sh_ah,
    uint32_t* __restrict__ sh_as,
    int slice_stride,
    float* __restrict__ sh_red,
    int kb0,
    int nrows,
    const int a_f32
)
{
    int t = threadIdx.x;
    int nel = nrows * 16;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        uint32_t* as = sh_as + r * slice_stride;
        uint32_t* as2 = sh_as + (M + r) * slice_stride;
        float* qsr = qs + 4 * r;
        __syncthreads();
        if (r >= size_m)
        {
            for (int i = t; i < nel; i += NUM_THREADS)
            {
                as[i] = 0;
                if constexpr (residual) as2[i] = 0;
            }
            if (t == 0)
            {
                qsr[0] = 1.0f;
                ((int*) qsr)[1] = 0;
                ((int*) qsr)[2] = 0;
            }
            continue;
        }
        // A holds f32 rows when a_f32 is set (cast folds into the prologue below),
        // fp16 rows otherwise: row stride differs, so resolve the base accordingly
        const half* Ar = a_f32
            ? (const half *) ((const float *) A + (size_t) r * size_k)
            : A + (size_t) r * size_k;
        for (int sp = t >> 5; sp < (nel >> 7); sp += NUM_THREADS >> 5) {
            if (a_f32) {
                had_hf_r_128_inner_f32<true, false>((const float *) Ar + (kb0 << 4) + (sp << 7), sh_ah + (sp << 7), suh + (kb0 << 4) + (sp << 7), 0.088388347648f);
            } else {
                had_hf_r_128_inner<true, false>(Ar + (kb0 << 4) + (sp << 7), sh_ah + (sp << 7), suh + (kb0 << 4) + (sp << 7), 0.088388347648f);
            }
        }
        __syncthreads();

        float mx = 0.0f;
        for (int i = t; i < nel; i += NUM_THREADS) mx = fmaxf(mx, fabsf(__half2float(sh_ah[i])));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        if ((t & 31) == 0) sh_red[t >> 5] = mx;
        __syncthreads();
        if (t < 32)
        {
            float v = t < (NUM_THREADS >> 5) ? sh_red[t] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
            if (t == 0) sh_red[32] = fmaxf(v, 1e-8f) / 127.0f;
        }
        __syncthreads();
        float q_s = sh_red[32];

        float rq = 1.0f / q_s;
        float rq2 = rq * 254.0f;
        int l1 = 0, l2 = 0;
        for (int i = t; i < nel; i += NUM_THREADS)
        {
            float a = __half2float(sh_ah[i]);
            int v = __float2int_rn(a * rq);
            v = max(-127, min(127, v));
            as[i] = ((uint32_t)(uint8_t)(int8_t) v) * 0x01010101u;
            l1 += v;
            if constexpr (residual)
            {
                float rr = a - q_s * (float) v;
                int v2 = __float2int_rn(rr * rq2);
                v2 = max(-127, min(127, v2));
                as2[i] = ((uint32_t)(uint8_t)(int8_t) v2) * 0x01010101u;
                l2 += v2;
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            l1 += __shfl_xor_sync(0xffffffff, l1, o);
            l2 += __shfl_xor_sync(0xffffffff, l2, o);
        }
        if ((t & 31) == 0) { ((int*) sh_red)[t >> 5] = l1; sh_red[16 + (t >> 5)] = __int_as_float(l2); }
        __syncthreads();
        if (t < 32)
        {
            int v1 = t < (NUM_THREADS >> 5) ? ((int*) sh_red)[t] : 0;
            int v2 = t < (NUM_THREADS >> 5) ? __float_as_int(sh_red[16 + t]) : 0;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                v1 += __shfl_xor_sync(0xffffffff, v1, o);
                v2 += __shfl_xor_sync(0xffffffff, v2, o);
            }
            if (t == 0)
            {
                qsr[0] = sh_red[32];
                ((int*) qsr)[1] = v1;
                ((int*) qsr)[2] = v2;
            }
        }
    }
    __syncthreads();
}

// Epilogue for one 256-column group: deterministic fixed-order combine over the
// per-slice partials, warp per (row, 128-span). __ldcg: contributions arrived
// from other blocks with no grid-wide barrier.
template <int M, bool c_fp32, bool residual>
__device__ __forceinline__ void gemv_int8_epilogue_group_sq
(
    const int* __restrict__ partials,
    const float* __restrict__ qsums,
    int pstride,
    int ksplit,
    int size_m,
    void* __restrict__ C,
    const half* __restrict__ svh,
    float* __restrict__ sh_tmp,
    int nb256,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    float k_inv  = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    float aff = 1024.0f * k_inv + k_bias;

    float* tmp = sh_tmp + warp * 128;
    // Cross-block inputs: partials/qsums are written by other CTAs (and the
    // pool lines may sit stale in this SM's L1 from a previous call), so read
    // them volatile (L1-bypass, L2-coherent with the contributors' fences).
    // Plain/__ldcg loads here intermittently fed last-layer's data to M > 1
    // decode batches (solo nt == 1 usually reuses its own lines: coherent).
    volatile const int* vpar = (volatile const int*) partials;
    volatile const float* vqs = (volatile const float*) qsums;
    volatile const int* vqsi = (volatile const int*) qsums;
    // 8 warps cover 2 rows each; stride the (row, span) pairs when M > 4
    // (single-shot warp>=2*M left rows >= 4 unwritten at M = 5..8).
    for (int p = warp; p < 2 * size_m; p += (NUM_THREADS >> 5))
    {
        int row = p >> 1;
        int base = nb256 * 256 + (p & 1) * 128;
        float acc[4] = {};
        float corr = 0.0f;
        for (int sl = 0; sl < ksplit; ++sl)
        {
            int idx = sl * M + row;
            float q_s = vqs[4 * idx];
            float suma = q_s * (float) vqsi[4 * idx + 1];
            volatile const int* q = vpar + (size_t) idx * pstride + base;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[i] += q_s * (float) q[lane * 4 + i];
            if constexpr (residual)
            {
                float q2_s = q_s * (1.0f / 254.0f);
                suma += q2_s * (float) vqsi[4 * idx + 2];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    acc[i] += q2_s * (float) q[size_n + lane * 4 + i];
            }
            corr += aff * suma;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            tmp[lane * 4 + i] = k_inv * acc[i] + corr;
        __syncwarp();
        if constexpr (c_fp32)
            had_ff_r_128_inner<false, true>(tmp, ((float*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
        else
            had_fh_r_128_inner<false, true>(tmp, ((half*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
    }
}

// ---------------------------------------------------------------------------------------------------------
// The sq kernel, m == 1, mode 2 (plain int8), regular launch

template <int bits, int M, bool c_fp32, bool residual>
__global__ __launch_bounds__(NUM_THREADS)
void exl3_gemv_int8_sq_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ suh,
    half* __restrict__ A_had,
    const half* __restrict__ svh,
    const int a_f32
)
{
    (void) A_had;  // unused in the sq path
    extern __shared__ uint32_t shmem[];

    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int r = (rows_total * nb256_total + (int) gridDim.x - 1) / (int) gridDim.x;
    int rows_per = (r > 2 * r ? r : (2 * r < 32 ? 2 * r : 32));
    rows_per = (rows_per + 7) & ~7;
    rows_per = rows_per < SQ_MINROWS ? SQ_MINROWS : rows_per;
    rows_per = rows_per > gemv_int8_sq_rows_max(M, residual) ? gemv_int8_sq_rows_max(M, residual) : rows_per;
    rows_per = rows_per > ((rows_total + 7) & ~7) ? ((rows_total + 7) & ~7) : rows_per;
    if constexpr (M > 1) {
        constexpr int smem_budget = 49152;
        constexpr int stage = bits == 3 ? 8 * GEMV_STAGE_D * 16 * bits * 4 : 0;
        int cap = (smem_budget - 1024 * M - stage) / (32 + 64 * M);
        cap &= ~7;
        if (cap < SQ_MINROWS) cap = SQ_MINROWS;
        rows_per = rows_per > cap ? cap : rows_per;
    }
    int ksplit = (rows_total + rows_per - 1) / rows_per;
    int units = nb256_total * ksplit;
    int slice_stride = rows_per * 16;
    int pstride = size_n * (residual ? 2 : 1);

    int* counters = locks;
    float* qsums = (float*) (locks + SQ_COUNTERS_CAP);
    int* partials = locks + SQ_WS_RESERVED;

    half* sh_ah = (half*) shmem;
    uint32_t* sh_as = shmem + rows_per * 8;
    uint32_t* sh_b = sh_as + slice_stride * M * (residual ? 2 : 1);
    float* sh_tmp = (float*) (sh_b + (gemv_int8_stage_smem(bits) ? 8 * GEMV_STAGE_D * 16 * bits : 0));
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    int t = threadIdx.x;
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
    {
        int slice = unit / nb256_total;
        int nb256 = unit % nb256_total;
        int kb0 = slice * rows_per;
        int nrows = rows_per < (rows_total - kb0) ? rows_per : (rows_total - kb0);
        if (slice != prev_slice)
        {
            gemv_int8_stage_slice<M, residual>(A, size_m, size_k, suh, qsums + 4 * slice * M,
                                               sh_ah, sh_as, slice_stride, sh_red, kb0, nrows, a_f32);
            prev_slice = slice;
        }
        int* pacc = partials + (size_t) slice * M * pstride;
        if constexpr (gemv_int8_stage_smem(bits))
            gemv_int8_unit_smem<bits, M, residual, false>(B, pacc, pstride, sh_as, slice_stride, sh_b, nb256, kb0, nrows, size_n);
        else
            gemv_int8_unit_narrow<bits, M, residual, false>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);

        // Completion counter: the ksplit-th contributor runs the epilogue
        __threadfence();
        __syncthreads();
        if (t == 0) sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
        __syncthreads();
        if (sh_last)
        {
            gemv_int8_epilogue_group_sq<M, c_fp32, residual>(partials, qsums, pstride, ksplit, size_m,
                                                             C, svh, sh_tmp, nb256, size_n);
            if (t == 0) counters[nb256] = 0;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Host side (mirror of exl3_gemv_int8.cu, sq path only)

static int g_num_sms = 82;


// --- adapted host side: plan cache + pool workspace (exl3_gemv_int8.cu) ---

template <int bits, int M, bool c_fp32>
static cudaFunction_t sq_kernel_fn()
{
    return (cudaFunction_t) exl3_gemv_int8_sq_kernel<bits, M, c_fp32, false>;
}

static size_t smem_for(int bits, int rows_per, int M)
{
    constexpr bool residual = false;
    size_t stage = gemv_int8_stage_smem(bits) ? (size_t) 8 * GEMV_STAGE_D * 16 * bits * 4 : 0;
    return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * M * (residual ? 2 : 1)
           + stage + (size_t) 2 * M * 128 * 4;
}

struct SqPlan
{
    cudaFunction_t fn;
    int grid, ksplit, rows_per;
    size_t smem;
};

// deterministic launch plan per matrix shape; occupancy query once
// M = batch rows (nt). M == 1 is the decode path; 2..SQ_M_MAX share one
// launch with the trellis read once. Cache key includes M (rows_max/smem
// are M-dependent); M > 1 clamps rows_per to the 48 KB smem budget.
#define SQ_M_MAX 8
#define SQ_SMEM_BUDGET 49152

static const SqPlan & plan_sq(int bits, int size_k, int size_n, int M)
{
    static std::map<std::tuple<int, int, int, int>, SqPlan> cache;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_tuple(bits, size_k, size_n, M);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    constexpr bool residual = false;
    int rows_max = gemv_int8_sq_rows_max(M, residual);
    int rows_total = size_k / 16;
    int nb256 = size_n / 256;

    auto decomp = [&] (int grid_, int & ksplit, int & rows_per)
    {
        int r = (rows_total * nb256 + grid_ - 1) / grid_;
        rows_per = (r > 2 * r ? r : (2 * r < 32 ? 2 * r : 32));
        rows_per = (rows_per + 7) & ~7;
        rows_per = rows_per < SQ_MINROWS ? SQ_MINROWS : rows_per;
        rows_per = rows_per > rows_max ? rows_max : rows_per;
        rows_per = rows_per > ((rows_total + 7) & ~7) ? ((rows_total + 7) & ~7) : rows_per;
        if (M > 1) {
            // clamp to the 48 KB smem budget (M == 1 keeps legacy behavior)
            size_t stage = gemv_int8_stage_smem(bits) ? (size_t) 8 * GEMV_STAGE_D * 16 * bits * 4 : 0;
            int cap = (int)((SQ_SMEM_BUDGET - 1024 * M - stage) / (32 + 64 * M));
            cap &= ~7;
            if (cap < SQ_MINROWS) cap = SQ_MINROWS;
            rows_per = rows_per > cap ? cap : rows_per;
        }
        ksplit = (rows_total + rows_per - 1) / rows_per;
    };

    SqPlan plan;
    if (bits == 1) {
        switch (M) {
            case 1: plan.fn = sq_kernel_fn<1, 1, true>(); break;
            case 2: plan.fn = sq_kernel_fn<1, 2, true>(); break;
            case 3: plan.fn = sq_kernel_fn<1, 3, true>(); break;
            case 4: plan.fn = sq_kernel_fn<1, 4, true>(); break;
            case 5: plan.fn = sq_kernel_fn<1, 5, true>(); break;
            case 6: plan.fn = sq_kernel_fn<1, 6, true>(); break;
            case 7: plan.fn = sq_kernel_fn<1, 7, true>(); break;
            default: plan.fn = sq_kernel_fn<1, 8, true>(); break;
        }
    } else if (bits == 4) {
        switch (M) {
            case 1: plan.fn = sq_kernel_fn<4, 1, true>(); break;
            case 2: plan.fn = sq_kernel_fn<4, 2, true>(); break;
            case 3: plan.fn = sq_kernel_fn<4, 3, true>(); break;
            case 4: plan.fn = sq_kernel_fn<4, 4, true>(); break;
            case 5: plan.fn = sq_kernel_fn<4, 5, true>(); break;
            case 6: plan.fn = sq_kernel_fn<4, 6, true>(); break;
            case 7: plan.fn = sq_kernel_fn<4, 7, true>(); break;
            default: plan.fn = sq_kernel_fn<4, 8, true>(); break;
        }
    } else if (bits == 2) {
        switch (M) {
            case 1: plan.fn = sq_kernel_fn<2, 1, true>(); break;
            case 2: plan.fn = sq_kernel_fn<2, 2, true>(); break;
            case 3: plan.fn = sq_kernel_fn<2, 3, true>(); break;
            case 4: plan.fn = sq_kernel_fn<2, 4, true>(); break;
            case 5: plan.fn = sq_kernel_fn<2, 5, true>(); break;
            case 6: plan.fn = sq_kernel_fn<2, 6, true>(); break;
            case 7: plan.fn = sq_kernel_fn<2, 7, true>(); break;
            default: plan.fn = sq_kernel_fn<2, 8, true>(); break;
        }
    } else {
        switch (M) {
            case 1: plan.fn = sq_kernel_fn<3, 1, true>(); break;
            case 2: plan.fn = sq_kernel_fn<3, 2, true>(); break;
            case 3: plan.fn = sq_kernel_fn<3, 3, true>(); break;
            case 4: plan.fn = sq_kernel_fn<3, 4, true>(); break;
            case 5: plan.fn = sq_kernel_fn<3, 5, true>(); break;
            case 6: plan.fn = sq_kernel_fn<3, 6, true>(); break;
            case 7: plan.fn = sq_kernel_fn<3, 7, true>(); break;
            default: plan.fn = sq_kernel_fn<3, 8, true>(); break;
        }
    }
    decomp(6 * g_num_sms, plan.ksplit, plan.rows_per);
    size_t smem_guess = smem_for(bits, plan.rows_per, M);
    int maxb;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, plan.fn, NUM_THREADS, smem_guess));
    plan.grid = (maxb * g_num_sms) < 1024 ? (maxb * g_num_sms) : 1024;
    decomp(plan.grid, plan.ksplit, plan.rows_per);
    // debug: clamp the launch grid (GGML_PAW_SQ_GRID_MAX=1 serializes all
    // units through one CTA: no cross-block partials/counters traffic)
    {
        const char * e = getenv("GGML_PAW_SQ_GRID_MAX");
        if (e && atoi(e) > 0 && plan.grid > atoi(e)) {
            plan.grid = atoi(e);
            decomp(plan.grid, plan.ksplit, plan.rows_per);
        }
    }
    plan.smem = smem_for(bits, plan.rows_per, M);
    GGML_ASSERT(plan.ksplit <= SQ_KSPLIT_CAP);
    GGML_ASSERT(nb256 <= SQ_COUNTERS_CAP);

    auto res = cache.emplace(key, plan);
    return res.first->second;
}

static void launch_sq(const SqPlan & plan, int bits, int M,
                      const float * A, const uint16_t * B, float * C,
                      int size_k, int size_n,
                      const half * suh, const half * svh,
                      int * locks, cudaStream_t stream)
{
    int g_size_m = M;                      // fused batch rows (was hardcoded 1)
    static half * const g_null_a_had = nullptr;  // residual-only operand
    const int a_f32 = 1;                   // A is f32: cast folds into the input prologue
    void * args[] =
    {
        (void *) &A, (void *) &B, (void *) &C,
        (void *) &g_size_m, (void *) &size_k, (void *) &size_n,
        (void *) &locks, (void *) &suh, (void *) &g_null_a_had, (void *) &svh,
        (void *) &a_f32
    };
    CUDA_CHECK(cudaLaunchKernel(plan.fn, dim3(plan.grid), dim3(NUM_THREADS), args, plan.smem, stream));
}

__global__ static void x3_cast_f32_f16_kernel(half * dst, const float * src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

static int x3_num_sms()
{
    static int n_sms = [] ()
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        return prop.multiProcessorCount;
    }();
    return n_sms;
}

// ---------------------------------------------------------------------------------------------------------
// EXL3 reconstruct_had_slice device port (x3 prefill path: trellis -> ORIGINAL-basis fp16 W).
// Verbatim logic from exllamav3_ext (ref 63b32f0)
//   quant/reconstruct.cu (reconstruct_had_kernel), quant/exl3_dq.cuh (dq8, dq8_aligned_2bits),
//   quant/codebook.cuh (decode fns, unions), quant/hadamard_inner.cuh (shuffle_had_h2x32).
// Only torch host wrappers are omitted; K is restricted to {2,3} and cb to mul1 (2).
// Already-defined paw_x3 names are reused, not redefined: fshift, FSHF_IMM, BFE16_IMM, half4.

union half2_uint32 { half2 as_half2; uint32_t as_uint32; };
union half_uint16 { half as_half; uint16_t as_uint16; __device__ half_uint16(uint16_t x) : as_uint16(x) {} };

struct FragB { half2 v[2]; __device__ half2 & operator[](int i) { return v[i]; } };

// Decode two mul1 codebook entries from precomputed products x0 = idx0 * 0x83DCD12D,
// x1 = idx1 * 0x83DCD12D
__device__ inline half2 decode_mul1_product_2(uint32_t x0, uint32_t x1)
{
    const uint32_t acc = 0x6400u;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));  //  0.00677 = 1/147.7
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));  // -10.39
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1)
{
    static_assert(cb == 2, "x3 reconstruct supports mul1 only");
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    return decode_mul1_product_2(x0, x1);
}

template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int b1 = (t_offset + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (align == 1)
    {
        w7 = fshift(b, a, s2);
        w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 2)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = w1 >> bits;
    }
    if constexpr (align == 4)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    if constexpr (align == 8)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = w4 >> bits;
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    half2 d0d1 = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    half2 d2d3 = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    half2 d4d5 = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    half2 d6d7 = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
    frag0[0] = d0d1;
    frag0[1] = d2d3;
    frag1[0] = d4d5;
    frag1[1] = d6d7;
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;
    i0 = (i1 + 15) & 15;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

// K=4 and K=1 use their own aligned extractors, NOT dq8<bits,cb,4>: at 4 bits
// the 8 values span 32 stream bits, which straddles three 32-bit words at the
// offsets the generic path loads only two of. Ported verbatim from
// exllamav3 quant/exl3_dq.cuh (dq8_aligned_4bits, dq8_aligned_1bit).
template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;
    i0 = (i1 + 31) & 31;
    a = ptr[i0];
    b = ptr[i1];
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_1bit(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 5;
    i0 = (i1 + 7) & 7;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 24));
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 1);
    BFE16_IMM(w5, b, 2);
    BFE16_IMM(w4, b, 3);
    BFE16_IMM(w3, b, 4);
    BFE16_IMM(w2, b, 5);
    BFE16_IMM(w1, b, 6);
    BFE16_IMM(w0, b, 7);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1)
{
    static_assert((bits == 1 || bits == 2 || bits == 3 || bits == 4) && cb == 2,
                  "x3 reconstruct supports K=1,2,3,4 mul1 only");
    if constexpr (bits == 1)
    {
        dq8_aligned_1bit<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 2)
    {
        dq8_aligned_2bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 4)
    {
        dq8_aligned_4bits<cb>(ptr, idx, frag0, frag1);
    }
    else
    {
        dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
    }
}

__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        half2 pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}

// Fused reconstruct + both-side Hadamard: emits W = diag(suh) . H128 . W_hat . H128 . diag(svh)
// (per 128x128 tile, 1/sqrt(128) per side), i.e. ORIGINAL-basis weights. See reconstruct.cu.
#define RH_THREADS 256

template <int K, int cb>
__global__ __launch_bounds__(RH_THREADS)
void reconstruct_had_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;
    constexpr float r_scale = 0.08838834764831845f;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&] (int R, int q, int p)
    {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS)
    {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const uint16_t* gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj)
    {
        int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        register FragB frag[2];
        dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq)
    {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        half2 a[4], b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v = *((const half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
        #pragma unroll
        for (int x = 0; x < 2; ++x)
        {
            half2* v = x == 0 ? a : b;
            half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    const half4 sv4 = ((const half4*) svh)[nb * 32 + lane_id];
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr)
    {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        half2 v01 = stile[base];
        half2 v23 = stile[base + 1];
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        half2 su2 = __half2half2(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        *((half4*) (g_unpacked + (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4)) = v;
    }
}

// Host launcher: W is (n, m) row-major fp16 (ORIGINAL basis); T is the
// (n//16, m//16, 16*bits) trellis; n and m must be multiples of 128.
static void x3r_reconstruct_ws(half * W, const uint16_t * T,
                               const half * suh, const half * svh,
                               int n, int m, int bits, cudaStream_t stream)
{
    GGML_ASSERT(n % 128 == 0 && m % 128 == 0);
    dim3 grid((m + 127) / 128, (n + 127) / 128);
    if (bits == 1) {
        reconstruct_had_kernel<1, 2><<<grid, RH_THREADS, 0, stream>>>(W, T, suh, svh, m / 16, 0);
    } else if (bits == 2) {
        reconstruct_had_kernel<2, 2><<<grid, RH_THREADS, 0, stream>>>(W, T, suh, svh, m / 16, 0);
    } else if (bits == 4) {
        reconstruct_had_kernel<4, 2><<<grid, RH_THREADS, 0, stream>>>(W, T, suh, svh, m / 16, 0);
    } else {
        reconstruct_had_kernel<3, 2><<<grid, RH_THREADS, 0, stream>>>(W, T, suh, svh, m / 16, 0);
    }
}


// ---------------------------------------------------------------------------------------------------------
// x3 tensor-core trellis GEMM (port of exllamav3 exl3_gemm_kernel / exl3_gemm_inner)
//
// Why this exists: the sq int8 GEMV above is only used by exllamav3 for m <= 2
// (`exl3_gemv_int8.cu: if (size_m > 2) return false;`). llama-paw widened it to
// M = 1..8 and paid for it: measured on RTX 3090 with nsys, the trellis matmul
// total for one B3.5 forward goes 17.95 ms (nt=1) -> 53.98 ms (nt=8), 3.01x,
// while exllamav3 on the same GPU goes 16.70 -> 21.29 ms, 1.27x. The two engines
// agree to within 0.6% at nt=2 (same kernel) and diverge from nt=4 (different
// kernel). This is that kernel: activations stay in mma fragments and the
// trellis is decoded straight into B fragments, so extra tokens ride the idle
// tensor pipe instead of re-running a dp4a pass per row.
// ---------------------------------------------------------------------------------------------------------

#define X3G_MIN(a, b) ((a) < (b) ? (a) : (b))
#define X3G_MAX(a, b) ((a) > (b) ? (a) : (b))
#define X3G_CEIL_DIVIDE(a, b) (((a) + (b) - 1) / (b))
#define X3G_BASE_THREADS 256
#define X3G_SMEM_MAX (90 * 1024)   // max opt-in dynamic shared memory, compute capability 8.6

// GA10x runs HMMA with fp32 accumulation at half rate; accumulate in fp16 and fold into the fp32
// accumulators once per k-slice (exllamav3 measures ~14% at bsz 1 on RTX 3090, error ~1% of output
// RMS at k=4096, well under quantization noise). Only enabled for sm_86, as upstream.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
    #define X3G_H_ACC 1
#else
    #define X3G_H_ACC 0
#endif

struct X3G_FragA { half2 v[4]; __device__ half2 & operator[](int i) { return v[i]; } };
struct X3G_FragC { float  v[4]; __device__ float  & operator[](int i) { return v[i]; } };
struct X3G_FragCh{ half2 v[2]; __device__ half2 & operator[](int i) { return v[i]; } };

// FP16 @ FP16 + FP32 -> FP32
__device__ inline void x3g_mma_m16n8k16(const X3G_FragA & frag_a, const FragB & frag_b, X3G_FragC & frag_c)
{
    const uint32_t * a = reinterpret_cast<const uint32_t *>(&frag_a);
    const uint32_t * b = reinterpret_cast<const uint32_t *>(&frag_b);
    float * c = reinterpret_cast<float *>(&frag_c);
    const float * d = reinterpret_cast<const float *>(&frag_c);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

// FP16 @ FP16 + FP16 -> FP16
__device__ inline void x3g_mma_m16n8k16(const X3G_FragA & frag_a, const FragB & frag_b, X3G_FragCh & frag_c)
{
    const uint32_t * a = reinterpret_cast<const uint32_t *>(&frag_a);
    const uint32_t * b = reinterpret_cast<const uint32_t *>(&frag_b);
    uint32_t * c = reinterpret_cast<uint32_t *>(&frag_c);
    const uint32_t * d = reinterpret_cast<const uint32_t *>(&frag_c);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(c[0]), "=r"(c[1])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "r"(d[0]), "r"(d[1]));
}

__device__ inline void x3g_barrier_acquire(int * lock, int stage)
{
    if (threadIdx.x == 0) {
        volatile int state = -1;
        do {
            asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        } while (state != stage);
    }
    __syncthreads();
}

__device__ inline void x3g_barrier_release(int * lock, int val, bool reset)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) { *lock = 0; return; }
        asm volatile ("fence.acq_rel.gpu;\n");
        asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
}

// Load 16x16 matrix fragment from shared memory, directly in tensor core layout
__device__ inline void x3g_ldsm4(X3G_FragA & frag_a, const void * smem_ptr)
{
    uint32_t * a = reinterpret_cast<uint32_t *>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem));
}

#define X3G_T_ARGS \
    const int bits, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define X3G_ARGS \
    const half * __restrict__ A, \
    const uint16_t * __restrict__ B, \
    void * __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int * __restrict__ locks, \
    const half * __restrict__ suh, \
    half * __restrict__ A_had, \
    const half * __restrict__ svh, \
    const int a_f32

typedef void (*fp_x3g_kernel) (X3G_ARGS);

template <X3G_T_ARGS, bool shmem_out_had>
inline __device__ void x3g_gemm_inner
(
    const half * __restrict__ A,
    const uint16_t * __restrict__ B,
    void * __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int * __restrict__ locks,
    const half * post_scale
)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (X3G_BASE_THREADS / 32);

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                        // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;  // in uint16s
    const int sh_c_size = X3G_MAX(4 * X3G_BASE_THREADS * FRAGS_N_PER_WARP,
                                  shmem_out_had ? TILESIZE_N * TILESIZE_M : 0);

    // XOR-swizzle constants for bank-conflict-free A fragment loads
    const int A_COLS = TILESIZE_K / 8;
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    static_assert(X3G_BASE_THREADS == 256);
    static_assert(TILESIZE_M == 16, "Invalid kernel params");
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert(X3G_SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
                  "Invalid kernel params (insufficient shared memory for shape)");

    extern __shared__ half shared_x3g[];
    half * sh_a = shared_x3g;
    uint16_t * sh_b = (uint16_t *) (sh_a + SH_STAGES * sh_a_stage_size);
    float * sh_c = (float *) (sh_b + sh_b_stage_size * SH_STAGES);

    int t = threadIdx.x % X3G_BASE_THREADS;
    int sub_k = threadIdx.x / X3G_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    int num_slices = gridDim.x;
    int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    const int slice_m = 0;

    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half * gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half * sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = X3G_CEIL_DIVIDE(sh0_a_stride_m / 8, X3G_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i) {
        int k = (i * X3G_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * X3G_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t * gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t * sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = X3G_CEIL_DIVIDE(sh0_b_stride_k / 8, X3G_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i) {
        int n = (i * X3G_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * X3G_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * X3G_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;
        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice0_k >= tiles_k) {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        } else {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    int slice1_k = slice0_k;
    int slice1_iters = slice0_iters;

    half * sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t * sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;
        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice1_k >= tiles_k) slice1_k = 0;
    };

    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n;

    half * gl_c_ptr_16 = ((half *) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float * gl_c_ptr_32 = ((float *) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    X3G_FragA frag_a[FRAG_STAGES];
    FragB     frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    X3G_FragC frag_c[FRAGS_N_PER_WARP];
    #if X3G_H_ACC
        X3G_FragCh frag_c_h[FRAGS_N_PER_WARP];
    #endif

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;
        if (slice2_k >= tiles_k) {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32) gl_c_ptr_32 += gl_c_stride_n;
            else                  gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    auto async_load_gl = [&] ()
    {
        if (sub_k) { cp_async_fence(); return; }
        if (slice0_iters) {
            {
                const int4 * gl = (const int4 *) gl_a_ptr;
                int4 * sh = (int4 *) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
            }
            {
                const int4 * gl = (const int4 *) gl_b_ptr;
                int4 * sh = (int4 *) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                    if (pred_b_gl[i]) cp_async(sh + X3G_BASE_THREADS * i + t, gl + load_b_gl[i]);
            }
            advance0();
        }
        cp_async_fence();
    };

    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m) {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                x3g_ldsm4(frag_a[buf], (int4 *) sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2) {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t * shb = (const uint32_t *) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);
            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }
        __syncthreads();
        advance1();
    };

    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) frag_c[n] = {};
        #if X3G_H_ACC
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n) frag_c_h[n] = {};
        #endif
    };

    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i)
        {
            if (sub_k == i) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[n][j];
            }
            __syncthreads();
        };
        auto add = [&] (int i)
        {
            if (sub_k == i) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[n][j] += *sh_red++;
            }
        };
        auto store_small = [&] (int i)
        {
            if (sub_k == i && lane_id / 4 < size_m) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                    *sh_red++ = frag_c[n][0];
                    *sh_red++ = frag_c[n][1];
                }
            }
            __syncthreads();
        };
        auto add_small = [&] (int i)
        {
            if (sub_k == i && lane_id / 4 < size_m) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                    frag_c[n][0] += *sh_red++;
                    frag_c[n][1] += *sh_red++;
                }
            }
        };

        if (size_m <= 8) {
            if constexpr (TILEBLOCKS_K == 2) { store_small(1); add_small(0); }
            if constexpr (TILEBLOCKS_K == 3) { store_small(1); add_small(0); store_small(2); add_small(0); }
            if constexpr (TILEBLOCKS_K == 4) { store_small(3); add_small(2); store_small(1); add_small(0); store_small(2); add_small(0); }
        } else {
            if constexpr (TILEBLOCKS_K == 2) { store(1); add(0); }
            if constexpr (TILEBLOCKS_K == 3) { store(1); add(0); store(2); add(0); }
            if constexpr (TILEBLOCKS_K == 4) { store(3); add(2); store(1); add(0); store(2); add(0); }
        }
    };

    auto write_sum_tile_sh = [&] ()
    {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        const int r0 = lane_id / 4;
        const int r1 = r0 + 8;
        if (r0 < size_m) {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                float * c_ptr = ((float *) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][0];
                *c_ptr++ = frag_c[n][1];
            }
        }
        if (r1 < size_m) {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                float * c_ptr = ((float *) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][2];
                *c_ptr++ = frag_c[n][3];
            }
        }
    };

    auto output_had_sh_gl = [&] ()
    {
        int sh_warp = warp_id;
        constexpr int active_warps = X3G_BASE_THREADS / 32;
        for (;; sh_warp += active_warps) {
            int col = sh_warp % (TILESIZE_N / 128);
            int row = sh_warp / (TILESIZE_N / 128);
            if (row >= size_m) break;
            const float * had_in = sh_c + row * TILESIZE_N + col * 128;
            const half * post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;
            if constexpr (c_fp32) {
                float * had_out = gl_c_ptr_32 + row * size_n + col * 128;
                had_ff_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            } else {
                half * had_out = gl_c_ptr_16 + row * size_n + col * 128;
                had_fh_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
        }
    };

    auto read_sum_gl = [&] ()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m) {
                if constexpr (c_fp32) {
                    float * c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    frag_c[n][0] += *c_ptr++;
                    frag_c[n][1] += *c_ptr++;
                } else {
                    half2 * c_ptr = (half2 *) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[n][0] += interm.x;
                    frag_c[n][1] += interm.y;
                }
            }
            if (r1 < size_m) {
                if constexpr (c_fp32) {
                    float * c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    frag_c[n][2] += *c_ptr++;
                    frag_c[n][3] += *c_ptr++;
                } else {
                    half2 * c_ptr = (half2 *) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[n][2] += interm.x;
                    frag_c[n][3] += interm.y;
                }
            }
        }
    };

    auto write_sum_gl = [&] ()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m) {
                if constexpr (c_fp32) {
                    float * c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][0];
                    *c_ptr++ = frag_c[n][1];
                } else {
                    half2 * c_ptr = (half2 *) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    *c_ptr = __floats2half2_rn(frag_c[n][0], frag_c[n][1]);
                }
            }
            if (r1 < size_m) {
                if constexpr (c_fp32) {
                    float * c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][2];
                    *c_ptr++ = frag_c[n][3];
                } else {
                    half2 * c_ptr = (half2 *) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    *c_ptr = __floats2half2_rn(frag_c[n][2], frag_c[n][3]);
                }
            }
        }
    };

    auto reduce = [&] ()
    {
        #if X3G_H_ACC
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                float2 f0 = __half22float2(frag_c_h[n][0]);
                float2 f1 = __half22float2(frag_c_h[n][1]);
                frag_c[n][0] += f0.x; frag_c[n][1] += f0.y;
                frag_c[n][2] += f1.x; frag_c[n][3] += f1.y;
            }
        #endif

        threadblock_reduce();

        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int * lock = &locks[slice_m * blocks_n + slice2_n];

        x3g_barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        if (!sub_k && !first) read_sum_gl();
        if (!sub_k && !last)  write_sum_gl();
        if (!sub_k && last) {
            if constexpr (shmem_out_had) write_sum_tile_sh();
            else                         write_sum_gl();
        }
        if constexpr (shmem_out_had) {
            if (last) __syncthreads();
            if (!sub_k && last) output_had_sh_gl();
        }

        x3g_barrier_release(lock, lock_d, last);
        clear_frag_c();
    };

    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            #if X3G_H_ACC
                x3g_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c_h[n]);
            #else
                x3g_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[n]);
            #endif
        }
    };

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i) async_load_gl();
    wait_stage();

    clear_frag_c();
    if constexpr (FRAG_STAGES > 1) load_frags(0);

    #define X3G_FSTAGE_OLD(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break;

    #define X3G_FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load);

    if constexpr (FRAG_STAGES == 1) { while (true) { X3G_FSTAGE_OLD(0, 0); } }
    if constexpr (FRAG_STAGES == 2) { while (true) { X3G_FSTAGE(1, 0); X3G_FSTAGE(0, 1); } }
    if constexpr (FRAG_STAGES == 3) { while (true) { X3G_FSTAGE(1, 0); X3G_FSTAGE(2, 1); X3G_FSTAGE(0, 2); } }
    if constexpr (FRAG_STAGES == 4) { while (true) { X3G_FSTAGE(1, 0); X3G_FSTAGE(2, 1); X3G_FSTAGE(3, 2); X3G_FSTAGE(0, 3); } }
    if constexpr (FRAG_STAGES == 5) { while (true) { X3G_FSTAGE(1, 0); X3G_FSTAGE(2, 1); X3G_FSTAGE(3, 2); X3G_FSTAGE(4, 3); X3G_FSTAGE(0, 4); } }

    #undef X3G_FSTAGE_OLD
    #undef X3G_FSTAGE
}

template <X3G_T_ARGS>
__global__ __launch_bounds__(X3G_BASE_THREADS * TILESIZE_K / 16)
void x3g_gemm_kernel(X3G_ARGS)
{
    namespace cg_ = cooperative_groups;
    auto grid = cg_::this_grid();

    // suh (input Hadamard scale) folded into A once, cooperatively, before the tiles are read
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for (; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>(A + this_warp * 128, A_had + this_warp * 128,
                                            suh + (this_warp * 128) % size_k, 0.088388347648f);
        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half * A_ = A;
    void * C_ = C;

    while (size_m_ > 0)
    {
        x3g_gemm_inner<bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
            (A_, B, C_, X3G_MIN(size_m_, 16), size_k, size_n, locks, svh);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void *) (((float *) C_) + 16 * size_n);
        else                  C_ = (void *) (((half  *) C_) + 16 * size_n);
        size_m_ -= 16;

        if (size_m_ > 0 || svh) grid.sync();
    }
}


// ---------------------------------------------------------------------------------------------------------
// Small-m GEMV path, ported from exllamav3 quant/exl3_gemv_kernel.cuh + quant/exl3_gemv.cu.
// QTIP-style structure on the unmodified EXL3 trellis format:
//
//   * warps split k and never synchronize in the main loop -- no block-wide pipeline barriers;
//     B streams straight to registers with ld.global.cs behind a register prefetch ring
//   * the two-word bit windows are resolved in-warp by lane shuffles, so the kernel needs NO
//     dynamic shared memory (only sh_red, ~16 KB static).  That is the whole point: x3g_gemm_kernel
//     asks for 90 KB dynamic and gets 1 block/SM on sm_86; this gets 4-6.
//   * one m16n8k16 MMA pair per 16x16 tile with fp16 accumulation, folded to fp32 on a cadence
//
// Same argument list as x3g_gemm_kernel, so the launch path is interchangeable.
// CFG 0 ("narrow", 512 threads, 2 n-tiles/warp, 16 k-splits) wins at attention-projection sizes;
// CFG 1 ("wide", 256 threads, 4 n-tiles/warp, 8 k-splits) wins at large-n FFN sizes.
// MMODE 0 is the m == 1 fast path, MMODE 1 covers 2 <= m <= 8.

#define X3V_MAX_M 8

// mma.m16n8k16 with the A operand supplied as two FragB halves, fp16 accumulate
__device__ __forceinline__ void x3v_mma_ab_h(const FragB & a01, const FragB & a23,
                                             const FragB & b, X3G_FragCh & c)
{
    const uint32_t * a0 = reinterpret_cast<const uint32_t *>(&a01);
    const uint32_t * a1 = reinterpret_cast<const uint32_t *>(&a23);
    const uint32_t * bb = reinterpret_cast<const uint32_t *>(&b);
    uint32_t * cc = reinterpret_cast<uint32_t *>(&c);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
           "r"(bb[0]), "r"(bb[1])
    );
}

template <int cb>
__device__ __forceinline__ void x3v_decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
                                            uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7,
                                            FragB & f0, FragB & f1)
{
    // decode_3inst_2<2> is bit-identical to exllamav3's decode_pair_cb2_dp4a_
    f0[0] = decode_3inst_2<cb>(w0, w1);
    f0[1] = decode_3inst_2<cb>(w2, w3);
    f1[0] = decode_3inst_2<cb>(w4, w5);
    f1[1] = decode_3inst_2<cb>(w6, w7);
}

// Register forms of the dq8 window extraction: same word order as ext8w<bits>, but the two
// source words arrive in registers (shuffled from a neighbour lane) rather than through smem.
template <int cb>
__device__ __forceinline__ void x3v_dq8_regs_4bits(uint32_t a, uint32_t b, FragB & f0, FragB & f1)
{
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    x3v_decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

template <int cb>
__device__ __forceinline__ void x3v_dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset,
                                                   FragB & f0, FragB & f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    x3v_decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

template <int cb>
__device__ __forceinline__ void x3v_dq8_regs_3bits(uint32_t a, uint32_t b, int s2,
                                                   FragB & f0, FragB & f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    x3v_decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                    w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

template <int bits, bool c_fp32, int cb, int MMODE, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256)
void x3v_gemv_kernel(X3G_ARGS)
{
    static_assert(bits == 2 || bits == 3 || bits == 4, "x3v_gemv_kernel supports 2, 3 and 4 bpw");
    namespace cg_ = cooperative_groups;

    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp
    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth
    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : X3V_MAX_M;
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS  = 8 * bits;                       // uint32 per 16x16 tile
    constexpr int LOADS   = bits == 2 ? WNT / 2 : WNT;      // warp loads per k-slice
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;            // uint32 per load
    static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

    auto grid = cg_::this_grid();

    // suh (input Hadamard scale) folded into A once, exactly as x3g_gemm_kernel does
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for (; this_warp < total_warps; this_warp += warps_grid) {
            if (a_f32) {
                had_hf_r_128_inner_f32<true, false>((const float *) A + this_warp * 128, A_had + this_warp * 128,
                                                suh + (this_warp * 128) % size_k, 0.088388347648f);
            } else {
                had_hf_r_128_inner<true, false>(A + this_warp * 128, A_had + this_warp * 128,
                                                suh + (this_warp * 128) % size_k, 0.088388347648f);
            }
        }
        grid.sync();
        A = A_had;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles     = size_n / 16;
    const int kslices    = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = X3G_CEIL_DIVIDE(kslices, WK);
    const int ks0   = warp * chunk;
    const int myn   = max(0, min(chunk, kslices - ks0));

    const uint32_t * B32 = (const uint32_t *) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half2 * A2 = (const half2 *) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A fragment row indices for this lane
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    // Per-lane extraction constants (mirror ext8w<bits>)
    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    (void) x_s2;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float sh_red[WK][ROWS][COLS];

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x)
    {
        const uint32_t * bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        X3G_FragCh ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    x3v_dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    // two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    x3v_dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    x3v_dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                x3v_mma_ab_h(a01, a23, f0, ch[t][0]);
                x3v_mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
        }

        // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
        // tile*16 + frag*8 + 2*(l%4) (+1)
        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = acc0[t][f].x;
                        sh_red[warp][sr0][col + 1] = acc0[t][f].y;
                    }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float *) C)[(size_t) r * size_n + col] = sum;
            else                  ((half  *) C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

    // svh (output Hadamard scale), same semantics as the GEMM epilogue
    {
        grid.sync();

        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for (; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true>(((const float *) C) + this_warp * 128,
                                                ((float *) C) + this_warp * 128,
                                                svh + (this_warp * 128) % size_n, 0.088388347648f);
            else
                had_hf_r_128_inner<false, true>(((const half *) C) + this_warp * 128,
                                                ((half *) C) + this_warp * 128,
                                                svh + (this_warp * 128) % size_n, 0.088388347648f);
        }
    }
}

// Shape table, mirrored from exllamav3 exl3_kernel_map.cuh (TILESIZE_M is always 16)
//                                   TS_M  TS_K  TS_N  SH  FRAG
#define X3G_SHAPE_1                    16,   16,  128,   6,   5
#define X3G_SHAPE_2                    16,   32,  128,   4,   3
#define X3G_SHAPE_3                    16,   32,  256,   4,   3
#define X3G_SHAPE_4                    16,   16,  512,   4,   3

static const int x3g_tilesize_k[5] = { 0, 16, 32, 32, 16 };
static const int x3g_tilesize_n[5] = { 0, 128, 128, 256, 512 };
static const int x3g_blockdim  [5] = { 0, 256, 512, 512, 256 };

template <int bits>
static fp_x3g_kernel x3g_kernel_for_shape(int shape_idx)
{
    switch (shape_idx) {
        case 1: return x3g_gemm_kernel<bits, true, 2, X3G_SHAPE_1>;
        case 2: return x3g_gemm_kernel<bits, true, 2, X3G_SHAPE_2>;
        case 3: return x3g_gemm_kernel<bits, true, 2, X3G_SHAPE_3>;
        case 4: return x3g_gemm_kernel<bits, true, 2, X3G_SHAPE_4>;
    }
    return nullptr;
}

static fp_x3g_kernel x3g_kernel_ptr(int bits, int shape_idx)
{
    switch (bits) {
        case 1: return x3g_kernel_for_shape<1>(shape_idx);
        case 2: return x3g_kernel_for_shape<2>(shape_idx);
        case 3: return x3g_kernel_for_shape<3>(shape_idx);
        case 4: return x3g_kernel_for_shape<4>(shape_idx);
    }
    return nullptr;
}

// exllamav3's Ampere branch of select_gemm_shape(), K <= 4 only (our codec's range)
static int x3g_select_shape(int size_k, int size_n, int bits)
{
    bool mod_256 = (size_n % 256 == 0);
    bool mod_512 = (size_n % 512 == 0);
    if (mod_256 && bits <= 4) {
        if (size_n <= 2048 || size_k <= 2048) return 2;
        return 3;
    }
    if (mod_256 && size_n < 4096) return size_k > 8192 ? 3 : 2;
    if (mod_512 && (size_t) size_n * size_k > (size_t) 4096 * 4096 && bits <= 6) return 4;
    if (mod_256) return 3;
    return 2;
}

static bool x3g_shape_compat(int shape_idx, int size_k, int size_n)
{
    return (size_k % x3g_tilesize_k[shape_idx] == 0) && (size_n % x3g_tilesize_n[shape_idx] == 0);
}

struct X3gPlan
{
    fp_x3g_kernel fn;
    int shape_idx;
    int block_dim;
    int grid;
    int lock_ints;
};

// deterministic plan per (bits, size_k, size_n); shape from exllamav3's Ampere
// rule, overridable with GGML_PAW_X3_GEMM_SHAPE for A/B measurement.
static const X3gPlan * plan_x3g(int bits, int size_k, int size_n)
{
    static std::map<std::tuple<int, int, int>, X3gPlan> cache;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_tuple(bits, size_k, size_n);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second.fn ? &it->second : nullptr;
    }

    static const int force_shape = []() {
        const char * e = getenv("GGML_PAW_X3_GEMM_SHAPE");
        return e ? atoi(e) : 0;
    }();

    X3gPlan plan = {};
    int shape_idx = force_shape > 0 ? force_shape : x3g_select_shape(size_k, size_n, bits);
    if (shape_idx < 1 || shape_idx > 4 || !x3g_shape_compat(shape_idx, size_k, size_n)) {
        // fall back to any compatible shape before giving up on the tensor-core path
        shape_idx = 0;
        for (int s = 4; s >= 1; --s) {
            if (x3g_shape_compat(s, size_k, size_n)) { shape_idx = s; break; }
        }
    }
    if (shape_idx == 0 || size_k % 128 != 0) {
        auto res = cache.emplace(key, plan);   // fn == nullptr: remembered as unsupported
        return res.first->second.fn ? &res.first->second : nullptr;
    }

    plan.fn        = x3g_kernel_ptr(bits, shape_idx);
    plan.shape_idx = shape_idx;
    plan.block_dim = x3g_blockdim[shape_idx];
    int max_slices = (size_k / x3g_tilesize_k[shape_idx]) * (size_n / x3g_tilesize_n[shape_idx]);
    plan.grid      = X3G_MAX(X3G_MIN(max_slices, g_num_sms), 1);
    // one lock per 16-wide output block column
    plan.lock_ints = size_n / 16;

    if (plan.fn) {
        CUDA_CHECK(cudaFuncSetAttribute((const void *) plan.fn,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, X3G_SMEM_MAX));
        // cooperative launch may not exceed the co-resident block count
        int maxb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, (const void *) plan.fn,
                                                                 plan.block_dim, X3G_SMEM_MAX));
        int cap = X3G_MAX(maxb * g_num_sms, 1);
        if (plan.grid > cap) plan.grid = cap;
    }

    auto res = cache.emplace(key, plan);
    return res.first->second.fn ? &res.first->second : nullptr;
}


// Kernel table: [bits][mmode][cfg].  cb is always 2 (mul1) and C is always fp32 here.
template <int bits, int mmode, int cfg>
static inline fp_x3g_kernel x3v_kernel_of() { return x3v_gemv_kernel<bits, true, 2, mmode, cfg>; }

static fp_x3g_kernel x3v_select_kernel(int bits, int mmode, int cfg)
{
    #define X3V_SEL(b_, m_, c_) if (bits == b_ && mmode == m_ && cfg == c_) return x3v_kernel_of<b_, m_, c_>();
    X3V_SEL(2, 0, 0) X3V_SEL(2, 0, 1) X3V_SEL(2, 1, 0) X3V_SEL(2, 1, 1)
    X3V_SEL(3, 0, 0) X3V_SEL(3, 0, 1) X3V_SEL(3, 1, 0) X3V_SEL(3, 1, 1)
    X3V_SEL(4, 0, 0) X3V_SEL(4, 0, 1) X3V_SEL(4, 1, 0) X3V_SEL(4, 1, 1)
    #undef X3V_SEL
    return nullptr;
}

// GGML_PAW_X3_GEMV: 0 = off (default), 1 = exllamav3's Ampere heuristic,
// 2 = take the path whenever eligible, 3 = force narrow, 4 = force wide.
static int x3v_mode()
{
    static const int v = []() {
        const char * e = getenv("GGML_PAW_X3_GEMV");
        return e ? atoi(e) : 0;
    }();
    return v;
}

// exl3_gemv_cfg, restricted to cb == 2 (mul1) and Ampere.  Returns the config index or -1.
// Upper bound on size_n for the GEMV path.  The 248320-wide LM head is 14x larger than any
// body tensor and measured slower through the GEMV than through the GEMM; capping keeps the
// head on the GEMM.  0 disables the cap.
static int x3v_nmax()
{
    static const int v = []() {
        const char * e = getenv("GGML_PAW_X3_GEMV_NMAX");
        return e ? atoi(e) : 0;
    }();
    return v;
}

static int x3v_cfg(int size_m, int size_k, int size_n, int bits, int mode, int narrow_coresident)
{
    if (mode == 0) return -1;
    if (bits < 2 || bits > 4) return -1;
    if (size_m > X3V_MAX_M) return -1;
    if (size_k % 128 || size_n % 128) return -1;
    const int nmax = x3v_nmax();
    if (nmax > 0 && size_n > nmax) return -1;
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;
    if (mode == 4) return 1;

    // The narrow config wins whenever its grid fits in a single co-resident wave; in the
    // 1..2-wave zone the trailing partial wave costs more than the kernel gains unless
    // per-group work is small.  The wide config covers large-n with small-to-mid k.
    if (bits == 2) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (bits == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120) return 1;
    return -1;
}

// Returns true if the GEMV path was launched.
static bool x3v_try_launch(const float * A, const uint16_t * B, float * C,
                           int size_m, int size_k, int size_n,
                           const half * suh, half * A_had, const half * svh,
                           int bits, cudaStream_t stream)
{
    const int mode = x3v_mode();
    if (mode == 0) return false;
    if (bits < 2 || bits > 4) return false;
    if (size_m > X3V_MAX_M) return false;
    if (size_k % 128 || size_n % 128) return false;

    const int mmode = size_m == 1 ? 0 : 1;

    // Cooperative launch: grid capped at full co-residency (cached per kernel).  The narrow
    // config's co-residency also feeds the shape heuristic, so resolve it first.
    static std::map<const void *, int> occ_cache;
    static std::mutex occ_mtx;
    auto occupancy = [&](const void * fn, int block_dim) -> int {
        std::lock_guard<std::mutex> lock(occ_mtx);
        auto it = occ_cache.find(fn);
        if (it != occ_cache.end()) return it->second;
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, block_dim, 0));
        occ_cache[fn] = blocks_per_sm;
        return blocks_per_sm;
    };

    fp_x3g_kernel narrow = x3v_select_kernel(bits, mmode, 0);
    if (!narrow) return false;
    const int narrow_coresident = occupancy((const void *) narrow, 512) * g_num_sms;

    const int cfg = x3v_cfg(size_m, size_k, size_n, bits, mode, narrow_coresident);
    if (cfg < 0) return false;

    fp_x3g_kernel fn = cfg == 0 ? narrow : x3v_select_kernel(bits, mmode, cfg);
    if (!fn) return false;

    const int block_dim = cfg == 0 ? 512 : 256;
    const int cols      = cfg == 0 ? 32  : 64;

    const int max_blocks = occupancy((const void *) fn, block_dim) * g_num_sms;
    const int grid = X3G_MIN(size_n / cols, max_blocks);
    if (grid < 1) return false;

    int * locks = nullptr;   // unused by the GEMV path, but kept in the arg list
    const int a_f32 = 1;       // A is f32: cast folds into the input prologue
    void * Cv = (void *) C;
    void * args[] = {
        (void *) &A, (void *) &B, (void *) &Cv,
        (void *) &size_m, (void *) &size_k, (void *) &size_n,
        (void *) &locks, (void *) &suh, (void *) &A_had, (void *) &svh,
        (void *) &a_f32
    };
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) fn, dim3(grid), dim3(block_dim),
                                           args, 0, stream));
    return true;
}

static void launch_x3g(const X3gPlan & plan,
                       const half * A, const uint16_t * B, float * C,
                       int size_m, int size_k, int size_n,
                       const half * suh, half * A_had, const half * svh,
                        int * locks, cudaStream_t stream)
{
    const int a_f32 = 0;   // x3g path keeps fp16 input (cast kernel output)
    void * args[] =
    {
        (void *) &A, (void *) &B, (void *) &C,
        (void *) &size_m, (void *) &size_k, (void *) &size_n,
        (void *) &locks, (void *) &suh, (void *) &A_had, (void *) &svh,
        (void *) &a_f32
    };
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) plan.fn,
                                           dim3(plan.grid), dim3(plan.block_dim),
                                           args, X3G_SMEM_MAX, stream));
}

} // namespace paw_x3


// scoped macros from the ported section; do not leak into the rest of paw.cu

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

void ggml_cuda_op_paw_x3_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace paw_x3;

    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * suh     = dst->src[1];
    const ggml_tensor * svh     = dst->src[2];
    const ggml_tensor * x       = dst->src[3];

    GGML_ASSERT(trellis->type == GGML_TYPE_I16);
    GGML_ASSERT(suh->type == GGML_TYPE_F16);
    GGML_ASSERT(svh->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(trellis) && ggml_is_contiguous(suh) &&
                ggml_is_contiguous(svh) && ggml_is_contiguous(x) && ggml_is_contiguous(dst));

    const int bits = (int) trellis->ne[0] / 16;   // words-per-tile = 16*K
    GGML_ASSERT(bits == 1 || bits == 2 || bits == 3 || bits == 4);

    const int n = (int) x->ne[0];
    const int m = (int) dst->ne[0];
    GGML_ASSERT((int64_t) trellis->ne[1] == (m / 16) * (n / 16));
    const int64_t nt64 = x->ne[1] * x->ne[2] * x->ne[3];
    GGML_ASSERT(nt64 <= INT_MAX);
    const int nt = (int) nt64;

    static bool sms_init = false;
    if (!sms_init) {
        g_num_sms = x3_num_sms();
        sms_init = true;
    }

    cudaStream_t stream = ctx.stream();

    // fp16 activation rows for the paths that still need them (x3g GEMM, x3 hgemm,
    // debug dump). The x3v and x3_sq paths take f32 x directly: the f32->f16 cast
    // folds into their input prologues (bit-identical, see had_hf_r_128_inner_f32).
    // NOTE: the cuda pool frees in exact reverse order of allocation, so xh is
    // allocated eagerly below (before any branch workspace) whenever any later path
    // may need it; allocating it lazily inside a branch breaks the pool LIFO invariant.
    ggml_cuda_pool_alloc<half> xh;

    // EXL3-style prefill switch: at nt >= GGML_PAW_X3_PREFILL_NT (default 16)
    // reconstruct W once and run one batched hgemm instead of looping the
    // single-row kernel per token. W is emitted in ORIGINAL basis (both
    // Hadamards + suh/svh folded), so the hgemm runs on the raw rows.
    static const int x3_prefill_nt = []() {
        const char * e = getenv("GGML_PAW_X3_PREFILL_NT");
        return e ? atoi(e) : 9;
    }();

    // Tensor-core trellis GEMM for nt >= GGML_PAW_X3_GEMM_NT (default 3).
    // exllamav3 abandons the int8 GEMV at exactly the same point
    // (exl3_gemv_int8.cu: `if (size_m > 2) return false;`); measured here, the
    // two engines' trellis-matmul cost agrees to 0.6% at nt=2 and diverges from
    // nt=4 (1.25x vs 1.66x), which is the whole speculative-verify gap.
    // GGML_PAW_X3_GEMM_NT=0 disables the path and restores the sq/hgemm split.
    static const int x3_gemm_nt = []() {
        const char * e = getenv("GGML_PAW_X3_GEMM_NT");
        return e ? atoi(e) : 3;
    }();
    // Upper cutoff: past ~128 rows the tile loop's grid.sync per 16 rows costs
    // more than reconstructing W once and handing the whole batch to cuBLAS.
    // Measured on B3.5 / RTX 3090 (t/s, gemm vs reconstruct+cuBLAS):
    //   nt=8 201.7/111.7  nt=16 362.4/78.0  nt=32 413.8/154.1
    //   nt=128 469.1/463.7 (tie)  nt=256 479.7/624.9  nt=512 482.4/798.3
    static const int x3_gemm_nt_max = []() {
        const char * e = getenv("GGML_PAW_X3_GEMM_NT_MAX");
        return e ? atoi(e) : 128;
    }();
    const X3gPlan * gplan = (x3_gemm_nt > 0 && nt >= x3_gemm_nt && nt <= x3_gemm_nt_max)
                          ? plan_x3g(bits, n, m) : nullptr;

    // Small-m GEMV path first: its prologue consumes f32 x directly, so a
    // successful launch needs no xh and no cast kernel at all. Tried before any
    // pool allocation (besides its own branch workspace, freed on exit).
    bool x3v_done = false;
    if (x3v_mode() != 0 && nt <= X3V_MAX_M) {
        char shpv[64];
        snprintf(shpv, sizeof(shpv), " m=%d n=%d K=%d nt=%d", m, n, bits, nt);
        paw_timed(stream, std::string("x3_gemv") + shpv, [&]() {
            ggml_cuda_pool_alloc<half> a_had(ctx.pool(), (size_t) n * nt);
            x3v_done = x3v_try_launch((const float *) x->data,
                                      (const uint16_t *) trellis->data,
                                      (float *) dst->data,
                                      nt, n, m,
                                      (const half *) suh->data, a_had.get(),
                                      (const half *) svh->data, bits, stream);
        });
    }

    // eager xh only for paths that actually consume fp16 rows (x3g GEMM, x3
    // hgemm, debug dump). Pool LIFO: allocated here, before any branch workspace
    // below; skipped entirely when x3v (or the folded sq path) handles the call.
    const bool need_xh = !x3v_done && (getenv("GGML_PAW_X3_DUMP") || gplan || nt >= x3_prefill_nt);
    if (need_xh) {
        xh.alloc(ctx.pool(), (size_t) n * nt);
        x3_cast_f32_f16_kernel<<<((size_t) n * nt + 255) / 256, 256, 0, stream>>>(
            (half *) xh.get(), (const float *) x->data, (int) ((size_t) n * nt));
    }

    if (x3v_done) {
        goto paw_x3_done;
    }

    if (gplan) {
        char shp[64];
        snprintf(shp, sizeof(shp), " m=%d n=%d K=%d nt=%d", m, n, bits, nt);
        paw_timed(stream, std::string("x3_gemm") + shp, [&]() {
            ggml_cuda_pool_alloc<half> a_had(ctx.pool(), (size_t) n * nt);
            ggml_cuda_pool_alloc<int>  locks(ctx.pool(), (size_t) gplan->lock_ints);
            CUDA_CHECK(cudaMemsetAsync(locks.get(), 0, (size_t) gplan->lock_ints * sizeof(int), stream));
            launch_x3g(*gplan,
                       xh.get(),
                       (const uint16_t *) trellis->data,
                       (float *) dst->data,
                       nt, n, m,
                       (const half *) suh->data, a_had.get(), (const half *) svh->data,
                       locks.get(), stream);
        });
    } else if (nt >= x3_prefill_nt) {
        char shp[64];
        snprintf(shp, sizeof(shp), " m=%d n=%d K=%d nt=%d", m, n, bits, nt);
        paw_timed(stream, std::string("x3_hgemm") + shp, [&]() {
            ggml_cuda_pool_alloc<half> wmat(ctx.pool(), (size_t) n * m);
            x3r_reconstruct_ws(wmat.get(), (const uint16_t *) trellis->data,
                               (const half *) suh->data, (const half *) svh->data,
                               n, m, bits, stream);
            // Y(nt,m) = X(nt,n) @ W(n,m), all row-major: column-major view is
            // Y_col(m,nt) = W_col(m,n) @ X_col(n,nt) with identical bytes.
            const float alpha = 1.0f, beta = 0.0f;
            cublasHandle_t h = ctx.cublas_handle();
            CUBLAS_CHECK(cublasSetStream(h, stream));
            cublasGemmAlgo_t algo = (m % 8 == 0 && n % 8 == 0 && nt % 8 == 0)
                ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
            CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, nt, n, &alpha,
                                      wmat.get(), CUDA_R_16F, m,
                                      xh.get(), CUDA_R_16F, n,
                                      &beta, (float *) dst->data, CUDA_R_32F, m,
                                      CUBLAS_COMPUTE_32F, algo));
        });
    } else {
    // small-nt fused path: one launch over all nt rows (trellis read once).
    // nt == 1 keeps the exact legacy single-row behavior.
    GGML_ASSERT(nt <= SQ_M_MAX);
    const SqPlan & plan = plan_sq(bits, n, m, nt);

    // fixed-layout workspace: [counters | qsums | partials]. The counters must
    // be zero at kernel start; they self-reset before return (graph-safe).
    // partials scale with the fused batch: ksplit slices x nt rows x output cols.
    // qsums needs 4 floats per (slice, row): 4*ksplit*nt must fit the
    // reserved qsums region (4*SQ_KSPLIT_CAP*8 ints).
    GGML_ASSERT((size_t) 4 * plan.ksplit * nt <= (size_t) 4 * SQ_KSPLIT_CAP * 8);
    ggml_cuda_pool_alloc<int> ws(ctx.pool(), SQ_WS_RESERVED + (size_t) plan.ksplit * nt * m);

    {
        char shp[64];
        snprintf(shp, sizeof(shp), " m=%d n=%d K=%d nt=%d", m, n, bits, nt);
        paw_timed(stream, std::string("x3_sq") + shp, [&]() {
            CUDA_CHECK(cudaMemsetAsync(ws.get(), 0, SQ_COUNTERS_CAP * sizeof(int), stream));
            launch_sq(plan, bits, nt,
                      (const float *) x->data,
                      (const uint16_t *) trellis->data,
                      (float *) dst->data,
                      n, m,
                      (const half *) suh->data, (const half *) svh->data,
                      ws.get(), stream);
        });
    }
    }

    // debug: dump the I/O of the first calls for offline verification against
    // the Phase-1 oracle (GGML_PAW_X3_DUMP=<dir>)
paw_x3_done:
    if (const char * dump_dir = getenv("GGML_PAW_X3_DUMP")) {
        static int dump_count = 0;
        static const int dump_skip = []() {
            const char * e = getenv("GGML_PAW_X3_DUMP_SKIP");
            return e ? atoi(e) : 0;
        }();
        static const int dump_cap = []() {
            const char * e = getenv("GGML_PAW_X3_DUMP_CAP");
            return e ? atoi(e) : 64;
        }();
        if (dump_count >= dump_skip && dump_count < dump_skip + dump_cap) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            char path[512];
            snprintf(path, sizeof(path), "%s/dump%02d", dump_dir, dump_count - dump_skip);
            FILE * fmeta = fopen((std::string(path) + "_meta.txt").c_str(), "w");
            fprintf(fmeta, "n %d m %d K %d nt %d\n", n, m, bits, nt);
            fclose(fmeta);
            const size_t tbytes = (size_t) trellis->ne[0] * trellis->ne[1] * 2;
            size_t hneed = (size_t) m * nt * sizeof(float);
            if ((size_t) n * nt * sizeof(float) > hneed) hneed = (size_t) n * nt * sizeof(float);
            if (tbytes > hneed) hneed = tbytes;
            std::vector<char> host(hneed);
            auto dump_dev = [&] (const std::string & suffix, const void * dev, size_t bytes) {
                CUDA_CHECK(cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost));
                FILE * f = fopen((std::string(path) + suffix).c_str(), "wb");
                fwrite(host.data(), 1, bytes, f);
                fclose(f);
            };
            dump_dev("_A.f32", x->data, (size_t) n * nt * sizeof(float));
            if (xh.get()) {
                dump_dev("_Ah.u16", xh.get(), (size_t) n * nt * 2);
            }
            dump_dev("_B.u16", trellis->data, tbytes);
            dump_dev("_suh.u16", suh->data, (size_t) n * 2);
            dump_dev("_svh.u16", svh->data, (size_t) m * 2);
            dump_dev("_y.f32", dst->data, (size_t) m * nt * sizeof(float));
        }
        ++dump_count;
    }
}

#undef NUM_THREADS
#undef GEMV_STAGE_D
#undef SQ_KSPLIT_CAP
#undef SQ_MINROWS
#undef SQ_ROWS_MAX
#undef SQ_COUNTERS_CAP
#undef SQ_WS_RESERVED
