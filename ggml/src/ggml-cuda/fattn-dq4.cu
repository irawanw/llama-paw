#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-dq4.cuh"

// Speculative-verify attention for the PAW-27B production shape on Ampere (sm_80..sm_86):
// D 256, 2..8 query rows (n_draft+1) over 24 Q heads grouped 6-per-KV-head, q4_0/q4_0 KV,
// causal mask, no sinks / ALiBi / logit softcap.  Dispatch is gated in fattn.cu; everything
// else about the shape is checked with GGML_ASSERT in the launcher.
//
// Why a dedicated kernel: at nt >= 2 the MMA path dequantizes the whole KV slice to f16
// scratch inside the op, which throttles long-context KV reads far below the device
// ceiling.  This kernel consumes the q4_0 KV directly: one CTA per (KV head, context
// split), 6 warps (one per sibling Q head), and each 16-token K/V tile is read exactly
// once per warp.  Splits that lie entirely beyond every row's causal prefix (the unused
// KV tail) are skipped without touching the cache.
//
// Data layout decisions:
// - Lane l of a warp covers the 8 dims [8l, 8l+8).  K/V q4_0 int32 #k of a 32-dim block
//   holds exactly the nibbles of dims [8k, 8k+8) of that block, so one int32 per lane.
// - K is staged in shared memory with the +8 nibble bias already subtracted (__vsubss4),
//   so the QK dot is a plain dp4a against Q q8_1 values with no correction terms.
// - Q is quantized per 32-dim block (same scheme as fattn-vec) but split into even/odd
//   nibble-aligned int32s so each lane's dp4a pair lines up with its 8 dims directly.
// - Online softmax per (row, lane): each lane tracks the tokens congruent to its token
//   slot t = lane & 3; the four partials are merged with one butterfly at split end.
// - Output partials land in the same [row][head][split][dim] layout that
//   flash_attn_combine_results consumes, so the standard combine kernel is reused.
template <bool need_combine>
__launch_bounds__(192, 2)
static __global__ void flash_attn_ext_dq4(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        float * __restrict__ dst_ptr,
        float2 * __restrict__ dst_meta_ptr,
        const float scale,
        const int nt,
        const int kv_len,
        const int split_len,
        const int64_t nb01, const int64_t nb02,
        const int64_t nb11, const int64_t nb12,
        const int64_t nb21, const int64_t nb22,
        const int64_t nb31) {
    

    constexpr int D     = 256;
    constexpr int NTMAX = 8;    // query rows, n_draft + 1 for n_max <= 7
    constexpr int NWARP = 6;    // Q heads per KV head
    constexpr int TILE  = 16;   // KV tokens per tile

    __shared__ int    K_qs_sh[TILE][8][4];        // raw q4_0 qs int32s, [token][block][int32]
    __shared__ float  K_d_sh [TILE][8];
    __shared__ float  V_d_sh [TILE][8];
    __shared__ half   V_sh   [TILE][D];           // dequantized V tile
    __shared__ int    Q_q8_sh[NWARP][NTMAX][64];  // Q q8_1 int32s: block*8 + j, int32 j = dims [4j, 4j+4)
    __shared__ float  Q_d_sh [NWARP][NTMAX][8];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    const int kvh   = blockIdx.y;
    const int head  = kvh*NWARP + warp;
    const int split = blockIdx.x;
    const int pb    = gridDim.x;

    const int kv0 = split*split_len;
    const int kv1 = min(kv_len, kv0 + split_len);

    const char * Qh = Q_ptr + nb02*head;
    const char * Kh = K_ptr + nb12*kvh;
    const char * Vh = V_ptr + nb22*kvh;

    // Quantize this warp's Q head rows to q8_1 with one scale per 32-dim block (the
    // same block partition the q4_0 K side uses): block jb covers dims [32jb, 32jb+32),
    // int32 j of a block holds dims [4j, 4j+4).  Lane (cb = lane >> 3, l7 = lane & 7)
    // stages int32 l7 of blocks cb and cb+4; the eight lanes of a block reduce its amax.
    const int cbq = lane >> 3;
    const int l7  = lane & 7;
    for (int r = 0; r < nt; ++r) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int jb = cbq + 4*p;
            float4 qv;
            ggml_cuda_memcpy_1<16, 4>(&qv, Qh + nb01*r + 128*jb + 16*l7);

            const float vals[4] = {qv.x, qv.y, qv.z, qv.w};

            float amax = fmaxf(fabsf(vals[0]), fmaxf(fabsf(vals[1]), fmaxf(fabsf(vals[2]), fabsf(vals[3]))));
#pragma unroll
            for (int mask = 1; mask <= 4; mask <<= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask));
            }

            const float d = amax/127;

            int u32 = 0;
            if (d != 0.0f) {
                int8_t * q8 = (int8_t *) &u32;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    q8[j] = (int) roundf(vals[j]/d);
                }
            }

            Q_q8_sh[warp][r][jb*8 + l7] = u32;
            if (l7 == 0) {
                Q_d_sh[warp][r][jb] = d;
            }
        }
    }

    // Row activity for this split: with a causal mask, row r attends tokens
    // [0, pos_r], so if the first token of the split is masked the whole split is.
    bool act[NTMAX];
    bool any = false;
#pragma unroll
    for (int r = 0; r < NTMAX; ++r) {
        if (r >= nt) {
            act[r] = false;
            continue;
        }
        half mh;
        ggml_cuda_memcpy_1<2>(&mh, mask_ptr + nb31*r + 2*kv0);
        act[r] = __half2float(mh) > -30000.0f;
        any = any || act[r];
    }

    if (!any) {
        // Split lies in the masked-out tail of the KV cache: emit dead partials so the
        // combine sees a full set, and skip all tile traffic.
        for (int r = 0; r < nt; ++r) {
            if (need_combine) {
                float * out = dst_ptr + ((r*(gridDim.y*NWARP) + head)*pb + split)*D;
                *(float4 *) (out + 8*lane)     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *(float4 *) (out + 8*lane + 4) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (lane == 0) {
                    dst_meta_ptr[(r*(gridDim.y*NWARP) + head)*pb + split] = make_float2(-FLT_MAX, 0.0f);
                }
            } else {
                float * out = dst_ptr + (r*(gridDim.y*NWARP) + head)*D;
                *(float4 *) (out + 8*lane)     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *(float4 *) (out + 8*lane + 4) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }

    float m_s[NTMAX];
    float l_s[NTMAX];
    half2 acc[NTMAX][4];
#pragma unroll
    for (int r = 0; r < NTMAX; ++r) {
        m_s[r] = -FLT_MAX;
        l_s[r] = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[r][j] = make_half2(0.0f, 0.0f);
        }
    }

    // K/V int32 #k of block ib covers dims [32ib + 8k, 32ib + 8k + 8); the lane's
    // slice is (ib = lane >> 2, k = lane & 3) -- identical mapping for loads and use.
    for (int kv = kv0; kv < kv1; kv += TILE) {
        for (int i = tid; i < TILE*8; i += NWARP*WARP_SIZE) {
            const int tk = i >> 3;
            const int ib = i & 7;
            half kd;
            ggml_cuda_memcpy_1<2>(&kd, Kh + nb11*(kv + tk) + 18*ib);
            K_d_sh[tk][ib] = __half2float(kd);
            half vd;
            ggml_cuda_memcpy_1<2>(&vd, Vh + nb21*(kv + tk) + 18*ib);
            V_d_sh[tk][ib] = __half2float(vd);
        }
        for (int i = tid; i < TILE*8*4; i += NWARP*WARP_SIZE) {
            const int tk   = i >> 5;
            const int ib   = (i >> 2) & 7;
            const int iq4  = i & 3;
            // q4_0 qs live at odd offsets (d takes the first 2 bytes), so copy in shorts.
            int v;
            ggml_cuda_memcpy_1<4, 2>(&v, Kh + nb11*(kv + tk) + 18*ib + 2 + 4*iq4);
            K_qs_sh[tk][ib][iq4] = v;
        }
        __syncthreads();
        for (int i = tid; i < TILE*8*4; i += NWARP*WARP_SIZE) {
            const int tk  = i >> 5;
            const int ib  = (i >> 2) & 7;
            const int iq4 = i & 3;
            int v;
            ggml_cuda_memcpy_1<4, 2>(&v, Vh + nb21*(kv + tk) + 18*ib + 2 + 4*iq4);
            const int v_lo = __vsubss4(v & 0x0F0F0F0F, 0x08080808);
            const int v_hi = __vsubss4((v >> 4) & 0x0F0F0F0F, 0x08080808);
            const float dV = V_d_sh[tk][ib];
            const int8_t * q8l = (const int8_t *) &v_lo;
            const int8_t * q8h = (const int8_t *) &v_hi;
            // q4_0 packs byte j as (dim j, dim j + 16): low nibbles are dims [4iq4, +4)
            // and high nibbles dims [16 + 4iq4, +4).
            half2 out_lo[2];
            half2 out_hi[2];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                out_lo[m] = make_half2(__float2half(dV*q8l[2*m + 0]), __float2half(dV*q8l[2*m + 1]));
                out_hi[m] = make_half2(__float2half(dV*q8h[2*m + 0]), __float2half(dV*q8h[2*m + 1]));
            }
            *(uint2 *) (V_sh[tk] + 32*ib + 4*iq4)       = *(const uint2 *) out_lo;
            *(uint2 *) (V_sh[tk] + 32*ib + 16 + 4*iq4)  = *(const uint2 *) out_hi;
        }
        __syncthreads();

        for (int r = 0; r < NTMAX; ++r) {
            if (r >= nt || !act[r]) {
                continue;
            }
            // Lane (t = lane >> 3, j = lane & 7): token slot t within a 4-token group,
            // Q/K block j.  The lane assembles block j's 32-dim dot in-registers (8 dp4a
            // over its four int32 pairs), the eight j-lanes of the token merge it with
            // three shuffles, and the four token dots are broadcast back so every lane
            // updates a full-row online softmax and accumulates all tokens of the tile
            // for its own 8 dims [8*lane, 8*lane + 8).
            const int tt = lane >> 3;
            const int jb = lane & 7;

            // Q q8 int32s of block jb: int32 j holds dims [4j, 4j+4); the K int32's low
            // nibbles are dims [4c4, 4c4+4) and its high nibbles dims [16+4c4, 16+4c4+4).
            int u[8];
#pragma unroll
            for (int c4 = 0; c4 < 8; ++c4) {
                u[c4] = Q_q8_sh[warp][r][jb*8 + c4];
            }
            const float dqb = Q_d_sh[warp][r][jb];

#pragma unroll
            for (int tg = 0; tg < TILE/4; ++tg) {
                const int tk = 4*tg + tt;

                int sumi = 0;
#pragma unroll
                for (int c4 = 0; c4 < 4; ++c4) {
                    const int v = K_qs_sh[tk][jb][c4];
                    const int k_lo = __vsubss4(v & 0x0F0F0F0F, 0x08080808);
                    const int k_hi = __vsubss4((v >> 4) & 0x0F0F0F0F, 0x08080808);
                    sumi = ggml_cuda_dp4a(k_lo, u[c4], sumi);
                    sumi = ggml_cuda_dp4a(k_hi, u[c4 + 4], sumi);
                }
                float blk = K_d_sh[tk][jb] * dqb * sumi;
                blk += __shfl_xor_sync(0xFFFFFFFF, blk, 1);
                blk += __shfl_xor_sync(0xFFFFFFFF, blk, 2);
                blk += __shfl_xor_sync(0xFFFFFFFF, blk, 4);

                // The four tokens of the group: own score in-lane, the other three via
                // one shuffle each (all eight lanes of a token carry the same merged dot).
                half mh[4];
                ggml_cuda_memcpy_1<8>(mh, mask_ptr + nb31*r + 2*(kv + 4*tg));
                float s_g[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const float blk_g = __shfl_sync(0xFFFFFFFF, blk, 8*g + jb);
                    s_g[g] = fmaf(g == tt ? blk : blk_g, scale, __half2float(mh[g]));
                }
                // One online-softmax step per token; every lane ends the tile with the
                // full-row (m, l) and the sum over all 16 tokens for its own 8 dims.
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const float m_new = fmaxf(m_s[r], s_g[g]);
                    const float c     = expf(m_s[r] - m_new);
                    const float w     = expf(s_g[g] - m_new);
                    l_s[r] = fmaf(c, l_s[r], w);
                    m_s[r] = m_new;

                    half2 vh[4];
                    *(uint4 *) vh = *(const uint4 *) (V_sh[4*tg + g] + 8*lane);
                    const half2 w2 = make_half2(w, w);
                    const half2 c2 = make_half2(c, c);
#pragma unroll
                    for (int j2 = 0; j2 < 4; ++j2) {
                        acc[r][j2] = __hfma2(w2, vh[j2], __hmul2(c2, acc[r][j2]));
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < nt; ++r) {
        float tmp[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tmp[2*j + 0] = __low2float(acc[r][j]);
            tmp[2*j + 1] = __high2float(acc[r][j]);
        }
        if (need_combine) {
            float * out = dst_ptr + ((r*(gridDim.y*NWARP) + head)*pb + split)*D;
            *(float4 *) (out + 8*lane)     = *(const float4 *) (tmp + 0);
            *(float4 *) (out + 8*lane + 4) = *(const float4 *) (tmp + 4);
            if (lane == 0) {
                dst_meta_ptr[(r*(gridDim.y*NWARP) + head)*pb + split] = make_float2(m_s[r], l_s[r]);
            }
        } else {
            float * out = dst_ptr + (r*(gridDim.y*NWARP) + head)*D;
            const float inv_l = 1.0f/l_s[r];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                tmp[j] *= inv_l;
            }
            *(float4 *) (out + 8*lane)     = *(const float4 *) (tmp + 0);
            *(float4 *) (out + 8*lane + 4) = *(const float4 *) (tmp + 4);
        }
    }
}


static bool ggml_cuda_fattn_dq4_split_env(int & pb_env) {
    static int val = -2;
    if (val == -2) {
        const char * env = getenv("GGML_PAW_DQ4_SPLIT");
        val = env ? atoi(env) : 0;
    }
    pb_env = val;
    return val > 0;
}

void ggml_cuda_flash_attn_ext_dq4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32 && KQV->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == 256 && Q->ne[1] >= 2 && Q->ne[1] <= 8 && Q->ne[3] == 1);
    GGML_ASSERT(K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(Q->ne[2] == 6*K->ne[2]);
    GGML_ASSERT(K->ne[1] == V->ne[1] && K->ne[1] % 256 == 0);
    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1);
    GGML_ASSERT(Q->nb[0] == 4);

    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f && logit_softcap == 0.0f);

    const int nt     = Q->ne[1];
    const int kv_len = K->ne[1];
    const int nsm    = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    int pb_env = 0;
    const bool pb_override = ggml_cuda_fattn_dq4_split_env(pb_env);

    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, flash_attn_ext_dq4<true>, 192, 0));
    GGML_ASSERT(max_blocks_per_sm > 0);

    int pb;
    if (pb_override) {
        pb = pb_env;
    } else {
        // Contract split length 8192, but keep the machine full at shorter contexts:
        // one (head, split) block per KV head, so ~2 CTAs/SM want pb ~= nsm/2.
        const int pb_fill  = (nsm + 1)/2;
        const int pb_want  = (kv_len + 8191)/8192;
        const int pb_max   = (kv_len + 15)/16;
        pb = std::max(pb_want, std::min(pb_fill, pb_max));
    }
    pb = std::min(pb, (kv_len + 15)/16);

    const int split_len = ((kv_len + pb - 1)/pb + 15) & ~15;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();

    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (pb > 1) {
        dst_tmp.alloc(pb*ggml_nelements(KQV));
        dst_tmp_meta.alloc(pb*ggml_nrows(KQV));
    }

    const dim3 block_dim(32, 6, 1);
    const dim3 blocks_num(pb, K->ne[2], 1);

    if (pb > 1) {
        ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_ext_dq4<true>, launch_params,
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            (const char *) mask->data,
            dst_tmp.ptr,
            dst_tmp_meta.ptr,
            scale, nt, kv_len, split_len,
            Q->nb[1], Q->nb[2],
            K->nb[1], K->nb[2],
            V->nb[1], V->nb[2],
            mask->nb[1]);
        CUDA_CHECK(cudaGetLastError());

        const dim3 block_dim_combine(256, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = pb*sizeof(float2);

        ggml_cuda_kernel_launch_params launch_params_combine =
            ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<256>, launch_params_combine,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, pb);
        CUDA_CHECK(cudaGetLastError());
    } else {
        ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_ext_dq4<false>, launch_params,
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            (const char *) mask->data,
            (float *) KQV->data,
            nullptr,
            scale, nt, kv_len, split_len,
            Q->nb[1], Q->nb[2],
            K->nb[1], K->nb[2],
            V->nb[1], V->nb[2],
            mask->nb[1]);
        CUDA_CHECK(cudaGetLastError());
    }
}
