// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"


//
// EXP_MM — 4 kernels: group pairs by storage expert -> u = H(su_e ⊙ x) per
// pair -> trellis walk/dot per group -> y = sv_e ⊙ H(v) per pair
// (paw_exp_group/exp_u/exp_walk/exp_out.comp). Scratch: int32
// [4*n_groups + P] (cnt | off | orig | cursor(unused) | pairs) and float
// [P*n + P*m] (u then v).
//

static __global__ void paw_exp_group_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t       * GGML_CUDA_RESTRICT scr,
        const int n_used, const int n_tok, const int n_kept, const int n_groups,
        const int ids_s0, const int ids_s1) {
    constexpr int WG = 256;
    __shared__ int sh_cnt[512];    // MAX_GROUPS = 512, enforced by supports_op
    __shared__ int sh_cur[512];
    __shared__ int sh_orig[512];

    const int tid = threadIdx.x;
    const int P   = n_used*n_tok;

    ggml_cuda_pdl_sync();
    for (int g = tid; g < n_groups; g += WG) {
        sh_cnt[g]  = 0;
        sh_orig[g] = 0;
    }
    __syncthreads();
    for (int p = tid; p < P; p += WG) {
        const int s = p % n_used;
        const int t = p / n_used;
        const int g = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
        atomicAdd(&sh_cnt[g], 1);
        atomicExch(&sh_orig[g], ids[s*ids_s0 + t*ids_s1]);   // orig id (same value per group)
    }
    __syncthreads();
    if (tid == 0) {
        int off = 0;
        for (int g = 0; g < n_groups; ++g) {
            const int c = sh_cnt[g];
            sh_cur[g] = off;
            scr[g]              = c;            // cnt
            scr[n_groups + g]   = off;          // off
            scr[2*n_groups + g] = sh_orig[g];   // orig id
            off += c;
        }
    }
    __syncthreads();
    for (int p = tid; p < P; p += WG) {
        const int g   = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
        const int pos = atomicAdd(&sh_cur[g], 1);
        scr[4*n_groups + pos] = p;              // flat pair rank p = t*n_used + s
    }
}

template <int WG>
static __global__ void paw_exp_u_kernel(
        const half    * GGML_CUDA_RESTRICT su,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT x,
        float         * GGML_CUDA_RESTRICT scr_u,
        const int n, const int n_used, const int xne1,
        const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;
    for (int i = tid; i < n; i += WG) {
        sh[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) p*n + i] = __fdiv_rn(sh[i], sc);
    }
}

// --- exp_group + exp_u launch fusion, GGML_PAW_EXP_GROUP_FUSE=1 ---
//
// exp_group_kernel is a single, tiny block (~10.8us average per call per
// GGML_PAW_TIME -- almost entirely kernel-launch overhead, the real work
// is a couple hundred int ops) that's otherwise independent of exp_u_kernel
// (exp_u reads only su/ids/x, never scr_i -- no data dependency between
// them). This shares one launch: blockIdx.x==0 && blockIdx.y==0 does
// exp_group's unchanged body; every other block does exp_u's unchanged
// body with s = blockIdx.x - 1. Grid grows from (n_used, n_tok, 1) to
// (n_used+1, n_tok, 1); the extra (0, y>0) blocks when n_tok>1 are
// harmless no-ops (exp_group's result doesn't depend on t). Both bodies
// are copied verbatim -- no numeric changes, purely a launch-count cut.
template <int WG>
static __global__ void paw_exp_group_u_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT scr,
        const half     * GGML_CUDA_RESTRICT su,
        const float    * GGML_CUDA_RESTRICT x,
        float          * GGML_CUDA_RESTRICT scr_u,
        const int n, const int n_used, const int n_tok, const int n_kept, const int n_groups,
        const int xne1, const int ids_s0, const int ids_s1) {
    if (blockIdx.x == 0) {
        if (blockIdx.y != 0) {
            return;   // exp_group's result is global, only run it once
        }
        __shared__ int sh_cnt[512];
        __shared__ int sh_cur[512];
        __shared__ int sh_orig[512];

        const int tid = threadIdx.x;
        const int P   = n_used*n_tok;

        ggml_cuda_pdl_sync();
        for (int g = tid; g < n_groups; g += WG) {
            sh_cnt[g]  = 0;
            sh_orig[g] = 0;
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int s = p % n_used;
            const int t = p / n_used;
            const int g = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            atomicAdd(&sh_cnt[g], 1);
            atomicExch(&sh_orig[g], ids[s*ids_s0 + t*ids_s1]);
        }
        __syncthreads();
        if (tid == 0) {
            int off = 0;
            for (int g = 0; g < n_groups; ++g) {
                const int c = sh_cnt[g];
                sh_cur[g] = off;
                scr[g]              = c;
                scr[n_groups + g]   = off;
                scr[2*n_groups + g] = sh_orig[g];
                off += c;
            }
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int g   = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            const int pos = atomicAdd(&sh_cur[g], 1);
            scr[4*n_groups + pos] = p;
        }
        return;
    }

    __shared__ float sh[2048];

    const int s   = blockIdx.x - 1;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;
    for (int i = tid; i < n; i += WG) {
        sh[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) p*n + i] = __fdiv_rn(sh[i], sc);
    }
}

// One-time host-side repack of the V8 tlut (F16 [8, 32768], 512 KB) into the
// Mach-1 engine's p4 form (PAW_V8_LUT=p4): the V8 alphabet is <= 16 distinct
// fp16 levels, so a codeword row becomes one nibble-packed u32 (128 KB table)
// resolved through a 16-entry level table. The levels are stored as the exact
// fp32 values of the distinct F16 entries, so the resolved value is
// bit-identical to __half2float of the original entry by construction (no
// step multiply at runtime; +0/-0 collapse to one level like np.unique in
// the reference). Falls back to the fp16-gather path (packed == nullptr)
// when the alphabet has more than 16 distinct values.
//
// The cache is keyed by the tlut's device pointer and is NEVER freed: one
// entry is 128 KB + 64 B per distinct tlut (one per model) and the key's
// lifetime matches the weight allocation — intentionally leaked.
struct paw_p4_table {
    const uint32_t * packed;   // device [32768], 8 x 4-bit level indices per row
    const float    * levels;   // device [16], exact fp32 of the fp16 alphabet
};

static paw_p4_table paw_exp_p4_table(const void * tlut_data, cudaStream_t stream) {
    struct entry { const uint32_t * packed; const float * levels; };
    static std::mutex paw_p4_mutex;
    static std::unordered_map<const void *, entry> paw_p4_cache;

    std::lock_guard<std::mutex> lock(paw_p4_mutex);
    const auto it = paw_p4_cache.find(tlut_data);
    if (it != paw_p4_cache.end()) {
        return {it->second.packed, it->second.levels};
    }

    constexpr int ROWS = 32768;
    std::vector<uint16_t> host(8*ROWS);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tlut_data, host.size()*sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float levels[16];
    int   n_levels = 0;
    bool  ok = true;
    for (size_t i = 0; i < host.size(); ++i) {
        const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[i]);
        if (v != v) {   // NaN can never match a level slot
            ok = false;
            break;
        }
        int j = 0;
        while (j < n_levels && levels[j] != v) {
            j++;
        }
        if (j == n_levels) {
            if (n_levels == 16) {
                ok = false;
                break;
            }
            levels[n_levels++] = v;
        }
    }

    entry e = {nullptr, nullptr};
    if (ok) {
        std::sort(levels, levels + n_levels);   // deterministic nibble order
        std::vector<uint32_t> packed(ROWS);
        for (int r = 0; r < ROWS; ++r) {
            uint32_t pk = 0;
            for (int c = 0; c < 8; ++c) {
                const float v = GGML_FP16_TO_FP32((ggml_fp16_t) host[8*r + c]);
                uint32_t idx = 0;
                while (levels[idx] != v) {
                    idx++;
                }
                pk |= idx << (4*c);
            }
            packed[r] = pk;
        }
        float lv16[16] = {0.0f};
        for (int i = 0; i < n_levels; ++i) {
            lv16[i] = levels[i];
        }
        void * buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, ROWS*sizeof(uint32_t) + 16*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(buf, packed.data(), ROWS*sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) buf + ROWS*sizeof(uint32_t), lv16,
                                   16*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        e.packed = (const uint32_t *) buf;
        e.levels = (const float *)((const char *) buf + ROWS*sizeof(uint32_t));
    }
    paw_p4_cache.emplace(tlut_data, e);
    if (paw_debug_on() || paw_time_on()) {
        fprintf(stderr, "paw: p4 tlut repack %s (n_levels=%d, tlut=%p)\n",
                ok ? "ENGAGED" : "FALLBACK (fp16 gather)", n_levels, tlut_data);
    }
    return {e.packed, e.levels};
}

// V2 serial walk over register-held trellis words. STEPB/WORDS are template
// constants so the fully-unrolled window indices are compile-time and wl
// stays in registers (kept: <4,32>, demoted: <2,16>). Numerics identical to
// the previous in-loop decode.
template <int STEPB, int WORDS>
static __device__ __forceinline__ void paw_exp_walk_v2_steps(
        const uint16_t * GGML_CUDA_RESTRICT wl,      // [WORDS] registers
        const float    * GGML_CUDA_RESTRICT tlut,    // F32 [512, 2]
        const float    * GGML_CUDA_RESTRICT ub,
        float          * GGML_CUDA_RESTRICT partial) {
#pragma unroll
    for (int i = 0; i < 128; ++i) {
        // direct-window state: 16 bits at stream bit STEPB*i (wrapping)
        const int      bb  = STEPB*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < WORDS ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) wl[wi] << 16) | (uint32_t) wl[wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        float v0 = tlut[2*row + 0];
        const float v1raw = tlut[2*row + 1];
        if (ph & 0x8000u) {
            v0 = -v0;                         // exact either side of the round
        }
        // hatWr is defined at fp16 precision — pinned round-trip
        const float w0 = __half2float(__float2half_rn(v0));
        const float w1 = __half2float(__float2half_rn(v1raw));
        const int ri = (2*i) >> 4;
        const int ci = (2*i) & 15;
        partial[ri] += w0*ub[ci] + w1*ub[ci + 1];
    }
}

// V8 = false: payload v2, V=2, K=2 kept (32 words) / K=1 demoted (16 words),
//             tlut F32 [512,2], row = (p>>6)&511, per-value fp16 round here.
// V8 = true:  payload v3, V=8, K=1.5 (24 words, 12 fresh bits per step), tlut
//             pre-rounded F16 [32768,8], row = p&0x7FFF, per-tile wave gamma
//             applied after the round, no demoted tier.
// P4 (V8 only): gather one nibble-packed u32 per state from the repacked
//             table and resolve levels from shared — bit-identical values.
// --- grid compaction for exp_walk at nt==1, GGML_PAW_EXP_WALK_COMPACT=1 ---
//
// consolidated P4 tables in a single allocation so an L2 persisting window
// can pin them: the fused kernel's gathers were going to DRAM whenever the
// live attention/KV traffic evicted the tables between launches
struct paw_l2_tables {
    uint32_t * packed = nullptr;
    float    * levels = nullptr;
    void     * base   = nullptr;
};
static paw_l2_tables g_paw_l2tab;

static const paw_l2_tables & paw_l2_tables_get(cudaStream_t stream,
        const paw_p4_table & t) {
    if (!g_paw_l2tab.base && t.packed) {
        constexpr size_t packed_b = (size_t) 32768*4;
        constexpr size_t levels_b = 16*4;
        void * base = nullptr;
        if (cudaMalloc(&base, packed_b + levels_b) == cudaSuccess) {
            CUDA_CHECK(cudaMemcpyAsync(base, t.packed, packed_b,
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync((char *) base + packed_b, t.levels,
                levels_b, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr = base;
            attr.accessPolicyWindow.num_bytes = packed_b + levels_b;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream,
                cudaStreamAttributeAccessPolicyWindow, &attr);
            size_t carve = 0;
            cudaDeviceGetLimit(&carve, cudaLimitPersistingL2CacheSize);
            if (carve < packed_b + levels_b) {
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                    packed_b + levels_b);
            }
            g_paw_l2tab.packed = (uint32_t *) base;
            g_paw_l2tab.levels = (float *)((char *) base + packed_b);
            g_paw_l2tab.base = base;
        }
    }
    return g_paw_l2tab;
}


// The walk kernels below launch grid.z=n_groups (up to 256) but at nt==1
// only n_used (~8) are ever active -- every other block reads scr_i[g]==0
// and exits immediately. This writes the n_used active group ids directly
// (no scan, no early-exit blocks needed) so the dispatcher can launch
// grid.z=n_used instead. Unlike the (reverted) per-slot bank cache, this
// changes NOTHING about what gets decoded or how -- same fused decode+
// accumulate, same scr_i-indexed arrays, just fewer blocks that would have
// scanned-and-skipped. Safe at nt==1 specifically because top-k routing
// guarantees no expert appears twice in one token's own routing list, so
// there's no cnt>1 sharing case this bypasses.
static __global__ void paw_exp_active_groups_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT active_g,
        const int n_used, const int n_kept, const int ids_s0) {
    const int s = threadIdx.x;
    if (s >= n_used) {
        return;
    }
    active_g[s] = (int32_t) paw_group_of(remap, ids, s, n_used, n_kept, ids_s0, 0);
}


// The tile's trellis words and (for V8) the whole per-step state stream are
// pair-invariant, so they are hoisted out of the pair loop (for V2 the full
// 128-step state precompute would cost ~64+ registers, so only the words are
// hoisted there).
template <bool V8, bool P4>
static __global__ void paw_exp_walk_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const uint16_t * GGML_CUDA_RESTRICT dem,     // placeholder (= kept) when absent
        const void     * GGML_CUDA_RESTRICT tlut,    // float (V2) or half (V8)
        const uint32_t * GGML_CUDA_RESTRICT p4,      // P4 only, nullptr otherwise
        const float    * GGML_CUDA_RESTRICT p4lv,    // P4 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const half     * GGML_CUDA_RESTRICT gamma,   // V8 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT active_g, // compacted group ids, nullptr = scan all n_groups
        const int m, const int n, const int n_kept, const int n_groups) {
    static_assert(V8 || !P4, "P4 is a V8-tlut repack");
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float lv[16];       // P4 level table

    (void) dem;
    (void) gamma;
    (void) p4;
    (void) p4lv;

    const int g   = active_g != nullptr ? active_g[blockIdx.z] : blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    int              words;
    int              stepb;        // fresh bits per step
    const uint16_t * trd;
    int64_t          tbase;
    float            gsc = 0.0f;
    if constexpr (V8) {
        words = 24;
        stepb = 12;
        trd   = kept;
        tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;
        // per-tile wave gamma: last anti-diagonal wavefront writing tile (tr, tid)
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        if (have) {
            const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                     : Mb + tiles_y - 2 - (tr + tid);
            gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
    } else {
        const bool is_dem = g >= n_kept;
        const int  ei     = is_dem ? g - n_kept : g;
        words = is_dem ? 16 : 32;
        stepb = is_dem ? 2 : 4;
        trd   = is_dem ? dem : kept;
        tbase = ((int64_t) ei*ntiles + (int64_t) tr*tiles_y)*words;
    }
    (void) stepb;

    // V2 stages the tile words in registers (pair-invariant hoist); V8's
    // window state is recomputed per-qq instead of cached (see exp_walk2's
    // matching change -- caching phv[32] only pays off when cnt is large,
    // which it typically isn't at decode; w8[24] is the actual global read
    // so that alone stays resident).
    uint16_t w8[V8 ? 24 : 1];
    uint16_t wl[V8 ? 1 : 32];
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
        if constexpr (V8) {
#pragma unroll
            for (int q = 0; q < 24; ++q) {
                w8[q] = trd[tw + q];
            }
        } else {
#pragma unroll
            for (int q = 0; q < 32; ++q) {   // predicated: no read past 16-word dem tiles
                wl[q] = q < words ? trd[tw + q] : (uint16_t) 0;
            }
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    // one pair per iteration on purpose — see the MEASURED note in
    // paw_exp_walk.comp: blocking pairs 2-at-a-time regressed 3x.
    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + tid*16;
            if constexpr (V8) {
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int      bb  = 12*i;
                    const int      wi  = bb >> 4;
                    const int      o   = bb & 15;
                    const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                    const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                    const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
                    const uint32_t ph  = reg*(reg + 1u);
                    const uint32_t row = ph & 0x7FFFu;
                    const int      ri  = (8*i) >> 4;
                    const int      ci  = (8*i) & 15;
                    float dotp = 0.0f;
                    if constexpr (P4) {
                        const uint32_t pk = p4[row];
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            // exact fp32 of the fp16 entry — bit-identical to
                            // the __half2float gather; gamma order unchanged
                            float vv = lv[(pk >> (4*c)) & 0xFu];
                            if (c == 0 && (ph & 0x8000u)) {
                                vv = -vv;                 // exact (sign bit)
                            }
                            dotp += (vv * gsc) * ub[ci + c];
                        }
                    } else {
                        const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            // tlut is pre-rounded fp16; wave gamma then
                            // multiplies in fp32 (reference op order)
                            float vv = __half2float(tl[c]);
                            if (c == 0 && (ph & 0x8000u)) {
                                vv = -vv;                 // exact in fp16 (sign bit)
                            }
                            dotp += (vv * gsc) * ub[ci + c];
                        }
                    }
                    partial[ri] += dotp;
                }
            } else {
                // stepb is block-uniform (kept vs demoted group); branch so
                // the unrolled walk sees compile-time window indices
                if (words == 32) {
                    paw_exp_walk_v2_steps<4, 32>(wl, (const float *) tlut, ub, partial);
                } else {
                    paw_exp_walk_v2_steps<2, 16>(wl, (const float *) tlut, ub, partial);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(partial[i]);
            if (lane == 0) {
                red[i][wid] = s;
            }
        }
        __syncthreads();
        if (tid < 16) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[tid][wj];
            }
            scr_v[(int64_t) p*m + tr*16 + tid] = sum;
        }
        if (qq + 1 < cnt) __syncthreads();   // red is reused by the next pair -- skip on last iter, nothing left to protect
    }
}

// V8 fused walk for narrow inputs (tiles_y <= warp_size): one WARP per row
// tile, lanes = column tiles, 4 row tiles per block — the block-per-row-tile
// kernel above leaves (WG - tiles_y) of its threads idle in the walk, which
// wastes 3/4 of the block for the real model's down projection (n = 512).
// Per-(row tile, column tile) state math, gathers, wave gamma and per-pair
// accumulation order are identical to paw_exp_walk_kernel<true, P4>; the
// warp reduction is the same shuffle tree, and the cross-warp shared step it
// replaces only ever added exact zeros there.
template <bool P4>
static __global__ void paw_exp_walk_v8_warp_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,    // half [32768, 8]
        const uint32_t * GGML_CUDA_RESTRICT p4,      // P4 only, nullptr otherwise
        const float    * GGML_CUDA_RESTRICT p4lv,    // P4 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const half     * GGML_CUDA_RESTRICT gamma,
        const int32_t  * GGML_CUDA_RESTRICT remap,
        const int32_t  * GGML_CUDA_RESTRICT ids,
        const int m, const int n, const int n_groups,
        const int n_used, const int n_kept, const int ids_s0) {
    constexpr int warp_size = 32;   // caller guards tiles_y <= 32
    __shared__ float lv[16];

    (void) p4;
    (void) p4lv;

    const int g   = remap != nullptr ? (int) paw_group_of(remap, ids, blockIdx.z,
                                                            n_used, n_kept, ids_s0, 0)
                                     : blockIdx.z;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const int  lane    = tid % warp_size;
    const int  tr      = blockIdx.x*4 + tid / warp_size;   // 4 warps = 4 row tiles
    const bool have    = lane < tiles_y;                    // tr < m/16 by grid

    constexpr int words = 24;   // V = 8, K = 1.5
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;

    // per-tile wave gamma: same closed form as the block kernel
    const int e_orig = scr_i[2*n_groups + g];
    const int Mb     = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + lane <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + lane)
                                                  : Mb + tiles_y - 2 - (tr + lane);
        gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
    }

    // w8 (the real global read) stays resident; phv is recomputed per-qq
    // instead of cached (see exp_walk2's matching change -- caching all 32
    // step states only pays off when cnt is large, which it typically
    // isn't at decode).
    uint16_t w8[24];
    if (have) {
        const int64_t tw = tbase + (int64_t) lane*words;
#pragma unroll
        for (int q = 0; q < 24; ++q) {
            w8[q] = kept[tw + q];
        }
    }

    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + lane*16;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
                const uint32_t ph  = reg*(reg + 1u);
                const uint32_t row = ph & 0x7FFFu;
                const int      ri  = (8*i) >> 4;
                const int      ci  = (8*i) & 15;
                float dotp = 0.0f;
                if constexpr (P4) {
                    const uint32_t pk = p4[row];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = lv[(pk >> (4*c)) & 0xFu];
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;                 // exact (sign bit)
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                } else {
                    const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = __half2float(tl[c]);
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;                 // exact in fp16 (sign bit)
                        }
                        dotp += (vv * gsc) * ub[ci + c];
                    }
                }
                partial[ri] += dotp;
            }
        }
        float sums[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sums[i] = warp_reduce_sum<warp_size>(partial[i]);
        }
        if (lane == 0) {
            float * vo = scr_v + (int64_t) p*m + tr*16;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                vo[i] = sums[i];
            }
        }
        // no __syncthreads(): warps share nothing across pairs
    }
}

// dense prefill: when P = n_used*n_tok is large, most pairs share a storage
// group and the fused walk re-decodes each group's trellis once per pair.
// Instead decode every ACTIVE group (cnt > 0) once into an fp16 bank
// [n_groups, m, n] and apply it with plain fp32 dots. The bank holds the
// exact hatWr halves of the fused path (V2 pins the fp16 round; V8 stores
// the pre-rounded tlut entries UNSCALED — wave gamma is applied in the
// apply kernel, folded into u per column tile, so no extra fp16 rounding
// enters the chain). exp_group/exp_u/exp_out are untouched.

template <bool V8, bool P4>
static __global__ void __launch_bounds__(256, 8) paw_exp_dense_decode_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const uint16_t * GGML_CUDA_RESTRICT dem,     // placeholder (= kept) when absent
        const void     * GGML_CUDA_RESTRICT tlut,    // float (V2) or half (V8)
        const uint32_t * GGML_CUDA_RESTRICT p4,      // P4 only, nullptr otherwise
        const float    * GGML_CUDA_RESTRICT p4lv,    // P4 only, nullptr otherwise
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        half           * GGML_CUDA_RESTRICT bank,    // [n_groups, m, n]
        const int m, const int n, const int n_kept, const int n_groups,
        const int g0,                                 // first group of this slab
        const half    * GGML_CUDA_RESTRICT wgamma) { // V8 wave gamma or nullptr
    static_assert(V8 || !P4, "P4 is a V8-tlut repack");
    constexpr int WG = 256;
    __shared__ float lv[16];

    (void) dem;
    (void) p4;
    (void) p4lv;
    (void) n_groups;

    const int g   = g0 + (int) blockIdx.z;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (scr_i[g] == 0) {           // block-uniform: skip inactive groups
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: adjacent threads write adjacent 16-half chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;
    const int tile   = tr*tiles_y + tc;
    const int ntiles = (m/16)*tiles_y;

    int              words;
    int              stepb;
    const uint16_t * trd;
    int64_t          tbase;
    if constexpr (V8) {
        words = 24;
        stepb = 12;
        trd   = kept;
        tbase = ((int64_t) g*ntiles + tile)*words;
    } else {
        const bool is_dem = g >= n_kept;
        const int  ei     = is_dem ? g - n_kept : g;
        words = is_dem ? 16 : 32;
        stepb = is_dem ? 2 : 4;
        trd   = is_dem ? dem : kept;
        tbase = ((int64_t) ei*ntiles + tile)*words;
    }

    half * dst = bank + ((int64_t)(g - g0)*m + tr*16 + rr)*n + tc*16;
    half tmp[16];

    if constexpr (V8) {
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {     // tile row rr = states 2*rr, 2*rr+1
            const int      i   = 2*rr + jj;
            const int      bb  = 12*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) trd[tbase + wi] << 16) | (uint32_t) trd[tbase + wn];
            const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = ph & 0x7FFFu;
            if constexpr (P4) {
                const uint32_t pk = p4[row];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = lv[(pk >> (4*c)) & 0xFu];
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;
                    }
                    // vv is an exact fp16 value: the round-trip stores its bits
                    tmp[8*jj + c] = __float2half_rn(vv);
                }
            } else {
                const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float vv = __half2float(tl[c]);
                    if (c == 0 && (ph & 0x8000u)) {
                        vv = -vv;                         // exact in fp16 (sign bit)
                    }
                    tmp[8*jj + c] = __float2half_rn(vv);
                }
            }
        }
    } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {        // tile row rr = steps 8*rr .. 8*rr+8
            const int      i   = 8*rr + s;
            const int      bb  = stepb*i;
            const int      wi  = bb >> 4;
            const int      o   = bb & 15;
            const int      wn  = wi + 1 < words ? wi + 1 : 0;
            const uint32_t w2  = ((uint32_t) trd[tbase + wi] << 16) | (uint32_t) trd[tbase + wn];
            const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
            const uint32_t ph  = reg*(reg + 1u);
            const uint32_t row = (ph >> 6) & 511u;
            const float * tl = (const float *) tlut;
            float v0 = tl[2*row + 0];
            const float v1 = tl[2*row + 1];
            if (ph & 0x8000u) {
                v0 = -v0;                                 // exact either side of the round
            }
            // hatWr is defined at fp16 precision — the fused path's pinned
            // round IS this store
            tmp[2*s + 0] = __float2half_rn(v0);
            tmp[2*s + 1] = __float2half_rn(v1);
        }
    }
    if (V8 && wgamma != nullptr) {
        // fold the wave gamma into the weights so the apply runs as a plain
        // GEMM; product terms match folding it into the activations up to a
        // contractible fp16 rounding site
        const int e_ori = scr_i[2*n_groups + g];
        const int Mb      = m/16;
        const int tiles_y = n/16;
        const int wv = (tr + tc <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tc)
                                                : Mb + tiles_y - 2 - (tr + tc);
        const float gv = __half2float(wgamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            tmp[c] = __float2half_rn(gv*__half2float(tmp[c]));
        }
    }
    paw_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}


// warp-per-tile variant of paw_exp_dense_decode_kernel (V8 layout): the 16
// rows of a tile all live in the same 24 stream words, so the warp stages
// them in shared once instead of every lane re-reading them from global;
// lanes 0..15 then decode one row each off shared. Removes the redundant
// global traffic and the per-row serial load chain that starved the old
// mapping (measured ~3.4 GB/s effective on RTX 3060).
template <bool P4>
static __global__ void paw_exp_dense_decode_kernel_v2(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        half           * GGML_CUDA_RESTRICT bank,
        const int m, const int n, const int n_groups,
        const int g0,
        const half    * GGML_CUDA_RESTRICT wgamma) {
    constexpr int WARPS = 8;
    constexpr int WG    = 24;                    // words per tile
    __shared__ float    lv[16];
    __shared__ uint16_t wsh[WARPS][WG];

    const int g    = g0 + (int) blockIdx.z;
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;

    ggml_cuda_pdl_sync();
    if (scr_i[g] == 0) {                         // block-uniform
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int ntiles  = (m / 16)*tiles_y;
    const int tile    = blockIdx.x*WARPS + wid;
    if (tile >= ntiles) {
        return;
    }

    // stage this tile's stream words once per warp
    if (lane < WG) {
        wsh[wid][lane] = kept[((int64_t) g*ntiles + tile)*WG + lane];
    }
    __syncwarp();

    half tmp16[16];

    const int tr   = tile / tiles_y;
    const int tc   = tile % tiles_y;
    const int rr   = lane & 15;
    half * dst = bank + ((int64_t)(g - g0)*m + tr*16 + rr)*n + tc*16;

#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int      i   = 2*rr + jj;
        const int      bb  = 12*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < WG ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) wsh[wid][wi] << 16) | (uint32_t) wsh[wid][wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t row = ph & 0x7FFFu;
        if constexpr (P4) {
            const uint32_t pk = p4[row];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = lv[(pk >> (4*c)) & 0xFu];
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp16[jj*8 + c] = __float2half_rn(vv);
            }
        } else {
            const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = __half2float(tl[c]);
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp16[jj*8 + c] = __float2half_rn(vv);
            }
        }
    }
    if (wgamma != nullptr) {
        const int e_ori = scr_i[2*n_groups + g];
        const int Mb    = m/16;
        const int wv    = (tr + tc <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tc)
                                                   : Mb + tiles_y - 2 - (tr + tc);
        const float gv = __half2float(wgamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            tmp16[c] = __float2half_rn(gv*__half2float(tmp16[c]));
        }
    }
    paw_store_half16(dst, tmp16);
}




// decode one 16-row weight strip of one k-tile straight into shared memory
template <bool P4>
static __device__ __forceinline__ void paw_fused_decode_strip(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * lv,
        const half     * GGML_CUDA_RESTRICT gamma,
        const int g, const int tr_r, const int tc,
        const int tiles_y, const int ntiles, const int m, const int n,
        const int n_groups, const int e_ori,
        uint16_t * wsh, half (* wout)[16], const int lane) {
    const int tile  = tr_r*tiles_y + tc;
    const int tbase = ((int64_t) g*ntiles + tile)*24;
    if (lane < 24) {
        wsh[lane] = kept[tbase + lane];
    }
    __syncwarp();
    const int rr  = lane & 15;
    const int Mb  = m/16;
    const int wv  = (tr_r + tc <= tiles_y - 1)
                        ? Mb + tiles_y - 1 - (tr_r + tc)
                        : Mb + tiles_y - 2 - (tr_r + tc);
    const float gv =
        __half2float(gamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
    if (lane >= 16) {
        return;
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int      i   = 2*rr + jj;
        const int      bb  = 12*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) wsh[wi] << 16) | (uint32_t) wsh[wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t lrow = ph & 0x7FFFu;
        if constexpr (P4) {
            const uint32_t pk = p4[lrow];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = lv[(pk >> (4*c)) & 0xFu];
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                wout[rr][8*jj + c] =
                    __float2half_rn(gv*__half2float(__float2half_rn(vv)));
            }
        } else {
            const half * tl = (const half *) tlut + 8*(int64_t) lrow;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = __half2float(tl[c]);
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                wout[rr][8*jj + c] =
                    __float2half_rn(gv*__half2float(__float2half_rn(vv)));
            }
        }
    }
}

// Fully-fused expert apply: the WS apply skeleton, but the weight tiles are
// decoded from the QTIP stream directly into shared memory right before the
// wmma load -- no fp16 bank materialization at all. Kills the ~2 GB/pass
// bank write+readback of the decode->GEMM pipeline and overlaps the LUT
// gather latency with tensor-core work.
template <bool P4, int BMT>
static __global__ void __launch_bounds__(128, 4) paw_exp_apply_kernel_fused(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const half     * GGML_CUDA_RESTRICT bank,   // debug: pre-decoded weights
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const half     * GGML_CUDA_RESTRICT xg,      // [P, n] grouped fp16
        float          * GGML_CUDA_RESTRICT scr_v,   // [P, m]
        const half     * GGML_CUDA_RESTRICT gamma,   // V8 wave gamma
        float          * GGML_CUDA_RESTRICT dbg,     // debug dump or nullptr
        const int m, const int n, const int n_groups) {
    using namespace nvcuda;

    constexpr int n_warps  = 4;
    constexpr int bk       = 16;                  // == the wmma K tile
    constexpr int bn       = 64;                  // token chunk, whole block
    constexpr int n_tt     = bn / 16;             // token subtiles per chunk

    const int g    = blockIdx.y;
    const int row0 = blockIdx.x*(n_warps*16);
    const int tid  = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int tiles_y = n/16;
    const int ntiles  = (m/16)*tiles_y;

    __shared__ half     Xsh[bn][bk];              // staged activation tile
    // double-buffered per-warp weight strips: tile k+1 decodes while tile k
    // is still being multiplied, hiding LUT gather latency behind tensor math
    __shared__ half     Wsh[2][n_warps][16][bk];
    __shared__ float    out_sh[n_warps][16*16];
    __shared__ int      pidx_sh[bn];
    __shared__ uint16_t wsh[n_warps][24];
    __shared__ float    lv[16];                   // staged P4 levels

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {
        return;
    }
    const int off   = scr_i[n_groups + g];
    const int e_ori = scr_i[2*n_groups + g];
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[n_tt];

    for (int q0 = 0; q0 < cnt; q0 += bn) {
#pragma unroll
        for (int t = 0; t < n_tt; ++t) {
            wmma::fill_fragment(acc[t], 0.0f);
        }
        for (int i = tid; i < bn; i += 128) {
            pidx_sh[i] = scr_i[4*n_groups + off + min(q0 + i, cnt - 1)];
        }
        __syncthreads();

        // prologue: decode strip k=0 into buffer 0 before the loop
        int cur = 0;
        paw_fused_decode_strip<P4>(kept, tlut, p4, lv, gamma,
            g, row0/16 + warp_id, 0, tiles_y, ntiles, m, n, n_groups, e_ori,
            wsh[warp_id], Wsh[0][warp_id], lane);

        for (int k0 = 0; k0 < n; k0 += bk, cur ^= 1) {
            // kick off the next strip into the other buffer; its LUT
            // gathers then overlap with this tile's mma chain
            if (k0 + bk < n) {
                paw_fused_decode_strip<P4>(kept, tlut, p4, lv, gamma,
                    g, row0/16 + warp_id, (k0 + bk)/16, tiles_y, ntiles, m, n,
                    n_groups, e_ori, wsh[warp_id], Wsh[cur ^ 1][warp_id], lane);
            }

            __syncwarp();
            wmma::load_matrix_sync(a_frag, &Wsh[cur][warp_id][0][0], bk);

            if (dbg && g == 0 && blockIdx.x == 0 && q0 == 0 && k0 == 0) {
                for (int idx = lane; idx < 16*bk; idx += 32) {
                    const int rrr = idx / bk;
                    const int kkk = idx % bk;
                    dbg[(warp_id*16 + rrr)*bk + kkk] =
                        __half2float(Wsh[cur][warp_id][rrr][kkk]);
                }
                __syncwarp();
            }

            // vectorized activation staging: 16B per lane, tail rows clamp
            // onto the last valid pair (store guards discard those slots)
            {
                const int last = off + cnt - 1;
                uint4 * xd = (uint4 *) &Xsh[0][0];
                for (int idx = tid; idx < bn*(bk/8); idx += 128) {
                    const int tt = idx / (bk/8);
                    const int vq = off + q0 + tt;
                    const uint4 * xs = (const uint4 *)
                        (xg + (int64_t) min(vq, last)*n + k0);
                    xd[idx] = xs[idx % (bk/8)];
                }
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < n_tt; ++t) {
                wmma::load_matrix_sync(b_frag, &Xsh[t*16][0], bk);
                wmma::mma_sync(acc[t], a_frag, b_frag, acc[t]);
            }
        }

        // store this warp's strip for every token subtile
#pragma unroll
        for (int t = 0; t < n_tt; ++t) {
            wmma::store_matrix_sync(&out_sh[warp_id][0], acc[t], 16,
                                    wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16*16; idx += 32) {
                const int row = idx / 16;
                const int tt  = idx % 16;
                if (q0 + t*16 + tt < cnt) {
                    scr_v[(int64_t) pidx_sh[t*16 + tt]*m + row0 + warp_id*16 + row] =
                        out_sh[warp_id][idx];
                }
            }
            __syncwarp();
        }
        // block-wide barrier before the next chunk refills pidx_sh: the
        // store loop above reads it under a mere per-warp sync
        __syncthreads();
    }
}

// ---- decode-kernel microbench (dev harness, not used by inference) -------
// Launches the v1 and v2 dense-decode kernels on synthetic data at real
// shapes, checks bit-exact agreement of the banks, and prints timings.


extern "C" void paw_decode_bench(int m, int n, int n_groups, int iters) {
    const size_t words = (size_t) n_groups*(m/16)*(n/16)*24;
    const size_t tlut_n = 32768*8;
    uint16_t * kept;  CUDA_CHECK(cudaMalloc(&kept,  words*2));
    half     * tlut;  CUDA_CHECK(cudaMalloc(&tlut,  tlut_n*2));
    uint32_t * p4t;   CUDA_CHECK(cudaMalloc(&p4t,   32768*4));
    float    * p4l;   CUDA_CHECK(cudaMalloc(&p4l,   16*4));
    int32_t  * scr;   CUDA_CHECK(cudaMalloc(&scr,   (size_t) 3*n_groups*4));
    half     * b1;    CUDA_CHECK(cudaMalloc(&b1,   (size_t) n_groups*m*n*2));
    half     * b2;    CUDA_CHECK(cudaMalloc(&b2,   (size_t) n_groups*m*n*2));
    half     * gam;   CUDA_CHECK(cudaMalloc(&gam,   (size_t) 256*(m/16 + n/16)*2));

    // deterministic pseudo-random fill
    std::vector<uint16_t> hk(words);
    unsigned rng = 12345;
    for (auto & w : hk) { rng = rng*1103515245u + 12345u; w = (uint16_t)(rng >> 9); }
    CUDA_CHECK(cudaMemcpy(kept, hk.data(), words*2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(tlut, 0, tlut_n*2));
    CUDA_CHECK(cudaMemset(p4t, 0, 32768*4));
    std::vector<float> hl(16, 0.5f);
    CUDA_CHECK(cudaMemcpy(p4l, hl.data(), 64, cudaMemcpyHostToDevice));
    std::vector<int32_t> hs(3*(size_t) n_groups, 8);   // all groups active
    CUDA_CHECK(cudaMemcpy(scr, hs.data(), hs.size()*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gam, 0, (size_t) 256*(m/16 + n/16)*2));

    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    half * gnullptr_h = nullptr;

    auto launch_v1 = [&](half * bank) {
        if (m % 16 == 0 && n % 16 == 0) {
            paw_launch(paw_exp_dense_decode_kernel<true, true>,
                ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups), dim3(256,1,1), 0, st),
                kept, kept, (const void *) tlut, p4t,
                p4l, scr, bank, m, n, n_groups, n_groups, 0,
                (const half *) gam);
        }
    };
    auto launch_v2 = [&](half * bank) {
        paw_launch(paw_exp_dense_decode_kernel_v2<true>,
            ggml_cuda_kernel_launch_params(dim3(((m/16)*(n/16) + 7)/8, 1, n_groups), dim3(256,1,1), 0, st),
            kept, (const void *) tlut, p4t,
            p4l, scr, bank, m, n, n_groups, 0,
            (const half *) gam);
    };

    // correctness: one pass each, compare banks
    CUDA_CHECK(cudaMemset(b1, 0xaa, (size_t) n_groups*m*n*2));
    CUDA_CHECK(cudaMemset(b2, 0xbb, (size_t) n_groups*m*n*2));
    launch_v1(b1); launch_v2(b2);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<half> h1((size_t) n_groups*m*n), h2((size_t) n_groups*m*n);
    CUDA_CHECK(cudaMemcpy(h1.data(), b1, h1.size()*2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), b2, h2.size()*2, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < h1.size(); ++i) {
        if (((uint16_t*)h1.data())[i] != ((uint16_t*)h2.data())[i]) ++diff;
    }

    // timing
    float ms1 = -1.f, ms2 = -1.f;
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int i = 0; i < iters; ++i) { launch_v1(b1); }
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms1, e0, e1));
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int i = 0; i < iters; ++i) { launch_v2(b2); }
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms2, e0, e1));
    (void) gnullptr_h;
    printf("decode-bench m=%d n=%d groups=%d | v1 %.3f ms | v2 %.3f ms | speedup %.2fx | bitdiff %zu / %zu\n",
           m, n, n_groups, ms1/iters, ms2/iters, ms1/ms2, diff, h1.size());
    fflush(stdout);

    cudaStreamDestroy(st);
    cudaFree(kept); cudaFree(tlut); cudaFree(p4t); cudaFree(p4l);
    cudaFree(scr); cudaFree(b1); cudaFree(b2); cudaFree(gam);
}

// grid (m/16, n_groups): each block owns 16 output rows of one group's bank
// tile and loops the group's pair list (group-ordered, so the 16 x n slice
// stays hot in L2 across pairs). V8 folds wave gamma into u per column tile
// (gamma is constant within a tile and W*(g*u) == (g*W)*u up to contractible
// fp32 association).
// Pairs are processed PC at a time: the 16-row W column slice is loaded into
// registers once per column iteration and reused across the chunk's pairs,
// amortizing the dominant bank traffic PC-fold (same trick as
// paw_rt_apply_kernel's token chunk). Per-pair accumulation order over the
// columns is unchanged (same WG stride, same warp reduction), so numerics are
// identical to the unchunked kernel; the tail chunk clamps to the last pair
// and write-guards (duplicate compute, no duplicate store). PC=1 reproduces
// the unchunked kernel exactly; GGML_PAW_EXP_PC=1 selects it (A/B knob).
template <bool V8, int PC>
static __global__ void paw_exp_apply_kernel(
        const half    * GGML_CUDA_RESTRICT bank,     // [n_groups, m, n]
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const float   * GGML_CUDA_RESTRICT scr_u,    // [P, n]
        float         * GGML_CUDA_RESTRICT scr_v,    // [P, m]
        const half    * GGML_CUDA_RESTRICT gamma,    // V8 only, nullptr otherwise
        const int m, const int n, const int n_groups) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[PC][16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float gsh[128];         // V8: per-column-tile wave gamma (tiles_y <= 128)

    (void) gamma;

    const int g   = blockIdx.y;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    const int off     = scr_i[n_groups + g];
    const int tiles_y = n / 16;

    if constexpr (V8) {
        // same closed-form wave index as the fused walk, per column tile
        const int e_orig = scr_i[2*n_groups + g];
        const int Mb     = m / 16;
        for (int i = tid; i < tiles_y; i += WG) {
            const int wv = (tr + i <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + i)
                                                   : Mb + tiles_y - 2 - (tr + i);
            gsh[i] = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
        }
        __syncthreads();
    }

    const half * W = bank + ((int64_t) g*m + tr*16)*n;

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    for (int q0 = 0; q0 < cnt; q0 += PC) {
        int pidx[PC];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            pidx[t] = scr_i[4*n_groups + off + min(q0 + t, cnt - 1)];
        }

        float acc[PC][16];
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[t][i] = 0.0f;
            }
        }
        // half2/float2 loads over column pairs (c0 even, so both columns are
        // in the same 16-wide gamma tile). Merging a pair's two terms into
        // one partial is a contractible fp32 reassociation, like the fused
        // walk's own ordering.
        for (int c0 = 2*tid; c0 < n; c0 += 2*WG) {
            float2 wc[16];
#pragma unroll
            for (int ri = 0; ri < 16; ++ri) {
                wc[ri] = __half22float2(*(const half2 *)(W + (int64_t) ri*n + c0));
            }
#pragma unroll
            for (int t = 0; t < PC; ++t) {
                float2 uc = *(const float2 *)(scr_u + (int64_t) pidx[t]*n + c0);
                if constexpr (V8) {
                    const float gv = gsh[c0 >> 4];
                    uc.x *= gv;
                    uc.y *= gv;
                }
#pragma unroll
                for (int ri = 0; ri < 16; ++ri) {
                    acc[t][ri] += wc[ri].x*uc.x + wc[ri].y*uc.y;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < PC; ++t) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float s = warp_reduce_sum<warp_size>(acc[t][i]);
                if (lane == 0) {
                    red[t][i][wid] = s;
                }
            }
        }
        __syncthreads();
        if (tid < PC*16) {
            const int t = tid >> 4;
            const int i = tid & 15;
            if (q0 + t < cnt) {
                float sum = 0.0f;
#pragma unroll
                for (int wj = 0; wj < n_warps; ++wj) {
                    sum += red[t][i][wj];
                }
                scr_v[(int64_t) pidx[t]*m + tr*16 + i] = sum;
            }
        }
        if (q0 + PC < cnt) __syncthreads();   // red is reused by the next chunk -- skip on last iter, nothing left to protect
    }
}



// Groups the routed-pair activation slab by expert: xg[(off_g + j)*n + k] =
// x[plist[off_g + j]*n + k]. Turns the apply kernels' random-row gathers
// into sequential reads; costs one extra pass over P*n floats.
static __global__ void paw_exp_permute_x_kernel(
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const float   * GGML_CUDA_RESTRICT x,
        half          * GGML_CUDA_RESTRICT xg,
        const int n, const int n_groups) {
    const int g  = blockIdx.y;
    const int kg = blockIdx.x*blockDim.x + threadIdx.x;
    if (kg >= n) {
        return;
    }
    const int cnt = scr_i[g];
    const int off = scr_i[n_groups + g];
    const int32_t * plist = scr_i + 4*n_groups;
    // halves on purpose: the ws apply staged these exact conversions anyway,
    // and the slab must fit next to the weights on small cards
    for (int j = 0; j < cnt; ++j) {
        xg[(int64_t)(off + j)*n + kg] = __float2half(x[(int64_t) plist[off + j]*n + kg]);
    }
}


// In-place wave-gamma scale of a decoded group bank: W'[r,k] = W[r,k] * g,
// with the closed-form diagonal wave index of the fused walk. Folding the
// gamma into the weights (instead of the activations) lets the apply run
// as a plain GEMM; the product terms are unchanged up to one fp16 rounding
// site, the same contractible class the dense paths already accept.
static __global__ void paw_exp_scale_bank_gamma_kernel(
        half          * GGML_CUDA_RESTRICT bank,
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const half    * GGML_CUDA_RESTRICT gamma,
        const int m, const int n, const int n_groups) {
    const int g = blockIdx.y;
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m*n) {
        return;
    }
    const int tr_r = (i / n) >> 4;
    const int kt   = (i % n) >> 4;
    const int e_ori = scr_i[2*n_groups + g];
    const int Mb      = m/16;
    const int tiles_y = n/16;
    const int wv = (tr_r + kt <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr_r + kt)
                                              : Mb + tiles_y - 2 - (tr_r + kt);
    const size_t o = (size_t) g*m*n + i;
    bank[o] = __float2half(__half2float(bank[o]) *
                           __half2float(gamma[(int64_t) e_ori*(Mb + tiles_y) + wv]));
}

// inverse of the grouped slab: scr_v[pidx] = yg[off + j]
static __global__ void paw_exp_unpermute_y_kernel(
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const half    * GGML_CUDA_RESTRICT yg,
        float         * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n_groups) {
    const int g   = blockIdx.y;
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col >= m) {
        return;
    }
    const int cnt = scr_i[g];
    const int off = scr_i[n_groups + g];
    const int32_t * plist = scr_i + 4*n_groups;
    for (int j = 0; j < cnt; ++j) {
        scr_v[(int64_t) plist[off + j]*m + col] =
            __half2float(yg[(int64_t)(off + j)*m + col]);
    }
}

// Weight-stationary tensor-core twin of paw_exp_apply_kernel for the dense
// prefill path (V8 and legacy payloads). One persistent block per (group,
// 64-row tile) sweeps that group's routed pairs in chunks of bn pairs; a
// full-K accumulation finishes each chunk before the next starts, so a
// group with cnt <= bn reads its bank strip exactly once instead of
// ceil(cnt/PC) times. The V8 wave gamma is row-strip dependent, so instead
// of folding it into the shared activation slab the B tile is rescaled per
// A-tile into a small scratch buffer -- the association is contractible in
// the same sense the fused walk documents. Requires m % 64 == 0.
template <bool V8, int BMT, bool GX>
static __global__ void paw_exp_apply_kernel_ws(
        const half    * GGML_CUDA_RESTRICT bank,     // [n_groups, m, n]
        const int32_t * GGML_CUDA_RESTRICT scr_i,
        const float   * GGML_CUDA_RESTRICT scr_u,    // [P, n]
        const half    * GGML_CUDA_RESTRICT xg,       // [P, n] grouped fp16 (GX)
        float         * GGML_CUDA_RESTRICT scr_v,    // [P, m]
        const half    * GGML_CUDA_RESTRICT gamma,    // V8 only, else nullptr
        const int m, const int n, const int n_groups) {
    using namespace nvcuda;

    constexpr int n_warps = 4;
    constexpr int bm_t    = BMT;                  // narrow tiles for small m
    constexpr int tpb     = 2;                    // BN = 128 pairs per chunk
    constexpr int bn      = n_warps * tpb * 16;
    constexpr int bk      = 16;                   // == the wmma K tile

    const int g    = blockIdx.y;
    const int row0 = blockIdx.x * (bm_t * 16);
    const int tid  = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int tiles_y = n / 16;
    const int Mb      = m / 16;

    __shared__ half Xsh[n_warps][tpb][16][bk];    // raw gathered activations
    __shared__ half Bsh[n_warps][16][bk];         // gamma-scaled B tile
    __shared__ float out_sh[n_warps][16*16];
    __shared__ int pidx_sh[bn];
    __shared__ int scnt, soff, e_ori;

    ggml_cuda_pdl_sync();
    if (tid == 0) {
        scnt   = scr_i[g];
        soff   = scr_i[n_groups + g];
        e_ori  = V8 ? scr_i[2*n_groups + g] : 0;
    }
    __syncthreads();
    const int cnt = scnt;
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    const int off = soff;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[bm_t][tpb];

    for (int q0 = 0; q0 < cnt; q0 += bn) {
#pragma unroll
        for (int r = 0; r < bm_t; ++r) {
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                wmma::fill_fragment(acc[r][t], 0.0f);
            }
        }
        for (int i = tid; i < bn; i += 128) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            pidx_sh[i] = scr_i[4*n_groups + off + min(q0 + i, cnt - 1)];
        }
        __syncthreads();

        for (int k0 = 0; k0 < n; k0 += bk) {
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                const int tt0 = (warp_id*tpb + t)*16;
                for (int idx = lane; idx < 16*bk; idx += 32) {
                    const int tt = idx / bk;
                    const int kk = idx % bk;
                    const int kg = k0 + kk;
                    float v = 0.0f;
                    if (kg < n && q0 + tt0 + tt < cnt) {
                        // GX: activations arrive pre-grouped (and pre-cast),
                        // rows are read sequentially instead of gathered
                        const int64_t row = GX ? (int64_t)(off + q0 + tt0 + tt)
                                               : (int64_t) pidx_sh[tt0 + tt];
                        v = GX ? __half2float(xg[row*n + kg])
                               : scr_u[(int64_t) pidx_sh[tt0 + tt]*n + kg];
                    }
                    Xsh[warp_id][t][tt][kk] = __float2half(v);
                }
            }
            __syncwarp();
#pragma unroll
            for (int r = 0; r < bm_t; ++r) {
                wmma::load_matrix_sync(a_frag,
                    bank + ((int64_t) g*m + row0 + r*16)*n + k0, n);
                float gv = 1.0f;
                if (V8) {
                    // closed-form wave index of the fused walk, per column
                    // tile, for this block's r-th output strip
                    const int tr_r = row0/16 + r;
                    const int wv = (tr_r + k0/16 <= tiles_y - 1)
                        ? Mb + tiles_y - 1 - (tr_r + k0/16)
                        : Mb + tiles_y - 2 - (tr_r + k0/16);
                    gv = __half2float(gamma[(int64_t) e_ori*(Mb + tiles_y) + wv]);
                }
#pragma unroll
                for (int t = 0; t < tpb; ++t) {
                    if (V8) {
                        for (int idx = lane; idx < 16*bk; idx += 32) {
                            const int tt = idx / bk;
                            const int kk = idx % bk;
                            Bsh[warp_id][tt][kk] = __float2half(
                                gv*__half2float(Xsh[warp_id][t][tt][kk]));
                        }
                        __syncwarp();
                        wmma::load_matrix_sync(b_frag, &Bsh[warp_id][0][0], bk);
                    } else {
                        wmma::load_matrix_sync(b_frag, &Xsh[warp_id][t][0][0], bk);
                    }
                    wmma::mma_sync(acc[r][t], a_frag, b_frag, acc[r][t]);
                }
            }
            __syncwarp();
        }

#pragma unroll
        for (int r = 0; r < bm_t; ++r) {
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                wmma::store_matrix_sync(&out_sh[warp_id][0], acc[r][t], 16, wmma::mem_row_major);
                __syncwarp();
                const int tt0 = (warp_id*tpb + t)*16;
                for (int idx = lane; idx < 16*16; idx += 32) {
                    const int row = idx / 16;
                    const int tt  = idx % 16;
                    if (q0 + tt0 + tt < cnt) {
                        scr_v[(int64_t) pidx_sh[tt0 + tt]*m + row0 + r*16 + row] =
                            out_sh[warp_id][idx];
                    }
                }
                __syncwarp();
            }
        }
    }
}

extern "C" int paw_fused_bench(int m, int n, int n_groups, int P) {
    const size_t words = (size_t) n_groups*(m/16)*(n/16)*24;
    uint16_t * kept;  CUDA_CHECK(cudaMalloc(&kept, words*2));
    half     * tlut;  CUDA_CHECK(cudaMalloc(&tlut, (size_t)32768*8*2));
    uint32_t * p4t;   CUDA_CHECK(cudaMalloc(&p4t,  32768*4));
    float    * p4l;   CUDA_CHECK(cudaMalloc(&p4l,  16*4));
    int32_t  * scr;   CUDA_CHECK(cudaMalloc(&scr,  (size_t)(5*n_groups + P)*4));
    half     * xg;    CUDA_CHECK(cudaMalloc(&xg,   (size_t) P*n*2));
    float    * v;     CUDA_CHECK(cudaMalloc(&v,    (size_t) P*m*4));
    half     * gam;   CUDA_CHECK(cudaMalloc(&gam,  (size_t) 256*(m/16 + n/16)*2));

    std::vector<uint16_t> hk(words);
    {
        unsigned rng = 9973;
        for (size_t i = 0; i < words; ++i) {
            rng = rng*1103515245u + 12345u;
            hk[i] = (uint16_t)((rng >> 8) | 1u);   // nonzero streams
        }
    }
    CUDA_CHECK(cudaMemcpy(kept, hk.data(), words*2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(tlut, 0, (size_t)32768*16));
    CUDA_CHECK(cudaMemset(p4t, 0, 32768*4));
    std::vector<float> hl(16, 0.5f);
    CUDA_CHECK(cudaMemcpy(p4l, hl.data(), 64, cudaMemcpyHostToDevice));
    // group g active with cnt=8, offsets staggered
    const int CNT = getenv("FUSED_CNT") ? atoi(getenv("FUSED_CNT")) : 8;
    std::vector<int32_t> hs(5*(size_t)n_groups + (size_t)(CNT>0?0:P), 0);
    // pair-list region sized by total routed pairs
    hs.resize(5*(size_t)n_groups + (size_t) n_groups*CNT);
    int32_t * pairs = hs.data() + 4*n_groups;
    for (int g = 0; g < n_groups; ++g) {
        hs[g] = CNT;
        hs[n_groups + g] = CNT*g;            // off
        hs[2*n_groups + g] = g % 256;        // e_ori
        for (int q = 0; q < CNT; ++q) {
            pairs[CNT*g + q] = (CNT*g + q) % P;         // pair list
        }
    }
    CUDA_CHECK(cudaMemcpy(scr, hs.data(), hs.size()*4, cudaMemcpyHostToDevice));
    {
        std::vector<half> hx((size_t)P*n);
        unsigned rng = 31;
        for (size_t i = 0; i < hx.size(); ++i) {
            rng = rng*1103515245u + 12345u;
            hx[i] = __float2half(((int)(rng >> 16 & 0xFF) - 128) / 512.0f);
        }
        CUDA_CHECK(cudaMemcpy(xg, hx.data(), hx.size()*2, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(v, 0, (size_t)P*m*4));
    std::vector<half> hg((size_t)256*(m/16 + n/16), __float2half(1.0f));
    CUDA_CHECK(cudaMemcpy(gam, hg.data(), hg.size()*2, cudaMemcpyHostToDevice));

    float * dbgf; CUDA_CHECK(cudaMalloc(&dbgf, 64*16*4));
    CUDA_CHECK(cudaMemset(dbgf, 0, 64*16*4));
    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    paw_launch(paw_exp_apply_kernel_fused<true, 4>,
        ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128,1,1), 0, st),
        kept, (const half *) nullptr, (const void *) tlut, (const uint32_t *) p4t,
        (const float *) p4l, (const int32_t *) scr, xg, v, (const half *) gam,
        (float *) nullptr, m, n, n_groups);
    cudaError_t le = cudaGetLastError();

    // structural twin: same loops, bank-fed weights
    half * bank2; CUDA_CHECK(cudaMalloc(&bank2, (size_t) n_groups*m*n*2));
    float * vf;   CUDA_CHECK(cudaMalloc(&vf, (size_t) P*m*4));
    CUDA_CHECK(cudaMemset(vf, 0, (size_t) P*m*4));
    {
        std::vector<uint16_t> hkz(words, 0x1555u);
        CUDA_CHECK(cudaMemcpy(kept, hkz.data(), words*2, cudaMemcpyHostToDevice));
    }
    paw_launch(paw_exp_dense_decode_kernel<true, true>,
        ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups), dim3(256,1,1), 0, st),
        kept, kept, (const void *) tlut, p4t, p4l,
        (const int32_t *) scr, bank2, m, n, n_groups, n_groups, 0,
        (const half *) gam);
    {
        std::vector<uint16_t> hkr(words);
        unsigned rng = 9973;
        for (size_t i = 0; i < words; ++i) {
            rng = rng*1103515245u + 12345u;
            hkr[i] = (uint16_t)((rng >> 8) | 1u);
        }
        CUDA_CHECK(cudaMemcpy(kept, hkr.data(), words*2, cudaMemcpyHostToDevice));
    }
    paw_launch(paw_exp_apply_kernel_fused<true, 4>,
        ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128,1,1), 0, st),
        kept, (const half *) bank2, (const void *) tlut, p4t, p4l,
        (const int32_t *) scr, xg, vf, (const half *) gam,
        dbgf, m, n, n_groups);

    // reference: decode into a bank, then WS apply over it
    half * bank; CUDA_CHECK(cudaMalloc(&bank, (size_t) n_groups*m*n*2));
    float * vref; CUDA_CHECK(cudaMalloc(&vref, (size_t) P*m*4));
    CUDA_CHECK(cudaMemset(vref, 0, (size_t) P*m*4));
    paw_launch(paw_exp_dense_decode_kernel<true, true>,
        ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups), dim3(256,1,1), 0, st),
        kept, kept, (const void *) tlut, p4t, p4l,
        (const int32_t *) scr, bank, m, n, n_groups, n_groups, 0,
        (const half *) gam);
    paw_launch(paw_exp_apply_kernel_ws<true, 4, true>,
        ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128,1,1), 0, st),
        (const half *) bank, (const int32_t *) scr, nullptr, xg, vref,
        (const half *) gam, m, n, n_groups);

    cudaError_t se = cudaStreamSynchronize(st);
    printf("fused-bench m=%d n=%d groups=%d P=%d | launch=%s sync=%s\n",
           m, n, n_groups, P, cudaGetErrorString(le), cudaGetErrorString(se));
    std::vector<float> hv((size_t)P*m), hr((size_t)P*m), hf((size_t)P*m);
    CUDA_CHECK(cudaMemcpy(hv.data(), v, hv.size()*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hr.data(), vref, hr.size()*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hf.data(), vf, hf.size()*4, cudaMemcpyDeviceToHost));
    size_t fdiff = 0;
    for (size_t i = 0; i < hf.size(); ++i) {
        if (fabs((double)hf[i] - (double)hr[i]) > 0) ++fdiff;
    }
    printf("bankfed-fused diff %zu / %zu\n", fdiff, hf.size());
    {
        std::vector<float> hd(64*16);
        CUDA_CHECK(cudaMemcpy(hd.data(), dbgf, hd.size()*4, cudaMemcpyDeviceToHost));
        std::vector<half> hb((size_t) n_groups*m*n);
        CUDA_CHECK(cudaMemcpy(hb.data(), bank2, hb.size()*2, cudaMemcpyDeviceToHost));
        int shown = 0;
        for (int w = 0; w < 4 && shown < 8; ++w) {
            for (int e = 0; e < 4 && shown < 8; ++e) {
                float got = hd[(w*16 + 0)*16 + e];      // row 0 of strip w, k=e
                float want = __half2float(hb[((size_t)0*m + (0*16 + w*16))*n + e]);
                if (fabs(got - want) > 1e-3) {
                    printf("  dbg warp%d k%d: Wsh=%.4f bank=%.4f\n", w, e, got, want);
                    ++shown;
                }
            }
        }
        if (!shown) printf("  dbg: all sampled W tiles match bank\n");
    }

    // ---- host ground truth over group 0's pairs only (fast enough) ----
    {
        std::vector<half> hb((size_t) n_groups*m*n);
        CUDA_CHECK(cudaMemcpy(hb.data(), bank2, hb.size()*2, cudaMemcpyDeviceToHost));
        std::vector<half> hx((size_t)P*n);
        CUDA_CHECK(cudaMemcpy(hx.data(), xg, hx.size()*2, cudaMemcpyDeviceToHost));
        std::vector<int32_t> hsc(5*(size_t)n_groups + (size_t)n_groups*CNT);
        CUDA_CHECK(cudaMemcpy(hsc.data(), scr, hsc.size()*4, cudaMemcpyDeviceToHost));
        double worst_f = 0, worst_r = 0;
        long long bad_f = 0, bad_r = 0, bad_f_row[4] = {0,0,0,0};
        const int GCHK = n_groups < 4 ? n_groups : 4;
        for (int g = 0; g < GCHK; ++g) {
            const int cg = hsc[g];
            const int og = hsc[n_groups + g];
            const int32_t * pp = hsc.data() + 4*n_groups + og;
            for (int i = 0; i < cg; ++i) {
                const int qp = pp[i];
                for (int rr = 0; rr < 64; rr += 16) {   // strips via blocks; check first block rows 0..63
                    for (int r16 = 0; r16 < 16; ++r16) {
                        const int row = rr + r16;
                        const half * wr = hb.data() + ((size_t) g*m + row)*n;
                        const half * xr = hx.data() + (size_t) qp*n;
                        double acc = 0;
                        for (int k = 0; k < n; ++k) {
                            acc += (double)__half2float(wr[k]) * (double)__half2float(xr[k]);
                        }
                        const size_t idx = (size_t) qp*m + row;
                        double df = fabs(acc - (double)hf[idx]);
                        double dr = fabs(acc - (double)hr[idx]);
                        if (df > worst_f) worst_f = df;
                        if (dr > worst_r) worst_r = dr;
                        if (df > 0.05) { ++bad_f; ++bad_f_row[rr/16]; }
                        if (dr > 0.05) ++bad_r;
                    }
                }
            }
        }
        printf("host-truth: fused bad=%lld worst=%.4f | ref bad=%lld worst=%.4f\n",
               bad_f, worst_f, bad_r, worst_r);
        printf("  fused bad by strip: %lld %lld %lld %lld\n",
               bad_f_row[0], bad_f_row[1], bad_f_row[2], bad_f_row[3]);
    }
    size_t diff = 0; double maxd = 0;
    int shown = 0;
    long long by_strip[8] = {0,0,0,0,0,0,0,0}, by_qmod[8] = {0,0,0,0,0,0,0,0};
    for (size_t i = 0; i < hv.size(); ++i) {
        double d = fabs((double)hv[i] - (double)hr[i]);
        if (d > 0) ++diff;
        if (d > maxd) maxd = d;
        if (d > 0 && shown < 5 && hr[i] != 0.0f) {
            size_t qq = i / m, rr = i % m;
            printf("  mismatch q=%zu row=%zu fused=%.4f ref=%.4f\n",
                   qq, rr, (double)hv[i], (double)hr[i]);
            ++shown;
        }
        if (d > 0) {
            by_strip[(i % m)/16 & 7]++;
            by_qmod[(i / m) & 7]++;
        }
    }
    printf("  by_strip:");
    for (int k2 = 0; k2 < 8; ++k2) printf(" %lld", by_strip[k2]);
    printf("\n  by_qmod: ");
    for (int k2 = 0; k2 < 8; ++k2) printf(" %lld", by_qmod[k2]);
    printf("\n");
    printf("out diff %zu / %zu max=%.6f\n", diff, hv.size(), maxd);
    {
        long long by_strip[8] = {0}, by_qmod[8] = {0};
        for (size_t i = 0; i < hv.size(); ++i) {
            if (((uint32_t)0)) break;
        }
        (void)by_strip; (void)by_qmod;
    }
    fflush(stdout);
    cudaFree(bank); cudaFree(vref);
    cudaStreamDestroy(st);
    cudaFree(kept); cudaFree(tlut); cudaFree(p4t); cudaFree(p4l);
    cudaFree(scr); cudaFree(xg); cudaFree(v); cudaFree(gam);
    return (int)(le != cudaSuccess || se != cudaSuccess);
}

// --- per-routing-slot expert bank cache (nt==1 only), GGML_PAW_EXP_CACHE=1 ---
//
// The dense decode/apply path above already only touches active groups
// (scr_i[g]==0 skips); the actual inefficiency is upstream of it: at nt=1,
// P=n_used (~8) is always below GGML_PAW_DENSE_MIN (default 1024), so the
// walk path runs instead, which fuses decode+apply and throws the decoded
// values away every single token even though routing is sticky (the same
// expert is often reused a few tokens in a row). This caches the decoded
// bank per ROUTING-RANK SLOT s (0..n_used-1), persistent across tokens: slot
// s holds whichever group last occupied rank s, tagged by that group's id.
// If rank s routes to the same group again next token, decode is skipped.
//
// Scoped to v8+p4 (reason8192's runtime config, confirmed ng=256=n_expert
// via GGML_PAW_TIME output -- no expert-group clustering, so group id ==
// remapped expert id, and per-rank caching cannot mis-attribute a pair to
// the wrong group). Falls back to the existing walk/dense path otherwise.
// If a future checkpoint DOES use clustering (n_groups < n_expert), two
// ranks sharing a group would just decode it twice into two slots --
// redundant, not incorrect, since each slot is independently tagged.
//
// Decode math below is a verbatim copy of paw_exp_dense_decode_kernel's
// V8/P4 branch (see its own numeric-identity comments) -- only the outer
// indexing (persistent slot instead of scanned n_groups block, tag-gated
// instead of scr_i-cnt-gated) differs. The tag write is a SEPARATE kernel
// launched after decode (stream-ordered, so it can never observe a
// partially-written bank) rather than done inline, to avoid one block
// racing another block's still-in-flight decode of the same slot.
struct paw_exp_slots {
    half    * banks;     // [n_used, m, n]
    int32_t * tags_dev;  // [n_used], -1 = empty
};
static std::mutex paw_exp_slots_mutex;
static std::unordered_map<const void *, paw_exp_slots> paw_exp_slot_map;

static bool paw_exp_cache_on() {
    static const bool on = paw_env_int("GGML_PAW_EXP_CACHE", 0) != 0;
    return on;
}

static paw_exp_slots paw_exp_slots_get(
        const void * kept, const int m, const int n, const int n_used, cudaStream_t stream) {
    {
        std::lock_guard<std::mutex> lock(paw_exp_slots_mutex);
        auto it = paw_exp_slot_map.find(kept);
        if (it != paw_exp_slot_map.end()) {
            return it->second;
        }
    }
    paw_exp_slots s;
    CUDA_CHECK(cudaMalloc(&s.banks, (size_t) n_used*m*n*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s.tags_dev, (size_t) n_used*sizeof(int32_t)));
    std::vector<int32_t> init_tags(n_used, -1);
    CUDA_CHECK(cudaMemcpyAsync(s.tags_dev, init_tags.data(), n_used*sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        std::lock_guard<std::mutex> lock(paw_exp_slots_mutex);
        auto it = paw_exp_slot_map.find(kept);
        if (it != paw_exp_slot_map.end()) {
            cudaFree(s.banks);
            cudaFree(s.tags_dev);
            return it->second;
        }
        paw_exp_slot_map.emplace(kept, s);
    }
    return s;
}

template <bool P4>
static __global__ void paw_exp_slot_decode_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT remap,
        const int32_t  * GGML_CUDA_RESTRICT ids,
        const int32_t  * GGML_CUDA_RESTRICT tags,     // [n_used]
        half           * GGML_CUDA_RESTRICT banks,    // [n_used, m, n]
        const int m, const int n, const int ids_s0) {
    constexpr int WG = 256;
    __shared__ float lv[16];

    (void) p4; (void) p4lv;

    const int sslot = blockIdx.z;
    const int tid   = threadIdx.x;

    ggml_cuda_pdl_sync();

    const uint32_t id = (uint32_t) ids[sslot*ids_s0];
    const int      g  = (int) (uint32_t) remap[id];   // no dem flag: reason8192 never demotes

    if (tags[sslot] == g) {   // block-uniform: whole block exits together
        return;
    }
    if constexpr (P4) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {
        return;
    }
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;
    const int tile    = tr*tiles_y + tc;
    const int ntiles  = (m/16)*tiles_y;

    const int64_t tbase = ((int64_t) g*ntiles + tile)*24;

    half * dst = banks + ((int64_t) sslot*m + tr*16 + rr)*n + tc*16;
    half tmp[16];

#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int      i   = 2*rr + jj;
        const int      bb  = 12*i;
        const int      wi  = bb >> 4;
        const int      o   = bb & 15;
        const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
        const uint32_t w2  = ((uint32_t) kept[tbase + wi] << 16) | (uint32_t) kept[tbase + wn];
        const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
        const uint32_t ph  = reg*(reg + 1u);
        const uint32_t row = ph & 0x7FFFu;
        if constexpr (P4) {
            const uint32_t pk = p4[row];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = lv[(pk >> (4*c)) & 0xFu];
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp[8*jj + c] = __float2half_rn(vv);
            }
        } else {
            const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float vv = __half2float(tl[c]);
                if (c == 0 && (ph & 0x8000u)) {
                    vv = -vv;
                }
                tmp[8*jj + c] = __float2half_rn(vv);
            }
        }
    }
    paw_store_half16(dst, tmp);
}

// stream-ordered after the decode launch above, before the apply launch
// below: safe to observe a bank as soon as its tag matches, since decode is
// fully complete (same stream) by the time this or apply runs.
static __global__ void paw_exp_slot_tag_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT tags,
        const int n_used, const int ids_s0) {
    const int s = threadIdx.x;
    if (s >= n_used) {
        return;
    }
    const uint32_t id = (uint32_t) ids[s*ids_s0];
    tags[s] = (int32_t) (uint32_t) remap[id];
}

// builds a trivial scr_i (cnt=1, off=s, orig_id=ids[s], pidx=s per slot) so
// the existing, unmodified paw_exp_apply_kernel<true,PC> can run over the
// slot banks as if n_groups==n_used and every "group" has exactly one pair.
static __global__ void paw_exp_slot_scri_kernel(
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT scr_i,
        const int n_used, const int ids_s0) {
    const int s = threadIdx.x;
    if (s >= n_used) {
        return;
    }
    scr_i[s]             = 1;
    scr_i[n_used + s]    = s;
    scr_i[2*n_used + s]  = ids[s*ids_s0];
    scr_i[4*n_used + s]  = s;
}

template <int WG>
static __global__ void paw_exp_out_kernel(
        const half    * GGML_CUDA_RESTRICT sv,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT scr_v,
        float         * GGML_CUDA_RESTRICT dst,
        const int m, const int n_used, const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int p   = t*n_used + s;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    for (int i = tid; i < m; i += WG) {
        __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) p*m + i], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    if (WG == m/16 && paw_fwht_v2_ok(m)) {
        paw_fwht_block_v2(sh, m, tid, WG);
    } else {
        paw_fwht_block(sh, m, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) m);
    const float inv_sc = __frcp_rn(sc);
    const int64_t obase = (int64_t) t*n_used*m + (int64_t) s*m;
    for (int i = tid; i < m; i += WG) {
        dst[obase + i] = sh[i] * inv_sc * __half2float(sv[e*m + i]);
    }
}

// ---------------------------------------------------------------------------
// batched EXP_MM for the gate+up pair (GGML_OP_PAW_EXP_MM_BATCH2,
// GGML_PAW_EXP_BATCH2=1). gate_exps and up_exps share one input (xexp),
// one routing decision (remap/ids), and one model-global tlut/p4 codebook --
// only the per-expert trellis (kept), wave_gamma, and su/sv scale vectors
// differ. Scoped by the caller (paw.cpp) to this checkpoint's actual
// runtime shape: V8+P4, no demotion, no low-rank basis correction,
// decode-only (n_tok==1) -- anything else falls back to two separate
// ggml_paw_exp_mm calls. group (routing) is shared and runs once; u/walk/
// out each run once for the whole pair via an extra grid dimension selecting
// matrix 0 (gate) vs 1 (up). Kernel bodies below are copied verbatim from
// the single-matrix versions per matrix -- see those kernels' comments for
// the math; only the per-matrix pointer/index selection is new, the same
// pattern as paw_exp_group_u_kernel's existing group+u fusion.

template <int WG>
static __global__ void paw_exp_group_u2_kernel(
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        int32_t        * GGML_CUDA_RESTRICT scr,
        int32_t        * GGML_CUDA_RESTRICT active_g,
        const half     * GGML_CUDA_RESTRICT su0,
        const half     * GGML_CUDA_RESTRICT su1,
        const float    * GGML_CUDA_RESTRICT x,
        float          * GGML_CUDA_RESTRICT scr_u0,
        float          * GGML_CUDA_RESTRICT scr_u1,
        const int n, const int n_used, const int n_tok, const int n_kept, const int n_groups,
        const int xne1, const int ids_s0, const int ids_s1) {
    if (blockIdx.x == 0) {
        if (blockIdx.y != 0) {
            return;   // exp_group's result is global, only run it once
        }
        __shared__ int sh_cnt[512];
        __shared__ int sh_cur[512];
        __shared__ int sh_orig[512];

        const int tid = threadIdx.x;
        const int P   = n_used*n_tok;

        ggml_cuda_pdl_sync();
        for (int g = tid; g < n_groups; g += WG) {
            sh_cnt[g]  = 0;
            sh_orig[g] = 0;
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int s = p % n_used;
            const int t = p / n_used;
            const int g = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            atomicAdd(&sh_cnt[g], 1);
            atomicExch(&sh_orig[g], ids[s*ids_s0 + t*ids_s1]);
        }
        __syncthreads();
        if (tid == 0) {
            int off = 0;
            for (int g = 0; g < n_groups; ++g) {
                const int c = sh_cnt[g];
                sh_cur[g] = off;
                scr[g]              = c;
                scr[n_groups + g]   = off;
                scr[2*n_groups + g] = sh_orig[g];
                off += c;
            }
        }
        __syncthreads();
        for (int p = tid; p < P; p += WG) {
            const int g   = (int) paw_group_of(remap, ids, p, n_used, n_kept, ids_s0, ids_s1);
            const int pos = atomicAdd(&sh_cur[g], 1);
            scr[4*n_groups + pos] = p;
        }
        if (active_g != nullptr && tid < n_used) {
            active_g[tid] = (int32_t) paw_group_of(remap, ids, tid, n_used, n_kept, ids_s0, 0);
        }
        return;
    }

    __shared__ float sh[2048];

    const int which = blockIdx.x > n_used ? 1 : 0;
    const int s      = which == 0 ? blockIdx.x - 1 : blockIdx.x - 1 - n_used;
    const int t      = blockIdx.y;
    const int p      = t*n_used + s;
    const int tid    = threadIdx.x;
    const half  * su    = which == 0 ? su0    : su1;
    float       * scr_u = which == 0 ? scr_u0 : scr_u1;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;
    for (int i = tid; i < n; i += WG) {
        sh[i] = __half2float(su[e*n + i]) * x[xbase + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) n);
    const float inv_sc = __frcp_rn(sc);
    for (int i = tid; i < n; i += WG) {
        scr_u[(int64_t) p*n + i] = sh[i] * inv_sc;
    }
}

// V8+P4 only (this batched path never sees the V2 codec or the narrow-input
// warp variant -- gate/up's n=n_embd is always > 512 in this checkpoint).
// blockIdx.y selects gate (0) / up (1); routing (scr_i, active_g) and grid
// geometry are shared since gate/up share one selected_experts/remap.
static __global__ void paw_exp_walk2_kernel(
        const uint16_t * GGML_CUDA_RESTRICT kept0,
        const uint16_t * GGML_CUDA_RESTRICT kept1,
        const void     * GGML_CUDA_RESTRICT tlut,
        const uint32_t * GGML_CUDA_RESTRICT p4,
        const float    * GGML_CUDA_RESTRICT p4lv,
        const int32_t  * GGML_CUDA_RESTRICT scr_i,
        const float    * GGML_CUDA_RESTRICT scr_u0,
        const float    * GGML_CUDA_RESTRICT scr_u1,
        float          * GGML_CUDA_RESTRICT scr_v0,
        float          * GGML_CUDA_RESTRICT scr_v1,
        const half     * GGML_CUDA_RESTRICT gamma0,
        const half     * GGML_CUDA_RESTRICT gamma1,
        const int32_t  * GGML_CUDA_RESTRICT active_g,
        const int m, const int n, const int n_groups) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][4];
    __shared__ float lv[16];

    const bool which = blockIdx.y != 0;
    const uint16_t * kept  = which ? kept1  : kept0;
    const float     * scr_u = which ? scr_u1 : scr_u0;
    float            * scr_v = which ? scr_v1 : scr_v0;
    const half       * gamma = which ? gamma1 : gamma0;

    const int g   = active_g != nullptr ? active_g[blockIdx.z] : blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const int cnt = scr_i[g];
    if (cnt == 0) {                // block-uniform: whole block exits together
        return;
    }
    const bool p4_on = p4 != nullptr;
    if (p4_on) {
        if (tid < 16) {
            lv[tid] = p4lv[tid];
        }
        __syncthreads();
    }
    const int  off     = scr_i[n_groups + g];
    const int  tiles_y = n / 16;
    const int  ntiles  = (m / 16)*tiles_y;
    const bool have    = tid < tiles_y;

    const int words = 24;
    const uint16_t * trd = kept;
    const int64_t tbase = ((int64_t) g*ntiles + (int64_t) tr*tiles_y)*words;
    const int e_orig = scr_i[2*n_groups + g];
    const int Mb     = m / 16;
    float gsc = 0.0f;
    if (have) {
        const int wv = (tr + tid <= tiles_y - 1) ? Mb + tiles_y - 1 - (tr + tid)
                                                 : Mb + tiles_y - 2 - (tr + tid);
        gsc = __half2float(gamma[(int64_t) e_orig*(Mb + tiles_y) + wv]);
    }

    uint16_t w8[24];
    if (have) {
        const int64_t tw = tbase + (int64_t) tid*words;
#pragma unroll
        for (int q = 0; q < 24; ++q) {
            w8[q] = trd[tw + q];
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;

    for (int qq = 0; qq < cnt; ++qq) {
        const int p = scr_i[4*n_groups + off + qq];

        float partial[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] = 0.0f;
        }
        if (have) {
            const float * ub = scr_u + (int64_t) p*n + tid*16;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int      bb  = 12*i;
                const int      wi  = bb >> 4;
                const int      o   = bb & 15;
                const int      wn  = wi + 1 < 24 ? wi + 1 : 0;
                const uint32_t w2  = ((uint32_t) w8[wi] << 16) | (uint32_t) w8[wn];
                const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
                const uint32_t ph  = reg*(reg + 1u);
                const uint32_t row = ph & 0x7FFFu;
                const int      ri  = (8*i) >> 4;
                const int      ci  = (8*i) & 15;
                float dotp = 0.0f;
                if (p4_on) {
                    const uint32_t pk = p4[row];
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = lv[(pk >> (4*c)) & 0xFu];
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;
                        }
                        dotp = fmaf(vv, ub[ci + c], dotp);
                    }
                } else {
                    const half * tl = (const half *) tlut + 8*(int64_t) row;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float vv = __half2float(tl[c]);
                        if (c == 0 && (ph & 0x8000u)) {
                            vv = -vv;
                        }
                        dotp = fmaf(vv, ub[ci + c], dotp);
                    }
                }
                // gsc is loop-invariant across every i/c/qq for this thread
                // (depends only on the thread's fixed row-tile); deferring
                // the scale to here instead of every inner c-iteration cuts
                // 256 redundant multiplies/token/thread down to 16.
                partial[ri] += dotp;
            }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i] *= gsc;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(partial[i]);
            if (lane == 0) {
                red[i][wid] = s;
            }
        }
        __syncthreads();
        if (tid < 16) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[tid][wj];
            }
            scr_v[(int64_t) p*m + tr*16 + tid] = sum;
        }
        if (qq + 1 < cnt) __syncthreads();   // red is reused by the next pair -- skip on last iter, nothing left to protect
    }
}

// blockIdx.z selects gate (0) / up (1). dst is [m, n_used, 2, n_tok], so
// each complete projection is contiguous and directly viewable.
template <int WG>
static __global__ void paw_exp_out2_kernel(
        const half    * GGML_CUDA_RESTRICT sv0,
        const half    * GGML_CUDA_RESTRICT sv1,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT scr_v0,
        const float   * GGML_CUDA_RESTRICT scr_v1,
        float         * GGML_CUDA_RESTRICT dst,
        const int m, const int n_used, const int ids_s0, const int ids_s1) {
    __shared__ float sh[2048];

    const int s     = blockIdx.x;
    const int t     = blockIdx.y;
    const int which = blockIdx.z;
    const int p     = t*n_used + s;
    const int tid   = threadIdx.x;

    const half  * sv    = which == 0 ? sv0    : sv1;
    const float * scr_v = which == 0 ? scr_v0 : scr_v1;

    ggml_cuda_pdl_sync();
    const int64_t e = ids[s*ids_s0 + t*ids_s1];

    for (int i = tid; i < m; i += WG) {
        __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) p*m + i], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    if (WG == m/16 && paw_fwht_v2_ok(m)) {
        paw_fwht_block_v2(sh, m, tid, WG);
    } else {
        paw_fwht_block(sh, m, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) m);
    const float inv_sc = __frcp_rn(sc);
    const int64_t obase = ((int64_t) t*2 + which)*n_used*m + (int64_t) s*m;
    for (int i = tid; i < m; i += WG) {
        dst[obase + i] = sh[i] * inv_sc * __half2float(sv[e*m + i]);
    }
}



//
// supports_op — mirrors the Vulkan predicate (ggml-vulkan.cpp)
//

// --- PAW_V_REORDER --------------------------------------------------------
//
// Row permutation for the v3 mach1 codec. Within the segment starting at
// seg_off (seg_rows = hd*K*r rows): out[(v*K + k)*hd + d] =
// in[(k*r + v)*hd + d]. All other rows copy through. Replaces the
// cont/permute/cont + concat chains the graph used per SSM layer.


void ggml_cuda_op_paw_exp_basis(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * b     = dst->src[1];
    const ggml_tensor * c     = dst->src[2];
    const ggml_tensor * remap = dst->src[3];
    const ggml_tensor * ids   = dst->src[4];
    const ggml_tensor * x     = dst->src[5];
    const ggml_tensor * accs  = dst->src[6];

    GGML_ASSERT(a->type     == GGML_TYPE_F16);
    GGML_ASSERT(b->type     == GGML_TYPE_F16);
    GGML_ASSERT(c->type     == GGML_TYPE_F16);
    GGML_ASSERT(remap->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(x->type     == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(c));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    if (accs != nullptr) {
        GGML_ASSERT(accs->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(accs));
    }

    const int n      = (int) a->ne[0];
    const int r      = (int) a->ne[1];
    const int m      = (int) b->ne[1];
    const int n_used = (int) ids->ne[0];
    const int n_tok  = (int) ids->ne[1];
    const int xne1   = (int) x->ne[1];
    const int ids_s0 = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1 = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(r <= 256);   // tv shared bound

    paw_launch(paw_exp_basis_kernel,
        ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, ctx.stream()),
        (const half    *) a->data,
        (const half    *) b->data,
        (const half    *) c->data,
        (const int32_t *) remap->data,
        (const int32_t *) ids->data,
        (const float   *) x->data,
        accs != nullptr ? (const float *) accs->data : nullptr,
        (float         *) dst->data,
        n, r, m, n_used, xne1, ids_s0, ids_s1, accs != nullptr ? 1 : 0);
}

void ggml_cuda_op_paw_exp_mm_batch2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kept0  = dst->src[0];
    const ggml_tensor * su0    = dst->src[1];
    const ggml_tensor * sv0    = dst->src[2];
    const ggml_tensor * gamma0 = dst->src[3];
    const ggml_tensor * kept1  = dst->src[4];
    const ggml_tensor * su1    = dst->src[5];
    const ggml_tensor * sv1    = dst->src[6];
    const ggml_tensor * gamma1 = dst->src[7];
    const ggml_tensor * tlut   = dst->src[8];
    const ggml_tensor * remap  = dst->src[9];
    const ggml_tensor * ids    = dst->src[10];
    const ggml_tensor * x      = dst->src[11];

    GGML_ASSERT(kept0->type  == GGML_TYPE_I16 && kept1->type  == GGML_TYPE_I16);
    GGML_ASSERT(su0->type    == GGML_TYPE_F16 && su1->type    == GGML_TYPE_F16);
    GGML_ASSERT(sv0->type    == GGML_TYPE_F16 && sv1->type    == GGML_TYPE_F16);
    GGML_ASSERT(gamma0->type == GGML_TYPE_F16 && gamma1->type == GGML_TYPE_F16);
    GGML_ASSERT(tlut->type   == GGML_TYPE_F16);
    GGML_ASSERT(remap->type  == GGML_TYPE_I32);
    GGML_ASSERT(ids->type    == GGML_TYPE_I32);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kept0)  && ggml_is_contiguous(kept1));
    GGML_ASSERT(ggml_is_contiguous(su0)    && ggml_is_contiguous(su1));
    GGML_ASSERT(ggml_is_contiguous(sv0)    && ggml_is_contiguous(sv1));
    GGML_ASSERT(ggml_is_contiguous(gamma0) && ggml_is_contiguous(gamma1));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(kept0->ne[0] == 24 && kept1->ne[0] == 24);   // V8 walk rate
    GGML_ASSERT(tlut->ne[0] == 8);

    const int n        = (int) su0->ne[0];
    const int m        = (int) sv0->ne[0];
    const int n_kept   = (int) kept0->ne[2];
    const int n_groups = n_kept;             // batched path requires no demotion
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int P        = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int ids_s0   = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1   = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(n <= 2048 && m <= 2048);
    GGML_ASSERT(n_groups <= 512);
    GGML_ASSERT(n_tok == 1);   // caller-scoped: decode-only

    static const bool walk_compact = paw_env_int("GGML_PAW_EXP_WALK_COMPACT", 1) != 0;
    ggml_cuda_pool_alloc<int32_t> scr_i_alloc(ctx.pool(), (size_t) 4*n_groups + P + (walk_compact ? n_used : 0));
    ggml_cuda_pool_alloc<float>   scr_f_alloc(ctx.pool(), 2*((size_t) P*n + (size_t) P*m));
    int32_t * scr_i  = scr_i_alloc.get();
    int32_t * active_g = walk_compact ? scr_i + 4*n_groups + P : nullptr;
    float   * scr_u0 = scr_f_alloc.get();
    float   * scr_v0 = scr_u0 + (size_t) P*n;
    float   * scr_u1 = scr_v0 + (size_t) P*m;
    float   * scr_v1 = scr_u1 + (size_t) P*n;

    cudaStream_t stream = ctx.stream();

    paw_p4_table p4t = paw_exp_p4_table(tlut->data, stream);
    const bool p4 = p4t.packed != nullptr;

    char shp[96];
    snprintf(shp, sizeof(shp), " m=%d n=%d n_used=%d n_tok=%d ng=%d p4=%d",
             m, n, n_used, n_tok, n_groups, (int) p4);

    paw_fwht_set_mode();

    paw_timed(stream, std::string("exp_group_u2") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
        paw_fwht_for_wg(n/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_exp_group_u2_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(2*n_used + 1, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i, active_g,
                (const half *) su0->data, (const half *) su1->data,
                (const float *) x->data, scr_u0, scr_u1,
                n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
        });
    } else if (paw_fwht_wg512()) {
        paw_launch(paw_exp_group_u2_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(2*n_used + 1, n_tok, 1), dim3(512, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i, active_g,
            (const half *) su0->data, (const half *) su1->data,
            (const float *) x->data, scr_u0, scr_u1,
            n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
    } else {
        paw_launch(paw_exp_group_u2_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(2*n_used + 1, n_tok, 1), dim3(256, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i, active_g,
            (const half *) su0->data, (const half *) su1->data,
            (const float *) x->data, scr_u0, scr_u1,
            n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
    }
    });

    const int grid_z = walk_compact ? n_used : n_groups;

    paw_timed(stream, std::string("exp_walk2") + shp, [&]() {
    paw_launch(paw_exp_walk2_kernel,
        ggml_cuda_kernel_launch_params(dim3(m/16, 2, grid_z), dim3(128, 1, 1), 0, stream),
        (const uint16_t *) kept0->data, (const uint16_t *) kept1->data,
        (const void *) tlut->data, p4t.packed, p4t.levels,
        (const int32_t *) scr_i, scr_u0, scr_u1, scr_v0, scr_v1,
        (const half *) gamma0->data, (const half *) gamma1->data,
        active_g, m, n, n_groups);
    });

    paw_timed(stream, std::string("exp_out2") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
        paw_fwht_for_wg(m/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_exp_out2_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 2), dim3(wg, 1, 1), 0, stream),
                (const half *) sv0->data, (const half *) sv1->data, (const int32_t *) ids->data,
                scr_v0, scr_v1, (float *) dst->data, m, n_used, ids_s0, ids_s1);
        });
    } else if (paw_fwht_wg512()) {
        paw_launch(paw_exp_out2_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 2), dim3(512, 1, 1), 0, stream),
            (const half *) sv0->data, (const half *) sv1->data, (const int32_t *) ids->data,
            scr_v0, scr_v1, (float *) dst->data, m, n_used, ids_s0, ids_s1);
    } else {
        paw_launch(paw_exp_out2_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 2), dim3(256, 1, 1), 0, stream),
            (const half *) sv0->data, (const half *) sv1->data, (const int32_t *) ids->data,
            scr_v0, scr_v1, (float *) dst->data, m, n_used, ids_s0, ids_s1);
    }
    });
}

void ggml_cuda_op_paw_exp_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kept  = dst->src[0];
    const ggml_tensor * dem   = dst->src[1];
    const ggml_tensor * su    = dst->src[2];
    const ggml_tensor * sv    = dst->src[3];
    const ggml_tensor * tlut  = dst->src[4];
    const ggml_tensor * remap = dst->src[5];
    const ggml_tensor * ids   = dst->src[6];
    const ggml_tensor * x     = dst->src[7];
    const ggml_tensor * gamma = dst->src[8];   // payload v3 (V8 walk) only

    const bool v8 = tlut->ne[0] == 8;
    GGML_ASSERT(v8 == (gamma != nullptr));

    GGML_ASSERT(kept->type  == GGML_TYPE_I16);
    GGML_ASSERT(su->type    == GGML_TYPE_F16);
    GGML_ASSERT(sv->type    == GGML_TYPE_F16);
    GGML_ASSERT(tlut->type  == (v8 ? GGML_TYPE_F16 : GGML_TYPE_F32));
    GGML_ASSERT(remap->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(x->type     == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kept));
    GGML_ASSERT(ggml_is_contiguous(su));
    GGML_ASSERT(ggml_is_contiguous(sv));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(remap));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // the walk kernel hardcodes the rates like the Vulkan shader
    GGML_ASSERT(kept->ne[0] == (v8 ? 24 : 32));
    if (dem != nullptr) {
        GGML_ASSERT(dem->type == GGML_TYPE_I16);
        GGML_ASSERT(ggml_is_contiguous(dem));
        GGML_ASSERT(dem->ne[0] == 16);
    }
    if (gamma != nullptr) {
        GGML_ASSERT(gamma->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(gamma));
    }

    const int n        = (int) su->ne[0];
    const int m        = (int) sv->ne[0];
    const int n_kept   = (int) kept->ne[2];
    const int n_groups = n_kept + (int)(dem ? dem->ne[2] : 0);
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int P        = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int ids_s0   = (int)(ids->nb[0]/sizeof(int32_t));
    const int ids_s1   = (int)(ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(n <= 2048 && m <= 2048);   // exp_u/exp_out shared bounds
    GGML_ASSERT(n_groups <= 512);          // exp_group shared bound

    ggml_cuda_pool_alloc<int32_t> scr_i_alloc(ctx.pool(), (size_t) 4*n_groups + P);
    ggml_cuda_pool_alloc<float>   scr_f_alloc(ctx.pool(), (size_t) P*n + (size_t) P*m);
    int32_t * scr_i = scr_i_alloc.get();
    float   * scr_u = scr_f_alloc.get();
    float   * scr_v = scr_u + (size_t) P*n;

    cudaStream_t stream = ctx.stream();

    // p4 repack of the V8 tlut (one-time, cached); nullptr => fp16 gathers
    paw_p4_table p4t = {nullptr, nullptr};
    if (v8) {
        GGML_ASSERT(tlut->ne[1] == 32768);
        p4t = paw_exp_p4_table(tlut->data, stream);
    }
    const bool p4 = p4t.packed != nullptr;

    char shp[96];
    snprintf(shp, sizeof(shp), " m=%d n=%d n_used=%d n_tok=%d ng=%d v8=%d p4=%d",
             m, n, n_used, n_tok, n_groups, (int) v8, (int) p4);

    paw_fwht_set_mode();

    static const bool group_fuse = paw_env_int("GGML_PAW_EXP_GROUP_FUSE", 0) != 0;
    if (group_fuse) {
        paw_timed(stream, std::string("exp_group_u") + shp, [&]() {
        if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
            paw_fwht_for_wg(n/16, [&](auto WG) {
                constexpr int wg = decltype(WG)::value;
                paw_launch(paw_exp_group_u_kernel<wg>,
                    ggml_cuda_kernel_launch_params(dim3(n_used + 1, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                    (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
                    (const half *) su->data, (const float *) x->data, scr_u,
                    n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
            });
        } else if (paw_fwht_wg512()) {
            paw_launch(paw_exp_group_u_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(n_used + 1, n_tok, 1), dim3(512, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
                (const half *) su->data, (const float *) x->data, scr_u,
                n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
        } else {
            paw_launch(paw_exp_group_u_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(n_used + 1, n_tok, 1), dim3(256, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
                (const half *) su->data, (const float *) x->data, scr_u,
                n, n_used, n_tok, n_kept, n_groups, xne1, ids_s0, ids_s1);
        }
        });
    } else {
        paw_timed(stream, std::string("exp_group") + shp, [&]() {
        paw_launch(paw_exp_group_kernel,
            ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(256, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, scr_i,
            n_used, n_tok, n_kept, n_groups, ids_s0, ids_s1);
        });

        paw_timed(stream, std::string("exp_u") + shp, [&]() {
        if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
            paw_fwht_for_wg(n/16, [&](auto WG) {
                constexpr int wg = decltype(WG)::value;
                paw_launch(paw_exp_u_kernel<wg>,
                    ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                    (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
                    scr_u, n, n_used, xne1, ids_s0, ids_s1);
            });
        } else if (paw_fwht_wg512()) {
            paw_launch(paw_exp_u_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
                (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
                scr_u, n, n_used, xne1, ids_s0, ids_s1);
        } else {
            paw_launch(paw_exp_u_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
                (const half *) su->data, (const int32_t *) ids->data, (const float *) x->data,
                scr_u, n, n_used, xne1, ids_s0, ids_s1);
        }
        });
    }

    const uint16_t * kept_d = (const uint16_t *) kept->data;
    const uint16_t * dem_d  = dem != nullptr ? (const uint16_t *) dem->data : kept_d;

    if (paw_exp_cache_on() && v8 && p4 && n_tok == 1 && dem == nullptr) {
        // single-token decode, the case the walk path never amortizes.
        // Scoped to v8+p4+no-demotion (reason8192's actual runtime config)
        // -- see paw_exp_slot_decode_kernel's header comment.
        paw_exp_slots slots = paw_exp_slots_get(kept_d, m, n, n_used, stream);
        ggml_cuda_pool_alloc<int32_t> scr_i_slot_alloc(ctx.pool(), (size_t) 5*n_used);
        int32_t * scr_i_slot = scr_i_slot_alloc.get();

        const ggml_cuda_kernel_launch_params dec_params =
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_used),
                                           dim3(256, 1, 1), 0, stream);
        paw_timed(stream, std::string("exp_slot_decode") + shp, [&]() {
        paw_launch(paw_exp_slot_decode_kernel<true>, dec_params,
            kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
            (const int32_t *) remap->data, (const int32_t *) ids->data,
            slots.tags_dev, slots.banks, m, n, ids_s0);
        });
        paw_launch(paw_exp_slot_tag_kernel,
            ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(n_used, 1, 1), 0, stream),
            (const int32_t *) remap->data, (const int32_t *) ids->data, slots.tags_dev, n_used, ids_s0);
        paw_launch(paw_exp_slot_scri_kernel,
            ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(n_used, 1, 1), 0, stream),
            (const int32_t *) ids->data, scr_i_slot, n_used, ids_s0);

        static const int exp_pc = paw_env_int("GGML_PAW_EXP_PC", 8);
        paw_timed(stream, std::string("exp_slot_apply") + shp, [&]() {
        if (exp_pc == 1) {
            paw_launch(paw_exp_apply_kernel<true, 1>,
                ggml_cuda_kernel_launch_params(dim3(m/16, n_used, 1), dim3(128, 1, 1), 0, stream),
                (const half *) slots.banks, scr_i_slot, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_used);
        } else if (exp_pc == 4) {
            paw_launch(paw_exp_apply_kernel<true, 4>,
                ggml_cuda_kernel_launch_params(dim3(m/16, n_used, 1), dim3(128, 1, 1), 0, stream),
                (const half *) slots.banks, scr_i_slot, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_used);
        } else {
            paw_launch(paw_exp_apply_kernel<true, 8>,
                ggml_cuda_kernel_launch_params(dim3(m/16, n_used, 1), dim3(128, 1, 1), 0, stream),
                (const half *) slots.banks, scr_i_slot, (const float *) scr_u, scr_v,
                (const half *) gamma->data, m, n, n_used);
        }
        });

        paw_timed(stream, std::string("exp_out") + shp, [&]() {
        if (paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
            paw_fwht_for_wg(m/16, [&](auto WG) {
                constexpr int wg = decltype(WG)::value;
                paw_launch(paw_exp_out_kernel<wg>,
                    ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                    (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                    m, n_used, ids_s0, ids_s1);
            });
        } else if (paw_fwht_wg512()) {
            paw_launch(paw_exp_out_kernel<512>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        } else {
            paw_launch(paw_exp_out_kernel<256>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        }
        });
        return;
    }

    static const int dense_min = paw_env_int("GGML_PAW_DENSE_MIN", 1024);
    ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());
    if (P >= dense_min) {
        // dense prefill: decode active groups once, then apply
        static const bool exp_ws  = paw_env_int("GGML_PAW_EXP_APPLY_WS", 1) != 0;
        static const bool exp_gp  = paw_env_int("GGML_PAW_EXP_XGROUP", 1) != 0;
        static const int exp_pc   = paw_env_int("GGML_PAW_EXP_PC", 8);
        static const bool exp_blas = paw_env_int("GGML_PAW_EXP_BLAS", 1) != 0;
        // slab width for the chunked blas pipeline; bounds staging VRAM
        static const int exp_chunk = paw_env_int("GGML_PAW_EXP_BLAS_CHUNK_G", 16);
        // pool is strict LIFO: bank must be allocated before xg so the
        // destructor order (xg first, bank last) unwinds it correctly
        // the blas branch pulls group counts to host (uncaptured stream);
        // under graph capture fall back to the fully device-driven path
        cudaStreamCaptureStatus cst = cudaStreamCaptureStatusNone;
        const bool capturing = cudaStreamIsCapturing(stream, &cst) == cudaSuccess &&
                               cst == cudaStreamCaptureStatusActive;
        // single-token passes go through the fused WS kernel instead: no
        // staging round-trip and no host sync (measured +3 t/s generation)
        // below this, micro-batches skip the cuBLAS pipeline (host sync
        // per pass dominates); measured neutral-to-better at 29k ctx
        static const int exp_blas_min_nt = paw_env_int("GGML_PAW_EXP_BLAS_MIN_NT", 128);
        // fully-fused trellis->wmma apply: no bank materialization, no host
        // sync; supersedes the chunked cuBLAS pipeline where it applies
        // correct and graph-safe, but on RTX 3060 the cuBLAS split still
        // beats it at large nt (see paw_fused_bench); flip when the kernel
        // closes the activation-restage gap
        static const bool exp_fused = paw_env_int("GGML_PAW_EXP_FUSED", 0) != 0;
        const bool use_fused = v8 && exp_fused && exp_ws && exp_gp &&
                               m % 16 == 0 && n_tok >= exp_blas_min_nt;
        const bool use_blas = v8 && exp_blas && exp_ws && exp_gp &&
                              m % 16 == 0 && n_tok >= exp_blas_min_nt &&
                              !use_fused && !capturing;
        // overlap decode of slab i+1 with GEMMs of slab i on a second stream
        // off by default: measured no gain (decode and GEMM contend for the same
// bandwidth on this part); kept for parts where that does not hold
        static const bool exp_ovl = paw_env_int("GGML_PAW_EXP_OVERLAP", 0) != 0;
        const bool do_ovl = use_blas && exp_ovl && n_groups > exp_chunk;
        const int n_slabs = use_blas ? (n_groups + exp_chunk - 1)/exp_chunk : 1;
        const int bank_bufs = do_ovl ? 2 : 1;
        const int bank_g = use_blas ? (exp_chunk < n_groups ? exp_chunk : n_groups)
                                    : n_groups;
        half * bank = use_fused ? nullptr
                                : bank_alloc.alloc((size_t) bank_g*bank_bufs*m*n);
        ggml_cuda_pool_alloc<half> xg_alloc(ctx.pool());
        half * xg = nullptr;
        if (v8 && exp_ws && exp_gp) {
            xg = xg_alloc.alloc((size_t) P*n);
            paw_timed(stream, std::string("exp_xgroup") + shp, [&]() {
            paw_launch(paw_exp_permute_x_kernel,
                ggml_cuda_kernel_launch_params(
                    dim3((n + 255)/256, n_groups, 1), dim3(256, 1, 1), 0, stream),
                (const int32_t *) scr_i, (const float *) scr_u, xg, n, n_groups);
            });
        }
        if (use_fused) {
            const ggml_cuda_kernel_launch_params fparams =
                ggml_cuda_kernel_launch_params(
                    dim3(m/64, n_groups, 1), dim3(128, 1, 1), 0, stream);
            paw_timed(stream, std::string("exp_fused") + shp, [&]() {
                if (p4) {
                    const paw_l2_tables & lt = paw_l2_tables_get(stream, p4t);
                    paw_launch(paw_exp_apply_kernel_fused<true, 4>, fparams,
                        kept_d, (const half *) nullptr, (const void *) tlut->data,
                        lt.packed ? lt.packed : p4t.packed,
                        lt.levels ? lt.levels : p4t.levels,
                        (const int32_t *) scr_i, xg, scr_v,
                        (const half *) gamma->data,
                        (float *) nullptr, m, n, n_groups);
                } else {
                    paw_launch(paw_exp_apply_kernel_fused<false, 4>, fparams,
                        kept_d, (const half *) nullptr, (const void *) tlut->data,
                        (const uint32_t *) nullptr, (const float *) nullptr,
                        (const int32_t *) scr_i, xg, scr_v,
                        (const half *) gamma->data,
                        (float *) nullptr, m, n, n_groups);
                }
            });
        } else if (!use_blas) {
        static const bool dec_v2 = paw_env_int("GGML_PAW_EXP_DECODE_V2", 1) != 0;
        const ggml_cuda_kernel_launch_params dec_params =
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, n_groups),
                                           dim3(256, 1, 1), 0, stream);
        paw_timed(stream, std::string("exp_dense_decode") + shp, [&]() {
        if (v8) {
            // warp-per-tile decode: same bits out, fewer passes over the
            // stream words
            const ggml_cuda_kernel_launch_params v2_params =
                ggml_cuda_kernel_launch_params(
                    dim3(((m/16)*(n/16) + 7)/8, 1, n_groups), dim3(256, 1, 1), 0, stream);
            if (p4 && dec_v2) {
                paw_launch(paw_exp_dense_decode_kernel_v2<true>, v2_params,
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, bank, m, n, n_groups, 0,
                    nullptr);
            } else if (dec_v2) {
                paw_launch(paw_exp_dense_decode_kernel_v2<false>, v2_params,
                    kept_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, bank, m, n, n_groups, 0,
                    nullptr);
            }
        } else {
            paw_launch(paw_exp_dense_decode_kernel<false, false>, dec_params,
                kept_d, dem_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const int32_t *) scr_i, bank, m, n, n_kept, n_groups, 0,
                nullptr);
        }
        });
        }
        const ggml_cuda_kernel_launch_params app_params =
            ggml_cuda_kernel_launch_params(dim3(m/16, n_groups, 1), dim3(128, 1, 1), 0, stream);
        if (use_blas) {
            // tall-skinny shape: per-expert dense GEMMs over the grouped
            // slab beat every custom kernel here; the wave gamma is folded
            // into the decoded banks first
            // fp16 output keeps the padded slab small enough for 12GB cards;
            // the unpermute pass widens back to fp32 for downstream ops
            ggml_cuda_pool_alloc<half> ygf_alloc(ctx.pool());
            half * ygf = ygf_alloc.alloc((size_t) P*m);
            // the per-group GEMM dims live on device; pull the count table
            // once (a few hundred bytes). Requires uncaptured stream, so the
            // caller keeps this branch behind GGML_CUDA_DISABLE_GRAPHS=1.
            std::vector<int32_t> hscr(2*n_groups);
            CUDA_CHECK(cudaMemcpyAsync(hscr.data(), scr_i,
                (size_t) 2*n_groups*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            // diagnostic: GGML_PAW_EXP_BLAS_SKIP_DECODE=1 times GEMMs alone
            // (output is garbage; perf signal only)
            static const bool blas_skip_dec =
                paw_env_int("GGML_PAW_EXP_BLAS_SKIP_DECODE", 0) != 0;
            paw_timed(stream, std::string("exp_blas") + shp, [&]() {
            std::vector<cudaEvent_t> ev(do_ovl ? 2*n_slabs + 1 : 0);
            if (do_ovl) {
                for (auto & e : ev) {
                    CUDA_CHECK(cudaEventCreate(&e));
                }
                CUDA_CHECK(cudaEventRecord(ev[0], stream));
            }
            cudaStream_t ds = do_ovl ? paw_aux_stream() : stream;
            for (int si = 0; si < n_slabs; ++si) {
                const int g0 = si*exp_chunk;
                const int gc = exp_chunk < n_groups - g0 ? exp_chunk : n_groups - g0;
                half * buf = bank + (size_t)(do_ovl ? (si % 2) : 0)*bank_g*m*n;
                const ggml_cuda_kernel_launch_params dp =
                    ggml_cuda_kernel_launch_params(
                        dim3((m/16*n + 255)/256, 1, gc), dim3(256, 1, 1), 0, ds);
                if (!blas_skip_dec) {
                    if (do_ovl) {
                        // gate: previous slab's GEMMs released this buffer
                        CUDA_CHECK(cudaStreamWaitEvent(ds, ev[2*si]));
                    }
                    static const bool dec_v2 = paw_env_int("GGML_PAW_EXP_DECODE_V2", 1) != 0;
                    const ggml_cuda_kernel_launch_params v2p =
                        ggml_cuda_kernel_launch_params(
                            dim3(((m/16)*(n/16) + 7)/8, 1, gc), dim3(256, 1, 1), 0, ds);
                    if (p4 && dec_v2) {
                        paw_launch(paw_exp_dense_decode_kernel_v2<true>, v2p,
                            kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                            (const int32_t *) scr_i, buf, m, n, n_groups, g0,
                            (const half *) gamma->data);
                    } else if (dec_v2) {
                        paw_launch(paw_exp_dense_decode_kernel_v2<false>, v2p,
                            kept_d, (const void *) tlut->data,
                            (const uint32_t *) nullptr, (const float *) nullptr,
                            (const int32_t *) scr_i, buf, m, n, n_groups, g0,
                            (const half *) gamma->data);
                    }
                    if (do_ovl) {
                        CUDA_CHECK(cudaEventRecord(ev[2*si + 1], ds));
                        CUDA_CHECK(cudaStreamWaitEvent(stream, ev[2*si + 1]));
                    }
                }
                for (int gi = 0; gi < gc; ++gi) {
                    const int cnt = hscr[g0 + gi];
                    if (cnt == 0) {
                        continue;
                    }
                    const int off = hscr[n_groups + g0 + gi];
                    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(),
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            m, cnt, n,
                            &alpha,
                            buf + (size_t) gi*m*n,                  CUDA_R_16F, n,
                            xg + (int64_t) off*n,                   CUDA_R_16F, n,
                            &beta,
                            ygf + (int64_t) off*m,                  CUDA_R_16F, m,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
                if (do_ovl && si + 1 < n_slabs) {
                    CUDA_CHECK(cudaEventRecord(ev[2*si + 2], stream));
                }
            }
            if (do_ovl) {
                // pool frees ride on the main stream: pin aux work to it too
                CUDA_CHECK(cudaEventRecord(ev[2*n_slabs], ds));
                CUDA_CHECK(cudaStreamWaitEvent(stream, ev[2*n_slabs]));
                CUDA_CHECK(cudaStreamWaitEvent(ds, ev[2*n_slabs]));
                for (auto & e : ev) {
                    CUDA_CHECK(cudaEventDestroy(e));
                }
            }
            });
            paw_timed(stream, std::string("exp_unperm") + shp, [&]() {
            paw_launch(paw_exp_unpermute_y_kernel,
                ggml_cuda_kernel_launch_params(
                    dim3((m + 255)/256, n_groups, 1), dim3(256, 1, 1), 0, stream),
                (const int32_t *) scr_i, ygf, scr_v, m, n_groups);
            });
        } else if (!use_fused) {
        // fused handled everything above; this branch applies over a
        // materialized bank and must not run when no bank exists
        paw_timed(stream, std::string("exp_apply") + shp, [&]() {
        static const int ws_min_m = paw_env_int("GGML_PAW_EXP_WS_MIN_M", 512);
        if (v8 && exp_ws && xg && m % 64 == 0 && m >= ws_min_m) {
            paw_launch(paw_exp_apply_kernel_ws<true, 4, true>,
                ggml_cuda_kernel_launch_params(dim3(m/64, n_groups, 1), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const int32_t *) scr_i, nullptr, xg, scr_v,
                (const half *) gamma->data, m, n, n_groups);
        } else if (v8) {
        // m < 1024 keeps the scalar PC path: measured slower than WS there
        // (the pair-gather latency chain dominates the short row dimension)
            if (exp_pc == 1) {
                paw_launch(paw_exp_apply_kernel<true, 1>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else if (exp_pc == 4) {
                paw_launch(paw_exp_apply_kernel<true, 4>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            } else {
                paw_launch(paw_exp_apply_kernel<true, 8>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) gamma->data, m, n, n_groups);
            }
        } else {
            if (exp_pc == 1) {
                paw_launch(paw_exp_apply_kernel<false, 1>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            } else if (exp_pc == 4) {
                paw_launch(paw_exp_apply_kernel<false, 4>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            } else {
                paw_launch(paw_exp_apply_kernel<false, 8>, app_params,
                    (const half *) bank, (const int32_t *) scr_i, (const float *) scr_u, scr_v,
                    (const half *) nullptr, m, n, n_groups);
            }
        }
        });
        }
    } else {
        static const bool walk_compact = paw_env_int("GGML_PAW_EXP_WALK_COMPACT", 1) != 0;
        const bool do_compact = walk_compact && n_tok == 1;
        const bool walk_warp = v8 && n/16 <= 32 && m % 64 == 0;
        const bool fuse_warp_group = do_compact && walk_warp;
        ggml_cuda_pool_alloc<int32_t> active_g_alloc(ctx.pool());
        const int32_t * active_g = nullptr;
        const int grid_z = do_compact ? n_used : n_groups;
        if (do_compact && !fuse_warp_group) {
            int32_t * ag = active_g_alloc.alloc(n_used);
            paw_launch(paw_exp_active_groups_kernel,
                ggml_cuda_kernel_launch_params(dim3(1, 1, 1), dim3(n_used, 1, 1), 0, stream),
                (const int32_t *) remap->data, (const int32_t *) ids->data, ag, n_used, n_kept, ids_s0);
            active_g = ag;
        }

        const ggml_cuda_kernel_launch_params walk_params =
            ggml_cuda_kernel_launch_params(dim3(m/16, 1, grid_z), dim3(128, 1, 1), 0, stream);
        // narrow inputs: warp-per-row-tile variant keeps whole blocks busy
        const ggml_cuda_kernel_launch_params warp_params =
            ggml_cuda_kernel_launch_params(dim3(m/64, 1, grid_z), dim3(128, 1, 1), 0, stream);
        paw_timed(stream, std::string("exp_walk") + shp, [&]() {
        if (walk_warp) {
            if (p4) {
                paw_launch(paw_exp_walk_v8_warp_kernel<true>, warp_params,
                    kept_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data,
                    fuse_warp_group ? (const int32_t *) remap->data : nullptr,
                    fuse_warp_group ? (const int32_t *) ids->data : nullptr,
                    m, n, n_groups, n_used, n_kept, ids_s0);
            } else {
                paw_launch(paw_exp_walk_v8_warp_kernel<false>, warp_params,
                    kept_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data,
                    fuse_warp_group ? (const int32_t *) remap->data : nullptr,
                    fuse_warp_group ? (const int32_t *) ids->data : nullptr,
                    m, n, n_groups, n_used, n_kept, ids_s0);
            }
        } else if (v8) {
            if (p4) {
                paw_launch(paw_exp_walk_kernel<true, true>, walk_params,
                    kept_d, dem_d, (const void *) tlut->data, p4t.packed, p4t.levels,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, active_g, m, n, n_kept, n_groups);
            } else {
                paw_launch(paw_exp_walk_kernel<true, false>, walk_params,
                    kept_d, dem_d, (const void *) tlut->data,
                    (const uint32_t *) nullptr, (const float *) nullptr,
                    (const int32_t *) scr_i, scr_u, scr_v,
                    (const half *) gamma->data, active_g, m, n, n_kept, n_groups);
            }
        } else {
            paw_launch(paw_exp_walk_kernel<false, false>, walk_params,
                kept_d, dem_d, (const void *) tlut->data,
                (const uint32_t *) nullptr, (const float *) nullptr,
                (const int32_t *) scr_i, scr_u, scr_v,
                (const half *) nullptr, active_g, m, n, n_kept, n_groups);
        }
        });
    }

    paw_timed(stream, std::string("exp_out") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
        paw_fwht_for_wg(m/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_exp_out_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(wg, 1, 1), 0, stream),
                (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
                m, n_used, ids_s0, ids_s1);
        });
    } else if (paw_fwht_wg512()) {
        paw_launch(paw_exp_out_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(512, 1, 1), 0, stream),
            (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
            m, n_used, ids_s0, ids_s1);
    } else {
        paw_launch(paw_exp_out_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_used, n_tok, 1), dim3(256, 1, 1), 0, stream),
            (const half *) sv->data, (const int32_t *) ids->data, scr_v, (float *) dst->data,
            m, n_used, ids_s0, ids_s1);
    }
    });
}

