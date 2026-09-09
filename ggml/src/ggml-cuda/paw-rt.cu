// Split from paw.cu; see docs/paw/README.md for the file map.
#include "paw-common.cuh"

void ggml_cuda_op_paw_ne_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * packed = dst->src[0];
    const ggml_tensor * gscale = dst->src[1];
    const ggml_tensor * lut    = dst->src[2];
    const ggml_tensor * x      = dst->src[3];

    GGML_ASSERT(packed->type == GGML_TYPE_I8);
    GGML_ASSERT(gscale->type == GGML_TYPE_F16);
    GGML_ASSERT(lut->type    == GGML_TYPE_F16);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(packed));
    GGML_ASSERT(ggml_is_contiguous(gscale));
    GGML_ASSERT(ggml_is_contiguous(lut));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int T   = (int) x->ne[0];
    const int B   = (int) packed->ne[1];
    const int k   = (int)(packed->ne[0]*8 / T);
    const int ng  = (int) gscale->ne[0];
    const int rpc = (int)(B / lut->ne[1]);
    const int nt  = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    const dim3 grid((unsigned)((B + 127)/128), (unsigned)((nt + 7)/8), 1);
    char shp[64];
    snprintf(shp, sizeof(shp), " B=%d T=%d k=%d nt=%d", B, T, k, nt);
    paw_timed(ctx.stream(), std::string("ne_mm") + shp, [&]() {
    paw_launch(paw_ne_mm_kernel,
        ggml_cuda_kernel_launch_params(grid, dim3(128, 1, 1), 0, ctx.stream()),
        (const uint8_t *) packed->data,
        (const half    *) gscale->data,
        (const half    *) lut->data,
        (const float   *) x->data,
        (float         *) dst->data,
        B, T, k, ng, rpc, nt);
    });
}

//
// RT_MM — 3 kernels: u = H(su ⊙ x) per token -> K4 V2 trellis walk ->
// y = sv ⊙ H(v) (paw_rt_u/rt_walk/rt_out.comp). Scratch: u [nt, n] then
// v [nt, m] in one pool allocation.
//

// WG is a template knob (GGML_PAW_FWHT_WG): every loop strides by WG and
// each FWHT pass is elementwise, so the results are bit-identical for any WG.
template <int WG>
static __global__ void paw_rt_u_kernel(
        const float * GGML_CUDA_RESTRICT su,
        const float * GGML_CUDA_RESTRICT x,
        float       * GGML_CUDA_RESTRICT scr_u,
        const int n,
        const int blk) {
    // blk == n is the legacy single-Hadamard path: the loop runs once with
    // off == 0 and the arithmetic is identical to before blocking existed.
    // blockIdx.y selects the rotation chunk so chunks run on separate SMs
    // instead of serializing inside one block (per-chunk math is unchanged).
    __shared__ float sh[4096];

    const int t   = blockIdx.x;
    const int off = blockIdx.y * blk;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float sc     = __fsqrt_rn((float) blk);
    const float inv_sc = __frcp_rn(sc);
    {
        for (int i = tid; i < blk; i += WG) {
            sh[i] = su[off + i] * x[(int64_t) t*n + off + i];
        }
        __syncthreads();
        if (WG == blk/16 && paw_fwht_v2_ok(blk)) {
            paw_fwht_block_v2(sh, blk, tid, WG);
        } else {
            paw_fwht_block(sh, blk, tid, WG);
        }
        for (int i = tid; i < blk; i += WG) {
            scr_u[(int64_t) t*n + off + i] = sh[i] * inv_sc;
        }
    }
}

// WG covers one thread per column tile (tid < tiles_y walk); WG=128 avoids
// launching idle warps when tiles_y <= 128 (n <= 2048). The dropped warps
// only ever contributed exact zeros to the cross-warp reduction.
// WORDS is the trellis rate in int16 words per 16x16 tile: bits-per-tile is
// WORDS*16 over 128 states, so step = WORDS/8 fresh bits per state and the rate
// is K = step/V = WORDS/16 at V=2. 64 -> K=4 (the shipped 35B payload),
// 48 -> K=3, 32 -> K=2, 24 -> K=1.5, 16 -> K=1.
//
// Only WORDS=64 gives byte-aligned state windows, and only 16/32/64 give
// word-aligned tile rows, so the window is extracted by global bit offset
// rather than from a per-row register cache. At WORDS=64 that folds back to
// exactly the hand-unrolled arithmetic this replaced.
template <int WG, int WORDS>
static __global__ void paw_rt_walk_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[16][8];   // >= n_warps for warp_size 32 (and 64 on HIP)
    __shared__ float slut[1024];   // full F16 [2,512] tlut, staged as fp32

    const int t   = blockIdx.z;
    const int tr  = blockIdx.x;
    const int tid = threadIdx.x;

    static_assert(WORDS % 8 == 0 && WORDS >= 16 && WORDS <= 64, "unsupported trellis rate");
    constexpr int step  = WORDS/8;   // K*V fresh bits per state, V = 2
    constexpr int words = WORDS;     // int16 words per 16x16 tile
    const int tiles_y = n / 16;

    ggml_cuda_pdl_sync();

    // 2 KB table: gather from shared instead of global (same fp32 values —
    // the __half2float just moves ahead of the gather)
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = __half2float(tlut[i]);
    }
    __syncthreads();

    float partial[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        partial[i] = 0.0f;
    }

    // One thread per tile column, strided so n is not capped at WG*16. This
    // used to be a bare `if (tid < tiles_y)`, which silently dropped every
    // column past 4096 -- invisible until now because K=4 takes the bank path
    // and the walk only ran on the narrow GDN projections. For tiles_y <= WG
    // the loop runs exactly once at c == tid, so the arithmetic is unchanged.
    for (int c = tid; c < tiles_y; c += WG) {
        const int64_t tw = ((int64_t) tr*tiles_y + c)*words;
        const float * ub = scr_u + (int64_t) t*n + c*16;

        // the tile column's 16 u values are shared by all 16 tile rows
        float uu[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uu[c] = ub[c];
        }

        for (int ri = 0; ri < 16; ++ri) {
            // 8 states per tile row, each a 16-bit window starting at bit
            // step*(8*ri + j) of the tile's MSB-first stream, wrapping at the end
            uint32_t ph[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int b   = step*(8*ri + j);
                const int wi  = (b >> 4) % words;
                const int off = b & 15;
                const uint32_t w0 = trellis[tw + wi];
                const uint32_t st = off == 0 ? w0
                    : (((w0 << off) | (trellis[tw + (wi + 1) % words] >> (16 - off)))
                       & 0xFFFFu);
                ph[j] = st*(st + 1u);
            }
            // same accumulation order as the serial walk: per state, w0 then w1
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t row = (ph[j] >> 6) & 511u;
                float a0 = slut[2*row + 0];
                const float a1 = slut[2*row + 1];
                if (ph[j] & 0x8000u) {
                    a0 = -a0;                             // exact in fp16 (sign bit)
                }
                acc += a0*uu[2*j] + a1*uu[2*j + 1];
            }
            partial[ri] += acc;
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
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
        scr_v[(int64_t) t*m + tr*16 + tid] = sum;
    }
}

// --- QTIP-style tensor-core walk: decode-from-compressed-trellis fused
// with mma.sync accumulate, GGML_PAW_RT_WALK_QTIP=1, nt==1 only ---
//
// rt_bank_gemv (the current default at nt<4) reads a pre-decoded fp16 bank
// (2 bytes/weight) -- confirmed memory-bandwidth-bound (~590 GB/s
// effective). The earlier rt_apply_kernel_mma attempts (both reverted,
// see "Already tried") tried to speed up compute on top of THAT same
// bandwidth-bound bank read -- Amdahl's law says that can't win: if a
// kernel is bandwidth-bound, faster compute doesn't help, you still stall
// on the same DRAM traffic. That's the likely reason both attempts came
// back negative regardless of design.
//
// This is a different bet: read the COMPRESSED trellis directly (K4 =
// 4 bits/weight = 0.5 bytes -- 4x less than the fp16 bank) and decode
// on the fly, fused with the matvec via raw-PTX mma.sync (not wmma --
// wmma has no m16n8k16 fp16 shape, confirmed by an earlier failed build
// attempt; QTIP's own reference kernel uses this exact raw-PTX shape for
// the same reason). Even wasting 7/8 of the N=8 MMA tile (nt=1 padded)
// should still win on bandwidth alone, since tensor-core throughput has
// enormous headroom over what's needed here -- the goal is fewer DRAM
// bytes, not faster FMA.
//
// The A/B (m16n8k16) fragment-to-lane mapping below was verified against
// a CPU reference in an isolated standalone test before being written
// here (not just recalled from the PTX ISA docs from memory) -- see
// mma_layout_test.cu in the session scratch dir. Decode math (per-row
// state walk, tlut lookup, sign flip) is copied verbatim from
// paw_rt_walk_kernel above, just reorganized: one lane decodes one full
// row (16 columns at once, same j=0..7 loop) into shared memory, then all
// 32 lanes read out the specific (row,col) pairs mma.sync needs for their
// fragment slot -- necessary because decode naturally produces "one row,
// all columns" per step, while the MMA fragment layout needs "one lane,
// two arbitrary (row,col) pairs" -- these are different axes, so a
// shared-memory handoff (not a redistribution I can avoid) sits between
// decode and mma. One block = one warp = one 16-row output tile; grid.x =
// m/16 (matches rt_bank_gemv's block count, avoiding the v1 occupancy
// mistake from the rt_apply attempts).
// rate-templated twin of paw_rt_walk_qtip_kernel: same mma.sync epilogue,
// but the per-row state windows come from the generic bit-stream math
// (paw_rt_walk_kernel) instead of the K4-only (4*ri+q)&63 wrap, so the
// dense payload's K1/K1.5 rates get the register-walk fast path too.
template <int WORDS>
static __global__ void paw_rt_walk_qtip_rate_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16, [2,512]
        const float    * GGML_CUDA_RESTRICT scr_u,    // [nt=1, n]
        float          * GGML_CUDA_RESTRICT scr_v,    // [nt=1, m]
        const int m, const int n) {
    // stride-3 padding: a packed [512][2] float table makes every warp's
    // random row gather land on even banks only (2+ way conflicts); with a
    // stride of 3 the bank index (3*row)%32 covers all 32 banks uniformly
    __shared__ float  slut[512*3];
    // per-warp decode scratch: each warp owns an independent output tile
    __shared__ half   tile16[8][16][16];
    __shared__ half   bcol[8][16];   // this K-step's 16 activation values (col 0 real, rest 0)

    constexpr int step  = WORDS/8;   // fresh bits per state, V = 2
    // one BLOCK per output tile; four warps SPLIT the tile-column range so
    // small-m matrices (down proj: only m/16 tiles) still fill the machine
    // -- the walk is latency-bound, memory-level parallelism is everything
    const int wid  = threadIdx.x >> 5;
    const int tr   = blockIdx.x;
    const int lane = threadIdx.x & 31;

    const int tiles_y = n / 16;

    ggml_cuda_pdl_sync();

    for (int i = threadIdx.x; i < 512; i += 256) {
        slut[3*i + 0] = __half2float(tlut[2*i + 0]);
        slut[3*i + 1] = __half2float(tlut[2*i + 1]);
    }
    __syncthreads();   // all four warps share this table

    const int groupID = lane >> 2;
    const int tid4     = lane & 3;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // the per-lane state windows (bit offsets into the tile's stream) do not
    // depend on ct -- only the tile base word moves -- so the trellis loads
    // are software-pipelined one ct iteration ahead: issue iteration ct+8's
    // loads before consuming iteration ct's, giving them a full MMA-plus-
    // decode of latency slack instead of stalling the loop every step
    const int ri = lane >> 1;
    const int jb = (lane & 1)*4;
    int      wi0[4], wi1[4], offv[4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        const int b = step*(8*ri + jb + jj);
        wi0[jj]  = (b >> 4) % WORDS;
        offv[jj] = b & 15;
        wi1[jj]  = (wi0[jj] + 1) % WORDS;
    }
    uint32_t c0[4], c1[4];
    float    ucur = 0.f;
    {
        const int64_t tw = ((int64_t) tr*tiles_y + wid)*WORDS;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            c0[jj] = trellis[tw + wi0[jj]];
            c1[jj] = offv[jj] != 0 ? trellis[tw + wi1[jj]] : 0u;
        }
        if (lane < 16) {
            ucur = scr_u[wid*16 + lane];
        }
    }

    // gridDim.y > 1: several blocks SHARE one output tile, each walking a
    // strided subset of the tile columns (the serial chain per warp shrinks
    // by gridDim.y); partials are atomic-added into scr_v
    for (int ct = wid + 8*blockIdx.y; ct < tiles_y; ct += 8*gridDim.y) {
        // --- prefetch this lane's windows for the NEXT ct this warp owns ---
        const int ctn = ct + 8*gridDim.y;
        uint32_t n0[4], n1[4];
        float    unext = 0.f;
        if (ctn < tiles_y) {
            const int64_t twn = ((int64_t) tr*tiles_y + ctn)*WORDS;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                n0[jj] = trellis[twn + wi0[jj]];
                n1[jj] = offv[jj] != 0 ? trellis[twn + wi1[jj]] : 0u;
            }
            if (lane < 16) {
                unext = scr_u[ctn*16 + lane];
            }
        }
        // --- decode this (tr, ct) 16x16 tile: lanes 0-15 each decode one
        // full row (16 columns), generic rate-aware window math ---
        {
            // all 32 lanes decode: two lanes per row, four states each --
            // halves the serial chain vs one lane doing the whole row
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j   = jb + jj;
                const int off = offv[jj];
                const uint32_t w0 = c0[jj];
                const uint32_t st = off == 0 ? w0
                    : (((w0 << off) | (c1[jj] >> (16 - off)))
                       & 0xFFFFu);
                const uint32_t ph = st*(st + 1u);
                const uint32_t row = (ph >> 6) & 511u;
                float a0 = slut[3*row + 0];
                const float a1 = slut[3*row + 1];
                if (ph & 0x8000u) {
                    a0 = -a0;
                }
                tile16[wid][ri][2*j + 0] = __float2half_rn(a0);
                tile16[wid][ri][2*j + 1] = __float2half_rn(a1);
            }
        }
        // --- stage this K-step's 16 activation values (nt==1: col 0 real) ---
        if (lane < 16) {
            bcol[wid][lane] = __float2half_rn(ucur);   // nt==1: one shared u vector
        }
        __syncwarp();

        // --- build A/B fragments per the validated m16n8k16 layout ---
        auto Aat = [&](int r, int c) -> half { return tile16[wid][r][c]; };
        half2 a01 = __halves2half2(Aat(groupID,   tid4*2+0), Aat(groupID,   tid4*2+1));
        half2 a23 = __halves2half2(Aat(groupID+8, tid4*2+0), Aat(groupID+8, tid4*2+1));
        half2 a45 = __halves2half2(Aat(groupID,   tid4*2+8), Aat(groupID,   tid4*2+9));
        half2 a67 = __halves2half2(Aat(groupID+8, tid4*2+8), Aat(groupID+8, tid4*2+9));
        uint32_t ra0 = *(uint32_t*)&a01;
        uint32_t ra1 = *(uint32_t*)&a23;
        uint32_t ra2 = *(uint32_t*)&a45;
        uint32_t ra3 = *(uint32_t*)&a67;

        // B is [16,1] logically (nt=1, this K-step's 16 real activation
        // values), broadcast to all 8 MMA columns by reading the SAME bcol
        // regardless of groupID -- since B doesn't vary by column, every
        // output column ends up numerically identical, so any one of them
        // (tid4==0 below) is a valid read of the real result. Wasteful
        // (computes the same dot product 8x redundantly across tid4) but
        // not incorrect, and simpler than truly zero-padding 7 columns.
        half2 b01 = __halves2half2(bcol[wid][tid4*2+0], bcol[wid][tid4*2+1]);
        half2 b23 = __halves2half2(bcol[wid][tid4*2+8], bcol[wid][tid4*2+9]);
        uint32_t rb0 = *(uint32_t*)&b01;
        uint32_t rb1 = *(uint32_t*)&b23;

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1)
        );
        __syncwarp();   // tile16/bcol reused by the next ct

        // slide the prefetched windows down for the next iteration
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            c0[jj] = n0[jj];
            c1[jj] = n1[jj];
        }
        ucur = unext;
    }

    // only tid4==0 (groupID's col 0 slot) is the real token; d0 -> row
    // groupID, d2 -> row groupID+8 (per the validated D-fragment layout).
    // eight warps covered disjoint ct subsets -- reduce through shared
    __shared__ float xred[8][16];
    if (tid4 == 0) {
        xred[wid][groupID]     = d0;
        xred[wid][groupID + 8] = d2;
    }
    __syncthreads();
    if (wid == 0 && tid4 == 0) {
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float sum = xred[0][r] + xred[1][r] + xred[2][r] + xred[3][r]
                            + xred[4][r] + xred[5][r] + xred[6][r] + xred[7][r];
            if (gridDim.y == 1) {
                scr_v[(int64_t) blockIdx.z*m + tr*16 + r] = sum;
            } else {
                atomicAdd(&scr_v[(int64_t) blockIdx.z*m + tr*16 + r], sum);
            }
        }
    }
}


// fragment-direct qtip walk: each lane decodes EXACTLY its own mma.sync
// fragment elements straight out of the trellis stream -- the four states a
// lane needs are fixed by the m16n8k16 layout ((row=g,g+8) x (j=tid4,tid4+4)),
// so no shared-memory staging, no syncwarp in the loop, no cross-lane
// exchange at all. Verified bit-exact against paw_rt_walk_qtip_rate_kernel
// in an isolated probe before landing here.
// The HYB codebook gather is what this kernel is bound on: ncu puts l1tex at
// 60.7% with dram at 16.0% and the tensor pipe at 9.1%.  Reading `tlut` from
// global memory is 32 divergent addresses into a 2 KiB table, which the
// coalescer cannot merge -- ~25 distinct sectors, so the LSU replays the one
// instruction that matters ~25 times, four times per lane per mma.
//
// QTIP's own kernel (Cornell-RelaxML/qtip, qtip-kernels/src/inference.cu:352)
// duplicates the codebook 32x in shared memory, one private copy per lane:
//     masked_idx = (idx & 0b0111111111000000) | (laneId << 1)
// so the half2 index is (row << 5) | lane and the bank is
// (row*32 + lane) % 32 == lane -- a distinct bank per lane for *every* row,
// conflict-free by construction.  QTIP S4: "2KiB codebook, which fits in L1
// cache even after duplication for bank conflicts (32x)".  The duplication is
// free in instruction count: the `| lane` replaces the `>> 6`.
//
// Staging alone (one shared copy, no duplication) was measured earlier and is
// a null -- it trades a global replay for a ~4-way bank replay.  Replication,
// not relocation, is the fix.
//
// 64 KiB of shared forces one block per SM, so the block is widened from 8 to
// 32 warps to hold occupancy, matching QTIP's BLOCK_SIZE 1024 with
// __launch_bounds__(BLOCK_SIZE, 1).
// Replication factor is a tradeoff, not a constant.  ncu on the 32x/1024-thread
// build: shared bank conflicts 0 and global sectors/request 1.01 -- the gather
// is exactly as clean as intended -- but l1tex only fell 60.68% -> 59.14% and
// the time did not move, because 64 KiB of shared plus 40 regs/thread pins one
// block per SM and warps_active fell 73.4% -> 62.6%.  The counters also show
// why the ceiling was low: global loads outnumber shared loads 1,234,944 to
// 491,520, so the codebook gather was only ~28% of LSU instructions and could
// never have been the whole 81%.
//
// 16 copies at 512 threads restores full occupancy: 32 KiB and 40 regs both
// allow 3 blocks/SM = 1536 threads = 48 warps, the SM 8.6 maximum.  Lanes l
// and l+16 share a copy, so a conflict needs row_l == row_{l+16} (mod 2) --
// ~1.5 wavefronts expected instead of 1.0, against a 1.5x occupancy gain.
#define PAW_WALK_COPIES 16
#define PAW_WALK_LOG2C  4
#define PAW_WALK_WPB    16                             // warps per block
#define PAW_WALK_NTHR   (32*PAW_WALK_WPB)              // 512 threads
#define PAW_WALK_BPSM   3                              // blocks per SM
#define PAW_WALK_CBSZ   ((size_t) 512*PAW_WALK_COPIES*sizeof(uint32_t))

template <int WORDS, bool MT, bool COMPUTED = false>
__launch_bounds__(PAW_WALK_NTHR, PAW_WALK_BPSM)
static __global__ void paw_rt_walk_qtip_frag_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,
        const float    * GGML_CUDA_RESTRICT scr_u,   // [nt, n]
        float          * GGML_CUDA_RESTRICT scr_v,   // [nt, m]
        const int m, const int n, const int nt) {
    constexpr int step = WORDS/8;
    const int wid  = threadIdx.x >> 5;
    const int tr   = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int tiles_y = n / 16;

    // 32 private lane-copies of the 512-entry half2 codebook.  The XOR
    // swizzle on the write side keeps the fill itself conflict-free (all 32
    // lanes would otherwise hit one bank at each c); permuting c is harmless
    // because every c is written.
    // COMPUTED variant: no codebook at all -- the emission is a short integer
    // chain (see dec_state), so shared stays empty and occupancy rises.
    extern __shared__ uint32_t paw_smem_cb[];
    if constexpr (!COMPUTED) {
        const uint32_t * GGML_CUDA_RESTRICT cb = (const uint32_t *) tlut;
        const int sw = lane & (PAW_WALK_COPIES - 1);
        for (int r = threadIdx.x; r < 512; r += PAW_WALK_NTHR) {
            const uint32_t v = cb[r];
    #pragma unroll
            for (int c = 0; c < PAW_WALK_COPIES; ++c)
                paw_smem_cb[(r << PAW_WALK_LOG2C) | (c ^ sw)] = v;
        }
        __syncthreads();
    }
    const int groupID = lane >> 2;
    const int tid4    = lane & 3;

    const int rrows[2] = {groupID, groupID + 8};
    const int rjs[2]   = {tid4, tid4 + 4};
    int wi0[2][2], wi1[2][2], offv[2][2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int b = step*(8*rrows[k] + rjs[q]);
            wi0[k][q]  = (b >> 4) % WORDS;
            offv[k][q] = b & 15;
            wi1[k][q]  = (wi0[k][q] + 1) % WORDS;
        }

    // B fragment: this K-step's (== this tile column's) 16 activation
    // values; the lane needs k in {tid4*2,+1, tid4*2+8,+9}
    // The mma's B fragment is k16 x n8 and its column index IS groupID, so
    // the eight columns can carry eight different tokens for one walk of the
    // trellis. ncu says the walk is LSU-bound (l1tex 60.7%) with the tensor
    // pipe idle at 9.1%: the gathers, not the MMA, are the cost, and the
    // gather count per *token* is what multi-token divides. nt==1 keeps the
    // old numerics exactly -- only column 0 is ever read out, and the other
    // columns going to zero instead of a redundant copy cannot change it.
    auto load_b = [&](int ct, half2& h0, half2& h1) {
        // ub + tid4*2 and +8 are each two consecutive floats, 8-byte aligned
        // (n is a multiple of 16), so one LDG.64 each replaces two LDG.32.
        auto pack = [&](const float * ub, half2& a, half2& b) {
            const float2 v0 = *(const float2 *)(const void *)(ub + tid4*2);
            const float2 v1 = *(const float2 *)(const void *)(ub + tid4*2 + 8);
            a = __floats2half2_rn(v0.x, v0.y);
            b = __floats2half2_rn(v1.x, v1.y);
        };
        if constexpr (!MT) {
            pack(scr_u + ct*16, h0, h1);
        } else if (groupID < nt) {
            pack(scr_u + (int64_t) groupID*n + ct*16, h0, h1);
        } else {
            h0 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
            h1 = h0;
        }
    };
    half2 hb0 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
    half2 hb1 = hb0;
    if (wid < tiles_y) load_b(wid, hb0, hb1);
    const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;

    // WORDS==16 packed path: a lane's four states live in just two unit
    // pairs ((ri,ri+1) for ri = g and g+8), so each pair is fetched once
    // per column -- one aligned u32 load when the pair does not wrap --
    // instead of eight separate u16 gathers.
    auto fetch_pair = [&](int64_t tw, int u, uint32_t& lo_u, uint32_t& hi_u) {
        const int v = (u + 1) % WORDS;
        if ((u & 1) == 0 && v == u + 1) {
            const uint32_t W = *(const uint32_t*)(const void*)(trellis + tw + u);
            lo_u = W & 0xFFFFu;
            hi_u = W >> 16;
        } else {
            lo_u = __ldg(trellis + tw + u);
            hi_u = __ldg(trellis + tw + v);
        }
    };
    auto dec_state = [&](uint32_t lo, uint32_t hi, int off) {
        const uint32_t st = off == 0 ? lo
            : (((lo << off) | (hi >> (16 - off))) & 0xFFFFu);
        if constexpr (COMPUTED) {
            // Lookup-free emission prototype (Track B timing probe): the
            // table gather is replaced by an integer mixing chain on the
            // ALU pipe -- zero LSU traffic, zero shared memory, no bank
            // conflicts, and the 2 KiB codebook disappears from the kernel
            // entirely. Values are NOT the codec's (encoder not re-run);
            // this exists to measure the speed of the gather-free walk.
            const uint32_t x  = st * 2246822519u;   // 1 IMAD
            const uint32_t s  = (x >> 4) & 0x8000u; // a0 sign
            const uint32_t a0 = 0x3400u | ((x >> 5) & 0x3FFu);
            const uint32_t a1 = 0x3400u | ((x >> 9) & 0x3FFu);
            return (a1 << 16) | s | a0;
        } else {
            const uint32_t ph  = st*(st + 1u);
            // ((ph >> 6) & 511) << LOG2C | (lane % COPIES), one shift + and + or
            const uint32_t pair = paw_smem_cb[
                ((ph >> (6 - PAW_WALK_LOG2C)) & (511u << PAW_WALK_LOG2C))
                | (lane & (PAW_WALK_COPIES - 1))];
            return pair ^ (ph & 0x8000u);   // == (ph & 0x8000) ? pair ^ 0x8000 : pair
        }
    };

    auto load_at = [&](int64_t tw, uint32_t (&a)[2][2], uint32_t (&b)[2][2]) {
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                a[k][q] = __ldg(trellis + tw + wi0[k][q]);
                b[k][q] = offv[k][q] ? __ldg(trellis + tw + wi1[k][q]) : 0u;
            }
    };
    // gridDim.y > 1: several blocks share an output tile, each walking a
    // strided subset of its columns (small-m matrices otherwise leave the
    // machine idle); partials fold through the atomicAdd epilogue
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const int ct_first = wid + PAW_WALK_WPB*(int) blockIdx.y;

    if constexpr (WORDS == 16) {
        uint32_t L0, H0, L8, H8, nL0, nH0, nL8, nH8;
        L0 = H0 = L8 = H8 = nL0 = nH0 = nL8 = nH8 = 0;
        const bool has = wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + PAW_WALK_WPB - 1)/PAW_WALK_WPB);
        if (has) {
            const int64_t tw = (int64_t) tr*tiles_y*WORDS + ct_first*WORDS;
            fetch_pair(tw, groupID, L0, H0);
            fetch_pair(tw, (groupID + 8) % WORDS, L8, H8);
        }
        for (int ct = ct_first; ct < tiles_y; ct += PAW_WALK_WPB*gridDim.y) {
            const int ctn = ct + PAW_WALK_WPB*gridDim.y;
            if (ctn < tiles_y) {
                const int64_t twn = (int64_t) tr*tiles_y*WORDS + ctn*WORDS;
                fetch_pair(twn, groupID, nL0, nH0);
                fetch_pair(twn, (groupID + 8) % WORDS, nL8, nH8);
            }
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            // ra order: [k + 2*q]; k=row-group (g,g+8), q=j-pair (t,t+4)
            const uint32_t ra0 = dec_state(L0, H0, offv[0][0]);
            const uint32_t ra1 = dec_state(L8, H8, offv[1][0]);
            const uint32_t ra2 = dec_state(L0, H0, offv[0][1]);
            const uint32_t ra3 = dec_state(L8, H8, offv[1][1]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
            if (ctn < tiles_y) { L0=nL0; H0=nH0; L8=nL8; H8=nH8; }
            hb0 = nb0; hb1 = nb1;
        }
    } else if constexpr (WORDS == 24) {
        // w24: state (ri, j=t) starts at bit 24*ri + 3t; the j=t+4 state
        // starts 12 bits later -- same unit pair unless that crosses into
        // the next unit (then one extra unit covers it). Two fetch_pairs
        // plus occasional single units replace eight per-state gathers.
        uint32_t L0, H0, X0, L8, H8, X8, nL0, nH0, nX0, nL8, nH8, nX8;
        uint32_t o0[2], o2[2]; int u0[2], u2[2];
        L0=H0=X0=L8=H8=X8=nL0=nH0=nX0=nL8=nH8=nX8=0;
        const bool has = wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + PAW_WALK_WPB - 1)/PAW_WALK_WPB);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int bk = 3*(8*rrows[k] + tid4);
            u0[k]  = (bk >> 4) % WORDS;
            o0[k]  = bk & 15u;
            const int cross = (o0[k] + 12) >> 4;   // 0 or 1
            u2[k]  = (u0[k] + cross) % WORDS;
            o2[k]  = (o0[k] + 12) & 15u;
        }
        if (has) {
            const int64_t tw = (int64_t) tr*tiles_y*WORDS + ct_first*WORDS;
            fetch_pair(tw, u0[0], L0, H0);
            fetch_pair(tw, u0[1], L8, H8);
            X0 = __ldg(trellis + tw + ((u0[0] + 2) % WORDS));
            X8 = __ldg(trellis + tw + ((u0[1] + 2) % WORDS));
            (void) u2; (void) o2;
        }
        for (int ct = ct_first; ct < tiles_y; ct += PAW_WALK_WPB*gridDim.y) {
            const int ctn = ct + PAW_WALK_WPB*gridDim.y;
            if (ctn < tiles_y) {
                const int64_t twn = (int64_t) tr*tiles_y*WORDS + ctn*WORDS;
                fetch_pair(twn, u0[0], nL0, nH0);
                fetch_pair(twn, u0[1], nL8, nH8);
                nX0 = __ldg(trellis + twn + ((u0[0] + 2) % WORDS));
                nX8 = __ldg(trellis + twn + ((u0[1] + 2) % WORDS));
            }
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t lo = k ? L8 : L0, hi = k ? H8 : H0;
                const uint32_t xx = k ? X8 : X0;
                // j=t state: units (lo,hi) at offset o0[k]
                ra[k + 2*0] = dec_state(lo, hi, o0[k]);
                // j=t+4 state: starts 12 bits later
                ra[k + 2*1] = ((o0[k] + 12) >> 4)
                    ? dec_state(hi, xx, o2[k])     // crossed into next unit
                    : dec_state(lo, hi, o2[k]);    // still inside the pair
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
            if (ctn < tiles_y) {
                L0=nL0; H0=nH0; X0=nX0; L8=nL8; H8=nH8; X8=nX8;
            }
            hb0 = nb0; hb1 = nb1;
        }
    } else if constexpr (WORDS == 32) {
        // w32: state (ri, j) starts at bit 32*ri + 4*j -> unit
        // 2*ri + j/4, offset 4*(j&3). The j=t+4 state lives one unit
        // later at the SAME offset, so each row-group needs units
        // {u, u+1, u+2}: one fetch_pair plus a single extra unit.
        uint32_t L0, H0, X0, L8, H8, X8, nL0, nH0, nX0, nL8, nH8, nX8;
        uint32_t ov[2]; int uv[2];
        L0=H0=X0=L8=H8=X8=nL0=nH0=nX0=nL8=nH8=nX8=0;
        const bool has = wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + PAW_WALK_WPB - 1)/PAW_WALK_WPB);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int b = 4*(8*rrows[k] + tid4);
            uv[k] = (b >> 4) % WORDS;
            ov[k] = b & 15u;
        }
        if (has) {
            const int64_t tw = (int64_t) tr*tiles_y*WORDS + ct_first*WORDS;
            fetch_pair(tw, uv[0], L0, H0);
            fetch_pair(tw, uv[1], L8, H8);
            X0 = __ldg(trellis + tw + ((uv[0] + 2) % WORDS));
            X8 = __ldg(trellis + tw + ((uv[1] + 2) % WORDS));
        }
        for (int ct = ct_first; ct < tiles_y; ct += PAW_WALK_WPB*gridDim.y) {
            const int ctn = ct + PAW_WALK_WPB*gridDim.y;
            if (ctn < tiles_y) {
                const int64_t twn = (int64_t) tr*tiles_y*WORDS + ctn*WORDS;
                fetch_pair(twn, uv[0], nL0, nH0);
                fetch_pair(twn, uv[1], nL8, nH8);
                nX0 = __ldg(trellis + twn + ((uv[0] + 2) % WORDS));
                nX8 = __ldg(trellis + twn + ((uv[1] + 2) % WORDS));
            }
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t lo = k ? L8 : L0, hi = k ? H8 : H0;
                const uint32_t xx = k ? X8 : X0;
                // j=t state from units (lo,hi); j=t+4 from (hi,x) -- same offset
                ra[k + 2*0] = dec_state(lo, hi, ov[k]);
                ra[k + 2*1] = dec_state(hi, xx, ov[k]);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
            if (ctn < tiles_y) {
                L0=nL0; H0=nH0; X0=nX0; L8=nL8; H8=nH8; X8=nX8;
            }
            hb0 = nb0; hb1 = nb1;
        }
    } else {
        uint32_t c0[2][2], c1[2][2];
        if (wid < tiles_y && blockIdx.y < (unsigned)((tiles_y + PAW_WALK_WPB - 1)/PAW_WALK_WPB))
            load_at((int64_t) tr*tiles_y*WORDS + ct_first*WORDS, c0, c1);

        for (int ct = ct_first; ct < tiles_y; ct += PAW_WALK_WPB*gridDim.y) {
            const int ctn = ct + PAW_WALK_WPB*gridDim.y;
            uint32_t n0[2][2], n1[2][2];
            if (ctn < tiles_y) load_at((int64_t) tr*tiles_y*WORDS + ctn*WORDS, n0, n1);
            half2 nb0 = hb0, nb1 = hb1;
            if (ctn < tiles_y) load_b(ctn, nb0, nb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k)
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int off = offv[k][q];
                    const uint32_t w0 = c0[k][q];
                    const uint32_t st = off == 0 ? w0
                        : (((w0 << off) | (c1[k][q] >> (16 - off))) & 0xFFFFu);
                    const uint32_t ph  = st*(st + 1u);
                    const uint32_t row = (ph >> 6) & 511u;
                        // packed u32 tlut access: lo half = a0 (sign-flippable by
                        // xor on the fp16 sign bit), hi half = a1
                    const uint32_t pair = *(const uint32_t*)(const void*)(tlut + 2*row);
                    const uint32_t a0h  = (ph & 0x8000u) ? (pair ^ 0x00008000u) : pair;
                    half2 h = __halves2half2(__ushort_as_half((unsigned short) a0h),
                                             __ushort_as_half((unsigned short)(pair >> 16)));
                    ra[k + 2*q] = *(uint32_t*)&h;   // (row-group, j-pair): q is the outer axis
                }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
#pragma unroll
            for (int k = 0; k < 2; ++k)
#pragma unroll
                for (int q = 0; q < 2; ++q) { c0[k][q] = n0[k][q]; c1[k][q] = n1[k][q]; }
            hb0 = nb0; hb1 = nb1;
        }
    }
    // D fragment m16n8: (row groupID, col 2*tid4) -> d0, (col 2*tid4+1) -> d1;
    // rows groupID+8 -> d2/d3. Token t lives in column t, i.e. the lanes with
    // tid4 == t>>1, taking the even/odd register of each row pair.
    if constexpr (!MT) {
        if (tid4 == 0) {
            atomicAdd(&scr_v[(size_t) tr*16 + groupID],     d0);
            atomicAdd(&scr_v[(size_t) tr*16 + groupID + 8], d2);
        }
    } else {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < nt && tid4 == (t >> 1)) {
                const float vlo = (t & 1) ? d1 : d0;
                const float vhi = (t & 1) ? d3 : d2;
                atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID],     vlo);
                atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID + 8], vhi);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Staged walk: cp.async the trellis into shared, then walk out of shared.
//
// Why this kernel exists. ncu on the frag walk says global/L2 memory latency
// is 36.8-54.1% of stalls while DRAM runs at ~20% of peak (6.435 GB/token in
// 33.6 ms = 191 GB/s of 936). Those coexist because of bytes per load
// instruction: at 1.0-1.26 sectors/request each LDG moves ~32 bytes for the
// whole warp, since all 32 lanes are picking bits out of one 32-48 byte tile.
// No instruction rate the SM can issue turns that into bandwidth. Four
// attempts that did not address this were null or negative -- see
// reports/paw27b_speed_baseline_20260824.md.
//
// For a fixed output row-block tr the tiles ct = 0..tiles_y-1 are CONTIGUOUS
// in the payload (tr*tiles_y*WORDS + ct*WORDS), so a whole chunk of them can
// be pulled in with 16-byte-per-thread cp.async: 512 threads x 16 B = 8 KiB
// per issue, perfectly coalesced, instead of ~6-8 narrow LDG per tile. Every
// shipped rate has a tile size divisible by 16 bytes (WORDS*2 = 32/48/64), so
// both ends stay aligned for any tile count.
//
// The codebook drops to 8 copies (16 KiB) to leave room: bank conflicts on
// that gather are worth ~1.8% of stalls, so trading a little of it for the
// staging buffer is the right side of the trade.
// Tile columns per chunk, picked per rate so the staging buffer stays ~7.5 KiB
// (a tile is WORDS*2 bytes), and a multiple of the warp count so every warp
// gets whole tiles. 16 KiB codebook + 2 x ~7.5 KiB still leaves 3 blocks/SM.
template <int WORDS> struct paw_stage_tiles { static constexpr int v = 160; };
template <> struct paw_stage_tiles<16> { static constexpr int v = 240; };  // 7680 B
template <> struct paw_stage_tiles<24> { static constexpr int v = 160; };  // 7680 B
template <> struct paw_stage_tiles<32> { static constexpr int v = 112; };  // 7168 B
template <> struct paw_stage_tiles<64> { static constexpr int v =  64; };  // 8192 B
#define PAW_STAGE_COPIES 8
#define PAW_STAGE_LOG2C  3

static __device__ __forceinline__ void paw_cp_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
static __device__ __forceinline__ void paw_cp_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif
}

template <int WORDS, bool MT, bool COMPUTED = false, bool ALIGNED = false, bool REGRES = false>
__launch_bounds__(PAW_WALK_NTHR, PAW_WALK_BPSM)
static __global__ void paw_rt_walk_qtip_stage_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,
        const float    * GGML_CUDA_RESTRICT scr_u,   // [nt, n]
        float          * GGML_CUDA_RESTRICT scr_v,   // [nt, m]
        const int m, const int n, const int nt) {
    static_assert(WORDS == 16 || WORDS == 24 || WORDS == 32 || WORDS == 64,
                  "staged walk covers the shipped rates only");
    const int wid     = threadIdx.x >> 5;
    const int tr      = blockIdx.x;
    const int lane    = threadIdx.x & 31;
    const int tiles_y = n / 16;
    const int groupID = lane >> 2;
    const int tid4    = lane & 3;

    constexpr int CB_U32  = 512*PAW_STAGE_COPIES;
    constexpr int TILES   = paw_stage_tiles<WORDS>::v;
    constexpr int CHUNK_W = TILES*WORDS;                // uint16 per buffer

    extern __shared__ uint32_t paw_stage_smem[];
    uint32_t * GGML_CUDA_RESTRICT cb  = paw_stage_smem;
    uint16_t * GGML_CUDA_RESTRICT tsm = (uint16_t *)(paw_stage_smem + CB_U32);

    if constexpr (!COMPUTED) {   // 8 lane-copies of the codebook, XOR-swizzled writes
        const uint32_t * GGML_CUDA_RESTRICT src = (const uint32_t *) tlut;
        const int sw = lane & (PAW_STAGE_COPIES - 1);
        for (int r = threadIdx.x; r < 512; r += PAW_WALK_NTHR) {
            const uint32_t v = src[r];
#pragma unroll
            for (int c = 0; c < PAW_STAGE_COPIES; ++c)
                cb[(r << PAW_STAGE_LOG2C) | (c ^ sw)] = v;
        }
    }

    // Same window math as the frag kernel: state (ri, j) starts at bit
    // step*(8*ri + j) with step = WORDS/8, and the j+4 state starts
    // 2*step*4 bits later. Only the wrap indices differ per rate.
    constexpr int step = WORDS/8;
    const int rrows[2] = {groupID, groupID + 8};
    uint32_t o0[2], o2[2]; int u0[2], u1w[2], uxw[2];
    int gw0[2][2], gw1[2][2]; uint32_t gof[2][2];
    if constexpr (WORDS == 64) {
        // generic window, one independent unit pair per (row-group, j-pair):
        // at step 8 the j+4 state is two units on at the same offset, so the
        // 16/24/32 shortcut (which reuses u+1,u+2) does not hold.
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int b = step*(8*rrows[k] + (q ? tid4 + 4 : tid4));
                gw0[k][q] = (b >> 4) % WORDS;
                gof[k][q] = b & 15u;
                gw1[k][q] = (gw0[k][q] + 1) % WORDS;
            }
        (void) o0; (void) o2; (void) u0; (void) u1w; (void) uxw;
    } else {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int bk = step*(8*rrows[k] + tid4);
            u0[k]  = (bk >> 4) % WORDS;
            o0[k]  = bk & 15u;
            o2[k]  = (o0[k] + 4*step) & 15u;
            u1w[k] = (u0[k] + 1) % WORDS;
            uxw[k] = (u0[k] + 2) % WORDS;
        }
        (void) gw0; (void) gw1; (void) gof;
    }

    auto dec_state = [&](uint32_t lo, uint32_t hi, int off) {
        const uint32_t st = off == 0 ? lo
            : (((lo << off) | (hi >> (16 - off))) & 0xFFFFu);
        if constexpr (COMPUTED) {
            // Gather-free emission (timing probe): removes the codebook LDS op
            // from the staged walk, which ncu shows L1TEX-bound at 84%.
            // Values are NOT the codec's -- this measures the gather's cost.
            const uint32_t x  = st * 2246822519u;
            const uint32_t sg = (x >> 4) & 0x8000u;
            const uint32_t a0 = 0x3400u | ((x >> 5) & 0x3FFu);
            const uint32_t a1 = 0x3400u | ((x >> 9) & 0x3FFu);
            return (a1 << 16) | sg | a0;
        } else {
            const uint32_t ph = st*(st + 1u);
            const uint32_t pair = cb[((ph >> (6 - PAW_STAGE_LOG2C)) & (511u << PAW_STAGE_LOG2C))
                                     | (lane & (PAW_STAGE_COPIES - 1))];
            return pair ^ (ph & 0x8000u);
        }
    };
    auto load_b = [&](int ct, half2& h0, half2& h1) {
        auto pack = [&](const float * ub) {
            const float2 v0 = *(const float2 *)(const void *)(ub + tid4*2);
            const float2 v1 = *(const float2 *)(const void *)(ub + tid4*2 + 8);
            h0 = __floats2half2_rn(v0.x, v0.y);
            h1 = __floats2half2_rn(v1.x, v1.y);
        };
        if constexpr (!MT) {
            pack(scr_u + ct*16);
        } else if (groupID < nt) {
            pack(scr_u + (int64_t) groupID*n + ct*16);
        } else {
            h0 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
            h1 = h0;
        }
    };

    // ---- Stage 1: register-resident walk (EXL3 dq8_regs_* shape) ----------
    // K=2 gives step 4, so state (ri, j) starts at bit 4*(8*ri + j): unit
    // u0 = 2*ri (always EVEN) at offset 4*tid4, and the j+4 state is 16 bits
    // on, i.e. units u0+1, u0+2.  Both pairs therefore live inside the two
    // ALIGNED uint32 at indices ri and ri+1 -- so the lane can load them
    // straight to registers with no shared staging and no barrier, which is
    // exactly what exllamav3 does and what our staged walk never tried.
    if constexpr (REGRES && WORDS == 32) {
        __syncthreads();                       // cb fill above must be visible
        const uint16_t * GGML_CUDA_RESTRICT tb = trellis + (int64_t) tr*tiles_y*WORDS;
        float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
        for (int ct = wid; ct < tiles_y; ct += PAW_WALK_WPB) {
            half2 hb0, hb1;
            load_b(ct, hb0, hb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            const uint32_t * GGML_CUDA_RESTRICT g32 =
                (const uint32_t * GGML_CUDA_RESTRICT)(tb + (int64_t) ct*WORDS);
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int bi = groupID + 8*k;              // uint32 index, u0/2
                const uint32_t A = __ldg(g32 + bi);        // units u0, u0+1
                const uint32_t B = __ldg(g32 + ((bi + 1) & 15));  // units u0+2, u0+3
                ra[k + 2*0] = dec_state(A & 0xFFFFu, A >> 16,      o0[k]);
                ra[k + 2*1] = dec_state(A >> 16,     B & 0xFFFFu,  o2[k]);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(r0), "+f"(r1), "+f"(r2), "+f"(r3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
        }
        if constexpr (!MT) {
            if (tid4 == 0) {
                atomicAdd(&scr_v[(size_t) tr*16 + groupID],     r0);
                atomicAdd(&scr_v[(size_t) tr*16 + groupID + 8], r2);
            }
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t < nt && tid4 == (t >> 1)) {
                    atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID],     (t & 1) ? r1 : r0);
                    atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID + 8], (t & 1) ? r3 : r2);
                }
            }
        }
        return;
    }

    const uint16_t * GGML_CUDA_RESTRICT base = trellis + (int64_t) tr*tiles_y*WORDS;
    auto issue = [&](int buf, int c0) {
        const int tiles = (tiles_y - c0) < TILES ? (tiles_y - c0) : TILES;
        if (tiles <= 0) return;
        const int bytes = tiles*WORDS*2;                 // WORDS*2 == 48, a multiple of 16
        char       * dstb = (char *)(tsm + buf*CHUNK_W);
        const char * srcb = (const char *)(base + (int64_t) c0*WORDS);
        for (int off = threadIdx.x*16; off < bytes; off += PAW_WALK_NTHR*16)
            cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(dstb + off), srcb + off);
    };

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    issue(0, 0); paw_cp_commit();
    int buf = 0;
    for (int c0 = 0; c0 < tiles_y; c0 += TILES, buf ^= 1) {
        const int nxt = c0 + TILES;
        if (nxt < tiles_y) { issue(buf ^ 1, nxt); paw_cp_commit(); paw_cp_wait_group<1>(); }
        else               { paw_cp_wait_group<0>(); }
        __syncthreads();

        const uint16_t * GGML_CUDA_RESTRICT sh = tsm + buf*CHUNK_W;
        const int lim = (tiles_y - c0) < TILES ? (tiles_y - c0) : TILES;
        for (int l = wid; l < lim; l += PAW_WALK_WPB) {
            const uint16_t * GGML_CUDA_RESTRICT tw = sh + l*WORDS;
            half2 hb0, hb1;
            load_b(c0 + l, hb0, hb1);
            const uint32_t rb0 = *(uint32_t*)&hb0, rb1 = *(uint32_t*)&hb1;
            uint32_t ra[4];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                if constexpr (ALIGNED) {
                    // Integer-rate access-pattern probe: EXL3 at an integer bpw
                    // pulls 8 weights from ONE aligned 32-bit word (dq8_regs_*),
                    // where our fractional rate needs three scattered uint16
                    // reads.  This swaps in that access pattern -- values are
                    // WRONG, only the L1TEX traffic is meaningful.
                    const uint32_t * GGML_CUDA_RESTRICT tw32 =
                        (const uint32_t * GGML_CUDA_RESTRICT) tw;
                    const uint32_t w32 = tw32[(lane + 8*k) % (WORDS/2)];
                    const uint32_t alo = w32 & 0xFFFFu, ahi = w32 >> 16;
                    ra[k + 2*0] = dec_state(alo, ahi, o0[k]);
                    ra[k + 2*1] = dec_state(alo, ahi, o2[k]);
                    continue;
                }
                if constexpr (WORDS == 64) {
#pragma unroll
                    for (int q = 0; q < 2; ++q)
                        ra[k + 2*q] = dec_state(tw[gw0[k][q]], tw[gw1[k][q]], gof[k][q]);
                    continue;
                }
                const uint32_t lo = tw[u0[k]], hi = tw[u1w[k]];
                ra[k + 2*0] = dec_state(lo, hi, o0[k]);
                if constexpr (WORDS == 16) {
                    // step 2: both j and j+4 sit inside the same unit pair
                    ra[k + 2*1] = dec_state(lo, hi, o2[k]);
                } else {
                    // step 3 and 4 can run past the pair into one more unit
                    const uint32_t xx = tw[uxw[k]];
                    ra[k + 2*1] = ((o0[k] + 4*step) >> 4)
                        ? dec_state(hi, xx, o2[k])
                        : dec_state(lo, hi, o2[k]);
                }
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
        }
        __syncthreads();
    }

    if constexpr (!MT) {
        if (tid4 == 0) {
            atomicAdd(&scr_v[(size_t) tr*16 + groupID],     d0);
            atomicAdd(&scr_v[(size_t) tr*16 + groupID + 8], d2);
        }
    } else {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < nt && tid4 == (t >> 1)) {
                const float vlo = (t & 1) ? d1 : d0;
                const float vhi = (t & 1) ? d3 : d2;
                atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID],     vlo);
                atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID + 8], vhi);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EXL3-style fused-int8 AR walk (nt == 1).
//
// exllamav3's fast K <= 5 GEMV quantizes the activation to signed int8 and
// uses dp4a to evaluate four weights per instruction.  Its mul1 codebook is
// affine in a byte sum, so the decoded weights are already integer-friendly.
// PAW's shipped HYB codebook is different (a 512-entry lookup), but it is a
// 63-level signed lattice: the largest level is 31 times the lattice step.
// Preserve the existing bitstream and hash, quantize that tiny table once per
// CTA from max_abs/31, and use the same int8 dot-product machinery.  For the
// PAW-27B-v4 F16 table the measured table-only RMS error of this conversion is
// 2.19e-4 relative (max absolute error 8.20e-4).
//
// A warp owns one output row and its lanes own different 16-value input
// tiles.  This is deliberately an AR kernel: unlike the tensor-core path it
// computes no seven unused MMA columns.  Trellis chunks retain the proven
// coalesced cp.async double buffer.  Activation quantization is done once per
// GEMV into op-local scratch; doing it independently in every output CTA was
// measured at 18.35 tok/s and rejected before this version.
#define PAW_I8_CB_COPIES 8

static __global__ void paw_rt_quant_i8_kernel(
        const float * GGML_CUDA_RESTRICT x,
        int8_t       * GGML_CUDA_RESTRICT q,
        float        * GGML_CUDA_RESTRICT s,
        int          * GGML_CUDA_RESTRICT z,
        const int n) {
    const int lane = threadIdx.x & 31;
    const int half_lane = lane & 15;
    const int tile = blockIdx.x*16 + (threadIdx.x >> 5)*2 + (lane >> 4);
    if (tile >= n/16) return;
    const unsigned mask = lane < 16 ? 0x0000ffffu : 0xffff0000u;
    const float v = x[tile*16 + half_lane];
    float am = fabsf(v);
#pragma unroll
    for (int d = 8; d > 0; d >>= 1)
        am = fmaxf(am, __shfl_down_sync(mask, am, d));
    const int first = lane & 16;
    float scale = am > 0.0f ? am/127.0f : 1.0f;
    scale = __shfl_sync(mask, scale, first);
    int qi = __float2int_rn(v/scale);
    qi = max(-127, min(127, qi));
    q[tile*16 + half_lane] = (int8_t)qi;
    int sum = qi;
#pragma unroll
    for (int d = 8; d > 0; d >>= 1)
        sum += __shfl_down_sync(mask, sum, d);
    if (half_lane == 0) {
        s[tile] = scale;
        z[tile] = sum;
    }
}

static __device__ __forceinline__ int paw_dp4a_us(uint32_t a, uint32_t b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

template <int WORDS, bool MUL1>
__launch_bounds__(PAW_WALK_NTHR, PAW_WALK_BPSM)
static __global__ void paw_rt_walk_int8_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,
        const int8_t   * GGML_CUDA_RESTRICT q_u,
        const float    * GGML_CUDA_RESTRICT q_s,
        const int      * GGML_CUDA_RESTRICT q_z,
        float          * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n) {
    static_assert(WORDS == 16 || WORDS == 24 || WORDS == 32,
                  "int8 AR walk covers the PAW-27B shipped rates");
    constexpr int TILES   = paw_stage_tiles<WORDS>::v;
    constexpr int CHUNK_W = TILES*WORDS;
    constexpr int STEP    = WORDS/8;

    const int wid     = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tr      = blockIdx.x;
    const int tiles_y = n/16;

    extern __shared__ uint32_t paw_i8_smem[];
    constexpr int CB_U16 = MUL1 ? 0 : 512*PAW_I8_CB_COPIES;
    uint16_t * GGML_CUDA_RESTRICT cb = (uint16_t *) paw_i8_smem;
    uint16_t * GGML_CUDA_RESTRICT tsm = cb + CB_U16;
    int8_t   * GGML_CUDA_RESTRICT aq = (int8_t *)(tsm + 2*CHUNK_W);
    float    * GGML_CUDA_RESTRICT as = (float *)(aq + 2*TILES*16);
    int      * GGML_CUDA_RESTRICT az = (int *)(as + 2*TILES);
    float    * GGML_CUDA_RESTRICT red = (float *)(az + 2*TILES); // 16 warp maxima + scale

    // One thread owns one half2 table row.  Derive the lattice scale from the
    // model table itself instead of baking a model-specific floating constant
    // into the kernel.
    half2 hp;
    float2 fp;
    float mx = 0.0f;
    if constexpr (!MUL1) {
        hp = ((const half2 *) tlut)[threadIdx.x];
        fp = __half22float2(hp);
        mx = fmaxf(fabsf(fp.x), fabsf(fp.y));
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, d));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (wid == 0) {
        mx = lane < PAW_WALK_WPB ? red[lane] : 0.0f;
#pragma unroll
        for (int d = 16; d > 0; d >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, d));
        if (lane == 0) red[PAW_WALK_WPB] = mx > 0.0f ? mx/31.0f : 1.0f;
    }
    __syncthreads();
    const float cb_scale = MUL1 ? __half2float(__ushort_as_half(0x20e6)) :
                                  red[PAW_WALK_WPB];
    if constexpr (!MUL1) {
        int q0 = __float2int_rn(fp.x/cb_scale);
        int q1 = __float2int_rn(fp.y/cb_scale);
        q0 = max(-127, min(127, q0));
        q1 = max(-127, min(127, q1));
        const uint16_t qp = (uint16_t)(uint8_t)(int8_t)q0 |
                            ((uint16_t)(uint8_t)(int8_t)q1 << 8);
        const int sw = lane & (PAW_I8_CB_COPIES - 1);
#pragma unroll
        for (int c = 0; c < PAW_I8_CB_COPIES; ++c)
            cb[(threadIdx.x << 3) | (c ^ sw)] = qp;
    }

    const uint16_t * GGML_CUDA_RESTRICT base =
        trellis + (int64_t)tr*tiles_y*WORDS;
    auto issue = [&](int buf, int c0) {
        const int tiles = min(tiles_y - c0, TILES);
        if (tiles <= 0) return;
        const int bytes = tiles*WORDS*2;
        char       * dstb = (char *)(tsm + buf*CHUNK_W);
        const char * srcb = (const char *)(base + (int64_t)c0*WORDS);
        for (int off = threadIdx.x*16; off < bytes; off += PAW_WALK_NTHR*16)
            cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(dstb + off), srcb + off);
        char       * dsta = (char *)(aq + buf*TILES*16);
        const char * srca = (const char *)(q_u + (int64_t)c0*16);
        for (int off = threadIdx.x*16; off < tiles*16; off += PAW_WALK_NTHR*16)
            cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(dsta + off), srca + off);
    };

    float sum = 0.0f;
    issue(0, 0);
    paw_cp_commit();
    int buf = 0;
    for (int c0 = 0; c0 < tiles_y; c0 += TILES, buf ^= 1) {
        const int nxt = c0 + TILES;
        if (nxt < tiles_y) {
            issue(buf ^ 1, nxt);
            paw_cp_commit();
            paw_cp_wait_group<1>();
        } else {
            paw_cp_wait_group<0>();
        }
        __syncthreads();

        const int lim = min(tiles_y - c0, TILES);
        int8_t * GGML_CUDA_RESTRICT qab = aq + buf*TILES*16;
        float  * GGML_CUDA_RESTRICT qsb = as + buf*TILES;
        int    * GGML_CUDA_RESTRICT qzb = az + buf*TILES;

        // The activation is quantized once per GEMV, not once per output
        // block.  Only the tiny scale vector needs a regular global load;
        // packed values arrived coalesced in the cp.async group above.
        for (int l = threadIdx.x; l < lim; l += PAW_WALK_NTHR) {
            qsb[l] = q_s[c0 + l];
            qzb[l] = q_z[c0 + l];
        }
        __syncthreads();

        const uint16_t * GGML_CUDA_RESTRICT sh = tsm + buf*CHUNK_W;
        const int ri = wid;
        for (int l = lane; l < lim; l += 32) {
            const uint16_t * GGML_CUDA_RESTRICT tw = sh + l*WORDS;
            if constexpr (MUL1) {
                int dot = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int bit = STEP*(8*ri + j);
                    const int u0 = (bit >> 4) % WORDS;
                    const int u1 = (u0 + 1) % WORDS;
                    const int off = bit & 15;
                    const uint32_t lo = tw[u0], hi = tw[u1];
                    const uint32_t st = off == 0 ? lo :
                        (((lo << off) | (hi >> (16 - off))) & 0xffffu);
                    const uint32_t a0 = (uint8_t)qab[l*16 + 2*j];
                    const uint32_t a1 = (uint8_t)qab[l*16 + 2*j + 1];
                    const uint32_t splat = a0 | (a0 << 8) | (a1 << 16) | (a1 << 24);
                    dot = paw_dp4a_us(st*0x83DCD12Du, splat, dot);
                }
                sum += (float)(dot - 255*qzb[l])*(qsb[l]*cb_scale);
                continue;
            }
            uint32_t wp[4] = {0, 0, 0, 0};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int bit = STEP*(8*ri + j);
                const int u0 = (bit >> 4) % WORDS;
                const int u1 = (u0 + 1) % WORDS;
                const int off = bit & 15;
                const uint32_t lo = tw[u0], hi = tw[u1];
                const uint32_t st = off == 0 ? lo :
                    (((lo << off) | (hi >> (16 - off))) & 0xffffu);
                const uint32_t ph = st*(st + 1u);
                const uint16_t p = cb[(((ph >> 6) & 511u) << 3) |
                                      (lane & (PAW_I8_CB_COPIES - 1))];
                int w0 = (int)(int8_t)(p & 0xffu);
                const int w1 = (int)(int8_t)(p >> 8);
                if (ph & 0x8000u) w0 = -w0;
                const int shb = (j & 1)*16;
                wp[j >> 1] |= (uint32_t)(uint8_t)(int8_t)w0 << shb;
                wp[j >> 1] |= (uint32_t)(uint8_t)(int8_t)w1 << (shb + 8);
            }
            const uint32_t * ap = (const uint32_t *)(const void *)(qab + l*16);
            int dot = 0;
#pragma unroll
            for (int g = 0; g < 4; ++g)
                dot = __dp4a((int)wp[g], (int)ap[g], dot);
            sum += (float)dot*(qsb[l]*cb_scale);
        }
        __syncthreads(); // both double buffers are reused two iterations later
    }

#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, d);
    if (lane == 0) scr_v[(size_t)tr*16 + wid] = sum;
    (void)m;
}

// ---------------------------------------------------------------------------
// Stage2 walk: the staged design plus (a) a software-pipelined inner loop --
// the next column's trellis units and b-fragment are prefetched into
// registers before the current column's mma, halving the exposed
// LDS->emit->mma chain (the frag kernel pipelines exactly this way; the
// stage kernel does not); (b) for the 32-word rate one aligned 32-bit LDS
// per unit pair instead of three 16-bit loads (both rows' windows start on
// even units: state (ri, j=t) begins at bit 4*(8*ri + t) = unit 2*ri,
// offset 4*t, and the j+4 state is 16 bits later, crossing into unit
// 2*ri+2); (c) optional GGML_PAW_WALK_MBAL=1 swaps the codebook gather for
// a computed MUL-BAL-style emission (Proteus Eqs. 2-5 adapted to pair
// emission: two differently-mixed vabsdiff4 byte-sum Gaussians per 16-bit
// window, per-layer multiplier from the pre-vetted 3-candidate pool).
// MBAL values are NOT the codec's on an existing bitstream -- timing probe
// for the lookup-free walk, correctness needs the re-encode.
// gridDim.y (GGML_PAW_WALK_STAGE_SPLIT) splits each output tile's columns
// across blocks; partials fold through the atomicAdd epilogue.
template <int WORDS, bool MT, bool MBAL = false>
__launch_bounds__(PAW_WALK_NTHR, PAW_WALK_BPSM)
static __global__ void paw_rt_walk_qtip_stage2_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,
        const float    * GGML_CUDA_RESTRICT scr_u,   // [nt, n]
        float          * GGML_CUDA_RESTRICT scr_v,   // [nt, m]
        const int m, const int n, const int nt) {
    static_assert(WORDS == 32, "stage2 pipelined path covers the 32-word rate");
    const int wid     = threadIdx.x >> 5;
    const int tr      = blockIdx.x;
    const int lane    = threadIdx.x & 31;
    const int tiles_y = n / 16;
    const int groupID = lane >> 2;
    const int tid4    = lane & 3;

    constexpr int CB_U32  = 512*PAW_STAGE_COPIES;
    constexpr int TILES   = paw_stage_tiles<WORDS>::v;
    constexpr int CHUNK_W = TILES*WORDS;

    extern __shared__ uint32_t paw_stage_smem[];
    uint32_t * GGML_CUDA_RESTRICT cb  = paw_stage_smem;
    uint16_t * GGML_CUDA_RESTRICT tsm = (uint16_t *)(paw_stage_smem + CB_U32);

    if constexpr (!MBAL) {
        {   // 8 lane-copies of the codebook, XOR-swizzled writes
            const uint32_t * GGML_CUDA_RESTRICT src = (const uint32_t *) tlut;
            const int sw = lane & (PAW_STAGE_COPIES - 1);
            for (int r = threadIdx.x; r < 512; r += PAW_WALK_NTHR) {
                const uint32_t v = src[r];
#pragma unroll
                for (int c = 0; c < PAW_STAGE_COPIES; ++c)
                    cb[(r << PAW_STAGE_LOG2C) | (c ^ sw)] = v;
            }
        }
    }

    // byte offsets of the (lo, hi) window pair and the crossing pair inside
    // the 64-byte shared tile (units 2*ri / 2*ri+1 / 2*ri+2; the +8 row's
    // crossing unit wraps at g == 7)
    const int lo_off[2] = {4*groupID, 4*groupID + 32};
    const int xx_off[2] = {(4*groupID + 4) & 63, (4*groupID + 36) & 63};
    const int off0      = 4*tid4;   // bit offset, identical for both rows

    auto emit = [&](uint32_t st) -> uint32_t {
        if constexpr (MBAL) {
            const uint32_t x1 = st * 1927765585u;      // pool multiplier a
            const uint32_t x2 = x1 ^ (x1 >> 17);       // pool shift s
            uint32_t ua, ub, z = 0;
            asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(ua) : "r"(x2), "r"(z), "r"(z));
            const uint32_t x3 = x2 ^ (x2 >> 7);        // second mixing for the pair
            asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(ub) : "r"(x3), "r"(z), "r"(z));
            const __half2 h = __halves2half2(__uint2half_rn(ua), __uint2half_rn(ub));
            return *(const uint32_t *) &h;
        }
        const uint32_t ph = st*(st + 1u);
        const uint32_t pair = cb[((ph >> (6 - PAW_STAGE_LOG2C)) & (511u << PAW_STAGE_LOG2C))
                                 | (lane & (PAW_STAGE_COPIES - 1))];
        return pair ^ (ph & 0x8000u);
    };
    auto load_b = [&](int ct, half2& h0, half2& h1) {
        auto pack = [&](const float * ub) {
            const float2 v0 = *(const float2 *)(const void *)(ub + tid4*2);
            const float2 v1 = *(const float2 *)(const void *)(ub + tid4*2 + 8);
            h0 = __floats2half2_rn(v0.x, v0.y);
            h1 = __floats2half2_rn(v1.x, v1.y);
        };
        if constexpr (!MT) {
            pack(scr_u + ct*16);
        } else if (groupID < nt) {
            pack(scr_u + (int64_t) groupID*n + ct*16);
        } else {
            h0 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
            h1 = h0;
        }
    };

    const uint16_t * GGML_CUDA_RESTRICT base = trellis + (int64_t) tr*tiles_y*WORDS;
    auto issue = [&](int buf, int c0) {
        const int tiles = (tiles_y - c0) < TILES ? (tiles_y - c0) : TILES;
        if (tiles <= 0) return;
        const int bytes = tiles*WORDS*2;                 // WORDS*2 == 64, multiple of 16
        char       * dstb = (char *)(tsm + buf*CHUNK_W);
        const char * srcb = (const char *)(base + (int64_t) c0*WORDS);
        for (int off = threadIdx.x*16; off < bytes; off += PAW_WALK_NTHR*16)
            cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(dstb + off), srcb + off);
    };

    // column split across blocks sharing the output tile (blockIdx.y)
    const int tpb   = (tiles_y + (int) gridDim.y - 1)/(int) gridDim.y;
    const int c_beg = tpb*(int) blockIdx.y;
    const int c_end = tiles_y < c_beg + tpb ? tiles_y : c_beg + tpb;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    issue(0, c_beg); paw_cp_commit();
    int buf = 0;
    for (int c0 = c_beg; c0 < c_end; c0 += TILES, buf ^= 1) {
        const int nxt = c0 + TILES;
        if (nxt < c_end) { issue(buf ^ 1, nxt); paw_cp_commit(); paw_cp_wait_group<1>(); }
        else             { paw_cp_wait_group<0>(); }
        __syncthreads();

        const uint16_t * GGML_CUDA_RESTRICT sh = tsm + buf*CHUNK_W;
        const int lim = (c_end - c0) < TILES ? (c_end - c0) : TILES;
        const int l0  = wid;
        uint32_t plo[2], pxx[2]; half2 phb0, phb1;
        if (l0 < lim) {
            const char * tw = (const char *)(sh + l0*WORDS);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                plo[k] = *(const uint32_t *)(tw + lo_off[k]);
                pxx[k] = *(const uint32_t *)(tw + xx_off[k]);
            }
            load_b(c0 + l0, phb0, phb1);
        }
        for (int l = l0; l < lim; l += PAW_WALK_WPB) {
            const int ln = l + PAW_WALK_WPB;
            uint32_t nlo[2], nxx[2]; half2 nhb0, nhb1;
            if (ln < lim) {
                const char * twn = (const char *)(sh + ln*WORDS);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    nlo[k] = *(const uint32_t *)(twn + lo_off[k]);
                    nxx[k] = *(const uint32_t *)(twn + xx_off[k]);
                }
                load_b(c0 + ln, nhb0, nhb1);
            }
            uint32_t ra[4];
            const uint32_t rb0 = *(const uint32_t *) &phb0, rb1 = *(const uint32_t *) &phb1;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                // units are read MSB-first, so a 16-bit window starting at
                // bit `off0` of the (lo, hi) pair is a 32-bit rotate-left of
                // the little-endian pair, masked to 16 bits: for off0 == 0
                // the (32-off0)&31 shift degenerates to lo32 & 0xFFFF == lo,
                // matching dec_state's off == 0 branch.
                const uint32_t st1 = ((plo[k] << off0) | (plo[k] >> ((32 - off0) & 31))) & 0xFFFFu;
                const uint32_t hx  = (plo[k] >> 16) | (pxx[k] << 16);
                const uint32_t st2 = ((hx << off0) | (hx >> ((32 - off0) & 31))) & 0xFFFFu;
                ra[k]     = emit(st1);
                ra[k + 2] = emit(st2);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb0), "r"(rb1));
            if (ln < lim) {
#pragma unroll
                for (int k = 0; k < 2; ++k) { plo[k] = nlo[k]; pxx[k] = nxx[k]; }
                phb0 = nhb0; phb1 = nhb1;
            }
        }
        __syncthreads();
    }

    if constexpr (!MT) {
        if (tid4 == 0) {
            atomicAdd(&scr_v[(size_t) tr*16 + groupID],     d0);
            atomicAdd(&scr_v[(size_t) tr*16 + groupID + 8], d2);
        }
    } else {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < nt && tid4 == (t >> 1)) {
                const float vlo = (t & 1) ? d1 : d0;
                const float vhi = (t & 1) ? d3 : d2;
                atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID],     vlo);
                atomicAdd(&scr_v[(int64_t) t*m + tr*16 + groupID + 8], vhi);
            }
        }
    }
}

// dense prefill: decode the K4/V2 trellis ONCE into an fp16 bank [m, n] —
// one thread per (tile, tile-row), same window/state math as rt_walk, same
// pre-rounded tlut halves (sign flip is exact), then apply the bank with a
// plain fp32 GEMM-ish kernel. Replaces only the walk stage; rt_u/rt_out are
// untouched.

// rate-templated twin of paw_rt_dense_decode_kernel: same bank layout,
// generic per-row state windows (paw_rt_walk_kernel math) so the dense
// payload's K1/K1.5 rates can materialize fp16 banks for batched apply.
template <int WORDS>
static __global__ void paw_rt_dense_decode_rate_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        half           * GGML_CUDA_RESTRICT bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    constexpr int step = WORDS/8;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: a warp writes 16 consecutive 16-half row chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*WORDS;
    half * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    half tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int b   = step*(8*rr + j);
        const int wi  = (b >> 4) % WORDS;
        const int off = b & 15;
        const uint32_t w0 = trellis[tw + wi];
        const uint32_t st = off == 0 ? w0
            : (((w0 << off) | (trellis[tw + (wi + 1) % WORDS] >> (16 - off)))
               & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        // negate-then-round-trip is bit-exact for fp16 values (sign bit)
        tmp[2*j + 0] = __float2half_rn((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = slut[2*row + 1];
    }
    paw_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}

// e5m2 twin of paw_rt_dense_decode_rate_kernel: rate-generic decode into a
// 1-byte/weight bank, so the DOL path (GGML_PAW_DOL_K2) can cache non-K4
// payloads in VRAM. Same emission math, e5m2 stores like the K4 fp8 twin.
template <int WORDS>
static __global__ void paw_rt_dense_decode_rate_kernel_fp8(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        uint8_t        * GGML_CUDA_RESTRICT bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    constexpr int step = WORDS/8;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {
        return;
    }
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*WORDS;
    uint8_t * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    uint8_t tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int b   = step*(8*rr + j);
        const int wi  = (b >> 4) % WORDS;
        const int off = b & 15;
        const uint32_t w0 = trellis[tw + wi];
        const uint32_t st = off == 0 ? w0
            : (((w0 << off) | (trellis[tw + (wi + 1) % WORDS] >> (16 - off)))
               & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        const float a1 = __half2float(slut[2*row + 1]);
        tmp[2*j + 0] = paw_f32_to_e5m2((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = paw_f32_to_e5m2(a1);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i] = tmp[i];
    }
}

static __global__ void paw_rt_dense_decode_kernel(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        half           * GGML_CUDA_RESTRICT bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {   // one thread per (tile, tile-row)
        return;
    }
    // column tile fastest: a warp writes 16 consecutive 16-half row chunks
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*64;   // K = 4: 64 words/tile
    uint32_t w[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w[q] = trellis[tw + ((4*rr + q) & 63)];   // only q=4,rr=15 wraps
    }
    half * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    half tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t hi = w[j >> 1];
        const uint32_t st = (j & 1) == 0 ? hi
            : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        // negate-then-round-trip is bit-exact for fp16 values (sign bit)
        tmp[2*j + 0] = __float2half_rn((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = slut[2*row + 1];
    }
    paw_store_half16(dst, tmp);   // same bits, 2x16B instead of 16x2B
}

// fp8 twin of paw_rt_dense_decode_kernel: same decode, output stored as
// e5m2 (1 byte/weight) instead of fp16.
static __global__ void paw_rt_dense_decode_kernel_fp8(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        uint8_t        * GGML_CUDA_RESTRICT bank,     // [m, n]
        const int m, const int n) {
    constexpr int WG = 256;
    __shared__ half slut[1024];

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += WG) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {
        return;
    }
    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;

    const int64_t tw = ((int64_t) tr*tiles_y + tc)*64;
    uint32_t w[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w[q] = trellis[tw + ((4*rr + q) & 63)];
    }
    uint8_t * dst = bank + (int64_t)(tr*16 + rr)*n + tc*16;
    uint8_t tmp[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t hi = w[j >> 1];
        const uint32_t st = (j & 1) == 0 ? hi
            : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
        const uint32_t ph  = st*(st + 1u);
        const uint32_t row = (ph >> 6) & 511u;
        const float a0 = __half2float(slut[2*row + 0]);
        const float a1 = __half2float(slut[2*row + 1]);
        tmp[2*j + 0] = paw_f32_to_e5m2((ph & 0x8000u) ? -a0 : a0);
        tmp[2*j + 1] = paw_f32_to_e5m2(a1);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i] = tmp[i];
    }
}

static __global__ void paw_rt_dense_decode_kernel_idx80(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        uint16_t       * GGML_CUDA_RESTRICT bank,
        const int m, const int n) {
    constexpr int WG = 256;
    const int tid = threadIdx.x;
    const int tiles_y = n / 16;
    const int gx = blockIdx.x*WG + tid;
    if (gx >= (m/16)*16*tiles_y) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int tc     = gx % tiles_y;
    const int rowall = gx / tiles_y;
    const int tr     = rowall >> 4;
    const int rr     = rowall & 15;
    const int64_t tw = ((int64_t) tr*tiles_y + tc)*64;

    uint32_t w[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        w[q] = trellis[tw + ((4*rr + q) & 63)];
    }

    uint16_t packed[5] = {0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t hi = w[j >> 1];
        const uint32_t st = (j & 1) == 0 ? hi
            : (((hi << 8) | (w[(j >> 1) + 1] >> 8)) & 0xFFFFu);
        const uint32_t code = (st*(st + 1u) >> 6) & 0x3FFu;
        const int bit = 10*j;
        const int wi = bit >> 4;
        const int off = bit & 15;
        packed[wi] |= (uint16_t) (code << off);
        if (off > 6) {
            packed[wi + 1] |= (uint16_t) (code >> (16 - off));
        }
    }

    uint16_t * dst = bank + ((int64_t)(tr*16 + rr)*tiles_y + tc)*5;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        dst[i] = packed[i];
    }
}

// grid (m/16, 1, ceil(nt/TC)): each block owns 16 output rows and a TC-token
// chunk; per column-stride iteration the 16-row W slice is held in registers
// and reused across the chunk's tokens, so bank traffic is amortized TC-fold
// (per-chunk L2 re-reads keep grid-level parallelism, which a single token
// loop over grid (m/16) blocks would not). TC is an A/B knob
// (GGML_PAW_RT_TC, 4 or 8); numerics are TC-independent (per-token order
// unchanged, tail clamps are write-guarded).
template <int TC>
static __global__ void paw_rt_apply_kernel(
        const half  * GGML_CUDA_RESTRICT bank,     // [m, n]
        const float * GGML_CUDA_RESTRICT scr_u,    // [nt, n]
        float       * GGML_CUDA_RESTRICT scr_v,    // [nt, m]
        const int m, const int n, const int nt) {
    constexpr int WG        = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = WG / warp_size;
    __shared__ float red[TC][16][4];   // >= n_warps for warp_size 32 (and 64 on HIP)

    const int tr  = blockIdx.x;
    const int t0  = blockIdx.z*TC;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    const half * W = bank + (int64_t) tr*16*n;

    float acc[TC][16];
#pragma unroll
    for (int t = 0; t < TC; ++t) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc[t][i] = 0.0f;
        }
    }

    // half2/float2 loads: each thread covers column pair (c0, c0+1) — halves
    // the load instruction count. The two columns' terms are accumulated into
    // the same per-thread partial instead of two lanes' partials, a
    // contractible fp32 reassociation (the apply path's summation order is
    // already declared contractible).
    for (int c0 = 2*tid; c0 < n; c0 += 2*WG) {
        float2 wc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            wc[i] = __half22float2(*(const half2 *)(W + (int64_t) i*n + c0));
        }
#pragma unroll
        for (int t = 0; t < TC; ++t) {
            // tail chunk: clamp (duplicate compute, write-guarded below)
            const int tt = min(t0 + t, nt - 1);
            const float2 uc = *(const float2 *)(scr_u + (int64_t) tt*n + c0);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[t][i] += wc[i].x*uc.x + wc[i].y*uc.y;
            }
        }
    }

    const int lane = tid % warp_size;
    const int wid  = tid / warp_size;
#pragma unroll
    for (int t = 0; t < TC; ++t) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float s = warp_reduce_sum<warp_size>(acc[t][i]);
            if (lane == 0) {
                red[t][i][wid] = s;
            }
        }
    }
    __syncthreads();
    if (tid < TC*16) {
        const int t = tid >> 4;
        const int i = tid & 15;
        if (t0 + t < nt) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += red[t][i][wj];
            }
            scr_v[(int64_t)(t0 + t)*m + tr*16 + i] = sum;
        }
    }
}










// MODE 5: weight-stationary across tokens AND rows. The first WS cut kept
// one 16-row strip per block, so every row-strip block still re-read the
// whole nt x n activation slab (~8 GB per call on the shared-expert shape).
// Staging BM=64 rows per block cuts that by 4x; activations are cast once
// per K-chunk and reused by all four A tiles held live in registers.
static __global__ void paw_rt_apply_kernel_mma(
        const half  * GGML_CUDA_RESTRICT bank,
        const float * GGML_CUDA_RESTRICT scr_u,
        float       * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n, const int nt) {
    using namespace nvcuda;

    constexpr int n_warps = 4;
    constexpr int bm_t    = 4;                  // 16-row tiles -> BM = 64
    constexpr int tpb     = 2;                  // token subtiles -> BN = 128
    constexpr int bn      = n_warps * tpb * 16;
    constexpr int bk      = 16;                 // == the wmma K tile

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row0 = blockIdx.x * (bm_t * 16);  // first output row
    const int tok0 = blockIdx.z * bn;

    __shared__ half Xsh[n_warps][tpb][16][bk];         // 4 KB, warp-private
    __shared__ float out_sh[n_warps][16*16];           // 4 KB

    ggml_cuda_pdl_sync();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[bm_t][tpb];
#pragma unroll
    for (int r = 0; r < bm_t; ++r) {
#pragma unroll
        for (int t = 0; t < tpb; ++t) {
            wmma::fill_fragment(acc[r][t], 0.0f);
        }
    }

    for (int k0 = 0; k0 < n; k0 += bk) {
#pragma unroll
        for (int t = 0; t < tpb; ++t) {
            const int tt0 = tok0 + (warp_id*tpb + t)*16;
            for (int idx = lane; idx < 16*bk; idx += 32) {
                const int tt = idx / bk;
                const int kk = idx % bk;
                const int tok = tt0 + tt;
                const int kg = k0 + kk;
                Xsh[warp_id][t][tt][kk] =
                    (tok < nt && kg < n) ? __float2half(scr_u[(int64_t) tok*n + kg])
                                         : __float2half(0.0f);
            }
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < bm_t; ++r) {
            wmma::load_matrix_sync(a_frag,
                bank + (int64_t)(row0 + r*16)*n + k0, n);
#pragma unroll
            for (int t = 0; t < tpb; ++t) {
                wmma::load_matrix_sync(b_frag, &Xsh[warp_id][t][0][0], bk);
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
            const int tt0 = tok0 + (warp_id*tpb + t)*16;
            for (int idx = lane; idx < 16*16; idx += 32) {
                const int row = idx / 16;
                const int tt  = idx % 16;
                const int tok = tt0 + tt;
                if (tok < nt) {
                    scr_v[(int64_t) tok*m + row0 + r*16 + row] = out_sh[warp_id][idx];
                }
            }
            __syncwarp();
        }
    }
}

// host-side mode dispatch shared by every rt_apply mma call site so the
// bisect modes stay consistent across the cached-bank, fresh-bank and
// batched paths.

// fp32 -> fp16 activation cast feeding the cuBLAS dense apply
static __global__ void paw_cast_f32_f16_kernel(
        const float * GGML_CUDA_RESTRICT x, half * GGML_CUDA_RESTRICT h, const int k) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < k) {
        h[i] = __float2half(x[i]);
    }
}

// host dispatch shared by every rt_apply call site. The weight-stationary
// tensor-core kernel needs m tiled by 64; anything else falls back to the
// scalar dense apply.
static void paw_launch_rt_apply_mma(ggml_backend_cuda_context & ctx,
        cudaStream_t stream, const half * bank,
        const float * scr_u, float * scr_v, const int m, const int n, const int nt,
        const int voff = 0) {
    // voff: first output ROW of this chunk inside the full-m strided dst
    // (the two-half decode/apply overlap writes scr_v[m0..m) from a bank
    // slice while keeping the full-row leading dimension)
    static const int blas_min_tok = paw_env_int("GGML_PAW_RT_BLAS_MIN_TOK", 128);
    // the host count/cast staging is incompatible with graph capture
    cudaStreamCaptureStatus rcst = cudaStreamCaptureStatusNone;
    const bool rt_capturing =
        cudaStreamIsCapturing(stream, &rcst) == cudaSuccess &&
        rcst == cudaStreamCaptureStatusActive;
    if (!rt_capturing && m % 16 == 0 && n % 8 == 0 && nt >= blas_min_tok) {
        // dense apply against a materialized bank is an ordinary GEMM; hand
        // large batches to the tensor-core BLAS. Activations are cast once
        // to fp16 -- the same conversion the custom kernels stage anyway.
        ggml_cuda_pool_alloc<half> u_h_alloc(ctx.pool());
        half * u_h = u_h_alloc.alloc((size_t) nt*n);
        paw_launch(paw_cast_f32_f16_kernel,
            ggml_cuda_kernel_launch_params(
                dim3(((size_t) nt*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            scr_u, u_h, (int)((size_t) nt*n));
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
        CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, nt, n,
                &alpha,
                bank, CUDA_R_16F, n,
                u_h,  CUDA_R_16F, n,
                &beta,
                scr_v + voff, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return;
    }
    GGML_ASSERT(voff == 0 && "row-offset apply only supported on the cublas path");
    // The WS-mma kernel tiles tokens by 128; at DFlash verify sizes (nt ~ 8)
    // nearly the whole tile is padding and the lane regresses ~45% versus the
    // scalar apply (188.9 -> 105 tok/s code-lane, bisected to 05b8814). Keep
    // mma for the large-batch shapes it wins; hand small ubatches to the
    // scalar kernel<8> path. GGML_PAW_APPLY_MMA_MIN_TOK overrides the cutoff.
    static const int mma_min_tok = paw_env_int("GGML_PAW_APPLY_MMA_MIN_TOK", 32);
    if (m % 64 == 0 && nt >= mma_min_tok) {
        paw_launch(paw_rt_apply_kernel_mma,
            ggml_cuda_kernel_launch_params(dim3(m/64, 1, (nt + 127)/128), dim3(128, 1, 1), 0, stream),
            bank, scr_u, scr_v, m, n, nt);
    } else {
        paw_launch(paw_rt_apply_kernel<8>,
            ggml_cuda_kernel_launch_params(dim3(m/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
            bank, (const float *) scr_u, scr_v, m, n, nt);
    }
}

// --- experimental: tensor-core walk, opt-in via GGML_PAW_RT_WALK_MMA=1 ---
//
// paw_rt_walk_kernel above is a scalar matvec: per (token, 16-row tile) the
// trellis is decoded once per column-tile and dotted against the token's u
// values with FMA + warp reduce. This variant keeps the exact same decode
// math (the fp16 values match paw_rt_dense_decode_kernel, the dense-path
// reference) but stages each decoded 16x16 tile in shared memory and
// multiplies it onto the tokens' activations with m16n8k16 tensor-core MMA
// instead of scalar FMA. The K dimension (n columns) is split across the
// block's 4 warps, each accumulating a partial 16x8 tile; a small cross-warp
// reduction finishes the 16 output rows. Activations are cast fp32->fp16 for
// the B operand (standard for tensor-core GEMM; the one deliberate numeric
// difference the correctness gate checks). The walk still materializes only
// the nt x m output -- no full m x n bank -- which is the point of the walk
// over the dense path at small nt.
static __global__ void paw_rt_walk_kernel_mma(
        const uint16_t * GGML_CUDA_RESTRICT trellis,
        const half     * GGML_CUDA_RESTRICT tlut,     // pre-rounded fp16
        const float    * GGML_CUDA_RESTRICT scr_u,
        float          * GGML_CUDA_RESTRICT scr_v,
        const int m, const int n, const int nt) {
    using namespace nvcuda;
    constexpr int warp_size = 32;
    constexpr int n_warps   = 128 / warp_size;

    __shared__ half  slut[1024];
    __shared__ __align__(16) half shW[n_warps][16*16];   // decoded tile, row-major
    __shared__ __align__(16) half shU[n_warps][16*16];   // [16 tokens][16 K], col-major B
    __shared__ __align__(16) float out_sh[n_warps][16*16];

    const int t      = blockIdx.z*16;   // first token of this block's 16-tile
    const int tr     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane    = tid % warp_size;

    ggml_cuda_pdl_sync();
    for (int i = tid; i < 1024; i += 128) {
        slut[i] = tlut[i];
    }
    __syncthreads();

    const int tiles_y = n / 16;
    const int64_t tw0 = (int64_t) tr*tiles_y;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // K split across warps: warp w owns K-chunks {w, w+4, w+8, ...}
    for (int kt = warp_id; kt < tiles_y; kt += n_warps) {
        const int64_t tw = (tw0 + kt)*64;
        // decode this 16x16 tile into this warp's shared region, row-major.
        // lane (2r) covers states 0..3 of row r, lane (2r+1) states 4..7.
        const int r  = lane >> 1;
        const int st0 = (lane & 1) * 4;
        uint32_t w[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            w[q] = trellis[tw + ((4*r + q) & 63)];
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int kk  = st0 + s;
            const uint32_t hi = w[kk >> 1];
            const uint32_t st = (kk & 1) == 0 ? hi
                : (((hi << 8) | (w[(kk >> 1) + 1] >> 8)) & 0xFFFFu);
            const uint32_t ph  = st*(st + 1u);
            const uint32_t row = (ph >> 6) & 511u;
            const float a0 = __half2float(slut[2*row + 0]);
            shW[warp_id][r*16 + 2*kk]     = __float2half_rn((ph & 0x8000u) ? -a0 : a0);
            shW[warp_id][r*16 + 2*kk + 1] = slut[2*row + 1];
        }
        // stage 16 token columns' 16 activations into this warp's shared,
        // col-major B (element (k, tt) at shU[tt*16 + k]); columns past nt
        // are zero-padded (block z-tile is 16 tokens).
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int e   = lane*8 + q;   // 0..255
            const int tt  = e >> 4;
            const int kk  = e & 15;
            float v = 0.0f;
            if (t + tt < nt) {
                v = scr_u[(int64_t)(t + tt)*n + kt*16 + kk];
            }
            shU[warp_id][tt*16 + kk] = __float2half(v);
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, shW[warp_id], 16);
        wmma::load_matrix_sync(b_frag, shU[warp_id], 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(&out_sh[warp_id][0], acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < 16*16; idx += 128) {
        const int row = idx >> 4;
        const int tt  = idx & 15;
        const int tok = t + tt;
        if (tok < nt) {
            float sum = 0.0f;
#pragma unroll
            for (int wj = 0; wj < n_warps; ++wj) {
                sum += out_sh[wj][idx];
            }
            scr_v[(int64_t) tok*m + tr*16 + row] = sum;
        }
    }
}

// --- pre-decoded bank cache (Marlin-style: decode static weights once, then
// GEMM fast) ---
//
// The trellis weights are static, so the decoded fp16 bank for each RT matrix
// is identical across steps; the dense path has been re-decoding it every
// step. Caching it here turns the per-step walk/decode into a plain
// bandwidth-bound fp16 GEMV/GEMM over the cached bank. Keyed by the trellis
// data pointer *and shape*; entries are intentionally leaked (one per matrix,
// matching the weight allocation's lifetime). GGML_PAW_BANK_CACHE=0 falls back
// to the original per-step walk/decode.
//
// The shape belongs in the key because a pointer alone is only unique for the
// lifetime of the allocation behind it. Model weights live forever, so the old
// pointer-only key was safe in a server; test-backend-ops recycles buffers
// between cases, so a later case with a different shape used to be handed the
// previous case's bank -- wrong numbers, and an out-of-bounds read as soon as
// the new shape was larger.
struct paw_bank_key {
    const void * trellis;
    int m;
    int n;
    bool operator==(const paw_bank_key & o) const {
        return trellis == o.trellis && m == o.m && n == o.n;
    }
};
struct paw_bank_key_hash {
    size_t operator()(const paw_bank_key & k) const {
        return std::hash<const void *>()(k.trellis) ^ (std::hash<int>()(k.m) << 1)
                                                    ^ (std::hash<int>()(k.n) << 2);
    }
};
static std::mutex paw_rt_bank_mutex;
static std::unordered_map<paw_bank_key, const void *, paw_bank_key_hash> paw_rt_banks;
// DOL (GGML_PAW_DOL_K2) bank budget accounting, guarded by paw_rt_bank_mutex.
// Counts only banks the DOL path created; K4 bank-cache users are unbounded
// exactly as before.
static int64_t paw_dol_banked = 0;

// rate-dispatch wrapper: any supported trellis rate through the templated
// tensor-core walk (K4 included -- the generic window math is bit-identical)
template <int WORDS>
static void paw_rt_walk_qtip_frag_dispatch(const uint16_t * trellis,
        const half * tlut, const float * scr_u, float * scr_v,
        int8_t * q_u, float * q_s, int * q_z,
        const int m, const int n, const int nt, cudaStream_t stream) {
    // The same fused-int8/dp4a decode strategy that is enabled by default in
    // exllamav3, adapted to PAW's existing lattice lookup and bitstream.
    // Keep it opt-in until end-to-end quality and speed are measured.
    // mode 1 adapts the current LUT payload. Mode 2 is a lookup-free paired
    // mul1 timing prototype, not EXL3 format compatibility: PAW V=2 emits a
    // pair per state, whereas EXL3 mul1 emits one scalar per state. It needs a
    // matching re-encode and remains timing-only until that encoder exists.
    static const int int8_mode = paw_env_int("GGML_PAW_WALK_INT8", 0);
    if (int8_mode != 0 && nt == 1 &&
        (WORDS == 16 || WORDS == 24 || WORDS == 32)) {
        if constexpr (WORDS == 16 || WORDS == 24 || WORDS == 32) {
            paw_launch(paw_rt_quant_i8_kernel,
                ggml_cuda_kernel_launch_params(dim3((n/16 + 15)/16, 1, 1),
                    dim3(256, 1, 1), 0, stream),
                scr_u, q_u, q_s, q_z, n);
            constexpr int TILES = paw_stage_tiles<WORDS>::v;
            const size_t cb_bytes = int8_mode == 2 ? 0 :
                (size_t)512*PAW_I8_CB_COPIES*sizeof(uint16_t);
            const size_t int8_smem = cb_bytes +
                (size_t)2*TILES*WORDS*sizeof(uint16_t) +
                (size_t)2*TILES*16*sizeof(int8_t) +
                (size_t)2*TILES*sizeof(float) +
                (size_t)2*TILES*sizeof(int) +
                (size_t)(PAW_WALK_WPB + 1)*sizeof(float);
            static bool si8_lut = false, si8_mul1 = false;
            if (int8_mode == 2 && !si8_mul1) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    (const void *)paw_rt_walk_int8_kernel<WORDS, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)int8_smem));
                si8_mul1 = true;
            } else if (int8_mode != 2 && !si8_lut) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    (const void *)paw_rt_walk_int8_kernel<WORDS, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)int8_smem));
                si8_lut = true;
            }
            if (int8_mode == 2)
                paw_launch(paw_rt_walk_int8_kernel<WORDS, true>,
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1),
                        dim3(PAW_WALK_NTHR, 1, 1), int8_smem, stream),
                    trellis, tlut, q_u, q_s, q_z, scr_v, m, n);
            else
                paw_launch(paw_rt_walk_int8_kernel<WORDS, false>,
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1),
                        dim3(PAW_WALK_NTHR, 1, 1), int8_smem, stream),
                    trellis, tlut, q_u, q_s, q_z, scr_v, m, n);
            return;
        }
    }
    // enough blocks that no warp walks a long serial chain of tile columns
    // (measured neutral on PAW-27B -- small-m attention projections are not
    // on the critical path; kept for narrow-m payloads)
    static const int split_min_blocks =
        paw_env_int("GGML_PAW_WALK_SPLIT_BLOCKS", 0);
    static const int split_max   = paw_env_int("GGML_PAW_WALK_SPLIT_MAX", 8);
    int S = 1;
    if (split_min_blocks > 0 && m/16 < split_min_blocks && n >= 256) {
        S = (split_min_blocks + m/16 - 1)/(m/16);
        if (S > split_max) S = split_max;
        if (S < 1)         S = 1;
    }
    if (S > 1) {
        CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) nt*m*sizeof(float), stream));
    }
    // Staged walk (cp.async the trellis into shared). w24 alone measured
    // +12.0% on tg64 (23.88 -> 26.75) with byte-identical output, so it is
    // extended to the other two shipped rates here.
    // Stage2 (GGML_PAW_WALK_STAGE2=1): pipelined + 32-bit-unit stage walk
    // (w32 only; other rates fall through to stage1). Optional knobs:
    //   GGML_PAW_WALK_MBAL=1         computed MUL-BAL emission (timing only
    //                                on an existing bitstream)
    //   GGML_PAW_WALK_STAGE_SPLIT=N  split each output tile's columns across
    //                                blocks when m/16 < N (atomicAdd folds)
    static const bool stage2_on  = paw_env_int("GGML_PAW_WALK_STAGE2", 0) != 0;
    static const bool mbal_on    = paw_env_int("GGML_PAW_WALK_MBAL", 0) != 0;
    static const int  ssplit_min = paw_env_int("GGML_PAW_WALK_STAGE_SPLIT", 0);
    if (stage2_on && WORDS == 32) {
        if constexpr (WORDS == 32) {
        int S2 = 1;
        if (ssplit_min > 0 && m/16 < ssplit_min && n >= 256) {
            S2 = (ssplit_min + m/16 - 1)/(m/16);
            if (S2 > split_max) S2 = split_max;
            if (S2 < 1)         S2 = 1;
        }
        if (S2 > 1) {
            CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) nt*m*sizeof(float), stream));
        }
        constexpr size_t stage_smem =
            (size_t) 512*PAW_STAGE_COPIES*sizeof(uint32_t)
            + (size_t) 2*paw_stage_tiles<WORDS>::v*WORDS*sizeof(uint16_t);
        auto launch2 = [&](auto mt, auto mb) {
            constexpr bool mtb = decltype(mt)::value, mbb = decltype(mb)::value;
            paw_launch((paw_rt_walk_qtip_stage2_kernel<WORDS, mtb, mbb>),
                ggml_cuda_kernel_launch_params(dim3(m/16, S2, 1),
                    dim3(PAW_WALK_NTHR, 1, 1), stage_smem, stream),
                trellis, tlut, scr_u, scr_v, m, n, nt);
        };
        if (mbal_on) {
            if (nt <= 1) launch2(std::false_type{}, std::true_type{});
            else         launch2(std::true_type{},  std::true_type{});
        } else {
            if (nt <= 1) launch2(std::false_type{}, std::false_type{});
            else         launch2(std::true_type{},  std::false_type{});
        }
        return;
        }
    }
    static const bool stage_on = paw_env_int("GGML_PAW_WALK_STAGE", 1) != 0;
    static const bool computed = paw_env_int("GGML_PAW_WALK_COMPUTED", 0) != 0;
    static const bool aligned  = paw_env_int("GGML_PAW_WALK_ALIGNED", 0) != 0;
    static const bool regres   = paw_env_int("GGML_PAW_WALK_REGRES", 0) != 0;
    if constexpr (WORDS == 16 || WORDS == 24 || WORDS == 32 || WORDS == 64) {
        if (stage_on && S == 1) {
            constexpr size_t stage_smem =
                (size_t) 512*PAW_STAGE_COPIES*sizeof(uint32_t)
                + (size_t) 2*paw_stage_tiles<WORDS>::v*WORDS*sizeof(uint16_t);
            // GGML_PAW_WALK_COMPUTED=1 also drives the staged walk now: it is
            // the default path and ncu measures it L1TEX-bound at 84%, so the
            // codebook gather has to be tested here, not only in the frag kernel.
            if (nt <= 1 && regres && WORDS == 32) {
                static bool sr = false;
                if (!sr) { CUDA_CHECK(cudaFuncSetAttribute(
                    (const void *) paw_rt_walk_qtip_stage_kernel<WORDS, false, false, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) stage_smem)); sr = true; }
                paw_launch((paw_rt_walk_qtip_stage_kernel<WORDS, false, false, false, true>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1),
                        dim3(PAW_WALK_NTHR, 1, 1), stage_smem, stream),
                    trellis, tlut, scr_u, scr_v, m, n, nt);
            } else if (nt <= 1 && aligned) {
                static bool sa = false;
                if (!sa) { CUDA_CHECK(cudaFuncSetAttribute(
                    (const void *) paw_rt_walk_qtip_stage_kernel<WORDS, false, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) stage_smem)); sa = true; }
                paw_launch((paw_rt_walk_qtip_stage_kernel<WORDS, false, false, true>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1),
                        dim3(PAW_WALK_NTHR, 1, 1), stage_smem, stream),
                    trellis, tlut, scr_u, scr_v, m, n, nt);
            } else if (nt <= 1 && computed) {
                static bool sc = false;
                if (!sc) { CUDA_CHECK(cudaFuncSetAttribute(
                    (const void *) paw_rt_walk_qtip_stage_kernel<WORDS, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) stage_smem)); sc = true; }
                paw_launch((paw_rt_walk_qtip_stage_kernel<WORDS, false, true>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1),
                        dim3(PAW_WALK_NTHR, 1, 1), stage_smem, stream),
                    trellis, tlut, scr_u, scr_v, m, n, nt);
            } else if (nt <= 1) {
                static bool sf = false;
                if (!sf) { CUDA_CHECK(cudaFuncSetAttribute(
                    (const void *) paw_rt_walk_qtip_stage_kernel<WORDS, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) stage_smem)); sf = true; }
                paw_launch((paw_rt_walk_qtip_stage_kernel<WORDS, false>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1),
                        dim3(PAW_WALK_NTHR, 1, 1), stage_smem, stream),
                    trellis, tlut, scr_u, scr_v, m, n, nt);
            } else {
                static bool st = false;
                if (!st) { CUDA_CHECK(cudaFuncSetAttribute(
                    (const void *) paw_rt_walk_qtip_stage_kernel<WORDS, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) stage_smem)); st = true; }
                paw_launch((paw_rt_walk_qtip_stage_kernel<WORDS, true>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1),
                        dim3(PAW_WALK_NTHR, 1, 1), stage_smem, stream),
                    trellis, tlut, scr_u, scr_v, m, n, nt);
            }
            return;
        }
    }
    // 64 KiB of dynamic shared is above the 48 KiB default and has to be
    // opted into per kernel (once per instantiation).
    // GGML_PAW_WALK_COMPUTED=1: lookup-free emission prototype (Track B) --
    // no shared codebook (0 dynamic smem), the gather is an integer chain.
    auto launch_frag = [&](auto MT_C, auto C, int nt_) {
        constexpr bool mt  = decltype(MT_C)::value;
        constexpr bool cmp = decltype(C)::value;
        constexpr size_t smem = cmp ? 0 : PAW_WALK_CBSZ;
        static bool set_attr = false;
        if (!set_attr) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void *) paw_rt_walk_qtip_frag_kernel<WORDS, mt, cmp>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
            set_attr = true;
        }
        paw_launch((paw_rt_walk_qtip_frag_kernel<WORDS, mt, cmp>),
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1),
                dim3(PAW_WALK_NTHR, 1, 1), smem, stream),
            trellis, tlut, scr_u, scr_v, m, n, nt_);
    };
    if (nt <= 1) {
        if (computed) launch_frag(std::false_type{}, std::true_type{}, nt);
        else          launch_frag(std::false_type{}, std::false_type{}, nt);
    } else {
        if (computed) launch_frag(std::true_type{}, std::true_type{}, nt);
        else          launch_frag(std::true_type{}, std::false_type{}, nt);
    }
}

static void paw_rt_walk_qtip_rate_launch(const uint16_t * trellis,
        const half * tlut, const float * scr_u, float * scr_v,
        const int m, const int n, const int words, cudaStream_t stream) {
    // column-split factor: enough blocks in flight that no single warp walks
    // a long serial chain of tile columns (the walk is latency-bound); the
    // machine holds ~42k threads = ~168 of these 256-thread blocks, so any
    // matrix with fewer than ~1500 output-tile blocks gets split
    static const int split_min_blocks =
        paw_env_int("GGML_PAW_WALK_SPLIT_BLOCKS", 1500);
    static const int split_max   = paw_env_int("GGML_PAW_WALK_SPLIT_MAX", 8);
    int S = 1;
    if (split_min_blocks > 0 && m/16 < split_min_blocks && n >= 256) {
        S = (split_min_blocks + m/16 - 1)/(m/16);
        if (S > split_max)  S = split_max;
        if (S*16 > n/16*8)  S = std::max(1, (n/16*8)/16);   // keep every warp busy
        if (S < 1)          S = 1;
    }
    if (S > 1) {
        CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) m*sizeof(float), stream));
    }
    switch (words) {
        case 16: paw_launch(paw_rt_walk_qtip_rate_kernel<16>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 24: paw_launch(paw_rt_walk_qtip_rate_kernel<24>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 32: paw_launch(paw_rt_walk_qtip_rate_kernel<32>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 40: paw_launch(paw_rt_walk_qtip_rate_kernel<40>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 56: paw_launch(paw_rt_walk_qtip_rate_kernel<56>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        case 64: paw_launch(paw_rt_walk_qtip_rate_kernel<64>,
            ggml_cuda_kernel_launch_params(dim3(m/16, S, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, scr_u, scr_v, m, n); break;
        default: GGML_ABORT("paw: unsupported trellis rate for qtip walk");
    }
}

static void paw_rt_dense_decode_rate_launch(const uint16_t * trellis,
        const half * tlut, half * bank, const int m, const int n,
        const int words, cudaStream_t stream) {
    switch (words) {
        case 16: paw_launch(paw_rt_dense_decode_rate_kernel<16>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 24: paw_launch(paw_rt_dense_decode_rate_kernel<24>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 32: paw_launch(paw_rt_dense_decode_rate_kernel<32>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 40: paw_launch(paw_rt_dense_decode_rate_kernel<40>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 56: paw_launch(paw_rt_dense_decode_rate_kernel<56>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 64: paw_launch(paw_rt_dense_decode_rate_kernel<64>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        default: GGML_ABORT("paw: unsupported trellis rate for dense decode");
    }
}

// fp8 twin of paw_rt_dense_decode_rate_launch (rate-generic e5m2 banks)
static void paw_rt_dense_decode_rate_launch_fp8(const uint16_t * trellis,
        const half * tlut, uint8_t * bank, const int m, const int n,
        const int words, cudaStream_t stream) {
    switch (words) {
        case 16: paw_launch(paw_rt_dense_decode_rate_kernel_fp8<16>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 24: paw_launch(paw_rt_dense_decode_rate_kernel_fp8<24>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 32: paw_launch(paw_rt_dense_decode_rate_kernel_fp8<32>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 40: paw_launch(paw_rt_dense_decode_rate_kernel_fp8<40>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 56: paw_launch(paw_rt_dense_decode_rate_kernel_fp8<56>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        case 64: paw_launch(paw_rt_dense_decode_rate_kernel_fp8<64>,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            trellis, tlut, bank, m, n); break;
        default: GGML_ABORT("paw: unsupported trellis rate for dense decode fp8");
    }
}


static bool paw_bank_cache_on() {
    static const bool on = paw_env_int("GGML_PAW_BANK_CACHE", 1) != 0;
    return on;
}

static size_t paw_rt_idx_bytes(const int m, const int n) {
    return (size_t) m*(n/16)*5*sizeof(uint16_t);
}

// decode-once and return the cached bank [m, n] for this trellis (fp16 or
// e5m2 fp8). Caller must not hold the mutex; the first call per matrix syncs
// the stream to make the decode visible before the dependent GEMM launches.
// words is the trellis rate (16*rt_K): 64 selects the original K=4 kernels,
// any other supported rate goes through the rate-templated decode twins.
static const void * paw_rt_bank_get(
        const void * trellis, const void * tlut, const int m, const int n, cudaStream_t stream,
        const int words = 64) {
    const bool idx = words == 64 && paw_rt_bank_idx_on();
    const bool fp8 = !idx && paw_rt_bank_fp8_on();
    const paw_bank_key key{trellis, m, n};
    {
        std::lock_guard<std::mutex> lock(paw_rt_bank_mutex);
        auto it = paw_rt_banks.find(key);
        if (it != paw_rt_banks.end()) {
            return it->second;
        }
    }
    void * bank = nullptr;
    const size_t idx_bytes = paw_rt_idx_bytes(m, n);
    const size_t bank_bytes = idx ? idx_bytes + (size_t) m*n*sizeof(half) : (size_t) m*n*(fp8 ? 1 : (int) sizeof(half));
    CUDA_CHECK(cudaMalloc(&bank, bank_bytes));
    if (idx) {
        paw_launch(paw_rt_dense_decode_kernel_idx80,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis, (uint16_t *) bank, m, n);
        paw_launch(paw_rt_dense_decode_kernel,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis, (const half *) tlut, (half *) ((uint8_t *) bank + idx_bytes), m, n);
    } else if (fp8) {
        if (words == 64) {
            paw_launch(paw_rt_dense_decode_kernel_fp8,
                ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
                (const uint16_t *) trellis, (const half *) tlut, (uint8_t *) bank, m, n);
        } else {
            paw_rt_dense_decode_rate_launch_fp8((const uint16_t *) trellis,
                (const half *) tlut, (uint8_t *) bank, m, n, words, stream);
        }
    } else {
        if (words == 64) {
            paw_launch(paw_rt_dense_decode_kernel,
                ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
                (const uint16_t *) trellis, (const half *) tlut, (half *) bank, m, n);
        } else {
            paw_rt_dense_decode_rate_launch((const uint16_t *) trellis,
                (const half *) tlut, (half *) bank, m, n, words, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        std::lock_guard<std::mutex> lock(paw_rt_bank_mutex);
        auto it = paw_rt_banks.find(key);
        if (it != paw_rt_banks.end()) {
            cudaFree(bank);
            return it->second;
        }
        paw_rt_banks.emplace(key, bank);
    }
    return bank;
}

static const half * paw_rt_idx_fp16_bank(const void * bank, const int m, const int n) {
    return (const half *) ((const uint8_t *) bank + paw_rt_idx_bytes(m, n));
}

// bandwidth-bound fp16 GEMV/GEMM over a pre-decoded bank:
// scr_v[t, m] = scr_u[t, n] @ bank[m, n]^T. One warp per (output row, token);
// lanes read the row's half2s stride-32 (coalesced) and a warp-shuffle reduce
// finishes the row. Grid (m/8, 1, nt). This is the walk path's replacement at
// small nt: the decode work is amortized to one time per matrix instead of
// every step.
static __global__ void paw_rt_bank_gemv(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT scr_u,  // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    const int row  = blockIdx.x*8 + (threadIdx.x >> 5);
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;

    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();

    const float  * u = scr_u + (int64_t) t*n;
    const half   * W = bank   + (int64_t) row*n;
    const half2  * W2 = (const half2 *) W;
    const float2 * u2 = (const float2 *) u;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const float2 w = __half22float2(W2[i]);
        const float2 x = u2[i];
        acc += w.x*x.x + w.y*x.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}

// float4 variant of paw_rt_bank_gemv: wider loads (16 B/lane/iter instead of
// 4) to cut load-issue pressure, and u is staged in shared once per block so
// the 8 row-warps do not re-issue the same L2 loads. Same math, same output.
// Grid (m/16, 1, nt), 256 threads (8 warps, each handles 2 rows via a 4-row
// stride). GGML_PAW_RT_GEMV2=1 opt-in.
static __global__ void paw_rt_bank_gemv_v2(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT scr_u,  // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    constexpr int WARPS = 8;
    __shared__ float u_sh[4096];

    const int blk  = blockIdx.x*WARPS*2;    // first row of this block
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    ggml_cuda_pdl_sync();
    const float * u = scr_u + (int64_t) t*n;
    for (int i = tid; i < n; i += 256) {
        u_sh[i] = u[i];
    }
    __syncthreads();

    const int n4 = n/4;
    for (int r = wid; r < WARPS*2; r += WARPS) {   // 2 rows per warp
        const int row = blk + r;
        if (row < m) {
            const float4 * u4 = (const float4 *) u_sh;
            const uint2  * W4 = (const uint2 *)(bank + (int64_t) row*n);   // 4 halves per uint2
            float acc = 0.0f;
            for (int i = lane; i < n4; i += 32) {
                const half2 w0 = *((const half2 *) &W4[i].x);
                const half2 w1 = *((const half2 *) &W4[i].y);
                const float2 wf0 = __half22float2(w0);
                const float2 wf1 = __half22float2(w1);
                const float4 uu = u4[i];
                acc += wf0.x*uu.x + wf0.y*uu.y + wf1.x*uu.z + wf1.y*uu.w;
            }
            acc = warp_reduce_sum<32>(acc);
            if (lane == 0) {
                scr_v[(int64_t) t*m + row] = acc;
            }
        }
    }
}

// u-fused GEMV: computes u = FWHT(su .* x) internally (redundantly in every
// block, from the same su/x the rt_u kernel reads) instead of consuming the
// separate rt_u kernel's scr_u. This removes the rt_u stage's launch and
// serialization from every rt matrix's critical path -- the rt_u cost
// (~2.8 ms/token timed) becomes a few us of concurrent redundant FWHT work
// inside the gemv kernels. scr_v is the only output. The scale is applied
// inside the kernel so the gemv reads the exact normalized u. Grid
// (m/8, 1, nt), 256 threads (one row per warp). Requires n >= 256 and
// n % 2 == 0 (FWHT wg=256 + half2 gemv). GGML_PAW_RT_GEMVU=1 opt-in.
static __global__ void paw_rt_gemv_u_kernel(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT su,     // [n]
        const float * GGML_CUDA_RESTRICT x,      // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    __shared__ float sh[4096];

    const int row  = blockIdx.x*8 + (threadIdx.x >> 5);
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();
    for (int i = tid; i < n; i += 256) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    paw_fwht_block(sh, n, tid, 256);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += 256) {
        sh[i] = __fdiv_rn(sh[i], sc);
    }
    __syncthreads();

    const half2  * W2 = (const half2 *) (bank + (int64_t) row*n);
    const float2 * u2 = (const float2 *) sh;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const float2 w = __half22float2(W2[i]);
        const float2 uu = u2[i];
        acc += w.x*uu.x + w.y*uu.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}
// fused rt_u + fp8-bank GEMV: the e5m2 twin of paw_rt_gemv_u_kernel. The
// half-reading variant must not run against an fp8 cached bank, so when
// GGML_PAW_RT_BANK_FP8=1 this kernel handles the fused path instead.
static __global__ void paw_rt_gemv_u_fp8_kernel(
        const uint8_t * GGML_CUDA_RESTRICT bank,  // [m, n] row-major e5m2
        const float   * GGML_CUDA_RESTRICT su,    // [n]
        const float   * GGML_CUDA_RESTRICT x,     // [nt, n] row-major
        float         * GGML_CUDA_RESTRICT scr_v, // [nt, m] row-major
        const int m, const int n, const int nt) {
    __shared__ float sh[4096];
    __shared__ float lut[256];

    const int tid  = threadIdx.x;
    if (tid < 256) {
        lut[tid] = paw_e5m2_to_f32((uint8_t) tid);
    }
    __syncthreads();

    const int row  = blockIdx.x*8 + (tid >> 5);
    const int t    = blockIdx.z;
    const int lane = tid & 31;

    if (row >= m) {
        return;
    }
    ggml_cuda_pdl_sync();
    for (int i = tid; i < n; i += 256) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    paw_fwht_block(sh, n, tid, 256);
    const float sc = __fsqrt_rn((float) n);
    for (int i = tid; i < n; i += 256) {
        sh[i] = __fdiv_rn(sh[i], sc);
    }
    __syncthreads();

    const uint8_t * W = bank + (int64_t) row*n;
    const float4 * u4 = (const float4 *) sh;

    float acc = 0.0f;
    if (n % 4 == 0) {
        const uint32_t * W4 = (const uint32_t *) W;
        const int n4 = n/4;
        for (int i = lane; i < n4; i += 32) {
            const uint32_t w = W4[i];
            const uint8_t * wb = (const uint8_t *) &w;
            const float4 uu = u4[i];
            acc += lut[wb[0]]*uu.x + lut[wb[1]]*uu.y + lut[wb[2]]*uu.z + lut[wb[3]]*uu.w;
        }
    } else {
        for (int i = lane; i < n; i += 32) {
            acc += lut[W[i]]*sh[i];
        }
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[(int64_t) t*m + row] = acc;
    }
}
// bank GEMV. (1) u is staged in shared once per block, killing the 8x
// redundant per-warp u re-reads that were inflating DRAM traffic past the
// 32 MB bank. (2) the bank is read with __ldcs (evict-first streaming): it is
// touched exactly once per step, so polluting L2 with it only evicts the hot
// activations/scratch. (3) two independent accumulators give the FMA chain
// ILP. Same math, same output. Grid (m/16, 1, nt), 256 threads (2 rows/warp).
// GGML_PAW_RT_GEMV3=1 opt-in.
static __global__ void paw_rt_bank_gemv_v3(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] row-major
        const float * GGML_CUDA_RESTRICT scr_u,  // [nt, n] row-major
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] row-major
        const int m, const int n, const int nt) {
    constexpr int WARPS = 8;
    __shared__ float u_sh[4096];

    const int blk  = blockIdx.x*WARPS*2;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    ggml_cuda_pdl_sync();
    const float * u = scr_u + (int64_t) t*n;
    for (int i = tid; i < n; i += 256) {
        u_sh[i] = u[i];
    }
    __syncthreads();

    const int     n2 = n/2;
    const half2 * W2 = (const half2 *) bank;
    const float2 * u2 = (const float2 *) u_sh;

    for (int r = wid; r < WARPS*2; r += WARPS) {
        const int row = blk + r;
        if (row < m) {
            const half2 * Wr = W2 + (int64_t) row*n2;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            int i = lane;
            for (; i + 32 < n2; i += 64) {
                const float2 w0 = __half22float2(__ldcs(Wr + i));
                const float2 w1 = __half22float2(__ldcs(Wr + i + 32));
                const float2 x0 = u2[i];
                const float2 x1 = u2[i + 32];
                acc0 += w0.x*x0.x + w0.y*x0.y;
                acc1 += w1.x*x1.x + w1.y*x1.y;
            }
            for (; i < n2; i += 32) {
                const float2 w0 = __half22float2(__ldcs(Wr + i));
                const float2 x0 = u2[i];
                acc0 += w0.x*x0.x + w0.y*x0.y;
            }
            float acc = acc0 + acc1;
            acc = warp_reduce_sum<32>(acc);
            if (lane == 0) {
                scr_v[(int64_t) t*m + row] = acc;
            }
        }
    }
}

// --- fused RT_MM (cooperative): rt_u + bank GEMV + rt_out in ONE launch ---
//
// The RT op was three launches per matrix (rt_u, bank GEMV, rt_out), each
// tiny and mostly launch-latency-bound at nt=1. This variant runs all three
// stages in a single cooperative launch: block 0 does the u-FWHT, grid.sync,
// all blocks do the fp16 GEMV over the cached bank, grid.sync, block 0 does
// the v-FWHT + scale. The u/v scratch goes through the same global pool
// buffers (L2-hot, only 8-32KB per stage). Grid must be co-resident, so it is
// capped at nsm blocks with each block looping over its share of rows.
// GGML_PAW_RT_FUSED=1 opt-in.
static __global__ void paw_rt_fused_kernel(
        const half  * GGML_CUDA_RESTRICT bank,   // [m, n] pre-decoded fp16
        const float * GGML_CUDA_RESTRICT su,     // [n]
        const float * GGML_CUDA_RESTRICT sv,     // [m]
        const float * GGML_CUDA_RESTRICT x,      // [nt, n]
        float       * GGML_CUDA_RESTRICT scr_u,  // [nt, n] scratch
        float       * GGML_CUDA_RESTRICT scr_v,  // [nt, m] scratch
        float       * GGML_CUDA_RESTRICT dst,    // [nt, m]
        const int m, const int n, const int nt) {
    namespace cg = cooperative_groups;
    const cg::grid_group grid = cg::this_grid();

    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    __shared__ float sh[8192];   // reused for u (n <= 4096) and v (m <= 8192)

    // Phase 1: rt_u on block 0 only
    if (blockIdx.x == 0) {
        for (int i = tid; i < n; i += 256) {
            sh[i] = su[i] * x[(int64_t) t*n + i];
        }
        __syncthreads();
        paw_fwht_block(sh, n, tid, 256);
        const float sc = __fsqrt_rn((float) n);
        for (int i = tid; i < n; i += 256) {
            scr_u[(int64_t) t*n + i] = __fdiv_rn(sh[i], sc);
        }
    }
    grid.sync();

    // Phase 2: fp16 GEMV over the cached bank, rows striped across blocks
    {
        const float  * u  = scr_u + (int64_t) t*n;
        const float2 * u2 = (const float2 *) u;
        const half2  * b2 = (const half2 *) bank;
        const int n2 = n/2;
        for (int r0 = blockIdx.x*8; r0 < m; r0 += gridDim.x*8) {
            const int row = r0 + (tid >> 5);
            if (row < m) {
                const half2 * W2 = b2 + (int64_t) row*n2;
                float acc = 0.0f;
                for (int i = lane; i < n2; i += 32) {
                    const float2 w  = __half22float2(W2[i]);
                    const float2 uu = u2[i];
                    acc += w.x*uu.x + w.y*uu.y;
                }
                acc = warp_reduce_sum<32>(acc);
                if (lane == 0) {
                    scr_v[(int64_t) t*m + row] = acc;
                }
            }
        }
    }
    grid.sync();

    // Phase 3: rt_out on block 0 only
    if (blockIdx.x == 0) {
        for (int i = tid; i < m; i += 256) {
            sh[i] = scr_v[(int64_t) t*m + i];
        }
        __syncthreads();
        paw_fwht_block(sh, m, tid, 256);
        const float sc = __fsqrt_rn((float) m);
        for (int i = tid; i < m; i += 256) {
            dst[(int64_t) t*m + i] = __fdiv_rn(sh[i], sc) * sv[i];
        }
    }
}

template <int WG, bool EPILOGUE>
static __global__ void paw_rt_out_kernel(
        const float * GGML_CUDA_RESTRICT sv,
        const float * GGML_CUDA_RESTRICT scr_v,
        float       * GGML_CUDA_RESTRICT dst,
        const float * GGML_CUDA_RESTRICT gate,
        const float * GGML_CUDA_RESTRICT acc,
        const int m,
        const int blk) {
    // blk == m reproduces the pre-blocking kernel exactly (one iteration).
    // blockIdx.y selects the output chunk so chunks run on separate SMs
    // instead of serializing inside one block (per-chunk math is unchanged).
    __shared__ float sh[8192];

    const int t   = blockIdx.x;
    const int off = blockIdx.y * blk;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float sc     = __fsqrt_rn((float) blk);
    const float inv_sc = __frcp_rn(sc);
    {
        for (int i = tid; i < blk; i += WG) {
            __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) t*m + off + i], sizeof(float));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        if (WG == blk/16 && paw_fwht_v2_ok(blk)) {
            paw_fwht_block_v2(sh, blk, tid, WG);
        } else {
            paw_fwht_block(sh, blk, tid, WG);
        }
        for (int i = tid; i < blk; i += WG) {
            const int64_t oi = (int64_t) t*m + off + i;
            const float y = sh[i] * inv_sc * sv[off + i];
            if constexpr (EPILOGUE) {
                const float sigmoid = 1.0f / (1.0f + expf(-gate[t]));
                const float gated = __fmul_rn(y, sigmoid);
                dst[oi] = __fadd_rn(acc[oi], gated);
            } else {
                dst[oi] = y;
            }
        }
        __syncthreads();
    }
}

template <int WG>
static __global__ void paw_rt_out_epilogue_dot_kernel(
        const float * GGML_CUDA_RESTRICT sv,
        const float * GGML_CUDA_RESTRICT scr_v,
        float       * GGML_CUDA_RESTRICT dst,
        const float * GGML_CUDA_RESTRICT gate_w,
        const float * GGML_CUDA_RESTRICT gate_x,
        const float * GGML_CUDA_RESTRICT acc,
        const int m, const int gate_n, const int blk) {
    __shared__ float sh[8192];
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    float gate_sum = 0.0f;
    for (int i = tid; i < gate_n; i += WG) {
        gate_sum = fmaf(gate_w[i], gate_x[(int64_t) t*gate_n + i], gate_sum);
    }
    sh[tid] = gate_sum;
    __syncthreads();
    for (int stride = WG/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }
    const float sigmoid = 1.0f / (1.0f + expf(-sh[0]));
    __syncthreads();   // sh is about to be reused as the FWHT staging buffer

    const float inv_sc = __frcp_rn(__fsqrt_rn((float) blk));
    for (int off = 0; off < m; off += blk) {
        for (int i = tid; i < blk; i += WG) {
            __pipeline_memcpy_async(&sh[i], &scr_v[(int64_t) t*m + off + i], sizeof(float));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        if (WG == blk/16 && paw_fwht_v2_ok(blk)) {
            paw_fwht_block_v2(sh, blk, tid, WG);
        } else {
            paw_fwht_block(sh, blk, tid, WG);
        }
        for (int i = tid; i < blk; i += WG) {
            const int64_t oi = (int64_t) t*m + off + i;
            const float y = sh[i] * inv_sc * sv[off + i];
            dst[oi] = __fadd_rn(acc[oi], __fmul_rn(y, sigmoid));
        }
        __syncthreads();
    }
}

void ggml_cuda_op_paw_rt_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * su      = dst->src[1];
    const ggml_tensor * sv      = dst->src[2];
    const ggml_tensor * tlut    = dst->src[3];
    const ggml_tensor * x       = dst->src[4];
    const int epilogue_mode       = dst->op_params[0];
    const bool epilogue           = epilogue_mode != 0;
    const bool epilogue_dot       = epilogue_mode == 2;
    const ggml_tensor * gate      = epilogue ? dst->src[5] : nullptr;
    const ggml_tensor * gate_x    = epilogue_dot ? dst->src[6] : nullptr;
    const ggml_tensor * acc       = epilogue_dot ? dst->src[7] : epilogue ? dst->src[6] : nullptr;

    GGML_ASSERT(trellis->type == GGML_TYPE_I16);
    GGML_ASSERT(su->type   == GGML_TYPE_F32);
    GGML_ASSERT(sv->type   == GGML_TYPE_F32);
    GGML_ASSERT(tlut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!epilogue || (gate && acc && gate->type == GGML_TYPE_F32 && acc->type == GGML_TYPE_F32));
    GGML_ASSERT(!epilogue_dot || (gate_x && gate_x->type == GGML_TYPE_F32));
    GGML_ASSERT(ggml_is_contiguous(trellis));
    GGML_ASSERT(ggml_is_contiguous(su));
    GGML_ASSERT(ggml_is_contiguous(sv));
    GGML_ASSERT(ggml_is_contiguous(tlut));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // Trellis rate from the payload: words-per-tile = 16*K at V = 2.
    // 64 -> K=4 (the shipped 35B NE spine), 32 -> K=2, 16 -> K=1.
    const int rt_words = (int) trellis->ne[0];
    const int rt_K     = rt_words / 16;   // == 4 only for the shipped payload
    GGML_ASSERT(rt_words % 8 == 0 && rt_words >= 16 && rt_words <= 64);

    const int n  = (int) su->ne[0];
    const int m  = (int) sv->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);
    // Rotation block size. 0 means one Hadamard over the whole dimension --
    // the shipped power-of-two payloads. A blocked payload rotates within
    // blk-wide groups, which is what lets non-power-of-two dense shapes
    // (5120, 17408, ...) run at all: only one block is ever staged in shared.
    const int rht_blk = dst->op_params[GGML_PAW_RHT_BLK_SLOT];
    const int bn = rht_blk ? rht_blk : n;
    const int bm = rht_blk ? rht_blk : m;
    const bool blocked = (bn != n) || (bm != m);
    // The bank decode, the fused cooperative kernel, the gemv variants and the
    // qtip walk all bake in the K=4 stream layout. Only the generic walk below
    // is rate-aware, so every other rate is routed through it.
    const bool k4 = rt_words == 64;
    GGML_ASSERT(bn <= 4096);          // rt_u shared bound, now per block
    GGML_ASSERT(bm <= 8192);          // rt_out shared bound, now per block
    GGML_ASSERT(n % bn == 0 && m % bm == 0);

    // The fused-int8 AR walk needs one byte/activation plus one float scale
    // per 16 values.  Keep the workspace attached to this op so CUDA graph
    // capture and concurrent contexts do not share mutable global storage.
    const size_t i8_q_floats = ((size_t)n + sizeof(float) - 1)/sizeof(float);
    const size_t i8_s_floats = (size_t)n/16;
    ggml_cuda_pool_alloc<float> scr(ctx.pool(),
        (size_t)nt*n + (size_t)nt*m + i8_q_floats + 2*i8_s_floats);
    float * scr_u = scr.get();
    float * scr_v = scr_u + (size_t) nt*n;
    int8_t * q_u = (int8_t *)(scr_v + (size_t)nt*m);
    float * q_s = (float *)(q_u + i8_q_floats*sizeof(float));
    int * q_z = (int *)(q_s + i8_s_floats);

    cudaStream_t stream = ctx.stream();

    char shp[64];
    snprintf(shp, sizeof(shp), " m=%d n=%d nt=%d", m, n, nt);

    paw_fwht_set_mode();

    static const bool rt_fused = paw_env_int("GGML_PAW_RT_FUSED", 0) != 0;
    static const int  dense_min_tok = paw_env_int("GGML_PAW_DENSE_MIN_TOK", 4);
    // above this many tokens, a per-pass fp16 bank + batched apply beats the
    // generic per-token walk on non-K4 payloads
    static const int  rt_blas_min_nt = paw_env_int("GGML_PAW_RT_BLAS_MIN_NT", 8);
    if (k4 && !blocked && rt_fused && !paw_rt_bank_fp8_on() && !paw_rt_bank_idx_on() && nt < dense_min_tok && paw_bank_cache_on()) {
        // cooperative single-launch RT_MM (rt_u + bank GEMV + rt_out)
        const int id = ggml_cuda_get_device();
        const bool coop = ggml_cuda_info().devices[id].supports_cooperative_launch;
        if (coop) {
            const half * bank = (const half *) paw_rt_bank_get(trellis->data, tlut->data, m, n, stream);
            const float * su_p = (const float *) su->data;
            const float * sv_p = (const float *) sv->data;
            const float * x_p  = (const float *) x->data;
            float * dst_p      = (float *) dst->data;
            const dim3 block_nums(ggml_cuda_info().devices[id].nsm, 1, nt);
            const dim3 block_dims(256, 1, 1);
            void * args[] = { (void *) &bank, (void *) &su_p,
                                    (void *) &sv_p, (void *) &x_p,
                                    (void *) &scr_u, (void *) &scr_v,
                                    (void *) &dst_p, (void *) &m,
                                    (void *) &n, (void *) &nt };
            paw_timed(stream, std::string("rt_fused") + shp, [&]() {
                CUDA_CHECK(cudaLaunchCooperativeKernel((void *) paw_rt_fused_kernel,
                        block_nums, block_dims, args, 0, stream));
            });
            return;
        }
    }

    // K=4 fused walk, OFF by default -- and it must stay off.
    //
    // Measured +12.0% on PAW-35B (75.45 -> 84.54 tg64) and that number was an
    // artifact of benchmarking without the model's own serving flags. PAW-35B
    // ships every codec optimization opt-in and default-off (see
    // PAW-weights/SERVING.md and artifacts/mach1_reverse/v14/serve_best.sh);
    // with GGML_PAW_RT_BATCH and GGML_PAW_RT_BANK_IDX on, as they are meant to
    // be, the decode-to-bank path this replaces is *faster*:
    //
    //   tuned config, AR, no drafter:   qtip=0  92.37   qtip=1  90.53
    //   tuned config, dflash2 drafter:  qtip=0  84.81   qtip=1  84.80
    //
    // The staged WORDS=64 walk below is kept and still works, but it only ever
    // runs at nt == 1, which the drafter bypasses, and it loses to a properly
    // batched bank path. Turn it on only with a measurement to justify it.
    static const bool rt_walk_qtip = paw_env_int("GGML_PAW_RT_WALK_QTIP", 0) != 0;
    const bool bank_cache = paw_bank_cache_on();

    // u-fused gemv: skip the separate rt_u launch entirely; the gemv kernel
    // computes u internally. Valid when the gemv geometry matches the kernel's
    // assumptions (n >= 256, n % 2 == 0, m % 8 == 0).
    static const bool rt_gemvu = paw_env_int("GGML_PAW_RT_GEMVU", 0) != 0;
    const bool gemvu = k4 && !blocked && rt_gemvu && bank_cache && !paw_rt_bank_idx_on() &&
                       nt < dense_min_tok &&
                       n >= 256 && n % 2 == 0 && m % 8 == 0;

    static const bool uout_noop = paw_env_int("GGML_PAW_UOUT_NOOP", 0) != 0;
    if (!gemvu && !uout_noop) {
    paw_timed(stream, std::string("rt_u") + shp, [&]() {
    if (!blocked && paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
        paw_fwht_for_wg(n/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            paw_launch(paw_rt_u_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(nt, n/bn, 1), dim3(wg, 1, 1), 0, stream),
                (const float *) su->data, (const float *) x->data, scr_u, n, bn);
        });
    } else if (!blocked && paw_fwht_wg512()) {
        paw_launch(paw_rt_u_kernel<512>,
            ggml_cuda_kernel_launch_params(dim3(nt, n/bn, 1), dim3(512, 1, 1), 0, stream),
            (const float *) su->data, (const float *) x->data, scr_u, n, bn);
    } else {
        paw_launch(paw_rt_u_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(nt, n/bn, 1), dim3(256, 1, 1), 0, stream),
            (const float *) su->data, (const float *) x->data, scr_u, n, bn);
    }
    });
    }

    // the rate-templated walk is default-ON for non-K4 payloads: they have
    // no cached-bank alternative at nt==1 (banking the whole dense FFN would
    // not fit VRAM), and the generic walk is ~10x slower per call
    static const bool rt_walk_qtip_dense = paw_env_int("GGML_PAW_RT_WALK_QTIP_DENSE", 1) != 0;
    // Multi-token walk: the frag kernel's eight mma B-columns can carry eight
    // tokens for ONE walk of the trellis. The walk is 81% of a decode token
    // and is LSU-gather bound (ncu: l1tex 60.7%, dram 16.0%, tensor 9.1%), so
    // dividing the gathers across nt tokens is the only lever that scales --
    // the gather:weight ratio is fixed by the codec at 1:2 and cannot be cut
    // in place. Restricted to the dense frag path: the rate kernel is
    // documented [nt=1, m] and the k4 payloads have their own bank cache.
    // GGML_PAW_WALK_MT_MAX=1 disables and restores the nt==1-only behaviour.
    static const int  walk_mt_max  = paw_env_int("GGML_PAW_WALK_MT_MAX", 8);
    static const bool walk_frag_g  = paw_env_int("GGML_PAW_WALK_FRAG", 1) != 0;
    static const bool walk_noop_g  = paw_env_int("GGML_PAW_WALK_NOOP", 0) != 0;
    static const bool walk_skip_g  = paw_env_int("GGML_PAW_WALK_SKIP_M_ON", 0) != 0;
    const bool qtip_mt = !k4 && rt_walk_qtip_dense && walk_frag_g &&
                         !walk_noop_g && !walk_skip_g &&
                         nt > 1 && nt <= walk_mt_max;
    const bool use_qtip = (nt == 1 && (k4 ? rt_walk_qtip : rt_walk_qtip_dense) &&
                           !debug_diff_on()) ||
                          (qtip_mt && !debug_diff_on());
    // Decode-once (DOL) for non-K4 payloads -- the same deal the K=4 payloads
    // already get from the bank cache. The frag walk is LSU-gather bound
    // (ncu: l1tex 60.7%, dram 16.0%, tensor 9.1%) and is ~81% of a K=2
    // decode token; in-place gather tuning is exhausted (16x codebook
    // replication is gather-clean and did not move the time), so the
    // remaining lever is to not decode the trellis per token at all: the
    // first call for a matrix materializes a persistent fp16 (or e5m2) bank
    // and every later call is a plain bandwidth-bound GEMV over it.
    // GGML_PAW_DOL_K2=1 opt-in because the bank lives in VRAM next to the
    // model (2 bytes/weight fp16, 1 byte/weight e5m2) and can OOM a small
    // card -- the same trade the K4 bank cache makes.
    // No blocked check: rht_blk only shapes the rt_u/rt_out rotations; the
    // bank is the plain decoded [m, n] weight and the GEMV below is agnostic,
    // exactly like the K4 bank branch and the dense-apply branch.
    static const bool dol_on = paw_env_int("GGML_PAW_DOL_K2", 0) != 0;
    // Bank budget: on small cards the model + full banks do not fit (3060:
    // 7.27 GiB model + 5.62 GB fp8 banks > 11.9 GB). Bank whatever fits,
    // walk the rest -- benefit per banked byte is uniform (walk time and
    // bank bytes both scale with m*n), so greedy first-come banking is
    // near-optimal. GGML_PAW_DOL_MAX_MB caps the total; 0 = unbounded.
    static const int64_t dol_max_bytes =
        (int64_t) paw_env_int("GGML_PAW_DOL_MAX_MB", 0) << 20;
    // Per-matrix cap: bank small matrices first. The frag walk's cost per
    // weight depends on m/16 block count (small m = few blocks per SM =
    // latency-bound walk): on PAW-27B-v12d the small matrices walk at ~34
    // ns/weight vs ~4 ns/weight for the big ones, so banking smalls buys
    // ~8x more decode-time per bank byte than banking bigs (which are also
    // the ones a plain bandwidth-bound GEMV helps least).
    static const int64_t dol_max_bank_bytes =
        (int64_t) paw_env_int("GGML_PAW_DOL_MAX_BANK_MB", 0) << 20;
    const bool dol = !k4 && dol_on && !debug_diff_on() &&
                     nt < rt_blas_min_nt && !walk_noop_g && !walk_skip_g;
    bool dol_bank = false;
    if (dol) {
        const bool fp8b = !paw_rt_bank_idx_on() && paw_rt_bank_fp8_on();
        const int64_t need = (int64_t) m*n*(fp8b ? 1 : 2);
        std::lock_guard<std::mutex> lock(paw_rt_bank_mutex);
        const bool cached =
            paw_rt_banks.count(paw_bank_key{trellis->data, m, n}) != 0;
        // reserve at decision time so a single token cannot overshoot the
        // budget while its banks are still decoding; decode is single-threaded
        // per graph compute, so the reservation cannot double-count.
        dol_bank = cached ||
            (dol_max_bytes == 0 || paw_dol_banked + need <= dol_max_bytes);
        if (dol_bank && !cached && dol_max_bank_bytes != 0 &&
            need > dol_max_bank_bytes) {
            dol_bank = false;   // too big per-matrix: its walk is efficient
        }
        if (!cached && dol_bank) {
            paw_dol_banked += need;
            static const bool dol_trace = paw_env_int("GGML_PAW_DOL_TRACE", 0) != 0;
            if (dol_trace) {
                fprintf(stderr, "[dol-trace] reserve m=%d n=%d words=%d need=%.1fMB total=%.1fMB fp8=%d\n",
                        m, n, rt_words, need/1048576.0, paw_dol_banked/1048576.0, (int) fp8b);
            }
        }
    }
    if (dol && dol_bank) {
        const void * bank = paw_rt_bank_get(trellis->data, tlut->data, m, n, stream, rt_words);
        const bool fp8 = !paw_rt_bank_idx_on() && paw_rt_bank_fp8_on();
        paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
        // GGML_PAW_DOL_APPLY_MMA=1: tensor-core apply over the fp16 bank
        // (weight-stationary wmma, needs m tiled by 64). At batch 1 the
        // scalar gemvs are issue-bound on the per-weight FMA chain, while
        // the wmma apply is purely byte-bound -- 2 bytes/weight at DRAM
        // speed beats 1 byte/weight at issue speed on the big shapes.
        static const bool dol_apply_mma = paw_env_int("GGML_PAW_DOL_APPLY_MMA", 0) != 0;
        if (dol_apply_mma && !fp8 && m % 64 == 0) {
            paw_launch_rt_apply_mma(ctx, stream, (const half *) bank,
                (const float *) scr_u, scr_v, m, n, nt);
        } else
        // the v3 gemvs stage u in __shared__ float[4096]: only legal for
        // n <= 4096. This payload's dense shapes (5120/6144/17408) exceed
        // that, so they take the plain one-row-per-warp gemv (no u staging,
        // any n -- bandwidth-bound either way at these widths).
        if (fp8) {
            if (n <= 4096 && n % 4 == 0) {
                paw_launch(paw_rt_bank_gemv_fp8_v3,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_bank_gemv_fp8,
                    ggml_cuda_kernel_launch_params(dim3((m + 7)/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) bank, (const float *) scr_u, scr_v, m, n, nt);
            }
        } else {
            static const bool rt_gemv3 = paw_env_int("GGML_PAW_RT_GEMV3", 1) != 0;
            if (rt_gemv3 && n <= 4096 && n % 2 == 0) {
                paw_launch(paw_rt_bank_gemv_v3,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_bank_gemv,
                    ggml_cuda_kernel_launch_params(dim3((m + 7)/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
            }
        }
        });
        // numeric oracle: the first few nt==1 calls also compute the exact
        // fp16-bank GEMV and the rate-walk reference, and report max abs
        // deltas of the chosen path against both.
        static const bool dol_debug = paw_env_int("GGML_PAW_DOL_DEBUG", 0) != 0;
        if (dol_debug && nt == 1) {
            static int dol_dbg_calls = 0;
            if (dol_dbg_calls < 16) {
                ++dol_dbg_calls;
                ggml_cuda_pool_alloc<float> walk_v(ctx.pool(), (size_t) nt*m);
                ggml_cuda_pool_alloc<float> fp16_v(ctx.pool(), (size_t) nt*m);
                ggml_cuda_pool_alloc<half>  ref_bank(ctx.pool(), (size_t) m*n);
                paw_rt_dense_decode_rate_launch((const uint16_t *) trellis->data,
                    (const half *) tlut->data, ref_bank.get(), m, n, rt_words, stream);
                paw_launch(paw_rt_bank_gemv,
                    ggml_cuda_kernel_launch_params(dim3((m + 7)/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    ref_bank.get(), (const float *) scr_u, fp16_v.get(), m, n, nt);
                paw_rt_walk_qtip_rate_launch((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, walk_v.get(),
                    m, n, rt_words, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<float> hv(nt*m), hf(nt*m), hw(nt*m);
                CUDA_CHECK(cudaMemcpy(hv.data(), scr_v, nt*m*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hf.data(), fp16_v.get(), nt*m*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hw.data(), walk_v.get(), nt*m*sizeof(float), cudaMemcpyDeviceToHost));
                double e16 = 0.0, e8 = 0.0;
                for (size_t i = 0; i < hv.size(); ++i) {
                    const double d16 = (double) hv[i] - (double) hf[i];
                    const double d8  = (double) hv[i] - (double) hw[i];
                    const double a16 = d16 < 0 ? -d16 : d16;
                    const double a8  = d8  < 0 ? -d8  : d8;
                    if (a16 > e16) e16 = a16;
                    if (a8  > e8)  e8  = a8;
                }
                fprintf(stderr, "[dol-debug] m=%d n=%d words=%d fp8=%d"
                        " |fp8bank-fp16bank|max=%.3e |fp16bank-walk|max=%.3e\n",
                        m, n, rt_words, (int) fp8, e16, e8);
            }
        }
    } else if (use_qtip) {
        static int debug_calls = 0;
        static const bool debug_diff = paw_env_int("GGML_PAW_RT_WALK_QTIP_DEBUG", 0) != 0;
        if (debug_diff && debug_calls < 6) {
            ggml_cuda_pool_alloc<float> ref_v_alloc(ctx.pool(), (size_t) m);
            float * ref_v = ref_v_alloc.get();
            if (n/16 <= 128) {
                paw_launch((paw_rt_walk_kernel<128, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(128, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n);
            } else {
                paw_launch((paw_rt_walk_kernel<256, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n);
            }
            paw_rt_walk_qtip_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, (const float *) scr_u, scr_v,
                m, n, rt_words, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> hv(m), hr(m);
            CUDA_CHECK(cudaMemcpy(hv.data(), scr_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hr.data(), ref_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            float maxdiff = 0.0f; int maxi = -1;
            for (int i = 0; i < m; ++i) {
                float d = fabsf(hv[i] - hr[i]);
                if (d > maxdiff) { maxdiff = d; maxi = i; }
            }
            fprintf(stderr, "[rt_walk_qtip DEBUG] m=%d n=%d maxdiff=%g at i=%d qtip=%g ref=%g  first5: qtip=[%g,%g,%g,%g,%g] ref=[%g,%g,%g,%g,%g]\n",
                m, n, maxdiff, maxi, maxi>=0?hv[maxi]:0.0, maxi>=0?hr[maxi]:0.0,
                hv[0], hv[1], hv[2], hv[3], hv[4], hr[0], hr[1], hr[2], hr[3], hr[4]);
            // use the reference output so generation stays coherent while debugging
            CUDA_CHECK(cudaMemcpyAsync(scr_v, ref_v, m*sizeof(float), cudaMemcpyDeviceToDevice, stream));
            debug_calls++;
        } else if (debug_diff) {
            // past the debug window: use the known-good reference path
            if (n/16 <= 128) {
                paw_launch((paw_rt_walk_kernel<128, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(128, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            } else {
                paw_launch((paw_rt_walk_kernel<256, 64>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, scr_v, m, n);
            }
        } else {
        // bypasses the bank cache entirely -- reads the compressed trellis
        // directly instead of a pre-decoded fp16 bank. See
        // paw_rt_walk_qtip_kernel's header comment. Falls through to the
        // rt_out call below (same as every other branch here) -- does NOT
        // return early.
        static const bool walk_noop = paw_env_int("GGML_PAW_WALK_NOOP", 0) != 0;
        // fragment-direct variant (default ON): each lane decodes exactly its
        // own mma.sync fragment elements straight from the trellis -- no
        // shared staging, no syncwarp in the loop; scr_v is accumulated with
        // atomics so it must be zeroed first
        static const bool walk_frag = paw_env_int("GGML_PAW_WALK_FRAG", 1) != 0;
        static const bool frag_dbg_on = paw_env_int("GGML_PAW_WALK_FRAG_DEBUG", 0) != 0;
        static int frag_dbg_calls = 0;
        if (!walk_noop && walk_frag && frag_dbg_on &&
            frag_dbg_calls < paw_env_int("GGML_PAW_WALK_FRAG_DEBUG_MAX", 4) &&
            (rt_words == 16 || rt_words == 24 || rt_words == 32)) {
            // correctness probe: frag variant vs the generic reference walk
            ggml_cuda_pool_alloc<float> ref_v_alloc(ctx.pool(), (size_t) m);
            float * ref_v = ref_v_alloc.get();
            CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) m*sizeof(float), stream));
            switch (rt_words) {
                case 24: paw_rt_walk_qtip_frag_dispatch<24>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                case 32: paw_rt_walk_qtip_frag_dispatch<32>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                default: paw_rt_walk_qtip_frag_dispatch<16>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
            }
            #define PAW_FRAG_DBG(WG) \
                switch (rt_words) { \
                    case 16: paw_launch((paw_rt_walk_kernel<WG, 16>),\
                        ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(WG, 1, 1), 0, stream),\
                        (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n); break;\
                    case 24: paw_launch((paw_rt_walk_kernel<WG, 24>),\
                        ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(WG, 1, 1), 0, stream),\
                        (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n); break;\
                    case 32: paw_launch((paw_rt_walk_kernel<WG, 32>),\
                        ggml_cuda_kernel_launch_params(dim3(m/16, 1, 1), dim3(WG, 1, 1), 0, stream),\
                        (const uint16_t *) trellis->data, (const half *) tlut->data, scr_u, ref_v, m, n); break;\
                    default: GGML_ABORT("frag-dbg: rate");\
                }
            if (n/16 <= 128) { PAW_FRAG_DBG(128) } else { PAW_FRAG_DBG(256) }
            #undef PAW_FRAG_DBG
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> hv(m), hr(m);
            CUDA_CHECK(cudaMemcpy(hv.data(), scr_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hr.data(), ref_v, m*sizeof(float), cudaMemcpyDeviceToHost));
            double md=0, sc=0; int maxi=-1;
            for (int i = 0; i < m; ++i) { double d=fabs(hv[i]-hr[i]); if(d>md){md=d;maxi=i;} sc=fmax(sc,fabs(hr[i])); }
            fprintf(stderr, "[frag-dbg] m=%d n=%d words=%d maxdiff=%g scale=%g rel=%.2e at %d | frag[0..2]=%g,%g,%g ref=%g,%g,%g\n",
                m,n,rt_words,md,sc,md/(sc+1e-30),maxi,hv[0],hv[1],hv[2],hr[0],hr[1],hr[2]);
            frag_dbg_calls++;
            return;   // skip normal path this call; v already computed by frag
        }
        // timing-only attribution knob: skip walks whose m is in the list
        // (wrong results; use with GGML_PAW_WALK_NOOP-style runs only)
        static const bool walk_skip_on = paw_env_int("GGML_PAW_WALK_SKIP_M_ON", 0) != 0;
        static const std::string walk_skip_csv = std::string(",") +
            (getenv("GGML_PAW_WALK_SKIP_M") ? getenv("GGML_PAW_WALK_SKIP_M") : "") + ",";
        const bool walk_skip = walk_skip_on &&
            walk_skip_csv.find("," + std::to_string(m) + ",") != std::string::npos;
        if (walk_skip_on) {
            static std::string seen;
            const std::string key = "," + std::to_string(m) + "x" + std::to_string(n) + "w" + std::to_string(rt_words);
            if (seen.find(key) == std::string::npos) { seen += key; fprintf(stderr, "[walk-shape]%s\n", key.c_str()); }
        }
        if (!walk_noop && !walk_skip && walk_frag) {
            CUDA_CHECK(cudaMemsetAsync(scr_v, 0, (size_t) nt*m*sizeof(float), stream));
            switch (rt_words) {
                case 16: paw_rt_walk_qtip_frag_dispatch<16>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                case 24: paw_rt_walk_qtip_frag_dispatch<24>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                case 32: paw_rt_walk_qtip_frag_dispatch<32>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                case 40: paw_rt_walk_qtip_frag_dispatch<40>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                case 56: paw_rt_walk_qtip_frag_dispatch<56>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                case 64: paw_rt_walk_qtip_frag_dispatch<64>((const uint16_t *) trellis->data,
                    (const half *) tlut->data, (const float *) scr_u, scr_v, q_u, q_s, q_z, m, n, nt, stream); break;
                default: GGML_ABORT("paw: unsupported trellis rate for frag walk");
            }
        } else if (!walk_noop && !walk_skip) paw_timed(stream, std::string("rt_walk_qtip") + shp, [&]() {
        paw_rt_walk_qtip_rate_launch((const uint16_t *) trellis->data,
            (const half *) tlut->data, (const float *) scr_u, scr_v,
            m, n, rt_words, stream);
        });
        }
    } else if (k4 && bank_cache) {
        // Marlin-style: the bank is decoded once and cached; the per-step
        // decode is skipped and both paths run a plain GEMM/GEMV over it.
        const void * bank = paw_rt_bank_get(trellis->data, tlut->data, m, n, stream);
        const bool idx = paw_rt_bank_idx_on();
        const bool fp8 = !idx && paw_rt_bank_fp8_on();
        const half * dense_bank = idx ? paw_rt_idx_fp16_bank(bank, m, n) : (const half *) bank;
        static const bool rt_tc4     = paw_env_int("GGML_PAW_RT_TC", 8) == 4;
        static const bool rt_apply_mma = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
        if (gemvu) {
            // fused rt_u + bank GEMV; the half-reading variant must not run
            // against an fp8 cached bank, so pick the twin by bank format
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            if (fp8) {
                paw_launch(paw_rt_gemv_u_fp8_kernel,
                    ggml_cuda_kernel_launch_params(dim3(m/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) dense_bank, (const float *) su->data,
                    (const float *) x->data, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_gemv_u_kernel,
                    ggml_cuda_kernel_launch_params(dim3(m/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    dense_bank, (const float *) su->data, (const float *) x->data, scr_v, m, n, nt);
            }
            });
        } else if (idx && nt < dense_min_tok) {
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            paw_launch(paw_rt_bank_gemv_idx80,
                ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(512, 1, 1), 0, stream),
                (const uint16_t *) bank, (const half *) tlut->data,
                (const float *) scr_u, scr_v, m, n, nt);
            });
        } else if (fp8 && nt < dense_min_tok) {
            // fp8 bank gemv: decode-side only. Large batches go through the
            // dense-apply pipeline below, which re-decodes from trellis and
            // never touches the stored bank, so the bank format is free to
            // differ per batch size.
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            if (n % 4 == 0) {
                paw_launch(paw_rt_bank_gemv_fp8_v3,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_bank_gemv_fp8,
                    ggml_cuda_kernel_launch_params(dim3((m + 7)/8, 1, nt), dim3(256, 1, 1), 0, stream),
                    (const uint8_t *) bank, (const float *) scr_u, scr_v, m, n, nt);
            }
            });
        } else if (nt < dense_min_tok) {
            static const bool rt_gemv2 = paw_env_int("GGML_PAW_RT_GEMV2", 0) != 0;
            // float4+shared variant: measured ~3.5 t/s gen gain on RTX 3060
            static const bool rt_gemv3 = paw_env_int("GGML_PAW_RT_GEMV3", 1) != 0;
            paw_timed(stream, std::string("rt_bank_gemv") + shp, [&]() {
            if (rt_gemv3 && n % 2 == 0) {
                paw_launch(paw_rt_bank_gemv_v3,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    dense_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else if (rt_gemv2 && n % 4 == 0) {
                paw_launch(paw_rt_bank_gemv_v2,
                    ggml_cuda_kernel_launch_params(dim3((m + 15)/16, 1, nt), dim3(256, 1, 1), 0, stream),
                    dense_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
            paw_launch(paw_rt_bank_gemv,
                ggml_cuda_kernel_launch_params(dim3((m + 7)/8, 1, nt), dim3(256, 1, 1), 0, stream),
                dense_bank, (const float *) scr_u, scr_v, m, n, nt);
            }
            });
        } else {
            // an fp8 cached bank cannot feed the fp16 GEMM: decode a fresh
            // fp16 bank from trellis into scratch for this pass
            ggml_cuda_pool_alloc<half> fp16_bank_alloc(ctx.pool());
            const half * apply_bank = dense_bank;
            if (fp8) {
                apply_bank = fp16_bank_alloc.alloc((size_t) m*n);
                paw_launch(paw_rt_dense_decode_kernel,
                    ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data,
                    (half *) apply_bank, m, n);
            }
            paw_timed(stream, std::string("rt_apply") + shp, [&]() {
            if (rt_apply_mma) {
                paw_launch_rt_apply_mma(ctx, stream, apply_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else if (rt_tc4) {
                paw_launch(paw_rt_apply_kernel<4>,
                    ggml_cuda_kernel_launch_params(
                        dim3(m/16, 1, (nt + 3)/4), dim3(128, 1, 1), 0, stream),
                    apply_bank, (const float *) scr_u, scr_v, m, n, nt);
            } else {
                paw_launch(paw_rt_apply_kernel<8>,
                    ggml_cuda_kernel_launch_params(
                        dim3(m/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                    apply_bank, (const float *) scr_u, scr_v, m, n, nt);
            }
            });
        }
    } else if (!k4 && nt >= rt_blas_min_nt) {
        // dense payload prefill: materialize an fp16 bank ONCE per pass
        // (rate-templated decode) and run the same batched apply the K4
        // cached-bank branch uses. The generic walk this replaces re-decodes
        // every weight per token -- measured 7.4 t/s prompt at nt~1000.
        static const bool rt_apply_mma_d = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
        // split the matrix into two row halves and decode half 2 on the aux
        // stream while half 1 applies on the main stream -- hides roughly
        // half the decode traffic behind apply traffic. Same event pattern
        // as the expert blas pipeline; capture-unsafe, so graphs-off only.
        cudaStreamCaptureStatus dcst = cudaStreamCaptureStatusNone;
        const bool dcap = cudaStreamIsCapturing(stream, &dcst) == cudaSuccess &&
                          dcst == cudaStreamCaptureStatusActive;
        // default OFF: the two-half overlap produces corrupted logits after a
        // long prefill (generation degenerates to repeated '?'); bisected to
        // this commit. Re-enable only after the race is found.
        static const bool rt_ovl = paw_env_int("GGML_PAW_RT_DECODE_OVL", 0) != 0;
        // voff applies only on the cublas path, which needs nt >= its own
        // threshold; below that keep the serial single-bank flow
        const bool ovl = rt_ovl && !dcap && m >= 2048 &&
                         nt >= paw_env_int("GGML_PAW_RT_BLAS_MIN_TOK", 128);
        ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());
        half * apply_bank = bank_alloc.alloc((size_t) m*n);
        auto apply_fn = [&](const half * bk, int rows, int voff) {
            if (rt_apply_mma_d) {
                paw_launch_rt_apply_mma(ctx, stream, bk,
                    (const float *) scr_u, scr_v, rows, n, nt, voff);
            } else {
                paw_launch(paw_rt_apply_kernel<8>,
                    ggml_cuda_kernel_launch_params(
                        dim3(rows/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                    bk, (const float *) scr_u, scr_v, rows, n, nt);
            }
        };
        if (!ovl) {
            paw_rt_dense_decode_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, apply_bank, m, n, rt_words, stream);
            paw_timed(stream, std::string("rt_apply_dense") + shp, [&]() {
                apply_fn(apply_bank, m, 0);
            });
        } else {
            const int m0 = m/2;   // multiples of 16 guaranteed (m >= 2048)
            cudaEvent_t ev0, ev1;
            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));
            cudaStream_t ds = paw_aux_stream();
            paw_rt_dense_decode_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, apply_bank, m0, n, rt_words, stream);
            CUDA_CHECK(cudaEventRecord(ev0, stream));
            paw_rt_dense_decode_rate_launch((const uint16_t *) trellis->data,
                (const half *) tlut->data, apply_bank + (size_t) m0*n,
                m - m0, n, rt_words, ds);
            CUDA_CHECK(cudaEventRecord(ev1, ds));
            paw_timed(stream, std::string("rt_apply_dense") + shp, [&]() {
                apply_fn(apply_bank, m0, 0);
                CUDA_CHECK(cudaStreamWaitEvent(stream, ev1));
                apply_fn(apply_bank + (size_t) m0*n, m - m0, m0);
            });
            CUDA_CHECK(cudaEventDestroy(ev0));
            CUDA_CHECK(cudaEventDestroy(ev1));
        }
    } else {
    ggml_cuda_pool_alloc<half> bank_alloc(ctx.pool());
    // paw_rt_dense_decode_kernel bakes in the K=4 stream layout: 64 words per
    // tile and the byte-aligned state window that only WORDS=64 gives. On a
    // lower-rate payload (paw-dense runs K=1/1.5/2) it reads past the end of
    // the trellis -- an illegal access on the first prefill, since nt >= 4
    // there. Only the walk below is rate-aware, so every other rate takes it.
    // TODO: templating the decode kernel on WORDS the way the walk is would
    // restore the prefill fast path for dense payloads; it needs the generic
    // bit-offset window extraction and a numeric check against the walk.
    if (k4 && nt >= dense_min_tok) {
        half * bank = bank_alloc.alloc((size_t) m*n);
        paw_timed(stream, std::string("rt_dense_decode") + shp, [&]() {
        paw_launch(paw_rt_dense_decode_kernel,
            ggml_cuda_kernel_launch_params(dim3((m/16*n + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream),
            (const uint16_t *) trellis->data, (const half *) tlut->data, bank, m, n);
        });
        static const bool rt_tc4     = paw_env_int("GGML_PAW_RT_TC", 8) == 4;
        static const bool rt_apply_mma = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
        paw_timed(stream, std::string("rt_apply") + shp, [&]() {
        if (rt_apply_mma) {
            // experimental tensor-core path -- see paw_rt_apply_kernel_mma
            // comment above. TC fixed at 16 (wmma's fp16 tile shape). m must
            // be a multiple of 16 (same assumption the scalar path already
            // makes via m/16 grid dims).
            paw_launch_rt_apply_mma(ctx, stream, (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        } else if (rt_tc4) {
            paw_launch(paw_rt_apply_kernel<4>,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 3)/4), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        } else {
            paw_launch(paw_rt_apply_kernel<8>,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                (const half *) bank, (const float *) scr_u, scr_v, m, n, nt);
        }
        });
    } else {
        static const bool rt_walk_mma = paw_env_int("GGML_PAW_RT_WALK_MMA", 0) != 0;
        paw_timed(stream, std::string("rt_walk") + shp, [&]() {
        if (rt_walk_mma) {
            paw_launch(paw_rt_walk_kernel_mma,
                ggml_cuda_kernel_launch_params(
                    dim3(m/16, 1, (nt + 15)/16), dim3(128, 1, 1), 0, stream),
                (const uint16_t *) trellis->data, (const half *) tlut->data,
                scr_u, scr_v, m, n, nt);
        } else {
            const int wg = n/16 <= 128 ? 128 : 256;
            auto go = [&](auto WGC, auto KC) {
                paw_launch((paw_rt_walk_kernel<decltype(WGC)::value, decltype(KC)::value>),
                    ggml_cuda_kernel_launch_params(dim3(m/16, 1, nt), dim3(wg, 1, 1), 0, stream),
                    (const uint16_t *) trellis->data, (const half *) tlut->data,
                    scr_u, scr_v, m, n);
            };
            #define PAW_WALK_DISPATCH(WGV)                                        \
                switch (rt_words) {                                               \
                    case 16: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 16>{}); break;        \
                    case 24: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 24>{}); break;        \
                    case 32: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 32>{}); break;        \
                    case 40: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 40>{}); break;        \
                    case 48: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 48>{}); break;        \
                    case 56: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 56>{}); break;        \
                    case 64: go(std::integral_constant<int, WGV>{},               \
                                std::integral_constant<int, 64>{}); break;        \
                    default: GGML_ABORT("paw: unsupported trellis rate, "          \
                                        "%d words per tile", rt_words);           \
                }
            if (wg == 128) { PAW_WALK_DISPATCH(128) } else { PAW_WALK_DISPATCH(256) }
            #undef PAW_WALK_DISPATCH
        }
        });
    }
    }

    if (!uout_noop) paw_timed(stream, std::string("rt_out") + shp, [&]() {
    if (epilogue_dot) {
        GGML_ASSERT(!blocked && m == 2048 && paw_fwht_v2_on() && paw_fwht_v2_ok(m));
        paw_launch((paw_rt_out_epilogue_dot_kernel<128>),
            ggml_cuda_kernel_launch_params(dim3(nt, 1, 1), dim3(128, 1, 1), 0, stream),
            (const float *) sv->data, scr_v, (float *) dst->data,
            (const float *) gate->data, (const float *) gate_x->data,
            (const float *) acc->data, m, (int) gate->ne[0], bm);
    } else
    if (!blocked && paw_fwht_v2_on() && paw_fwht_v2_ok(m)) {
        paw_fwht_for_wg(m/16, [&](auto WG) {
            constexpr int wg = decltype(WG)::value;
            if (epilogue) {
                paw_launch((paw_rt_out_kernel<wg, true>),
                    ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(wg, 1, 1), 0, stream),
                    (const float *) sv->data, scr_v, (float *) dst->data,
                    (const float *) gate->data, (const float *) acc->data, m, bm);
            } else {
                paw_launch((paw_rt_out_kernel<wg, false>),
                    ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(wg, 1, 1), 0, stream),
                    (const float *) sv->data, scr_v, (float *) dst->data, nullptr, nullptr, m, bm);
            }
        });
    } else if (!blocked && paw_fwht_wg512()) {
        if (epilogue) {
            paw_launch((paw_rt_out_kernel<512, true>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(512, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data,
                (const float *) gate->data, (const float *) acc->data, m, bm);
        } else {
            paw_launch((paw_rt_out_kernel<512, false>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(512, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data, nullptr, nullptr, m, bm);
        }
    } else {
        if (epilogue) {
            paw_launch((paw_rt_out_kernel<256, true>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(256, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data,
                (const float *) gate->data, (const float *) acc->data, m, bm);
        } else {
            paw_launch((paw_rt_out_kernel<256, false>),
                ggml_cuda_kernel_launch_params(dim3(nt, m/bm, 1), dim3(256, 1, 1), 0, stream),
                (const float *) sv->data, scr_v, (float *) dst->data, nullptr, nullptr, m, bm);
        }
    }
    });
}

// ---------------------------------------------------------------------------
// batched RT_MM (GGML_OP_PAW_RT_MM_BATCH, GGML_PAW_RT_BATCH=1)
//
// K matrices sharing one input x run through three phase kernels (u, gemv,
// out) launched ONCE for the whole group instead of once per matrix. At
// nt=1 each per-matrix launch is latency/serialization-bound (~10-20 us for
// a few us of work); packing K independent matrices into one launch lets the
// blocks fill the GPU concurrently (measured 2.3x on a 3-matrix m=512 probe:
// 15.2 us serialized -> 6.6 us batched). The graph builder emits this op for
// adjacent same-x groups like (wq,wk,wv) and (wqkv,wqkv_gate); the op's
// output is the K outputs concatenated row-wise and sliced with views, so
// the batch is self-contained (flush-safe: consumers read the group's output
// only after this one op completes).
struct paw_rt_batch_desc {
    const void  * bank;   // cached bank [m, n]: fp16, or e5m2 fp8 (GGML_PAW_BANK_FP8=1)
    const half  * dense_bank;
    const half  * tlut;
    const float * su;     // [n]
    const float * sv;     // [m]
    float       * dst;    // output region (dst + row_off)
    int m, n;
    int u_off;            // offset into the batched scr_u
    int v_off;            // offset into the batched scr_v
    int row_off;          // output row offset
    int map_off;          // first row of an optional grouped-to-tiled segment
    int map_hd, map_k, map_r;
};

// u-phase: one block per (matrix, token); computes u = FWHT(su .* x) / sqrt(n)
// WG is a template knob (matches paw_rt_u_kernel's convention): n is the
// same for every matrix in a batch group (constructor-asserted), so unlike
// the out-kernel there's no per-matrix mismatch to guard against here --
// the call site can pick WG == n/16 for the whole launch whenever v2 applies.
template <int WG>
static __global__ void paw_rt_batch_u_kernel(
        paw_rt_batch_desc * GGML_CUDA_RESTRICT ddesc,
        const paw_rt_batch_desc d0, const paw_rt_batch_desc d1,
        const paw_rt_batch_desc d2, const paw_rt_batch_desc d3,
        const float * GGML_CUDA_RESTRICT x,
        float * GGML_CUDA_RESTRICT scr_u,
        const int nt) {
    const int mat = blockIdx.x;
    const int t   = blockIdx.z;
    const int tid = threadIdx.x;
    const paw_rt_batch_desc desc = mat == 0 ? d0 : mat == 1 ? d1 : mat == 2 ? d2 : d3;
    const int n = desc.n;
    __shared__ float sh[4096];

    if (t == 0 && tid == 0) {
        ddesc[mat] = desc;
    }
    ggml_cuda_pdl_sync();
    const float * su = desc.su;
    for (int i = tid; i < n; i += WG) {
        sh[i] = su[i] * x[(int64_t) t*n + i];
    }
    __syncthreads();
    if (WG == n/16 && paw_fwht_v2_ok(n)) {
        paw_fwht_block_v2(sh, n, tid, WG);
    } else {
        paw_fwht_block(sh, n, tid, WG);
    }
    const float sc     = __fsqrt_rn((float) n);
    const float inv_sc = __frcp_rn(sc);
    float * u = scr_u + desc.u_off + (int64_t) t*n;
    for (int i = tid; i < n; i += WG) {
        u[i] = sh[i] * inv_sc;
    }
}

// gemv phase: one block per 8 rows across the whole group; each block finds
// its matrix by scanning the (small) descriptor array.
static __global__ void paw_rt_batch_gemv_kernel(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v,
        const int nt, const int n_matrices) {
    const int b    = blockIdx.x;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;

    int mat = 0, base = 0;
    while (mat + 1 < n_matrices && b >= base + desc[mat].m/32) {
        base += desc[mat].m/32;
        mat++;
    }
    const int m = desc[mat].m;
    const int n = desc[mat].n;
    const int row = (b - base)*8 + (threadIdx.x >> 5);
    if (row >= m) {
        return;
    }

    const float  * u  = scr_u + desc[mat].u_off + (int64_t) t*n;
    const half2  * W2 = (const half2 *) (desc[mat].dense_bank + (int64_t) row*n);
    const float2 * u2 = (const float2 *) u;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const float2 w = __half22float2(W2[i]);
        const float2 uu = u2[i];
        acc += w.x*uu.x + w.y*uu.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[desc[mat].v_off + (int64_t) t*m + row] = acc;
    }
}

static __global__ void paw_rt_batch_gemv_kernel_idx80(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v,
        const int nt, const int n_matrices) {
    const int b    = blockIdx.x;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    int mat = 0, base = 0;
    while (mat + 1 < n_matrices && b >= base + desc[mat].m/16) {
        base += desc[mat].m/16;
        mat++;
    }

    __shared__ half2 slut[512];
    for (int i = tid; i < 512; i += blockDim.x) {
        slut[i] = ((const half2 *) desc[mat].tlut)[i];
    }
    __syncthreads();

    const int m = desc[mat].m;
    const int n = desc[mat].n;
    const int row = (b - base)*16 + (tid >> 5);
    if (row >= m) {
        return;
    }

    const float    * u = scr_u + desc[mat].u_off + (int64_t) t*n;
    const uint32_t * W = (const uint32_t *) desc[mat].bank + (int64_t) row*(n/32)*5;
    const float2   * u2 = (const float2 *) u;
    const int n2 = n/2;

    float acc = 0.0f;
    for (int i = lane; i < n2; i += 32) {
        const uint16_t code = paw_idx80_get(W, i);
        const int lr = code & 511;
        float2 w = __half22float2(slut[lr]);
        if (code & 512) {
            w.x = -w.x;
        }
        const float2 x = u2[i];
        acc += w.x*x.x + w.y*x.y;
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[desc[mat].v_off + (int64_t) t*m + row] = acc;
    }
}

// fp8 twin of paw_rt_batch_gemv_kernel: bank is e5m2 (1 byte/weight), same
// per-block software LUT + uint32/float4 wide loads as paw_rt_bank_gemv_fp8.
static __global__ void paw_rt_batch_gemv_kernel_fp8(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_u,
        float * GGML_CUDA_RESTRICT scr_v,
        const int nt, const int n_matrices) {
    __shared__ float lut[256];
    const int tid0 = threadIdx.x;
    if (tid0 < 256) {
        lut[tid0] = paw_e5m2_to_f32((uint8_t) tid0);
    }
    __syncthreads();

    const int b    = blockIdx.x;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;

    int mat = 0, base = 0;
    while (mat + 1 < n_matrices && b >= base + desc[mat].m/8) {
        base += desc[mat].m/8;
        mat++;
    }
    const int m = desc[mat].m;
    const int n = desc[mat].n;
    const int row = (b - base)*8 + (threadIdx.x >> 5);
    if (row >= m) {
        return;
    }

    const float   * u = scr_u + desc[mat].u_off + (int64_t) t*n;
    const uint8_t * W = (const uint8_t *) desc[mat].bank + (int64_t) row*n;

    float acc = 0.0f;
    if (n % 4 == 0) {
        const uint32_t * W4 = (const uint32_t *) W;
        const float4   * u4 = (const float4 *) u;
        const int n4 = n/4;
        for (int i = lane; i < n4; i += 128) {
            const uint32_t w = __ldcs(W4 + i);
            const uint8_t * wb = (const uint8_t *) &w;
            const float4 x = u4[i];
            acc += lut[wb[0]]*x.x + lut[wb[1]]*x.y + lut[wb[2]]*x.z + lut[wb[3]]*x.w;
            const int j = i + 32;
            if (j < n4) {
                const uint32_t wj = __ldcs(W4 + j);
                const uint8_t * wjb = (const uint8_t *) &wj;
                const float4 xj = u4[j];
                acc += lut[wjb[0]]*xj.x + lut[wjb[1]]*xj.y + lut[wjb[2]]*xj.z + lut[wjb[3]]*xj.w;
            }
            const int k = i + 64;
            if (k < n4) {
                const uint32_t wk = __ldcs(W4 + k);
                const uint8_t * wkb = (const uint8_t *) &wk;
                const float4 xk = u4[k];
                acc += lut[wkb[0]]*xk.x + lut[wkb[1]]*xk.y + lut[wkb[2]]*xk.z + lut[wkb[3]]*xk.w;
            }
            const int l = i + 96;
            if (l < n4) {
                const uint32_t wl = __ldcs(W4 + l);
                const uint8_t * wlb = (const uint8_t *) &wl;
                const float4 xl = u4[l];
                acc += lut[wlb[0]]*xl.x + lut[wlb[1]]*xl.y + lut[wlb[2]]*xl.z + lut[wlb[3]]*xl.w;
            }
        }
    } else {
        for (int i = lane; i < n; i += 32) {
            acc += lut[W[i]] * u[i];
        }
    }
    acc = warp_reduce_sum<32>(acc);
    if (lane == 0) {
        scr_v[desc[mat].v_off + (int64_t) t*m + row] = acc;
    }
}

// out phase: one block per (matrix, token); FWHT over m, scale by sv
static __global__ void paw_rt_batch_out_kernel(
        const paw_rt_batch_desc * GGML_CUDA_RESTRICT desc,
        const float * GGML_CUDA_RESTRICT scr_v,
        float * GGML_CUDA_RESTRICT dst,
        const int nt, const int m_sum) {
    const int mat = blockIdx.x;
    const int t   = blockIdx.z;
    const int tid = threadIdx.x;
    const int m   = desc[mat].m;
    const int wg  = blockDim.x;   // == m/16, so fwht_block sees nslots <= 16
    __shared__ float sh[8192];

    ggml_cuda_pdl_sync();
    const float * v = scr_v + desc[mat].v_off + (int64_t) t*m;
    for (int i = tid; i < m; i += wg) {
        __pipeline_memcpy_async(&sh[i], &v[i], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // wg is sized for the group's largest matrix (blockDim.x == max_m/16), so
    // v2's wg==m/16 precondition only holds when this matrix's own m equals
    // that max -- must check per-matrix, not assume it like a single-matrix
    // launch could (paw_fwht_block itself is wg-agnostic either way, so v1
    // here is always safe, just not maximally fast).
    if (wg == m/16 && paw_fwht_v2_ok(m)) {
        paw_fwht_block_v2(sh, m, tid, wg);
    } else {
        paw_fwht_block(sh, m, tid, wg);
    }
    const float sc     = __fsqrt_rn((float) m);
    const float inv_sc = __frcp_rn(sc);
    const float * sv = desc[mat].sv;
    // dst is the single [m_sum, T] concatenated output tensor (ggml.c
    // allocates it contiguous with ne[0]=m_sum), so its per-token stride is
    // m_sum, NOT this matrix's own m -- using m here aliased every token
    // after the first for any group with >1 matrix (invisible at nt==1,
    // corrupting every prompt-eval pass since nt there is the prompt length).
    float * y = dst + desc[mat].row_off + (int64_t) t*m_sum;
    for (int i = tid; i < m; i += wg) {
        int oi = i;
        if (i >= desc[mat].map_off && desc[mat].map_hd != 0) {
            const int j = i - desc[mat].map_off;
            const int d = j % desc[mat].map_hd;
            const int kr = j / desc[mat].map_hd;
            const int k = kr / desc[mat].map_r;
            const int vhead = kr % desc[mat].map_r;
            oi = desc[mat].map_off + (vhead*desc[mat].map_k + k)*desc[mat].map_hd + d;
        }
        y[oi] = sh[i] * inv_sc * sv[i];
    }
}

void ggml_cuda_op_paw_rt_mm_batch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!paw_bank_cache_on()) {
        GGML_ASSERT(false);   // batch op requires the bank cache (default on)
    }

    const int n_matrices = dst->op_params[0];
    GGML_ASSERT(n_matrices >= 2 && n_matrices <= 4);
    GGML_ASSERT(dst->op == GGML_OP_PAW_RT_MM_BATCH);
    // supports_op already refuses these, so reaching here means a caller built
    // the op directly; fail loudly rather than rotate with the wrong basis.
    GGML_ASSERT(dst->op_params[GGML_PAW_RHT_BLK_SLOT] == 0 &&
                "paw: the batched RT path has no blocked-rotation kernel yet");

    // tlut is src[3K], x is src[3K+1]
    const ggml_tensor * tlut = dst->src[3*n_matrices];
    const ggml_tensor * x    = dst->src[3*n_matrices + 1];
    const int n = (int) x->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(tlut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    // scratch: u[n_sum] + v[m_sum], per-matrix contiguous
    int n_sum = 0, m_sum = 0;
    for (int i = 0; i < n_matrices; ++i) {
        n_sum += (int) dst->src[3*i + 1]->ne[0];
        m_sum += (int) dst->src[3*i + 2]->ne[0];
    }

    // Persistent device descriptors are published by the u-phase kernel from
    // by-value arguments captured in the graph node.
    static paw_rt_batch_desc * ddesc = nullptr;
    if (ddesc == nullptr) {
        CUDA_CHECK(cudaMalloc(&ddesc, sizeof(paw_rt_batch_desc)*4));
    }

    ggml_cuda_pool_alloc<float> scr(ctx.pool(), (size_t) nt*n_sum + (size_t) nt*m_sum);
    float * scr_u = scr.get();
    float * scr_v = scr_u + (size_t) nt*n_sum;

    // build the descriptor array on the host (static, so the address is
    // stable; the desc-writer kernel copies it by value every call).
    static paw_rt_batch_desc hdesc[4];
    int n_off = 0, m_off = 0, row_off = 0;
    int nblk = 0, nblk16 = 0;
    for (int i = 0; i < n_matrices; ++i) {
        const ggml_tensor * trellis = dst->src[3*i + 0];
        const ggml_tensor * su      = dst->src[3*i + 1];
        const ggml_tensor * sv      = dst->src[3*i + 2];
        const int mi = (int) sv->ne[0];
        const int ni = (int) su->ne[0];
        GGML_ASSERT(ni == n);
        GGML_ASSERT(ni >= 256 && ni % 2 == 0 && mi % 8 == 0);
        hdesc[i].bank    = paw_rt_bank_get(trellis->data, tlut->data, mi, ni, stream);
        hdesc[i].dense_bank = paw_rt_bank_idx_on() ? paw_rt_idx_fp16_bank(hdesc[i].bank, mi, ni) : (const half *) hdesc[i].bank;
        hdesc[i].tlut    = (const half *) tlut->data;
        hdesc[i].su      = (const float *) su->data;
        hdesc[i].sv      = (const float *) sv->data;
        hdesc[i].dst     = (float *) dst->data;
        hdesc[i].m       = mi;
        hdesc[i].n       = ni;
        hdesc[i].u_off   = n_off;
        hdesc[i].v_off   = m_off;
        hdesc[i].row_off = row_off;
        if (dst->op_params[8] != 0) {
            hdesc[i].map_hd  = dst->op_params[9];
            hdesc[i].map_k   = dst->op_params[10];
            hdesc[i].map_r   = dst->op_params[11];
            hdesc[i].map_off = i == 0 ? 2*hdesc[i].map_hd*hdesc[i].map_k : 0;
            GGML_ASSERT(hdesc[i].map_off + hdesc[i].map_hd*hdesc[i].map_k*hdesc[i].map_r == mi);
        } else {
            hdesc[i].map_off = mi;
            hdesc[i].map_hd  = 0;
            hdesc[i].map_k   = 0;
            hdesc[i].map_r   = 0;
        }
        // NOTE: the scratch is [nt, n_sum] / [nt, m_sum]; the kernels index it
        // as [u_off + t*n + i], so per-matrix offsets must stride by nt*n /
        // nt*m (NOT n / m -- that aliases matrix regions when nt > 1).
        n_off  += nt*ni;
        m_off  += nt*mi;
        row_off += mi;
        nblk   += mi/8;
        nblk16 += mi/16;
    }

    char shp[64];
    snprintf(shp, sizeof(shp), " batch=%d nt=%d n=%d", n_matrices, nt, n);

    paw_timed(stream, std::string("rt_batch_u") + shp, [&]() {
    if (paw_fwht_v2_on() && paw_fwht_v2_ok(n)) {
        paw_fwht_for_wg(n/16, [&](auto WGC) {
            constexpr int wg = decltype(WGC)::value;
            paw_launch(paw_rt_batch_u_kernel<wg>,
                ggml_cuda_kernel_launch_params(dim3(n_matrices, 1, nt), dim3(wg, 1, 1), 0, stream),
                ddesc, hdesc[0], hdesc[1], hdesc[2], hdesc[3], (const float *) x->data, scr_u, nt);
        });
    } else {
        paw_launch(paw_rt_batch_u_kernel<256>,
            ggml_cuda_kernel_launch_params(dim3(n_matrices, 1, nt), dim3(256, 1, 1), 0, stream),
            ddesc, hdesc[0], hdesc[1], hdesc[2], hdesc[3], (const float *) x->data, scr_u, nt);
    }
    });

    static const int  dense_min_tok = paw_env_int("GGML_PAW_DENSE_MIN_TOK", 4);
    static const bool rt_tc4        = paw_env_int("GGML_PAW_RT_TC", 8) == 4;
    static const bool rt_apply_mma  = paw_env_int("GGML_PAW_RT_APPLY_MMA", 1) != 0;
    const bool idx = paw_rt_bank_idx_on();
    const bool fp8 = !idx && paw_rt_bank_fp8_on();
    if (idx && nt < dense_min_tok) {
        paw_timed(stream, std::string("rt_batch_gemv") + shp, [&]() {
        paw_launch(paw_rt_batch_gemv_kernel_idx80,
            ggml_cuda_kernel_launch_params(dim3(nblk16, 1, nt), dim3(512, 1, 1), 0, stream),
            ddesc, scr_u, scr_v, nt, n_matrices);
        });
    } else if (fp8) {
        // fp8 bank: the dense-apply kernels (paw_rt_apply_kernel*) read the
        // wrong element size, so one gemv kernel serves every nt (decode and
        // prefill alike), same simplification as the non-batched fp8 path.
        paw_timed(stream, std::string("rt_batch_gemv") + shp, [&]() {
        paw_launch(paw_rt_batch_gemv_kernel_fp8,
            ggml_cuda_kernel_launch_params(dim3(nblk, 1, nt), dim3(256, 1, 1), 0, stream),
            ddesc, scr_u, scr_v, nt, n_matrices);
        });
    } else if (nt >= dense_min_tok) {
        // prompt eval / batched decode: use the same dense apply kernels as
        // the per-matrix path (the gemv path below is a different summation
        // order; switching nt>=4 to it changes prompt-eval rounding and
        // diverges generation).
        for (int i = 0; i < n_matrices; ++i) {
            const int mi = hdesc[i].m;
            const int ni = hdesc[i].n;
            const float * u_i = scr_u + hdesc[i].u_off;
            float * v_i       = scr_v + hdesc[i].v_off;
            paw_timed(stream, std::string("rt_apply") + shp, [&]() {
            if (rt_apply_mma) {
                paw_launch_rt_apply_mma(ctx, stream, hdesc[i].dense_bank, u_i, v_i, mi, ni, nt);
            } else if (rt_tc4) {
                paw_launch(paw_rt_apply_kernel<4>,
                    ggml_cuda_kernel_launch_params(
                        dim3(mi/16, 1, (nt + 3)/4), dim3(128, 1, 1), 0, stream),
                    hdesc[i].dense_bank, u_i, v_i, mi, ni, nt);
            } else {
                paw_launch(paw_rt_apply_kernel<8>,
                    ggml_cuda_kernel_launch_params(
                        dim3(mi/16, 1, (nt + 7)/8), dim3(128, 1, 1), 0, stream),
                    hdesc[i].dense_bank, u_i, v_i, mi, ni, nt);
            }
            });
        }

    } else {
        paw_timed(stream, std::string("rt_batch_gemv") + shp, [&]() {
        paw_launch(paw_rt_batch_gemv_kernel,
            ggml_cuda_kernel_launch_params(dim3(nblk, 1, nt), dim3(256, 1, 1), 0, stream),
            ddesc, scr_u, scr_v, nt, n_matrices);
        });
    }

    // out phase: FWHT(m) + sv-scale + write into dst. Must run regardless of
    // which branch above filled scr_v -- the nt>=dense_min_tok branch used to
    // fall straight through to the function's end without this, leaving dst
    // as stale pool memory for every prompt-eval pass (nt = prompt length,
    // virtually always >= dense_min_tok=4). That corrupted the hidden state
    // from the very first token of every request while remaining invisible
    // to nt==1 decode-only correctness checks, which never took this branch.
    paw_timed(stream, std::string("rt_batch_out") + shp, [&]() {
    int max_m = 0;
    for (int i = 0; i < n_matrices; ++i) {
        max_m = std::max(max_m, hdesc[i].m);
    }
    const int out_wg = max_m/16;
    paw_launch(paw_rt_batch_out_kernel,
        ggml_cuda_kernel_launch_params(dim3(n_matrices, 1, nt), dim3(out_wg, 1, 1), 0, stream),
        ddesc, scr_v, (float *) dst->data, nt, m_sum);
    });
}


//
// EXP_BASIS — one block per (slot, token) pair (paw_exp_basis.comp)
//

static __global__ void paw_exp_basis_kernel(
        const half    * GGML_CUDA_RESTRICT a,
        const half    * GGML_CUDA_RESTRICT b,
        const half    * GGML_CUDA_RESTRICT c,
        const int32_t * GGML_CUDA_RESTRICT remap,
        const int32_t * GGML_CUDA_RESTRICT ids,
        const float   * GGML_CUDA_RESTRICT x,
        const float   * GGML_CUDA_RESTRICT acc_in,   // nullptr when has_acc == 0
        float         * GGML_CUDA_RESTRICT dst,
        const int n, const int r, const int m, const int n_used, const int xne1,
        const int ids_s0, const int ids_s1, const int has_acc) {
    constexpr int WG = 256;
    __shared__ float tv[256];

    const int s   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const uint32_t id = (uint32_t) ids[s*ids_s0 + t*ids_s1];
    const uint32_t rm = (uint32_t) remap[id];

    const int64_t obase = (int64_t) t*n_used*m + (int64_t) s*m;
    if ((rm & GGML_CUDA_PAW_DEM_FLAG) == 0u) {   // kept slot: block-uniform
        for (int i = tid; i < m; i += WG) {
            dst[obase + i] = has_acc ? acc_in[obase + i] : 0.0f;
        }
        return;
    }
    const int di = (int)(rm & ~GGML_CUDA_PAW_DEM_FLAG);
    const int64_t xbase = (int64_t)(xne1 == 1 ? 0 : s*n) + (int64_t) t*xne1*n;

    // t_j = c_j * (A_j . x)
    for (int j = tid; j < r; j += WG) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) {
            acc += __half2float(a[(int64_t) j*n + i]) * x[xbase + i];
        }
        // one rounding: t_j = fl(c_j * v_j)
        tv[j] = __half2float(c[(int64_t) di*r + j]) * acc;
    }
    __syncthreads();
    for (int i = tid; i < m; i += WG) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
            // reference rounds B_ij*t_j before the add — pinned, no fma
            const float pr = __fmul_rn(__half2float(b[(int64_t) i*r + j]), tv[j]);
            acc = __fadd_rn(acc, pr);
        }
        dst[obase + i] = has_acc ? acc_in[obase + i] + acc : acc;
    }
}

