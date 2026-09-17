// PAW_X3_MM_ID gate: routed-expert x3 matmul with per-expert rates against
// per-(token, slot) ggml_paw_x3_mm calls on views of the same weights.
//
// The trellis words are random: every bit pattern is a valid mul1 code, so no
// encoder is needed. Routing is skewed so single experts see 1..n_tokens rows
// and every dense x3 path (sq GEMV, tile GEMM, reconstruct) is exercised.
// The reference runs one row at a time, so it can take a different dense path
// than the batched op (sq GEMV vs GEMM); outputs are compared at a tolerance,
// not bitwise. Absolute accuracy of each path is checked separately with
// GGML_PAW_X3_DUMP + bonsai-pilot scripts/exl3_parity/phase4_verify_dump.py.
//
// usage: test-paw-x3-mm-id [n_tokens ...]    (CUDA device 0)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct shape { int64_t n, m; };

static int run_case(ggml_backend_t backend, shape sh, int64_t n_tokens, bool broadcast_x, std::mt19937 & rng) {
    const int64_t n_expert = 12;
    const int64_t n_used   = 4;
    const int64_t ntiles   = (sh.n / 16) * (sh.m / 16);

    std::vector<int32_t> meta(2 * n_expert);
    int64_t words = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        meta[2 * e]     = 1 + (int32_t) (e % 4);
        meta[2 * e + 1] = (int32_t) words;
        words += 16 * meta[2 * e] * ntiles;
    }

    // expert 0 takes slot 0 of every token (row count = n_tokens), the rest
    // are spread so some experts get a handful of rows
    std::vector<int32_t> ids(n_used * n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        ids[t * n_used] = 0;
        for (int64_t s = 1; s < n_used; ++s) {
            ids[t * n_used + s] = (int32_t) (1 + (t * 7 + s * 3 + (rng() % 2)) % (n_expert - 1));
        }
        // no expert repeats within a token
        for (int64_t s = 2; s < n_used; ++s) {
            for (int64_t r = 1; r < s; ++r) {
                if (ids[t * n_used + s] == ids[t * n_used + r]) {
                    ids[t * n_used + s] = (int32_t) (1 + ids[t * n_used + s] % (n_expert - 1));
                    r = 0;
                }
            }
        }
    }

    const int64_t x_slots = broadcast_x ? 1 : n_used;
    const size_t n_nodes = 4 + n_tokens * n_used * 8;
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (16 + n_nodes) + ggml_graph_overhead_custom(n_nodes, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * t_tr   = ggml_new_tensor_1d(ctx, GGML_TYPE_I16, words);
    ggml_tensor * t_meta = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_expert);
    ggml_tensor * t_suh  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, sh.n, n_expert);
    ggml_tensor * t_svh  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, sh.m, n_expert);
    ggml_tensor * t_ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * t_x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.n, x_slots, n_tokens);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);

    ggml_tensor * y = ggml_paw_x3_mm_id(ctx, t_tr, t_meta, t_suh, t_svh, t_ids, t_x);
    ggml_build_forward_expand(gf, y);

    std::vector<ggml_tensor *> refs(n_tokens * n_used);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t s = 0; s < n_used; ++s) {
            const int32_t e = ids[t * n_used + s];
            const int64_t k = meta[2 * e];
            ggml_tensor * w  = ggml_view_2d(ctx, t_tr, 16 * k, ntiles, 16 * k * t_tr->nb[0], (size_t) meta[2 * e + 1] * t_tr->nb[0]);
            ggml_tensor * su = ggml_view_1d(ctx, t_suh, sh.n, e * t_suh->nb[1]);
            ggml_tensor * sv = ggml_view_1d(ctx, t_svh, sh.m, e * t_svh->nb[1]);
            ggml_tensor * xr = ggml_view_1d(ctx, t_x, sh.n, t * t_x->nb[2] + (s % x_slots) * t_x->nb[1]);
            refs[t * n_used + s] = ggml_paw_x3_mm(ctx, w, su, sv, xr);
            ggml_build_forward_expand(gf, refs[t * n_used + s]);
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "buffer allocation failed\n");
        return 1;
    }

    {
        std::vector<int16_t> v(words);
        for (auto & w : v) w = (int16_t) (rng() & 0xffff);
        ggml_backend_tensor_set(t_tr, v.data(), 0, ggml_nbytes(t_tr));
    }
    ggml_backend_tensor_set(t_meta, meta.data(), 0, ggml_nbytes(t_meta));
    for (ggml_tensor * t_s : { t_suh, t_svh }) {
        // sign vectors scaled like real sidecars (|suh| ~ O(1e-2..1))
        std::vector<ggml_fp16_t> v(ggml_nelements(t_s));
        std::uniform_real_distribution<float> mag(0.25f, 1.0f);
        for (auto & h : v) h = ggml_fp32_to_fp16(((rng() & 1) ? 1.0f : -1.0f) * mag(rng));
        ggml_backend_tensor_set(t_s, v.data(), 0, ggml_nbytes(t_s));
    }
    ggml_backend_tensor_set(t_ids, ids.data(), 0, ggml_nbytes(t_ids));
    {
        std::vector<float> v(ggml_nelements(t_x));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & f : v) f = nd(rng);
        ggml_backend_tensor_set(t_x, v.data(), 0, ggml_nbytes(t_x));
    }

    // the allocator may overwrite inputs it considers intermediate; they are
    // leaves here, so compute once and compare
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> yv(ggml_nelements(y));
    ggml_backend_tensor_get(y, yv.data(), 0, ggml_nbytes(y));

    double worst_rel = 0.0;
    double worst_cos = 1.0;
    int64_t worst_e = -1;
    std::vector<float> rv(sh.m);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t s = 0; s < n_used; ++s) {
            ggml_backend_tensor_get(refs[t * n_used + s], rv.data(), 0, sh.m * sizeof(float));
            const float * a = yv.data() + (t * n_used + s) * sh.m;
            double dd = 0.0, rr = 0.0, aa = 0.0, ar = 0.0;
            for (int64_t i = 0; i < sh.m; ++i) {
                const double d = (double) a[i] - rv[i];
                dd += d * d;
                rr += (double) rv[i] * rv[i];
                aa += (double) a[i] * a[i];
                ar += (double) a[i] * rv[i];
            }
            const double rel = std::sqrt(dd / std::max(rr, 1e-30));
            const double cos = ar / std::sqrt(std::max(aa * rr, 1e-30));
            if (!std::isfinite(rel) || rel > worst_rel) {
                worst_rel = rel;
                worst_e   = ids[t * n_used + s];
            }
            worst_cos = std::min(worst_cos, cos);
        }
    }

    // both sides are within the 0.009 exl3 parity gate of the exact product, but
    // on different paths (sq GEMV vs GEMM), so path-vs-path allows twice that
    const bool ok = std::isfinite(worst_rel) && worst_rel <= 0.018 && worst_cos >= 0.9998;
    printf("n=%-5lld m=%-5lld tokens=%-4lld x_slots=%lld  worst rel_rms=%.5f (expert %lld, K=%d) worst cos=%.8f  %s\n",
           (long long) sh.n, (long long) sh.m, (long long) n_tokens, (long long) x_slots,
           worst_rel, (long long) worst_e, worst_e >= 0 ? meta[2 * worst_e] : 0, worst_cos, ok ? "PASS" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::vector<int64_t> token_counts;
    for (int i = 1; i < argc; ++i) {
        token_counts.push_back(atoll(argv[i]));
    }
    if (token_counts.empty()) {
        // expert 0 row counts: sq GEMV (1, 6), tile GEMM (40), reconstruct (200)
        token_counts = { 1, 6, 40, 200 };
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "no CUDA device\n");
        return 2;
    }

    std::mt19937 rng(1234);
    int fails = 0;
    for (shape sh : { shape{ 2560, 640 }, shape{ 640, 2560 } }) {
        for (int64_t nt : token_counts) {
            fails += run_case(backend, sh, nt, false, rng);
            fails += run_case(backend, sh, nt, true, rng);
        }
    }

    ggml_backend_free(backend);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
