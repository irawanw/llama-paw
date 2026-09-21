// PAW_X3_MOE gate: fused routed-expert SwiGLU FFN with per-expert rates against the unfused graph
//   paw_x3_mm_id(gate), paw_x3_mm_id(up) -> swiglu_split -> paw_x3_mm_id(down) -> paw_moe_reduce
// on the same random weights (every trellis bit pattern is a valid mul1 code).
//
// Routing is skewed so expert 0 takes one slot of every token: with enough tokens it exceeds the
// fused row limit (128) and runs the per-matrix fallback, while the other experts stay fused.
// Run once more with GGML_PAW_X3_MOE_FUSED_ROWS=0 to push every expert through the fallback.
// The fused kernel keeps fp16 intermediates, so outputs are compared at a tolerance.
//
// usage: test-paw-x3-moe [n_tokens ...]    (CUDA device 0)
//
// Set X3MOE_BENCH_REPS=N to also time the fused op and the unfused chain
// separately and report effective trellis bandwidth. Use X3MOE_TEST_EXPERTS=512
// for the real per-layer shape (0.762 GiB of trellis, fits in <1 GiB of VRAM).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

static int64_t env_int(const char * name, int64_t def) {
    const char * e = getenv(name);
    return e ? atoll(e) : def;
}

static int run_case(ggml_backend_t backend, int64_t n_tokens, std::mt19937 & rng) {
    const int64_t n_embd   = 2560;
    const int64_t n_ff     = 640;
    const int64_t n_expert = env_int("X3MOE_TEST_EXPERTS", 24);
    const int64_t n_used   = env_int("X3MOE_TEST_USED", 10);

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 64 + ggml_graph_overhead() * 3,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    // per projection: in, out, per-expert rate
    struct proj_host { int64_t in, out; std::vector<int32_t> meta; int64_t words; };
    proj_host ph[3];
    ggml_tensor * proj[12];
    for (int p = 0; p < 3; ++p) {
        ph[p].in  = p == 2 ? n_ff : n_embd;
        ph[p].out = p == 2 ? n_embd : n_ff;
        ph[p].meta.resize(2 * n_expert);
        const int64_t ntiles = (ph[p].in / 16) * (ph[p].out / 16);
        int64_t words = 0;
        const int64_t kfix = env_int("X3MOE_TEST_KFIX", 0);
        for (int64_t e = 0; e < n_expert; ++e) {
            ph[p].meta[2 * e]     = kfix ? (int32_t) kfix : 1 + (int32_t) ((e + p) % 4);
            ph[p].meta[2 * e + 1] = (int32_t) words;
            words += 16 * ph[p].meta[2 * e] * ntiles;
        }
        ph[p].words = words;
        proj[4 * p + 0] = ggml_new_tensor_1d(ctx, GGML_TYPE_I16, words);
        proj[4 * p + 1] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_expert);
        proj[4 * p + 2] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ph[p].in, n_expert);
        proj[4 * p + 3] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ph[p].out, n_expert);
    }

    ggml_tensor * t_x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * t_w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tokens);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * fused = ggml_paw_x3_moe(ctx, t_x, t_ids, t_w, proj);
    ggml_build_forward_expand(gf, fused);

    ggml_tensor * x3d  = ggml_reshape_3d(ctx, t_x, n_embd, 1, n_tokens);
    ggml_tensor * gate = ggml_paw_x3_mm_id(ctx, proj[0], proj[1], proj[2], proj[3], t_ids, x3d);
    ggml_tensor * up   = ggml_paw_x3_mm_id(ctx, proj[4], proj[5], proj[6], proj[7], t_ids, x3d);
    ggml_tensor * par  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_paw_x3_mm_id(ctx, proj[8], proj[9], proj[10], proj[11], t_ids, par);
    ggml_tensor * ref  = ggml_paw_moe_reduce(ctx, down, t_w);
    ggml_build_forward_expand(gf, ref);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "buffer allocation failed\n");
        return 1;
    }

    std::uniform_real_distribution<float> mag(0.25f, 1.0f);
    for (int p = 0; p < 3; ++p) {
        std::vector<int16_t> tw(ph[p].words);
        for (auto & w : tw) w = (int16_t) (rng() & 0xffff);
        ggml_backend_tensor_set(proj[4 * p + 0], tw.data(), 0, ggml_nbytes(proj[4 * p + 0]));
        ggml_backend_tensor_set(proj[4 * p + 1], ph[p].meta.data(), 0, ggml_nbytes(proj[4 * p + 1]));
        for (int s = 2; s < 4; ++s) {
            ggml_tensor * t_s = proj[4 * p + s];
            // real sidecars carry the weight scale (|suh*svh| ~ 1e-3 for rms 0.02 weights); unit-scale
            // sidecars overflow the fp16 intermediates of the fused path, which a real encode never produces
            const float scale = (float) env_int("X3MOE_TEST_SCALE_MILLI", 50) * 1e-3f;
            std::vector<ggml_fp16_t> v(ggml_nelements(t_s));
            for (auto & h : v) h = ggml_fp32_to_fp16(((rng() & 1) ? scale : -scale) * mag(rng));
            ggml_backend_tensor_set(t_s, v.data(), 0, ggml_nbytes(t_s));
        }
    }

    std::vector<int32_t> ids(n_used * n_tokens);
    std::vector<float> w(n_used * n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        std::vector<int32_t> pick(n_expert - 1);
        for (int64_t e = 0; e < n_expert - 1; ++e) pick[e] = (int32_t) (e + 1);
        std::shuffle(pick.begin(), pick.end(), rng);
        ids[t * n_used] = 0;
        float sum = 0.0f;
        for (int64_t s = 0; s < n_used; ++s) {
            if (s > 0) ids[t * n_used + s] = pick[s - 1];
            w[t * n_used + s] = mag(rng);
            sum += w[t * n_used + s];
        }
        for (int64_t s = 0; s < n_used; ++s) w[t * n_used + s] /= sum;
    }
    ggml_backend_tensor_set(t_ids, ids.data(), 0, ggml_nbytes(t_ids));
    ggml_backend_tensor_set(t_w, w.data(), 0, ggml_nbytes(t_w));
    {
        std::vector<float> v(ggml_nelements(t_x));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & f : v) f = nd(rng);
        ggml_backend_tensor_set(t_x, v.data(), 0, ggml_nbytes(t_x));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> a(ggml_nelements(fused)), b(ggml_nelements(ref));
    ggml_backend_tensor_get(fused, a.data(), 0, ggml_nbytes(fused));
    ggml_backend_tensor_get(ref, b.data(), 0, ggml_nbytes(ref));

    double worst_rel = 0.0, worst_cos = 1.0, ref_rms = 0.0;
    int64_t nonfinite_a = 0, nonfinite_b = 0;
    for (int64_t t = 0; t < n_tokens; ++t) {
        double dd = 0.0, rr = 0.0, aa = 0.0, ar = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) {
            const double av = a[t * n_embd + i], bv = b[t * n_embd + i];
            nonfinite_a += !std::isfinite(av);
            nonfinite_b += !std::isfinite(bv);
            dd += (av - bv) * (av - bv);
            rr += bv * bv;
            aa += av * av;
            ar += av * bv;
        }
        worst_rel = std::max(worst_rel, std::sqrt(dd / std::max(rr, 1e-30)));
        worst_cos = std::min(worst_cos, ar / std::sqrt(std::max(aa * rr, 1e-30)));
        ref_rms += rr;
    }
    ref_rms = std::sqrt(ref_rms / (n_tokens * n_embd));

    // the fused path runs the down projection on the GEMM, the reference may run it on the int8
    // GEMV; both are within the 0.009 parity gate of the exact product, plus fp16 intermediates
    const bool ok = !nonfinite_a && !nonfinite_b && worst_rel <= 0.02 && worst_cos >= 0.9998;
    printf("tokens=%-4lld used=%lld ref_rms=%-9.4g worst rel_rms=%.5f worst cos=%.8f nonfinite fused/ref=%lld/%lld  %s\n",
           (long long) n_tokens, (long long) n_used, ref_rms, worst_rel, worst_cos,
           (long long) nonfinite_a, (long long) nonfinite_b, ok ? "PASS" : "FAIL");

    // Opt-in timing mode. The correctness graph above holds the fused op and the
    // reference chain together, so neither can be timed from it; build a graph per
    // variant over the same already-allocated tensors and time them separately.
    const int64_t reps = env_int("X3MOE_BENCH_REPS", 0);
    if (reps > 0) {
        ggml_cgraph * gf_fused = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_fused, fused);
        ggml_cgraph * gf_ref = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_ref, ref);

        auto time_graph = [&](ggml_cgraph * g) {
            ggml_backend_graph_compute(backend, g);   // warmup: JIT, caches, clocks
            ggml_backend_synchronize(backend);
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t i = 0; i < reps; ++i) ggml_backend_graph_compute(backend, g);
            ggml_backend_synchronize(backend);
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        };

        // Trellis bytes the op must read: each distinct routed expert once, at that
        // expert's own rate K (rates are mixed per expert). This is the quantity the
        // decode path is bandwidth-bound on.
        std::vector<char> seen(n_expert, 0);
        int64_t uniq = 0, tbytes = 0;
        for (int64_t i = 0; i < n_used * n_tokens; ++i) {
            const int32_t e = ids[i];
            if (seen[e]) continue;
            seen[e] = 1; ++uniq;
            for (int p = 0; p < 3; ++p) tbytes += ph[p].in * ph[p].out * ph[p].meta[2 * e] / 16 * 2;
        }

        const double ms_f = time_graph(gf_fused);
        const double ms_r = time_graph(gf_ref);
        printf("  bench reps=%lld experts=%lld uniq_routed=%lld trellis=%.3f GiB | "
               "fused %.3f ms (%.1f GiB/s) | unfused %.3f ms | fused is %.2fx\n",
               (long long) reps, (long long) n_expert, (long long) uniq,
               tbytes / 1073741824.0, ms_f, tbytes / 1073741824.0 / (ms_f / 1e3),
               ms_r, ms_r / ms_f);
    }

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
        // expert 0 rows: fused (1, 3, 17, 128), fallback (129, 300)
        token_counts = { 1, 3, 17, 128, 129, 300 };
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "no CUDA device\n");
        return 2;
    }

    std::mt19937 rng(4321);
    int fails = 0;
    for (int64_t nt : token_counts) {
        fails += run_case(backend, nt, rng);
    }

    ggml_backend_free(backend);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
