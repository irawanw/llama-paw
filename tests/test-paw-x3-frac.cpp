// PAW_X3_MM rate gate: ggml_paw_x3_mm against an independent host reference, for the integer
// rates K = 2, 3, 4 and the half-integer rate K = 3.5 (exllamav3 fractional trellis, KA = 3,
// MASK 0xAAAA: weight i takes 3 + (i & 1) fresh bits, so a 16x16 tile is 56 uint16).
//
// The host reference decodes every tile from the bit-stream definition shared by all rates
// (window i of a tile is the 16 ring bits ending at S(i) = sum_{j <= i} D(j), MSB-first 32-bit
// words; exllamav3 quant/pack.cu and quant/frac.cu), the mul1 codebook, the tensor-core tile
// permutation and W = diag(suh) H128 W_hat H128 diag(svh), then computes y = x W in double.
// The integer rates validate the reference against the existing kernels; K = 3.5 then checks the
// fractional decoder against the same reference. Random trellis words are valid mul1 codes.
//
// usage: test-paw-x3-frac [nt ...]    (CUDA device 0)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// rate as 2K (4, 6, 7, 8) -> uint16 words per tile = 8 * 2K
static int tile_u16(int k2) { return 8 * k2; }

static int bits_of(int k2, int i) {   // D(i): fresh bits of weight i
    return k2 % 2 ? k2 / 2 + (i & 1) : k2 / 2;
}

static float decode_mul1(uint32_t idx) {
    const uint32_t x = idx * 0x83DCD12Du;
    const uint32_t sum = 0x6400u + (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    const float h = ggml_fp16_to_fp32((ggml_fp16_t) (sum & 0xffff));
    const float k_inv = ggml_fp16_to_fp32((ggml_fp16_t) 0x1eee);
    const float k_bias = ggml_fp16_to_fp32((ggml_fp16_t) 0xc931);
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(h * k_inv + k_bias));
}

// W_hat (n x m, transformed basis) from the trellis
static void decode_what(const std::vector<uint16_t> & tr, int k2, int n, int m, std::vector<double> & W) {
    const int tw = tile_u16(k2);
    const int ring = tw * 16;
    W.assign((size_t) n * m, 0.0);
    for (int kb = 0; kb < n / 16; ++kb) {
        for (int nb = 0; nb < m / 16; ++nb) {
            const uint16_t * t16 = tr.data() + ((size_t) kb * (m / 16) + nb) * tw;
            const uint32_t * w32 = (const uint32_t *) t16;
            auto bit = [&] (int p) { p = ((p % ring) + ring) % ring; return (w32[p >> 5] >> (31 - (p & 31))) & 1u; };
            int s = 0;
            for (int i = 0; i < 256; ++i) {
                s += bits_of(k2, i);
                uint32_t v = 0;
                for (int p = s - 16; p < s; ++p) v = (v << 1) | bit(p);
                const int t = i >> 3, j = i & 7;
                const int r = (t & 3) * 2 + ((j & 1) ? 1 : 0) + ((j & 2) ? 8 : 0);
                const int c = (t >> 2) + ((j & 4) ? 8 : 0);
                W[(size_t) (kb * 16 + r) * m + nb * 16 + c] = decode_mul1(v);
            }
        }
    }
}

static void had128_rows(std::vector<double> & W, int n, int m) {   // H128 on the row index, per 128 block
    std::vector<double> col(128);
    for (int b = 0; b < n / 128; ++b)
        for (int c = 0; c < m; ++c) {
            for (int i = 0; i < 128; ++i) col[i] = W[(size_t) (b * 128 + i) * m + c];
            for (int h = 1; h < 128; h <<= 1)
                for (int i = 0; i < 128; i += 2 * h)
                    for (int j = i; j < i + h; ++j) { double a = col[j], d = col[j + h]; col[j] = a + d; col[j + h] = a - d; }
            for (int i = 0; i < 128; ++i) W[(size_t) (b * 128 + i) * m + c] = col[i] / std::sqrt(128.0);
        }
}

static void had128_cols(std::vector<double> & W, int n, int m) {
    for (int r = 0; r < n; ++r)
        for (int b = 0; b < m / 128; ++b) {
            double * row = W.data() + (size_t) r * m + b * 128;
            for (int h = 1; h < 128; h <<= 1)
                for (int i = 0; i < 128; i += 2 * h)
                    for (int j = i; j < i + h; ++j) { double a = row[j], d = row[j + h]; row[j] = a + d; row[j + h] = a - d; }
            for (int i = 0; i < 128; ++i) row[i] /= std::sqrt(128.0);
        }
}

static int run_case(ggml_backend_t backend, int k2, int n, int m, int nt, std::mt19937 & rng) {
    const int tw = tile_u16(k2);
    const int64_t ntiles = (int64_t) (n / 16) * (m / 16);
    std::vector<uint16_t> tr((size_t) ntiles * tw);
    for (auto & w : tr) w = (uint16_t) (rng() & 0xffff);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    std::vector<ggml_fp16_t> suh(n), svh(m);
    for (auto & s : suh) s = ggml_fp32_to_fp16((rng() & 1 ? 1.0f : -1.0f) * ud(rng) * 0.02f);
    for (auto & s : svh) s = ggml_fp32_to_fp16((rng() & 1 ? 1.0f : -1.0f) * ud(rng));
    std::vector<float> x((size_t) n * nt);
    for (auto & v : x) v = nd(rng);

    // host reference
    std::vector<double> W;
    decode_what(tr, k2, n, m, W);
    had128_rows(W, n, m);
    had128_cols(W, n, m);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < m; ++c)
            W[(size_t) r * m + c] *= (double) ggml_fp16_to_fp32(suh[r]) * ggml_fp16_to_fp32(svh[c]);
    std::vector<double> yref((size_t) m * nt, 0.0);
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < n; ++r) {
            const double xv = x[(size_t) t * n + r];
            const double * wr = W.data() + (size_t) r * m;
            double * yr = yref.data() + (size_t) t * m;
            for (int c = 0; c < m; ++c) yr[c] += xv * wr[c];
        }

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_tr  = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, tw, ntiles);
    ggml_tensor * t_suh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    ggml_tensor * t_svh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, m);
    ggml_tensor * t_x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, nt);
    ggml_tensor * y = ggml_paw_x3_mm(ctx, t_tr, t_suh, t_svh, t_x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t_tr, tr.data(), 0, tr.size() * 2);
    ggml_backend_tensor_set(t_suh, suh.data(), 0, suh.size() * 2);
    ggml_backend_tensor_set(t_svh, svh.data(), 0, svh.size() * 2);
    ggml_backend_tensor_set(t_x, x.data(), 0, x.size() * 4);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }
    std::vector<float> yo((size_t) m * nt);
    ggml_backend_tensor_get(y, yo.data(), 0, yo.size() * 4);

    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < yo.size(); ++i) { const double d = yo[i] - yref[i]; se += d * d; sr += yref[i] * yref[i]; }
    const double rel = std::sqrt(se / sr);
    // fp16 weights + fp16/int8 activations: every correct path lands well under 2e-2; a wrong
    // window or permutation decodes unrelated codebook values (rel ~ 1.4)
    const bool ok = std::isfinite(rel) && rel < 2e-2;
    printf("K=%-4g n=%-5d m=%-5d nt=%-5d rel_rms=%.3e %s\n", k2 / 2.0, n, m, nt, rel, ok ? "ok" : "FAIL");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::vector<int> nts;
    for (int i = 1; i < argc; ++i) nts.push_back(atoi(argv[i]));
    if (nts.empty()) nts = { 1, 2, 4, 8, 16, 64, 200, 512, 1100 };
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "no CUDA backend\n"); return 1; }
    std::mt19937 rng(20260923);
    int fails = 0;
    const int shapes[][2] = { { 512, 512 }, { 1024, 256 }, { 256, 1280 } };
    for (int k2 : { 4, 6, 8, 7 })
        for (auto & sh : shapes)
            for (int nt : nts)
                fails += run_case(backend, k2, sh[0], sh[1], nt, rng);
    printf("%s (%d failures)\n", fails ? "X3-FRAC-FAIL" : "X3-FRAC-PASS", fails);
    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
