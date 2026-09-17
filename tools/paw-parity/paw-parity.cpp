// Common cross-engine batch-1 AR protocol driver for PAW.
//
// Phase 0 of reports/paw_exl3_ar_parity_plan_20260902.md: one shared
// protocol that runs the same boundaries on both engines (this binary for
// llama.cpp/PAW, scripts/exl3_parity/engine_exl3.py for exllamav3).
//
// Protocol:
//   - explicit prompt token IDs (no tokenizer dependence)
//   - warmup: prefill + n_warmup greedy decode steps, discarded, memory reset
//   - timed region: prefill (outside timing) + n_timed greedy decode steps,
//     one device synchronization per step (mirrors the EXL3 native harness,
//     which forces a sync per step via .cpu() on the sampled token)
//   - deterministic next-token policy: argmax over logits
//
// Results are emitted as a single JSON object on the line prefixed
// "PARITY_JSON: " and optionally written to --parity-out.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct parity_params {
    std::vector<llama_token> prompt_tokens;
    int n_warmup = 16;
    int n_timed  = 128;
    std::string out_path;
};

static bool parse_parity_args(int & argc, char ** argv, parity_params & pparams) {
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val_of = [&]() -> const char * {
            const auto eq = arg.find('=');
            if (eq != std::string::npos) {
                return argv[i] + eq + 1;
            }
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", argv[i]);
                exit(1);
            }
            return argv[++i];
        };

        auto starts_with = [&](const char * prefix) {
            return arg.rfind(prefix, 0) == 0 && (arg.size() == strlen(prefix) || arg[strlen(prefix)] == '=');
        };

        if (starts_with("--parity-prompt-tokens")) {
            const char * v = val_of();
            pparams.prompt_tokens.clear();
            const char * s = v;
            while (*s) {
                char * end = nullptr;
                long id = strtol(s, &end, 10);
                if (end == s) {
                    LOG_ERR("bad --parity-prompt-tokens near '%s'\n", s);
                    exit(1);
                }
                pparams.prompt_tokens.push_back((llama_token) id);
                s = end;
                while (*s == ',') ++s;
            }
            continue;
        }
        if (starts_with("--parity-n-warmup")) { pparams.n_warmup = atoi(val_of()); continue; }
        if (starts_with("--parity-n-timed"))  { pparams.n_timed  = atoi(val_of()); continue; }
        if (starts_with("--parity-out"))      { pparams.out_path = val_of();     continue; }

        // keep non-parity args
        rest.push_back(argv[i]);
    }

    if (pparams.prompt_tokens.empty()) {
        LOG_ERR("--parity-prompt-tokens is required\n");
        return false;
    }
    if (pparams.n_timed <= 0 || pparams.n_warmup < 0) {
        LOG_ERR("--parity-n-timed must be > 0 and --parity-n-warmup >= 0\n");
        return false;
    }

    argc = (int) rest.size();
    for (int i = 0; i < argc; ++i) {
        argv[i] = rest[i];
    }
    argv[argc] = nullptr;
    return true;
}

static int argmax_logits(const float * logits, int32_t n) {
    int best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

static void emit_json(const parity_params & pparams,
                      const std::string & model_path,
                      int n_ctx, int n_vocab,
                      double timed_s, double warmup_s,
                      const std::vector<double> & lat_ms,
                      const std::vector<llama_token> & timed_tokens,
                      const char * out_path) {
    const double tok_s = timed_s > 0.0 ? (double) pparams.n_timed / timed_s : 0.0;
    const double ms_per_token = timed_s > 0.0 ? timed_s * 1000.0 / (double) pparams.n_timed : 0.0;

    printf("PARITY_JSON: {\"engine\": \"paw\", \"protocol\": \"exl3_parity_v1\", "
           "\"model\": \"%s\", \"n_ctx\": %d, \"n_vocab\": %d, "
           "\"n_prompt_tokens\": %zu, \"n_warmup\": %d, \"n_timed\": %d, "
           "\"warmup_s\": %.6f, \"timed_s\": %.6f, \"tok_s\": %.4f, \"ms_per_token\": %.6f, ",
           model_path.c_str(), n_ctx, n_vocab,
           pparams.prompt_tokens.size(), pparams.n_warmup, pparams.n_timed,
           warmup_s, timed_s, tok_s, ms_per_token);

    printf("\"latencies_ms\": [");
    for (size_t i = 0; i < lat_ms.size(); ++i) {
        printf("%.4f%s", lat_ms[i], i + 1 < lat_ms.size() ? ", " : "");
    }
    printf("], \"generated_token_ids\": [");
    for (size_t i = 0; i < timed_tokens.size(); ++i) {
        printf("%d%s", (int) timed_tokens[i], i + 1 < timed_tokens.size() ? ", " : "");
    }
    printf("]}\n");
    fflush(stdout);

    if (out_path && out_path[0]) {
        // Re-print into file (simplest reliable path: reopen stdout buffer)
        FILE * f = fopen(out_path, "w");
        if (!f) {
            LOG_ERR("failed to open --parity-out %s\n", out_path);
            return;
        }
        fprintf(f, "{\"engine\": \"paw\", \"protocol\": \"exl3_parity_v1\", "
                   "\"model\": \"%s\", \"n_ctx\": %d, \"n_vocab\": %d, "
                   "\"n_prompt_tokens\": %zu, \"n_warmup\": %d, \"n_timed\": %d, "
                   "\"warmup_s\": %.6f, \"timed_s\": %.6f, \"tok_s\": %.4f, \"ms_per_token\": %.6f, ",
                model_path.c_str(), n_ctx, n_vocab,
                pparams.prompt_tokens.size(), pparams.n_warmup, pparams.n_timed,
                warmup_s, timed_s, tok_s, ms_per_token);
        fprintf(f, "\"latencies_ms\": [");
        for (size_t i = 0; i < lat_ms.size(); ++i) {
            fprintf(f, "%.4f%s", lat_ms[i], i + 1 < lat_ms.size() ? ", " : "");
        }
        fprintf(f, "], \"generated_token_ids\": [");
        for (size_t i = 0; i < timed_tokens.size(); ++i) {
            fprintf(f, "%d%s", (int) timed_tokens[i], i + 1 < timed_tokens.size() ? ", " : "");
        }
        fprintf(f, "]}\n");
        fclose(f);
    }
}

static void print_usage(int, char ** argv) {
    LOG("\nusage:\n");
    LOG("\n    %s -m model.gguf -ngl 99 -fa 1 -c 1024 \\\n", argv[0]);
    LOG("        --parity-prompt-tokens 1,2,3 --parity-n-warmup 16 --parity-n-timed 128 [--parity-out out.json]\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();

    parity_params pparams;
    if (!parse_parity_args(argc, argv, pparams)) {
        return 1;
    }

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    params.warmup = false;  // protocol warmup is explicit below
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (auto t : pparams.prompt_tokens) {
        if (t < 0 || t >= n_vocab) {
            LOG_ERR("prompt token %d out of range [0, %d)\n", (int) t, (int) n_vocab);
            llama_model_free(model);
            return 1;
        }
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        LOG_ERR("%s: error: failed to create the llama_context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    llama_batch batch = llama_batch_init((int) pparams.prompt_tokens.size() + 2, 0, 1);
    auto * mem = llama_get_memory(ctx);

    // Runs one full greedy pass: prefill (untimed, synced at end) + n_gen decode
    // steps, each followed by a device sync. When `timed`, records wall time and
    // per-step latencies.
    std::vector<double> lat_ms;
    std::vector<llama_token> timed_tokens;
    double timed_s = 0.0;

    auto run_pass = [&](int n_gen, bool timed) -> bool {
        lat_ms.clear();
        timed_tokens.clear();
        llama_memory_clear(mem, true);

        const int P = (int) pparams.prompt_tokens.size();

        // prefill
        common_batch_clear(batch);
        for (int i = 0; i < P; ++i) {
            common_batch_add(batch, pparams.prompt_tokens[i], i, { 0 }, i == P - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("prefill decode failed\n");
            return false;
        }
        llama_synchronize(ctx);
        // logits were requested only on the last prompt row
        llama_token cur = (llama_token) argmax_logits(llama_get_logits_ith(ctx, P - 1), n_vocab);

        int64_t t_prev = 0;
        if (timed) {
            t_prev = ggml_time_us();
        }

        for (int i = 0; i < n_gen; ++i) {
            common_batch_clear(batch);
            common_batch_add(batch, cur, P + i, { 0 }, true);
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("decode failed at step %d\n", i);
                return false;
            }
            llama_synchronize(ctx);
            const int64_t t_now = ggml_time_us();
            if (timed) {
                lat_ms.push_back((double) (t_now - t_prev) / 1000.0);
                t_prev = t_now;
                timed_tokens.push_back(cur);
            }
            cur = (llama_token) argmax_logits(llama_get_logits_ith(ctx, 0), n_vocab);
        }
        if (timed) {
            // Timed-region boundaries equal the EXL3 native harness: one device
            // sync per step, so the region total is the sum of per-step deltas.
            double acc = 0.0;
            for (double v : lat_ms) {
                acc += v;
            }
            timed_s = acc / 1000.0;
        }
        return true;
    };

    const double warmup_t0 = ggml_time_us();
    if (!run_pass(pparams.n_warmup, false)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    const double warmup_s = (double) (ggml_time_us() - warmup_t0) / 1e6;

    if (!run_pass(pparams.n_timed, true)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    emit_json(pparams, params.model.path, llama_n_ctx(ctx), (int) n_vocab,
              timed_s, warmup_s, lat_ms, timed_tokens,
              pparams.out_path.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
