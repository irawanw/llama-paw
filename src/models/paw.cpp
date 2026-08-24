#include "models.h"
#include "llama-memory-recurrent.h"

#include <cmath>
#include <stdexcept>

// PAW-Ternary-35B (Windows port P2).
//
// The checkpoint is a qwen35moe topology whose quantized matrices ship as
// trellis code streams instead of dense weights (producer:
// llm-compression/src/llm_compression/qrec/apps/export_gguf.py — the schema
// constants there are the frozen contract for the tensor names used here):
//   * NE tier      — <base>.m1_packed / .m1_gscale / .m1_lut on the stock
//                    attention/GDN/shared-expert stems, plus lm_head under
//                    "output" (K=3, 8 LUT chunks). Decoded in-kernel by
//                    GGML_OP_PAW_NE_MM.
//   * embeddings   — token_embd.m1_q / .m1_mn / .m1_mx (int3 asymmetric),
//                    gathered by GGML_OP_PAW_EMBED_ROWS.
//   * experts      — blk.N.ffn_{gate,up,down}_exps.m1_* (QTIP trellis; kernels
//                    land in P3). Hybrid bring-up GGUFs may instead carry stock
//                    dequantized ffn_*_exps tensors, which run the stock path.
//   * everything else (norms, router, GDN scalars) is dense F32 on the stock
//     names and loads exactly like qwen35moe.
//
// hparams come verbatim from llama_model_qwen35moe::load_arch_hparams (the
// exporter re-prefixes qwen35moe.* KVs to paw.*).

// packed sidecar with shape taken from the GGUF metadata (kept/demoted
// expert counts vary per layer, so shapes cannot be derived from hparams)
ggml_tensor * llama_model_paw::m1_create(llama_model_loader & ml, const LLM_TN_IMPL & tnv, bool required) {
    const std::string name = tnv.str();
    ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
    if (meta == nullptr) {
        if (!required) {
            return nullptr;
        }
        throw std::runtime_error("paw: missing required tensor '" + name + "'");
    }
    switch (ggml_n_dims(meta)) {
        case 1: return create_tensor(tnv, { meta->ne[0] }, 0);
        case 2: return create_tensor(tnv, { meta->ne[0], meta->ne[1] }, 0);
        case 3: return create_tensor(tnv, { meta->ne[0], meta->ne[1], meta->ne[2] }, 0);
        default:
            throw std::runtime_error("paw: unexpected rank for tensor '" + name + "'");
    }
}

llama_model_paw::m1_ne llama_model_paw::m1_create_ne(llama_model_loader & ml, llm_tensor base, int il) {
    m1_ne w;
    if (m1_version >= 3) {
        w.rt_trellis = m1_create(ml, tn(base, "m1_rt_trellis", il), true);
        w.rt_su      = m1_create(ml, tn(base, "m1_rt_su",      il), true);
        w.rt_sv      = m1_create(ml, tn(base, "m1_rt_sv",      il), true);
        return w;
    }
    w.packed = m1_create(ml, tn(base, "m1_packed", il), true);
    w.gscale = m1_create(ml, tn(base, "m1_gscale", il), true);
    w.lut    = m1_create(ml, tn(base, "m1_lut",    il), true);
    return w;
}

void llama_model_paw::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    GGML_ASSERT(hparams.n_layer_nextn == 0 && "paw checkpoints do not ship the MTP head");

    auto create_m1 = [&](const LLM_TN_IMPL & tnv, bool required) {
        return m1_create(ml, tnv, required);
    };
    // payload version: 2 = QTIP V2 tiers, 3 = additive (V8 experts, rotated
    // NE spine, int5-g64 head, nibble-LUT embed)
    // new checkpoints use the paw.* prefix; pre-rename ones use mach1.*
    if (!ml.get_key("paw.format_version", m1_version, false)) {
        ml.get_key("mach1.format_version", m1_version, false);
    }
    // absent on every pre-dense checkpoint, which is exactly the 0 default
    ml.get_key(LLM_KV_PAW_RHT_BLOCK, m1_rht_blk, false);

    auto create_m1_ne = [&](llm_tensor base, int il) {
        return m1_create_ne(ml, base, il);
    };

    m1_layers.resize(n_layer);

    // global: shared expert trellis codebook + packed embedding + packed lm_head.
    // A dense checkpoint has no V=8 expert tier and ships embed/head as stock
    // k-quants (measured: q4_K and q5_K beat both PAW tiers on bits and error),
    // so it wants only the NE table and the ordinary tok_embd/output tensors.
    if (uses_paw_experts()) {
        m1_tlut = create_m1(tn(LLM_TENSOR_PAW_TLUT), true);
    }

    if (m1_version >= 3) {
        m1_ne_tlut = create_m1(tn(LLM_TENSOR_PAW_NE_TLUT), true);
    }

    if (uses_paw_embed()) {
        if (m1_version >= 3) {
            m1_embed_codes = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_codes"), true);
            m1_embed_lut   = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_lut"),   true);
        } else {
            m1_embed_q  = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_q"),  true);
            m1_embed_mn = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_mn"), true);
            m1_embed_mx = create_m1(tn(LLM_TENSOR_TOKEN_EMBD, "m1_mx"), true);
        }
    } else {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    if (!uses_paw_head()) {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);
    } else if (m1_version >= 3) {
        m1_head_qp     = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_qp"),     true);
        m1_head_gscale = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_gscale"), true);
    } else {
        m1_output.packed = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_packed"), true);
        m1_output.gscale = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_gscale"), true);
        m1_output.lut    = create_m1(tn(LLM_TENSOR_OUTPUT, "m1_lut"),    true);
    }

    // hybrid bring-up artifacts carry stock dequantized expert tensors
    m1_stock_experts = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", 0).str().c_str()) != nullptr;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        auto & m1l   = m1_layers[il];

        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t key_dim    = hparams.ssm_d_state * hparams.ssm_n_group;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // dense F32 keeps — identical to qwen35moe
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        if (!hparams.is_recr(il)) {
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

            m1l.wq = create_m1_ne(LLM_TENSOR_ATTN_Q,   il);
            m1l.wk = create_m1_ne(LLM_TENSOR_ATTN_K,   il);
            m1l.wv = create_m1_ne(LLM_TENSOR_ATTN_V,   il);
            m1l.wo = create_m1_ne(LLM_TENSOR_ATTN_OUT, il);
        } else {
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);

            m1l.wqkv      = create_m1_ne(LLM_TENSOR_ATTN_QKV,  il);
            m1l.wqkv_gate = create_m1_ne(LLM_TENSOR_ATTN_GATE, il);
            m1l.ssm_out   = create_m1_ne(LLM_TENSOR_SSM_OUT,   il);
        }

        load_arch_ffn_tensors(ml, il);
    }
}

void llama_model_paw::load_arch_ffn_tensors(llama_model_loader & ml, int il) {
    LLAMA_LOAD_LOCALS;
    auto & layer = layers[il];
    auto & m1l   = m1_layers[il];
    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

    // router (dense)
    layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, 0);

    if (m1_stock_experts) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
    } else {
        m1l.remap = m1_create(ml, tn(LLM_TENSOR_FFN_GATE_INP, "m1_remap", il), true);
        const llm_tensor exps[3] = { LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_DOWN_EXPS };
        for (int p = 0; p < 3; ++p) {
            auto & e = m1l.exps[p];
            e.kept_trellis = m1_create(ml, tn(exps[p], "m1_kept_trellis", il), true);
            e.dem_trellis  = m1_create(ml, tn(exps[p], "m1_dem_trellis",  il), m1_version < 3);
            e.su           = m1_create(ml, tn(exps[p], "m1_su", il), true);
            e.sv           = m1_create(ml, tn(exps[p], "m1_sv", il), true);
            e.basis_a      = m1_create(ml, tn(exps[p], "m1_basis_a", il), false);
            e.basis_b      = m1_create(ml, tn(exps[p], "m1_basis_b", il), false);
            e.basis_c      = m1_create(ml, tn(exps[p], "m1_basis_c", il), false);
            e.wave_gamma   = m1_create(ml, tn(exps[p], "m1_wave_gamma", il), m1_version >= 3);
        }
    }

    // shared expert (dense gate + packed projections)
    layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
    m1l.gate_shexp = m1_create_ne(ml, LLM_TENSOR_FFN_GATE_SHEXP, il);
    m1l.up_shexp   = m1_create_ne(ml, LLM_TENSOR_FFN_UP_SHEXP,   il);
    m1l.down_shexp = m1_create_ne(ml, LLM_TENSOR_FFN_DOWN_SHEXP, il);
}

std::unique_ptr<llm_graph_context> llama_model_paw::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("paw: no MTP head in this checkpoint");
    }
    return std::make_unique<graph>(*this, params);
}

// ---------------------------------------------------------------------------
// graph — qwen35moe topology with every dense projection routed through the
// paw codec ops. Structure mirrors llama_model_qwen35moe::graph so the two
// stay diffable.
// ---------------------------------------------------------------------------

ggml_tensor * llama_model_paw::graph::ne_mm(const m1_ne & w, ggml_tensor * x) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    if (w.rt_trellis) {   // payload v3: rotated int-lattice spine
        return ggml_paw_rt_mm(ctx0, w.rt_trellis, w.rt_su, w.rt_sv, model.m1_ne_tlut, x,
                              (int) model.m1_rht_blk);
    }
    return ggml_paw_ne_mm(ctx0, w.packed, w.gscale, w.lut, x);
}

// GGML_PAW_RT_BATCH=1: emit one GGML_OP_PAW_RT_MM_BATCH op for adjacent
// rt matmuls sharing the same input x (wq/wk/wv, wqkv/wqkv_gate,
// gate_shexp/up_shexp). The backend runs each phase once for the whole group
// instead of once per matrix — the per-matrix launches are latency-bound at
// nt=1. The output is the K outputs concatenated row-wise; callers slice it
// with views, so consumers only read the group's output after this one op.
// Env lookup with pre-rename fallback: these knobs shipped as GGML_MACH1_*
// before the PAW rename. Without the fallback an old script silently loses
// the fusions it asks for -- a performance regression with no error message.
static const char * paw_getenv(const char * name) {
    const char * v = getenv(name);
    if (v == nullptr && strncmp(name, "GGML_PAW_", 9) == 0) {
        char legacy[128];
        snprintf(legacy, sizeof(legacy), "GGML_MACH1_%s", name + 9);
        v = getenv(legacy);
    }
    return v;
}

static bool paw_rt_batch_on() {
    const char * v = paw_getenv("GGML_PAW_RT_BATCH");
    return v != nullptr && atoi(v) != 0;
}

// debug isolation: bit0=qkv+gate group, bit1=q+k+v group, bit2=mlp gate+up
// group. Falls back to paw_rt_batch_on() (all groups) when unset.
static bool paw_rt_batch_site(int bit) {
    const char * v = paw_getenv("GGML_PAW_RT_BATCH_MASK");
    if (v == nullptr) {
        return paw_rt_batch_on();
    }
    return (atoi(v) & (1 << bit)) != 0;
}

// GGML_PAW_EXP_BATCH2=1: fuse the gate+up routed-expert projections'
// group/u/walk/out launches into one call each (they share input, routing,
// and codebook -- only per-expert trellis/scale differ). Scoped by the call
// site to decode-only, no-demotion, no-basis, V8 -- this checkpoint's actual
// runtime shape for routed experts.
static bool paw_exp_batch2_on() {
    const char * v = paw_getenv("GGML_PAW_EXP_BATCH2");
    return v == nullptr || atoi(v) != 0;
}

// Fold the shared-expert sigmoid gate and residual add into the RT down
// projection's output transform. Decode-only; prompt graphs stay unchanged.
static bool paw_shared_epilogue_on() {
    const char * v = paw_getenv("GGML_PAW_SHARED_EPILOGUE");
    return v == nullptr || atoi(v) != 0;
}

static bool paw_shared_gate_dot_on() {
    const char * v = paw_getenv("GGML_PAW_SHARED_GATE_DOT");
    return v == nullptr || atoi(v) != 0;
}

// Collapse the MoE aggregation's ggml_mul + (n_expert_used-1) ggml_add chain
// into one op. At decode that chain is 8 tiny elementwise launches per layer
// (320/token over 40 layers) moving a few KB each -- almost pure launch
// overhead. Bit-exact: the fused kernel accumulates in the same slot order.
static bool paw_moe_reduce_on() {
    const char * v = paw_getenv("GGML_PAW_MOE_REDUCE");
    return v == nullptr || atoi(v) != 0;
}

bool llama_model_paw::graph::rt_batch_site(int bit) {
    return paw_rt_batch_site(bit);
}

ggml_tensor * llama_model_paw::graph::ne_mm_batch(
        const m1_ne * ws, int n_matrices, ggml_tensor * x) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    ggml_tensor * trellis[4], * su[4], * sv[4];
    for (int i = 0; i < n_matrices; ++i) {
        GGML_ASSERT(ws[i].rt_trellis);
        trellis[i] = ws[i].rt_trellis;
        su[i]      = ws[i].rt_su;
        sv[i]      = ws[i].rt_sv;
    }
    return ggml_paw_rt_mm_batch(ctx0, n_matrices, trellis, su, sv, model.m1_ne_tlut, x,
                                (int) model.m1_rht_blk);
}

// grouped -> tiled V-head row reorder on an activation vector segment
// [head_v_dim * n_v_heads, T]: new[(v*K + k)*hd + d] = old[(k*r + v)*hd + d].
// The v2 exporter bakes this into the packed NE streams; the v3 rotated codec
// cannot take row permutations (the Hadamard mixes rows), so it moves here.
ggml_tensor * llama_model_paw::graph::v_tiled(ggml_tensor * y) {
    const int64_t hd = hparams.ssm_d_state;                    // head_v_dim
    const int64_t K  = hparams.ssm_n_group;                    // num_k_heads
    const int64_t r  = hparams.ssm_dt_rank / K;                // v heads per k head
    const int64_t T  = y->ne[1];
    GGML_ASSERT(y->ne[0] == hd*K*r);
    ggml_tensor * t = ggml_reshape_4d(ctx0, y, hd, r, K, T);
    t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));    // [hd, K, r, T]
    return ggml_reshape_2d(ctx0, t, hd*K*r, T);
}

// build_inp_embd with the int3 embedding gather in place of get_rows
// (mirrors llm_graph_context::build_inp_embd; no lora on packed embeddings)
ggml_tensor * llama_model_paw::graph::build_inp_embd_paw() {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd_    = hparams.n_embd;
    GGML_ASSERT(n_embd_inp == n_embd_ && "paw: no deepstack inputs");

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    std::array<ggml_tensor *, 2> inps;
    inps[0] = model.m1_embed_codes
        ? ggml_paw_embed_gather(ctx0, model.m1_embed_codes, model.m1_embed_lut, inp->tokens)
        : model.m1_embed_q
        ? ggml_paw_embed_rows(ctx0, model.m1_embed_q, model.m1_embed_mn, model.m1_embed_mx, inp->tokens)
        // stock k-quant embedding (paw-dense): ordinary row gather
        : ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    inps[1] = inp->embd;

    GGML_ASSERT(ggml_are_same_shape(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    res->t_inp_embd = cur;
    res->add_input(std::move(inp));

    return cur;
}

llama_model_paw::graph::graph(const llama_model_paw & model, const llm_graph_params & params,
                              defer_build_t) :
    llm_build_delta_net_base(params), model(model) {
}

llama_model_paw::graph::graph(const llama_model_paw & model, const llm_graph_params & params) :
    graph(model, params, defer_build_t{}) {
    build();
}

void llama_model_paw::graph::build() {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd_paw();

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head: v2 = K=3 NE tier (8 LUT chunks); v3 = int5-g64 codec
    if (model.m1_head_qp) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_paw_head_mm(ctx0, model.m1_head_qp, model.m1_head_gscale, cur);
    } else if (model.m1_output.packed) {
        cur = ne_mm(model.m1_output, cur);
    } else {
        // stock k-quant head (paw-dense)
        cur = build_lora_mm(model.output, cur);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_paw::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed;
    ggml_tensor * z;
    bool rt_rows_mapped = false;
    if (paw_rt_batch_site(0) &&
            model.m1_layers[il].wqkv.rt_trellis && model.m1_layers[il].wqkv_gate.rt_trellis) {
        const m1_ne ws[2] = { model.m1_layers[il].wqkv, model.m1_layers[il].wqkv_gate };
        ggml_tensor * cat = ne_mm_batch(ws, 2, input);
        cat->op_params[8]  = 1;
        cat->op_params[9]  = (uint8_t) hparams.ssm_d_state;
        cat->op_params[10] = (uint8_t) hparams.ssm_n_group;
        cat->op_params[11] = (uint8_t) (hparams.ssm_dt_rank/hparams.ssm_n_group);
        rt_rows_mapped = true;
        ggml_build_forward_expand(gf, cat);
        const int64_t m_qkv = model.m1_layers[il].wqkv.rt_sv->ne[0];
        const int64_t m_z   = model.m1_layers[il].wqkv_gate.rt_sv->ne[0];
        const int64_t T = cat->ne[1];
        qkv_mixed = ggml_view_2d(ctx0, cat, m_qkv, T, cat->nb[1], 0);
        z         = ggml_view_2d(ctx0, cat, m_z,   T, cat->nb[1], m_qkv*sizeof(float));
        if (T != 1) {
            qkv_mixed = ggml_cont(ctx0, qkv_mixed);
            z         = ggml_cont(ctx0, z);
        }
    } else {
        qkv_mixed = ne_mm(model.m1_layers[il].wqkv, input);
        z         = ne_mm(model.m1_layers[il].wqkv_gate, input);
    }
    if (model.m1_layers[il].wqkv.rt_trellis && !rt_rows_mapped) {
        // v3: the grouped->tiled V-row reorder is not baked into the rotated
        // codec, so permute the V segment of the output activations here.
        // One fused row-permutation launch replaces the old
        // cont + cont + permute + concat chain (the head rows copy through).
        const int64_t key_dim = hparams.ssm_d_state * hparams.ssm_n_group;
        const int64_t val_dim = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t T = qkv_mixed->ne[1];
        const int64_t hd = hparams.ssm_d_state;
        const int64_t Kg = hparams.ssm_n_group;
        const int64_t rv = hparams.ssm_dt_rank / Kg;
        ggml_tensor * qkv_seg = ggml_view_2d(ctx0, qkv_mixed, 2*key_dim + val_dim, T,
                                             qkv_mixed->nb[1], 0);
        qkv_mixed = ggml_paw_v_reorder(ctx0, qkv_seg, (int)(2*key_dim), (int)hd, (int)Kg, (int)rv);
    }
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    if (model.m1_layers[il].wqkv_gate.rt_trellis && !rt_rows_mapped) {
        z = ggml_paw_v_reorder(ctx0, z, 0,
                (int) hparams.ssm_d_state, (int) hparams.ssm_n_group,
                (int)(hparams.ssm_dt_rank / hparams.ssm_n_group));   // v3: same reorder on the z gate rows
    }
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_paw::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_paw::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full;
    ggml_tensor * Kcur;
    ggml_tensor * Vcur;
    if (paw_rt_batch_site(1) &&
            model.m1_layers[il].wq.rt_trellis &&
            model.m1_layers[il].wk.rt_trellis &&
            model.m1_layers[il].wv.rt_trellis) {
        const m1_ne ws[3] = { model.m1_layers[il].wq, model.m1_layers[il].wk, model.m1_layers[il].wv };
        ggml_tensor * cat = ne_mm_batch(ws, 3, cur);
        ggml_build_forward_expand(gf, cat);
        const int64_t m_q = model.m1_layers[il].wq.rt_sv->ne[0];
        const int64_t m_k = model.m1_layers[il].wk.rt_sv->ne[0];
        const int64_t m_v = model.m1_layers[il].wv.rt_sv->ne[0];
        const int64_t T = cat->ne[1];
        Qcur_full = ggml_view_2d(ctx0, cat, m_q, T, cat->nb[1], 0);
        Kcur      = ggml_view_2d(ctx0, cat, m_k, T, cat->nb[1], m_q*sizeof(float));
        Vcur      = ggml_view_2d(ctx0, cat, m_v, T, cat->nb[1], (m_q + m_k)*sizeof(float));
        if (T != 1) {
            Qcur_full = ggml_cont(ctx0, Qcur_full);
            Kcur      = ggml_cont(ctx0, Kcur);
            Vcur      = ggml_cont(ctx0, Vcur);
        }
    } else {
        Qcur_full = ne_mm(model.m1_layers[il].wq, cur); // [ (n_embd_head * 2) * n_head, n_tokens ]
        Kcur = ne_mm(model.m1_layers[il].wk, cur);
        Vcur = ne_mm(model.m1_layers[il].wv, cur);
    }
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    cb(Kcur, "Kcur", il);

    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = ne_mm(model.m1_layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_paw::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    // alpha and beta share the input and shape; one fused launch instead of
    // two tiny GEMV launches per layer (60 -> 30 launches per decode token).
    // The halves are addressed as strided views of the packed [2R, T] output
    // (a plain view_2d slice would be non-contiguous in the row dimension).
    // decode-only: the fused kernel stages x in shared per block, which does
    // not pay off at prefill widths (measured pp regression)
    // the fused kernel stages x in fixed shared arrays of 4096 floats and
    // pays off only at decode widths (measured pp regression at prefill)
    const bool dual_ab = model.layers[il].ssm_beta_s == nullptr &&
                         model.layers[il].ssm_alpha_s == nullptr &&
                         ggml_is_contiguous(cur) &&
                         ubatch.n_tokens == 1 &&
                         cur->ne[0] <= 4096;
    ggml_tensor * alpha = nullptr;
    ggml_tensor * beta  = nullptr;
    if (dual_ab) {
        ggml_tensor * ba = ggml_paw_dual_mm(ctx0,
                model.layers[il].ssm_alpha, model.layers[il].ssm_beta, cur);
        alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                             ba->nb[0], ba->nb[1], 0);
        beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                             ba->nb[0], ba->nb[1], ba->nb[1]*cur->ne[1],
                             num_v_heads*sizeof(float));
    } else {
        beta  = build_lora_mm(model.layers[il].ssm_beta,  cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // The packed ssm_out stream keeps the HF column order, where V heads are
    // GROUPED by K head; the graph runs in ggml's TILED order (the exporter
    // row-reorders every other v-indexed tensor). A trellis stream's columns
    // cannot be permuted at export, so convert the activations tiled->grouped
    // here: heads [hd, r*K (k fastest)] -> [hd, K*r (v fastest)].
    ggml_tensor * final_output;
    if (num_k_heads != num_v_heads) {
        const int64_t r = num_v_heads / num_k_heads;
        ggml_tensor * t = ggml_reshape_4d(ctx0, attn_out_norm, head_v_dim, num_k_heads, r, n_seq_tokens*n_seqs);
        t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));   // [hd, r, K, T]
        final_output = ggml_reshape_3d(ctx0, t, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    } else {
        final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(final_output, "final_output", il);

    cur = ne_mm(model.m1_layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_paw::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out;
    if (model.m1_stock_experts) {
        moe_out =
            build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
                nullptr, model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
    } else {
        // routed experts from packed QTIP trellis streams. Routing mirrors
        // build_moe_ffn(SOFTMAX, norm_w=true, scale=expert_weights_scale) with
        // the three mul_mat_id calls replaced by the paw expert ops.
        const auto & m1l = model.m1_layers[il];

        // like build_moe_ffn, take the token count from the tensor: the
        // last-layer inp_out_ids gather may have shrunk cur below ubatch
        const int64_t n_tokens = cur->ne[1];

        ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * probs = ggml_soft_max(ctx0, logits);
        cb(probs, "ffn_moe_probs", il);

        // partial top-k (GGML_OP_TOP_K) instead of a full 256-wide argsort:
        // the routing only needs the n_expert_used best ids
        ggml_tensor * selected_experts = ggml_top_k(ctx0, probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts, "ffn_moe_topk", il);

        ggml_tensor * weights = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights", il);

        // norm_w: normalize the selected probabilities to sum to 1
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_norm", il);

        const float w_scale = hparams.expert_weights_scale;
        if (w_scale != 0.0f && w_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, w_scale);
            cb(weights, "ffn_moe_weights_scaled", il);
        }

        ggml_build_forward_expand(gf, weights);

        ggml_tensor * xexp = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

        auto exp_mm = [&](const m1_exp & e, ggml_tensor * xin) {
            if (!ggml_is_contiguous(xin)) {
                xin = ggml_cont(ctx0, xin);
            }
            ggml_tensor * out = ggml_paw_exp_mm(ctx0,
                e.kept_trellis, e.dem_trellis, e.su, e.sv, model.m1_tlut,
                m1l.remap, selected_experts, xin, e.wave_gamma);
            if (e.basis_a) {
                // fused acc form: dst = out + basis (no separate ggml_add node)
                out = ggml_paw_exp_basis(ctx0, e.basis_a, e.basis_b, e.basis_c,
                                           m1l.remap, selected_experts, xin, out);
            }
            return out;
        };

        ggml_tensor * gate;
        ggml_tensor * up;
        const m1_exp & eg = m1l.exps[0];
        const m1_exp & eu = m1l.exps[1];
        if (paw_exp_batch2_on() && n_tokens == 1 &&
                eg.kept_trellis && eu.kept_trellis &&
                eg.dem_trellis == nullptr && eu.dem_trellis == nullptr &&
                eg.basis_a == nullptr && eu.basis_a == nullptr &&
                eg.wave_gamma && eu.wave_gamma &&
                model.m1_tlut->ne[0] == 8) {
            ggml_tensor * xin = xexp;
            if (!ggml_is_contiguous(xin)) {
                xin = ggml_cont(ctx0, xin);
            }
            ggml_tensor * cat = ggml_paw_exp_mm_batch2(ctx0,
                eg.kept_trellis, eg.su, eg.sv, eg.wave_gamma,
                eu.kept_trellis, eu.su, eu.sv, eu.wave_gamma,
                model.m1_tlut, m1l.remap, selected_experts, xin);
            ggml_build_forward_expand(gf, cat);
            const int64_t m_gu = eg.sv->ne[0];
            gate = ggml_view_3d(ctx0, cat, m_gu, cat->ne[1], cat->ne[3],
                                cat->nb[1], cat->nb[3], 0);
            up   = ggml_view_3d(ctx0, cat, m_gu, cat->ne[1], cat->ne[3],
                                cat->nb[1], cat->nb[3], cat->nb[2]);
        } else {
            gate = exp_mm(eg, xexp); // [n_ff_exp, n_expert_used, n_tokens]
            up   = exp_mm(eu, xexp);
        }
        cb(gate, "ffn_moe_gate", il);
        cb(up, "ffn_moe_up", il);

        ggml_tensor * par;
        const float swiglu_limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
        if (swiglu_limit > 1e-6f) {
            up = ggml_clamp(ctx0, up, -swiglu_limit, swiglu_limit);
            ggml_tensor * gate_act = ggml_silu(ctx0, gate);
            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, swiglu_limit);
            par = ggml_mul(ctx0, gate_act, up);
            cb(par, "ffn_moe_swiglu_limited", il);
        } else {
            par = ggml_swiglu_split(ctx0, gate, up);
            cb(par, "ffn_moe_swiglu", il);
        }

        ggml_tensor * experts = exp_mm(m1l.exps[2], par); // [n_embd, n_expert_used, n_tokens]
        cb(experts, "ffn_moe_down", il);

        if (paw_moe_reduce_on() && ggml_is_contiguous(experts) && ggml_is_contiguous(weights)) {
            moe_out = ggml_paw_moe_reduce(ctx0, experts, weights);
            ggml_build_forward_expand(gf, moe_out);
        } else {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);

        ggml_build_forward_expand(gf, experts);

        // aggregate the expert slots (same view+add pattern as build_moe_ffn)
        ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
        for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
            cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
            ggml_build_forward_expand(gf, cur_experts[i]);
        }
        moe_out = cur_experts[0];
        for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
            ggml_build_forward_expand(gf, moe_out);
        }
        if (hparams.n_expert_used == 1) {
            moe_out = ggml_cont(ctx0, moe_out);
        }
        }
    }
    cb(moe_out, "ffn_moe_out", il);

    // shared expert: silu(gate(x)) * up(x) -> down, then sigmoid-gated
    ggml_tensor * g;
    ggml_tensor * u;
    if (paw_rt_batch_site(2) &&
            model.m1_layers[il].gate_shexp.rt_trellis && model.m1_layers[il].up_shexp.rt_trellis) {
        const m1_ne ws[2] = { model.m1_layers[il].gate_shexp, model.m1_layers[il].up_shexp };
        ggml_tensor * cat = ne_mm_batch(ws, 2, cur);
        ggml_build_forward_expand(gf, cat);
        const int64_t m_g = model.m1_layers[il].gate_shexp.rt_sv->ne[0];
        const int64_t m_u = model.m1_layers[il].up_shexp.rt_sv->ne[0];
        const int64_t T = cat->ne[1];
        g = ggml_view_2d(ctx0, cat, m_g, T, cat->nb[1], 0);
        u = ggml_view_2d(ctx0, cat, m_u, T, cat->nb[1], m_g*sizeof(float));
        if (T != 1) {
            g = ggml_cont(ctx0, g);
            u = ggml_cont(ctx0, u);
        }
    } else {
        g = ne_mm(model.m1_layers[il].gate_shexp, cur);
        u = ne_mm(model.m1_layers[il].up_shexp,   cur);
    }
    ggml_tensor * shared_inp = ggml_mul(ctx0, ggml_silu(ctx0, g), u);
    if (paw_shared_epilogue_on() && n_tokens == 1 &&
            model.m1_layers[il].down_shexp.rt_trellis) {
        const m1_ne & w = model.m1_layers[il].down_shexp;
        if (paw_shared_gate_dot_on()) {
            cur = ggml_paw_rt_mm_epilogue_dot(ctx0, w.rt_trellis, w.rt_su, w.rt_sv,
                    model.m1_ne_tlut, shared_inp,
                    model.layers[il].ffn_gate_inp_shexp, cur, moe_out,
                    (int) model.m1_rht_blk);
        } else {
            ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(shared_gate, "shared_expert_gate", il);
            cur = ggml_paw_rt_mm_epilogue(ctx0, w.rt_trellis, w.rt_su, w.rt_sv,
                                            model.m1_ne_tlut, shared_inp, shared_gate, moe_out,
                                            (int) model.m1_rht_blk);
        }
    } else {
        ggml_tensor * ffn_shexp = ne_mm(model.m1_layers[il].down_shexp, shared_inp);
        cb(ffn_shexp, "ffn_shexp", il);

        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    }
    cb(cur, "ffn_out", il);

    return cur;
}
