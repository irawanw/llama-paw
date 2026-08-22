#include "models.h"

#include <stdexcept>

// PAW-27B — the PAW codec on the dense qwen35 topology.
//
// Everything outside the FFN is shared with llama_model_paw: the same GDN /
// full-attention stack, the same nibble-LUT embedding, the same int-lattice
// lm_head. Only two things differ:
//   * the FFN is three rt matrices per layer (ffn_gate / ffn_up / ffn_down)
//     instead of the routed-expert tier plus a shared expert;
//   * the dimensions are not powers of two (5120, 6144, 10240, 17408), so the
//     payload carries paw-dense.rht_block = 1024 and every rotation runs
//     block-diagonally. m1_rht_blk is read by the shared loader.
//
// A separate arch string is required rather than a branch on n_expert:
// llama_model_create dispatches on arch alone, before any hparams exist.

void llama_model_paw_dense::load_arch_hparams(llama_model_loader & ml) {
    // the expert KVs are optional in the moe loader and simply stay 0 here
    llama_model_qwen35moe::load_arch_hparams(ml);

    GGML_ASSERT(hparams.n_expert == 0 && "paw-dense is the dense topology; use arch 'paw' for MoE");

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_paw_dense::load_arch_ffn_tensors(llama_model_loader & ml, int il) {
    auto & m1l = m1_layers[il];

    m1l.ffn_gate = m1_create_ne(ml, LLM_TENSOR_FFN_GATE, il);
    m1l.ffn_up   = m1_create_ne(ml, LLM_TENSOR_FFN_UP,   il);
    m1l.ffn_down = m1_create_ne(ml, LLM_TENSOR_FFN_DOWN, il);
}

void llama_model_paw_dense::load_arch_tensors(llama_model_loader & ml) {
    llama_model_paw::load_arch_tensors(ml);

    // the dense FFN has no router; leaving it set would send build_layer_ffn
    // down the MoE path in any shared code that keys off it
    for (int il = 0; il < (int) hparams.n_layer(); ++il) {
        GGML_ASSERT(layers[il].ffn_gate_inp == nullptr);
    }
}

std::unique_ptr<llm_graph_context> llama_model_paw_dense::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("paw-dense: no MTP head in this checkpoint");
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_paw_dense::graph::graph(const llama_model_paw_dense & model, const llm_graph_params & params) :
    llama_model_paw::graph(model, params, defer_build_t{}) {
    build();
}

ggml_tensor * llama_model_paw_dense::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    const auto & m1l = model.m1_layers[il];

    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    ggml_tensor * g;
    ggml_tensor * u;
    if (rt_batch_site(2) && m1l.ffn_gate.rt_trellis && m1l.ffn_up.rt_trellis &&
            m1l.ffn_gate.rt_sv->ne[0] == m1l.ffn_up.rt_sv->ne[0]) {
        // gate and up share the input and the shape — one batched op instead
        // of two latency-bound launches (same fusion as the 35B shared expert)
        const m1_ne ws[2] = { m1l.ffn_gate, m1l.ffn_up };
        ggml_tensor * cat = ne_mm_batch(ws, 2, cur);
        ggml_build_forward_expand(gf, cat);
        const int64_t m_g = m1l.ffn_gate.rt_sv->ne[0];
        const int64_t m_u = m1l.ffn_up.rt_sv->ne[0];
        const int64_t T   = cat->ne[1];
        g = ggml_view_2d(ctx0, cat, m_g, T, cat->nb[1], 0);
        u = ggml_view_2d(ctx0, cat, m_u, T, cat->nb[1], m_g*sizeof(float));
        if (T != 1) {
            g = ggml_cont(ctx0, g);
            u = ggml_cont(ctx0, u);
        }
    } else {
        g = ne_mm(m1l.ffn_gate, cur);
        u = ne_mm(m1l.ffn_up,   cur);
    }
    cb(g, "ffn_gate", il);
    cb(u, "ffn_up", il);

    ggml_tensor * par = ggml_swiglu_split(ctx0, g, u);
    cb(par, "ffn_swiglu", il);

    cur = ne_mm(m1l.ffn_down, par);
    cb(cur, "ffn_out", il);

    return cur;
}
