# PAW-35B -> SGLang Port: Session Report (2026-09-06)

## Objective
- Port `llama-paw` PAW-35B-A3B MoE weights to SGLang on `gpurental@192.168.18.18` RTX 3060 12GB, matching llama.cpp-PAW output; canonical test: templated ids `[248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198,248068,198]` -> SGLang first token should equal llama's `90700` (llama logits top8: `[90700:23.55, 8160:22.5, 31248:19.53, 760:18.4, ...]`, margin top1-top2 = 1.05).

## Important Details
- Box: `gpurental@192.168.18.18`, 15GB RAM; `tmux new-session -d -s sgl '<cmd>'` with absolute paths; `pkill -f sglang.launch_server` or kill tmux session to stop. **llama-paw-dump runs OOM while the SGL server holds the GPU - kill the server first, run the dump, then relaunch.**
- Server: `~/run_server.sh` = `/data/models/sglang-env/sgl/bin/python -m sglang.launch_server --model-path /data/models/paw35b-sgl --host 127.0.0.1 --port 30000 --context-length 4096 --mem-fraction-static 0.8 --disable-cuda-graph --skip-server-warmup` (+radix cache currently NOT disabled per latest server_args dump - recheck). Log `~/paw_sgl/sgl_fixed.log`. Dump arms need `PAW_DUMP_GDN=1` (+`PAW_DUMP_ATTN=1 PAW_DUMP_MOE=1`) at server start AND arm files present at request time.
- **Do NOT use `return_logprob`** - crashes the server (SGLang assert in prefill logprob path, `input_token_logprobs_val` length mismatch, EXITED_137).
- **Head-pairing fix COMPLETE and VERIFIED** for prefill layer 0 (all vs llama dumps, templated T=15): gdn_in 1.0, mixed 1.0, postconv v head p == llama v[sigma(p)] all 32 (sigma(p)=16*(p%2)+(p//2)), z[h']==llama z[sigma(h')] 1.0, q16/k16 identity 1.0, beta/g 1.0, **core[h']==llama core[sigma(h')] 1.0 all heads, gated == llama final_output 1.0 direct, gdn_out == llama linear_attn_out-0 1.0**.
- Final weight convention (all in-place patched into `paw_dense.safetensors`): conv1d = parent dense w/ v-channel rows sigma-permuted; out_proj = raw `so` (identity); in_proj_qkvz = **plain raw rt regroup** (q/k = raw identity, v = raw identity, z = raw identity - NO V_F transform anywhere); in_proj_ba = sigma-regroup (group g rows `[b g, b 16+g, a g, a 16+g]`, verified against the split `b = view(T,16,4)[...,0:2], a = [...,2:4]`); **A_log = log(-ssm_a[sigma]), dt_bias = ssm_dt[sigma]** (sigma-regrouped per-head [32] vectors - this was the last missing piece, patch8). `raw` = rt_bf16 decode (v/z/gate rows are ROTR-scrambled: raw block m = parent block sigma(m); q/k rows = parent identity).
- llama g formula (paw.cpp:745): `g = ssm_a_raw * softplus(alpha + dt_bias)` (raw multiply, ssm_a negative). ckpt A_log = log(-ssm_a) makes SGLang's `-exp(A_log)*softplus` identical. patch7 (A_log := raw ssm_a) was WRONG (g 27x too big) and was reverted.
- llama-side verified formulas (templated prompt, fp64 python == llama dumps 1.0): core = delta rule w/ mod pairing (k block h%16, v block h), l2norm eps 1e-6, decay-then-pred, out from POST-update state, scale 128^-0.5, g/beta in raw row order.
- **SGLang FLA chunk kernel facts**: mutates `initial_state` in place ([1,32,128,128]); returned `h` buffer is ZEROS (unused - runtime relies on in-place update + `ssm_states[cache_indices] = ssm_states_contig` scatter-back). **State layout = [V, K] v-major == exact S^T** (llama's S is k-major [K, V] - transpose for python comparisons). Kernel needs `use_qk_l2norm_in_kernel=True` when passing raw conv output. Deterministic (fresh state per call - repeat calls with the SAME state tensor pollute results via in-place mutation).
- **Packed decode kernel** (`fused_recurrent_gated_delta_rule_packed_decode` in `fla/fused_recurrent.py:186`): q/k/v split = `q_off=i_h*K, k_off=H*K+i_h*K, v_off=2*H*K+i_hv*V` with k head `i_h = i_hv//2` (interleave, same as chunk); state read [V,K] consistent with chunk write; gating internally `-exp(A_log[h'])*softplus(a[h']+dt_bias[h'])`, beta=sigmoid(b[h']). **Verified: standalone kernel == runtime decode output 1.0** (with dumped inputs). replayssm path disabled (`enable_linear_replayssm: False`).
- causal_conv1d: import from `sglang.kernels.ops.mamba.causal_conv1d_triton`; call needs x [dim, T] 2-dim, `seq_lens_cpu=[T]` list, fp32 conv_states (bf16 state triggers triton dtype assert); python call == sgl postconv exactly 1.0.
- **Prefill layer-by-layer vs llama (T=15)**: all 40 layer inputs (sgl `paw_gin_{il}` vs llama dump6 `attn_norm-{il}`) = 0.999+ (drift to 0.998 by layers 32-39, bf16 accumulation); full-attn layer 3 attn_out EXACT 1.0; moe_out == ffn_out-0 0.9996. Final: llama result_norm dump = last-token only [2048]; llama result_output = full LOGITS [248320].
- **T=15 first token**: sgl 92232 vs llama 90700 - plausibly bf16-flip on a 1.05-logit margin (hidden drift ~0.998 by end).
- **REMAINING PROBLEM - T=17 prefill diverges hugely**: prefill `P+[90700,8340]` -> sgl next 176872; llama (dump7, T=17) logits top8 = `[25:20.57, 279:15.67, 271:15.19, 290:14.15, 369:14.0, ...]` (5-logit margin) - sgl top8 completely different (203294...), sgl logit of 25 = 2.5 vs llama 20.57. Same for T=23 forced-prefill (want 27382, got 4979) and decode steps (garbage loops: 14876 repeated, mixed languages). Since T=15 prefill matched everywhere, the breakage is context/T-dependent - likely token-15+ embeddings (90700/8340 rows), a position-dependent component, or a layer that only misbehaves for this context. NOT yet localized.
- Off-server logit recomputation mismatch: final-residual normed (`gemma rms*(1+w)`) @ ckpt lm_head gives top1 158250 while the server chose 92232 - my final-norm formula != SGLang's actual final norm (or moe_out != true final residual) - unresolved, low priority.
- Dump arms current state (installed sglang, backups exist: `~/paw_sgl/{qwen3_next,gdn_backend}.py.bak`): qwen3_next.py gdn/attn arms dump per-layer `/tmp/paw_gin_{il}.pt`, `/tmp/paw_gout_{il}.pt`, `/tmp/paw_ain_{il}.pt`, `/tmp/paw_aout_{il}.pt` (T>1 guarded, never self-remove; arms `/tmp/paw_gdn_arm`, `/tmp/paw_attn_arm`); paw_qwen3next.py moe arm layer 39 -> `/tmp/paw_moe_out.pt` (arm `/tmp/paw_moe_arm`, self-removes); gdn_backend.py decode arm -> `/tmp/paw_dec_{mixed,a,b,ssm,conv,idx,core}.pt` (layer 0, T=1, arm `/tmp/paw_dec_arm`, self-removes); gdn_dbg/gcnv/ggt arms still un-guarded (overwritten by decode steps - stale). All `paw_gin/gout/ain/aout/moe_out` currently hold **T=17 data** from the last 20:02 request.
- SSM pool: [21 slots, 32, 128, 128]; request used slot idx=[2]. conv pool [21, 8192, 3].
- FLA direct-call quirk: needs bf16+cuda, q/k/v 4-dim [1,T,H,D] when cu_seqlens passed, returns (o, None, h) - h is index [2].
- Carry-over: norms Gemma +1 baked; rope IMROPE; 40L, full-attn {3,7,...,39}; no x3; raw 5-token prompt is OOD - only templated ids meaningful.

## Work State
### Completed
- **GDN prefill path fully fixed and verified exact vs llama** (layer-0 core/gated/out all 1.0): qkvz plain-raw regroup (patch5 fixed), ba sigma-regroup confirmed correct under the [2,2] split, **A_log/dt_bias sigma-regroup (patch8)** - the final piece; patch7 detour reverted.
- conv kernel python-reproduction validated (causal_conv1d_triton import path + calling convention).
- Kernel semantics established: chunk kernel exact (== fp64 recurrence 1.0 all heads), in-place state [V,K], zero returned h; decode kernel exact + consistent with chunk state layout; l2norm consistent.
- Prefill localization: layers 0-3 exact, all-layer inputs 0.999+ for T=15.
- llama dumps: `~/paw_dump6` (attn_norm/attn_residual all 40 layers + result_norm/result_output, T=15), `~/paw_dump7` (T=17 = prompt+[90700,8340]: attn_norm-16, result_norm, result_output).
- Decode-step capture: `/tmp/paw_dec_*.pt` (mixed [1,8192] post-conv, a/b [1,32], ssm pool, conv pool, idx=[2], core [1,1,32,128]) - kernel verified exact on them.

### Active
- **T=17 prefill divergence** (llama wants 25 @ 20.57 logit; sgl gives 176872 with 25 at 2.5) - structural, not noise. The `paw_gin_*` files from the 20:02 run hold the T=17 layer inputs, ready to compare vs llama dump7 (`attn_norm-16`) and (if needed) new llama dumps for intermediate layers.

### Blocked
- Token parity (90700) - blocked on the T=17 divergence (which also breaks all decode steps and multi-token generation quality).

## Next Move
1. Compare the T=17 dumps: `paw_gin_16.pt` vs llama dump7 `attn_norm-16` (both should be [17, 2048]) - per-layer cos for all layers using the T=17 sgl dumps vs dump6 (first 15 positions) and dump7 (position 16). If gin_16 diverges, step back layer by layer (need llama dump8 with attn_norm-4..15 for the T=17 prompt) to find the first divergent layer; if gin_16 matches, the divergence is in layers 17-39 for this context.
2. Prime suspect if early layers match: **embedding rows for tokens 90700/8340** - compare ckpt `embed_tokens` rows for those ids vs the PAW gguf `token_embd` rows (also check a handful of random ids for a partial vocab-order issue) - this would explain T=15 working (format tokens) and any context containing those tokens breaking.
3. After localization, apply the fix (likely another in-place ckpt patch), then re-verify: T=15 first token 90700, T=17 next 25, 16-token generation vs llama `[90700,8340,25,271,16,13,220,2972,27382,1386,279,6007,...]`.
4. Cleanup (after parity): restore qwen3_next.py/gdn_backend.py from `~/paw_sgl/*.bak` minus needed arms, remove `--disable-radix-cache` discrepancy check, consolidate all ckpt patches into a clean `build_paw_ckpt.py` (final convention listed above).

## Relevant Files
- `/data/models/paw35b-sgl/paw_dense.safetensors`: ckpt, in-place patched (final convention in Important Details). Header: 573 tensors, data_start 65096. Keys: `model.layers.N.linear_attn.{conv1d.weight, in_proj_qkvz.weight, in_proj_ba.weight, out_proj.weight, A_log, dt_bias}`, `model.embed_tokens.weight`, `lm_head.weight` [248320, 2048], `model.norm.weight`.
- Patch scripts (box `~/paw_sgl/`): `patch_ckpt5.py` (qkvz plain raw regroup), `patch_ckpt7.py` (A_log - now reverted to log(-ssm_a) semantics, superseded), `patch_ckpt8.py` (A_log/dt_bias sigma-regroup - CURRENT).
- `/data/models/paw35b/rt_bf16/`: `manifest.json`, `blk_0_attn_qkv.pt` [8192,2048], `blk_0_attn_gate.pt` [4096,2048], `blk_0_ssm_out.pt` [2048,4096] (fp32).
- `/data/models/paw35b/PAW-35B-A3B-paw.gguf`: dense F32 `ssm_conv1d.weight` [8192,4], `ssm_beta/ssm_alpha.weight` [32,2048], `ssm_norm.weight` [128], `ssm_a` [32] (NEGATIVE values; llama g = ssm_a*softplus), `ssm_dt.bias` [32], `token_embd.weight` (name TBD - no `output.weight` key).
- `/data/models/sglang-env/sgl/lib/python3.10/site-packages/sglang/srt/models/qwen3_next.py` (+`.bak`): per-layer dump arms (gin/gout/ain/aout), ba split at ~344-367, normgate ~455, fused_gdn_gating call ~892.
- `.../sglang/srt/layers/attention/linear/gdn_backend.py` (+`.bak`): decode arm (~624-645), extend/state scatter ~899-930, dispatcher use ~146.
- `.../sglang/srt/layers/attention/linear/kernels/gdn_triton.py`: `packed_decode` (46-136), `extend` (169-199).
- `.../sglang/kernels/ops/attention/fla/fused_recurrent.py:186`: packed decode kernel (index math documented above).
- `.../sglang/kernels/ops/attention/fla/chunk.py` / `chunk_fwd.py`: vendored FLA chunk; `.../fla/fused_gdn_gating.py`: gating kernel.
- `.../sglang/kernels/ops/mamba/causal_conv1d_triton.py`: conv kernel (2-dim x, seq_lens_cpu required).
- `.../sglang/srt/models/paw_qwen3next.py`: moe dump arm (layer 39).
- llama dumps: `~/paw_dump5` (T=15: attn_norm-0/3, attn_output-0/3, conv_output_silu-0, z-0, final_output-0, linear_attn_out-0/1, attn_residual-0, ffn_out-0, result_norm [1,2048], result_output [248320] = logits), `~/paw_dump6` (T=15: attn_norm/attn_residual all layers), `~/paw_dump7` (T=17: attn_norm-16, result_norm, result_output).
- SGLang dumps: `/tmp/paw_gin_{0..39}.pt`, `/tmp/paw_gout_*.pt`, `/tmp/paw_ain_*.pt`, `/tmp/paw_aout_*.pt`, `/tmp/paw_moe_out.pt` (all = T=17 data now), `/tmp/paw_dec_{mixed,a,b,ssm,conv,idx,core}.pt`, `/tmp/paw_dbg_postconv.pt` [15,8192] (clean layer-0 T=15; the mixed/core/z/gated/g/beta dbg files are decode-polluted).
- `~/run_server.sh`, `~/run_dump5.sh`, `~/run_dump6.sh`, `~/run_dump7.sh` (llama dump invocations; kill SGL first).
- `~/llama-paw-ref/build-cuda/bin/llama-paw-dump`; source refs `~/llama-paw/src/models/paw.cpp` (gate = softplus*ssm_a at 745, beta/alpha cb 726-746), `ggml/src/ggml-cuda/gated_delta_net.cu` (recurrence 84-110: `delta=(v-g*kv)*b`, `S=g*S+k*delta`, out from post-update S).
- `~/paw_sgl/build_paw_ckpt.py` (original builder - needs rewrite to encode the final convention), `~/paw_sgl/{qwen3_next,gdn_backend}.py.bak` (pristine).
