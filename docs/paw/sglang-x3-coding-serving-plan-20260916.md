# PAW X3 SGLang coding-serving execution plan

Date: 2026-09-16

## 1. Decision and objective

The port is worthwhile only if it is a faster coding server than llama-paw, not merely a second engine that can load the same GGUF.

The priority target is `/home/green-gpu/paw27b_out/B3.5.gguf` with the existing Qwen3.8-27B DFlash2 draft. The model is 11.50 GiB and the current llama-paw reference results are:

- PP512: 809.16 tok/s.
- AR TG128: 40.68 tok/s.
- DFlash2 short-context warm result: 100.45 tok/s, 78/120 accepted draft tokens, 15,292 MiB peak VRAM.
- Quality: MMLU-Pro 337/500, HumanEval 159/164, HumanEval+ 151/164, MBPP+ 297/378.

The SGLang implementation must retain these quality and speculative-acceptance results and beat llama-paw on concurrent coding traffic. Single-stream parity alone is not success.

Qwen4-Exp/Flash-Next X3 MoE is phase 2. It starts only after the dense B3.5 serving gate passes.

## 2. Fixed comparison contract

Benchmark both servers through their HTTP APIs on the same RTX 3090, using the same model files, tokenizer, prompts, sampling settings, context limits, and request arrival schedule. Do not compare SGLang HTTP results with `llama-bench` or an internal llama timing counter.

Use GPU 2 only unless the owner changes the assignment. Hold the existing GPU lock and renter watcher for every run. Every new executable first gets a run of at most 60 seconds. No multi-hour run is permitted before the short correctness and throughput gates pass.

Record for every run:

- Server revision and complete command line.
- Model and draft SHA256.
- Prompt tokens, generated tokens, accepted draft tokens, and output token IDs.
- Request throughput, output tok/s, TTFT, inter-token latency, and end-to-end latency at p50 and p95.
- GPU utilization, peak allocated/reserved VRAM, errors, and request aborts.
- Cold run followed by at least five warm repetitions. Report the median and worst warm result.

### 2.1 Coding workload corpus

Create one checked-in manifest containing fixed tokenized requests rather than relying only on synthetic lengths:

| Workload | Requests | Input | Output | Purpose |
|---|---:|---:|---:|---|
| C1 completion | 32 | 2K-4K | 256 | Short interactive code completion |
| C2 edit | 24 | 8K-16K | 512 | Repository-aware edit and explanation |
| C3 agent | 16 | 24K-32K | 768 | Long coding-agent turn |
| C4 continuation | 8 roots x 4 branches | 8K-24K shared | 256 | Radix/GDN-state reuse |
| C5 mixed | 64 | C1/C2/C3 mixture | mixed | Continuous batching under realistic arrivals |

Run C1-C3 at closed-loop concurrency 1, 2, 4, and 8. Run C5 once at a fixed request trace and once at the maximum stable offered load. Use greedy decoding for the primary comparison so output and acceptance differences are attributable. Add a smaller temperature 0.6 regression run after greedy passes.

### 2.2 Mandatory win gate

SGLang replaces llama-paw only if all of the following pass:

1. Concurrency 1: median output tok/s is at least 100.45 on the original short DFlash2 contract and no slower than the matched llama HTTP result by more than 3% on C1-C3.
2. Concurrency 4: aggregate output tok/s is at least 1.25x llama-paw, with p95 per-request inter-token latency no worse than llama-paw by more than 10%.
3. Concurrency 8: aggregate output tok/s is at least 1.50x llama-paw, with no OOM, scheduler stall, or incorrect state reuse.
4. Continuations: second-branch TTFT is at least 2x faster than recomputing the same prefix and at least 1.5x faster than llama-paw on C4.
5. Quality and acceptance pass section 8.
6. The server completes a two-hour mixed C5 soak with zero wrong-request state leakage, crashes, or failed requests. This soak is run only after all short gates pass.

The 1.25x and 1.50x concurrency targets are grounded in the existing X3 batch behavior. The current kernel measures 201.7 tok/s at 8 rows versus 111.7 for reconstructed GEMM, and the direct X3 path remains preferred through 128 rows. DFlash's 16-row verify block gives approximately 16, 32, 64, and 128 target rows at request concurrency 1, 2, 4, and 8. The implementation must preserve that packed batch through the PAW linears.

If concurrency 4 is below 1.15x after CUDA graphs and packed verification are confirmed, profile once and fix only a demonstrated dominant cost. If it remains below 1.15x, stop the dense port and keep llama-paw as the production engine. Do not proceed to MoE on the assumption that later tuning will recover the gap.

## 3. Scope

### Required for dense B3.5

- Direct PAW X3 GGUF sidecar loading.
- X3 dense linear operation for decode, DFlash verify, and prefill.
- Standard Q4_K token embedding and Q5_K language-model head.
- Qwen3.5 dense GDN/full-attention topology.
- DFlash2 draft loading and candidate selection.
- CUDA graph capture for target verify and draft decode.
- Continuous batching through concurrency 8.
- Hybrid GDN state management and radix-prefix reuse.
- OpenAI-compatible streaming serving and cancellation.

### Explicitly deferred

- Legacy PAW codecs not used by B3.5.
- Tensor parallel X3 sharding.
- Custom llama `q4_0`/`q8_0` KV-cache formats.
- 256K context support.
- Flash-Next MoE until the dense mandatory win gate passes.

This is not a literal port of every historical `GGML_OP_PAW_*` operation. It ports every operation on the B3.5 plus DFlash2 serving path. Porting unused operators would add time and maintenance without improving the target server.

## 4. Implementation architecture

Pin SGLang commit `444b29c932253d82b14ad4e09c4260b3d200bfbc` for the first result. Keep PAW as a private extension until performance and correctness are stable.

### 4.1 Shared CUDA core

Split the framework-independent X3 device code and launch planning from `ggml/src/ggml-cuda/paw-x3.cu`. Both wrappers must call the same core:

```text
PAW GGUF tensors
    -> SGLang PawX3LinearMethod
    -> Torch custom op
    -> shared PAW X3 launch planner
    -> x3v / x3g / reconstruction+GEMM kernel
```

The Torch operation needs:

- Inputs: activation, trellis, `suh`, `svh`, logical input/output dimensions, and K.
- BF16, FP16, and F32 activation entry points.
- BF16 or FP16 output for the SGLang residual stream, plus an F32 reference mode.
- Stream-correct launches with no implicit default-stream use.
- No host synchronization, CUDA allocation, or shape-dependent host work during graph replay.
- A shape-keyed plan created during model load or graph warmup.
- Workspace supplied from persistent graph-stable buffers.

Do not duplicate the 3,700-line kernel in a second implementation. Extract only the core necessary for X3 and leave the ggml tensor/pool wrapper in llama-paw.

### 4.2 SGLang quantization method

Add a private `PawX3Config` and `PawX3LinearMethod` following SGLang's existing `QuantizationConfig` and `LinearMethodBase` interface.

The loader must register the three sidecars as non-trainable parameters and preserve their bytes:

- `m3_trellis`
- `m3_suh`
- `m3_svh`

Matrix fusion must be explicit. For QKV, gate/up, and other merged SGLang linears, load each GGUF matrix and sidecars into the correct output slice. Preserve the individual K value for each component; never assume one K for a merged tensor.

Use the native SGLang Qwen3.5 implementation for RMSNorm, rotary attention, GDN, residuals, scheduler state, and KV/state pools. The old PAW-35B conversion's T=17 failure shows why model topology and state code should not be re-created in a converted checkpoint.

### 4.3 Direct GGUF loader

Implement a custom SGLang loader rather than materializing reconstructed weights:

- Read model metadata, tokenizer metadata, ordinary GGUF tensors, and X3 sidecars from the original file.
- Use a sibling Hugging Face `config.json` only to instantiate the native Qwen3.5 architecture.
- Map GGUF names to SGLang parameters with a checked table.
- Reject a missing sidecar, wrong shape, unsupported K, or ambiguous merged mapping at load time.
- Print one concise load summary containing counts and bytes for X3, Q4_K, Q5_K, and unquantized tensors.

No safetensors conversion is part of the serving path.

## 5. Kernel path required for serving wins

Preserve these existing decisions from `paw-x3.cu` unless direct SGLang measurements disprove them:

| Activation rows | Path | Reason |
|---:|---|---|
| 1-2 | x3v GEMV | Lowest decode latency; folded F32 input prologue |
| 3-128 | x3g tensor-core trellis GEMM | Best DFlash verify and small continuous batches |
| 129-1023 | plain reconstruction plus GEMM | Amortizes weight reconstruction and activation transforms |
| 1024+ | reconstructed GEMM | Prefill throughput |

SGLang should pass one flattened activation matrix containing all scheduled tokens. Do not loop over requests or call PAW once per sequence. Add counters for rows and selected path so the benchmark report can prove that concurrency 1/2/4/8 actually reaches 16/32/64/128-row verification.

First reproduce the F32-output result. Then add a BF16 input/output specialization that folds conversion into the X3 prologue/epilogue. Keep it only if:

- Operator relative RMS remains within the existing 0.2% block gate.
- Full-model quality is unchanged by section 8.
- It improves the matched workload by at least 2%; otherwise retain the simpler path.

## 6. Language-model head and DFlash2

The Q5_K head is performance-critical because DFlash2 uses it to select candidates for draft positions.

SGLang's current GGUF method already sends Q5_K matrices to `ggml_mul_mat_a8` once the activation row count exceeds the MMVQ threshold. DFlash2 produces 15 candidate rows per request for a 16-token block, so this should select MMQ even at concurrency 1. Verify this with a counter; do not assume it.

Required head sequence:

1. Load Q5_K directly with `GGUFLinearMethod`.
2. Confirm DFlash2 accepts the quantized `lm_head.quant_method` and folds selector execution into its draft CUDA graph.
3. Confirm FlashInfer radix top-k is active; SGLang explicitly warns that the Torch fallback roughly halves throughput on a large vocabulary.
4. Measure Q5_K projection plus top-k for 15, 30, 60, and 120 rows against llama-paw's MMQ head path.
5. Only if SGLang's head is more than 10% slower, port the narrow llama-paw Q5_K MMQ override as a custom head method. Do not port `PAW_HEAD_MM`; B3.5 uses the standard Q5_K head, not the legacy PAW head codec.

Load the DFlash2 GGUF without retraining. Its Q2_K/Q3_K/Q4_K matrices and selector tensors are already representable by SGLang GGUF. Add only its config and tensor-name mapping if required.

Tune the DFlash block only after block 16 matches the reference. Evaluate 8, 12, and 16 on the complete C1-C5 workload. Select by delivered output tok/s and p95 latency, not model-eval tok/s. A larger block is retained only when its lower acceptance is outweighed by fewer target calls.

## 7. Continuous batching and prefix reuse

Use SGLang's overlap scheduler and hybrid GDN radix cache. Do not use `mamba_radix_cache_strategy=no_buffer` for the winning configuration because current SGLang forces overlap scheduling off in that mode.

Start with:

```text
mamba_radix_cache_strategy=extra_buffer
mamba_track_interval=256
CUDA graph decode batch sizes=1,2,4,8
DFLASH block_size=16
pp_size=1
tp_size=1
```

Test `extra_buffer_lazy` only after the stable result. Test checkpoint intervals 128, 256, and 512 on C4; the selected value must minimize continuation TTFT without consuming enough state memory to reduce stable concurrency below 8.

Keep active GDN state in its accuracy-preserving dtype for the first gate. The int8 Mamba checkpoint pool is optional and is enabled only after cached-prefix quality comparison. It may increase cached-prefix capacity, but it must not be allowed to change the primary parity result silently.

Scheduler requirements:

- A newly arriving prefill must not starve active coding decodes.
- DFlash verify batches must remain packed across active requests.
- Cancellation must release attention KV, GDN state, and draft state.
- A radix hit must restore both attention KV and the corresponding GDN/conv state.
- Forked requests sharing a prefix must not mutate one another's GDN state.

## 8. Correctness and quality gates

### 8.1 Operator and layer gates

- Compare every B3.5 X3 shape and K against the llama-paw operator dump for rows 1, 2, 3, 8, 16, 32, 64, 128, 256, and 512.
- Relative RMS <= 0.2% for the serving dtype path.
- Verify one GDN layer and one full-attention layer at prompt lengths 15, 17, 256, and 4096.
- Verify continuous batches with unequal sequence lengths and independently initialized GDN states.
- Verify one prefix fork, cancellation, slot reuse, and cache eviction sequence.

### 8.2 Full-model gates

- On a fixed 200-prompt coding/parity set, greedy first-token agreement with llama-paw >= 99%.
- On the complete greedy continuations, token agreement >= 99.5%. Any systematic divergence, repeated-token loop, language corruption, or request cross-contamination is a failure regardless of the percentage.
- DFlash reference acceptance remains 78/120, allowing at most one token difference only when the corresponding target logits are within the documented numerical tolerance.
- Output IDs from eager and CUDA-graph SGLang runs must match exactly.
- Cached and uncached SGLang continuations must match exactly before optional int8 checkpointing is considered.
- Re-run MMLU-Pro 500, HumanEval+, and MBPP+. Accept no drop larger than two items on any set; investigate item-level changes rather than averaging them away.

## 9. Execution slices

### Slice 0: lock the baseline, 0.5-1 day

Deliverables:

- Reproducible llama-paw HTTP commands for AR and DFlash2.
- Tokenized C1-C5 manifest and runner.
- Baseline table at concurrency 1/2/4/8.
- GPU/VRAM trace and DFlash acceptance trace.

Gate: the runner must reproduce the existing single-stream result within 3%. Resolve measurement mismatch before port work.

### Slice 1: standalone Torch X3 operation, 3-5 days

Deliverables:

- Shared CUDA launch core and ggml wrapper using it.
- Torch custom operation.
- Shape/K oracle tests and row-path counters.
- CUDA graph capture/replay test with stable addresses.

Gate: all operator shapes pass, graph output equals eager output, and rows 16-128 are no slower than the llama-paw kernel by more than 3%.

### Slice 2: direct B3.5 load and eager forward, 4-7 days

Deliverables:

- `PawX3Config`, `PawX3LinearMethod`, and direct GGUF loader.
- Complete tensor mapping report.
- Full model eager logits and greedy generation.

Gate: T=15, T=17, 256, and 4096 layer/full-model checks pass. Do not move forward with an unexplained T-dependent divergence.

### Slice 3: AR server and graphs, 3-5 days

Deliverables:

- Streaming HTTP server.
- Decode graphs for concurrency 1/2/4/8.
- Correct GDN state allocation, cancellation, and slot reuse.
- AR benchmark against llama-paw.

Gate: AR concurrency 4 aggregate throughput >=1.20x llama-paw and concurrency 1 no worse than 3%. This is an intermediate gate; DFlash remains necessary for the final result.

### Slice 4: DFlash2 and packed verification, 4-7 days

Deliverables:

- Direct draft GGUF load.
- Quantized Q5_K candidate head inside draft graph.
- Packed target verification across requests.
- Acceptance, head/top-k, and target-kernel timing counters.

Gate: single stream >=100.45 tok/s, reference acceptance passes, and concurrency 4 >=1.25x llama-paw.

### Slice 5: coding scheduler and radix reuse, 3-6 days

Deliverables:

- Stable `extra_buffer` GDN radix configuration.
- Prefix fork/restore correctness tests.
- C1-C5 benchmark and configuration selection.
- Two-hour soak after all short gates.

Gate: every mandatory win condition in section 2.2 passes. This is the release decision for dense PAW X3.

Expected dense duration for one engineer is approximately 3-5 weeks. The work remains incremental: every slice leaves a runnable, measurable artifact, and no long quality run is started before the relevant smoke and performance gates pass.

## 10. Profiling order when a gate misses

Profile one representative C2 run at concurrency 4 and one C5 run at concurrency 8. Attribute GPU and CPU time before changing code. Investigate in this order:

1. Verify X3 row packing and selected path.
2. Q5_K head projection and radix top-k.
3. Draft CUDA graph coverage and graph breaks.
4. Target verify CUDA graph coverage.
5. GDN verify/state commit kernels.
6. Scheduler gaps and CPU synchronization.
7. Attention at the actual coding context length.
8. Prefill chunking and radix checkpoint restore.

Any proposed optimization must name the measured percentage of wall time it removes. Keep it only if the end-to-end C1-C5 result improves by at least 2% or it is necessary for correctness/capacity.

## 11. Flash-Next X3 MoE after dense success

Current SGLang already implements `Qwen4ExpForConditionalGeneration`, PLE, PLE host offload, hybrid linear/full attention, MoE, and embedded MTP. Reuse that architecture.

Add `PawX3MoEMethod` behind SGLang's `FusedMoEMethodBase`:

- Consume the dispatcher-provided GPU routing buffers.
- Keep counting, sorting, gather, expert execution, and combine on the GPU.
- Reuse the dense X3 device core for gate/up/down expert matrices.
- Support per-expert K1-K4 metadata.
- Never port the current `ggml_cuda_op_paw_x3_mm_id` host sort or its device-to-host synchronization into SGLang.
- Start with native embedded MTP, not DFlash, for Flash-Next speculative decoding.
- Start two-GPU deployment with pipeline-parallel layer placement. Consider TP/EP only after a measured PP bottleneck.

MoE has its own gate: it must beat the corresponding llama-paw Flash-Next server at concurrency 4 while preserving routing and quality. Estimated additional work after dense infrastructure is 3-6 weeks.

## 12. Expected outcome

The realistic single-stream target is 100-115 tok/s rather than a large multiplier over the current 100.45 tok/s. The reason to move to SGLang is the expected concurrency and prefix-reuse gain:

- Concurrency 4 target: at least 1.25x llama-paw aggregate output throughput.
- Concurrency 8 target: at least 1.50x llama-paw aggregate output throughput.
- Shared coding-session continuations: at least 1.5x llama-paw TTFT improvement.

These are release gates, not claimed results. Until measured, llama-paw remains the production reference.

## 13. Grounding references

- `ggml/src/ggml-cuda/paw-x3.cu`: X3 decode, verify, and prefill kernels and measured row thresholds.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: B3.5 Q5_K MMQ head override.
- `/home/green-gpu/bonsai-pilot/reports/paw27b_spec100_v2_residual_closed_20260908.md`: 100.45 tok/s DFlash2 reference.
- `/home/green-gpu/bonsai-pilot/reports/paw27b_full_comparison_final_20260906.md`: B3.5 size, quality, PP, and TG results.
- `docs/paw/sglang-port-report.md`: previous non-X3 PAW-35B experiment and unresolved T=17 failure.
- SGLang `python/sglang/srt/layers/quantization/gguf.py`: existing Q4_K/Q5_K loaders and MMVQ/MMQ selection.
- SGLang `python/sglang/srt/models/dflash.py`: quantized target-head candidate selection.
- SGLang `python/sglang/srt/speculative/dflash_worker_v2.py`: DFlash CUDA graph and continuous-batch integration.
- SGLang `python/sglang/srt/arg_groups/mamba_hook.py`: GDN radix strategies and overlap-scheduler restrictions.
- SGLang `python/sglang/srt/models/qwen4_exp.py`: existing Flash-Next/Qwen4-Exp architecture.
