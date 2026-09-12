# llama-paw

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs **PAW**
models — checkpoints whose weights ship as packed trellis code streams and are
decoded on the fly inside the compute kernels.

Get the weights from
[PAW-27B-X3](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF) (1.5-3.5
bit) and [PAW-35B-A3B](https://huggingface.co/lackonendes/PAW-35B-A3B-GGUF).
Direct downloads for the recommended 27B x3 artifact:

- [PAW-27B-X3-3.5bit.gguf](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/PAW-27B-X3-3.5bit.gguf) (11.50 GiB)
- [Qwen3.8-27B-DFlash2-Q2_K.gguf](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF/resolve/main/Qwen3.8-27B-DFlash2-Q2_K.gguf) (0.67 GiB, speculative drafter)

The PAW codec ops are implemented for **CPU, CUDA, and Vulkan**. There are no
Metal kernels: on Apple Silicon this fork runs the codec on CPU.

## Credit

PAW is inspired by and format-compatible with
[**Mach-1**](https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B) by
SyzygyResearch, and this fork descends from
[`llama.cpp-mach1`](https://github.com/SyzygyResearch/llama.cpp-mach1). The
trellis codec and container format are their design.

**Mach-1 checkpoints load directly in this fork** — the loader accepts both
`mach1.*` and `paw.*` naming, so nothing you already have stops working.

What this fork adds on top: fused and batched codec kernels (worth **+35%**
end to end, output-identical), DFlash speculative-decoding support for packed
embeddings and head, and a multi-token vocabulary head.

Additional projects that informed the codec and runtime work:

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3), including its EXL3
  format and reference quantization/inference implementation.
- Mia's ExLlamaV3 fork (exllamav3 1.4.2), the reference implementation used
  for our x3 parity work and the source of the
  `Qwen3.8-27B-DFlash2-EXL3-5.0bpw` comparison artifact.
- [EschaLabs Qwen3.8-27B Escha-W2](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2),
  a related packed-weight reference artifact.
- [SyzygyResearch’s llama.cpp-mach1](https://github.com/SyzygyResearch/llama.cpp-mach1)
  and the [SyzygyResearch organization](https://github.com/SyzygyResearch),
  whose Mach-1 work established the PAW codec lineage noted above.
- [PrismML Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) and its
  [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), for the Bonsai
  model and runtime work.

## PAW-27B-X3 evaluation

The [PAW-27B-X3](https://huggingface.co/lackonendes/PAW-27B-X3-GGUF) weights
are a 1.0-4.0 bpw x3 trellis-coded sweep of `Qwen/Qwen3.8-27B` with greedy
mixed-precision allocation. Every artifact measured on one RTX 3090, one
table. All MMLU-Pro rows use the same protocol, seed and subset hash, so that
column is comparable across every row. IFB-L/IFB-S are raw pass counts out of
64. `-` means no valid measurement exists; it is never an estimate.

| artifact | bpw / size | MMLU-Pro /500 | IFB-L /64 | IFB-S /64 | HumanEval /164 | HumanEval+ /164 | MBPP /378 | MBPP+ /378 | GSM8K | PP512 | TG128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8 parent | 8.51 | 326 | 20 | - | 159 | 150 | - | - | 94/100 | - | - |
| **B4.0** | 4.000 / 12.93 GiB | 333 | 21 | 20 | 157 | 150 | 350 | 295 | 94/100 | 811.24 | 38.23 |
| **B3.5** | 3.500 / 11.50 GiB | **337** | 22 | 21 | **159** | **151** | 352 | **297** | 94/100 | 809.16 | 40.68 |
| MIA / EXL3 | 3.50 / 14.2 GB | 326 | 22 | 21 | 157 | 149 | - | - | 93/100 | - | 45.67+ |
| **B3.0** | 3.000 / 10.10 GiB | 327 | 22 | 21 | 155 | 146 | 351 | 297 | 95/100 | 798.11 | 42.21 |
| **B2.5** | 2.500 / 8.67 GiB | 309 | 19 | 17 | 155 | 144 | 348 | 294 | 97/100 | 805.82 | 44.28 |
| Escha-W2 | 2.469 / 10.2 GB | 309 | 20 | 19 | 156 | 149 | - | - | 95/100 | - | - |
| csweep `reason_code_web` | 2.2966 / 8.70 GB | 319 | 20 | 19 | 153 | 142 | 345 | 294 | 97/100 | - | - |
| csweep `reason_raw` | 2.2966 / 8.70 GB | 316 | 16 | 13 | 150 | 141 | - | - | 95/100 | - | - |
| csweep `reason` (anchor) | 2.2966 / 8.70 GB | 311 | 17 | 14 | 154 | 144 | - | - | 96/100 | - | - |
| csweep `reason_code` | 2.2966 / 8.70 GB | 308 | 18 | 17 | 155 | 146 | - | - | 96/100 | - | - |
| csweep `code` | 2.2966 / 8.70 GB | 303 | 24 | 20 | 156 | 144 | 339 | 290 | 95/100 | - | - |
| csweep `web` | 2.2966 / 8.70 GB | 303 | 19 | 19 | 144 | 134 | - | - | 94/100 | - | - |
| reasonfull-x3 | 2.2966 / 8.70 GB | 307 | 20 | 19 | 150 | 141 | 334 | 284 | 195/200 | - | - |
| v6-x3 | 2.2966 / 8.70 GB | 272 | 17 | 14 | 151 | 138 | 326 | 283 | 189/200 | 843.20 | 47.11 |
| v12d (legacy K2) | 2.324 / 7.8 GB | 290 | 15 | 15 | 154 | 143 | 335 | 287 | - | 366.10 | 31.45 |
| **B2.0** | 2.000 / 7.26 GiB | 290 | 20 | 20 | 150 | 142 | 334 | 283 | 94/100 | 815.23 | 46.45 |
| v7 qkv-only | 1.980 | 244 | 15 | - | 137 | 127 | - | - | 72/100 | - | - |
| **B1.5** | 1.500 / 5.84 GiB | 118 | 10 | 8 | 101 | 94 | 245 | 206 | 42/100 | 813.25 | 46.96 |
| Unsloth IQ2XXS | ~2 / 7.3 GB | 231 | 15 | 9 | - | 138 | - | - | 93/100 | - | - |
| AtomicChat AD-IQ2XXS | ~2 / 9.0 GB | 265 | 13 | 13 | - | 121 | - | - | 94/100 | - | - |
| **B1.0** | 1.000 / 4.43 GiB | 0 | 0 | 0 | 0 | 0 | 44 | 38 | 0/100 | 813.55 | 47.57 |

- GSM8K denominators differ by harness generation: fast-gate rows are n=100,
  full-eval rows (reasonfull-x3, v6-x3) are n=200. Not normalized.
- `+` EXL3's 45.67 is EXL3 `perf.py` INT8 GEMV at context 0 - a different
  harness from the `llama-bench` figures in the same column.
- The seven bit-sweep rows (B1.0-B4.0) are `llama-bench -ngl 99 -p 512 -n 128
  -r 5 -sm none` on one pinned device - mutually comparable. The speed
  protocol differs by generation and is not uniformly comparable down the
  column.
- B3.5 is the first artifact to beat the Q8 parent outright (MMLU-Pro 337 vs
  326, HumanEval+ 151 vs 150) while uniform K4 (B4.0) loses to it on every
  quality column despite being larger - curated mixed precision beats uniform
  higher bit rate.
- Below ~1.5 bpw there is a hard coherence cliff, not a smooth decline: B1.0
  (uniform K1) is total structural incoherence (0 on every benchmark); B1.5
  is a real, partial recovery (MMLU-Pro 118, HumanEval 101).

### Inference speed (current build, B3.5, one RTX 3090)

Since `9f3ba0717` the fp16-accumulate x3 GEMM is on by default
(`GGML_PAW_X3_GEMM_F16ACC=0` restores fp32). Measured on B3.5:

| measurement | tok/s |
|---|---:|
| PP512 (short prompt) | 1076.5 |
| PP8192 (chat length) | **1215.4** |
| PP, 211k-token prompt at 262144 ctx (`-ub 2048`) | 563.0 |
| TG128 (generation, chat scale) | 33.5 |
| TG at 211k depth | 19.9 |
| speculative decode, 8k code context | 100.45 (median, output hash-identical) |

The PP512/TG128 column in the table above was measured on the older
fp32-accumulate build and is retained for cross-artifact comparability. The
fp16-accumulate change passed a paired quality A/B with no detectable
difference on MMLU-Pro / HumanEval+ / MBPP+.

## Quick start

```sh
git clone https://github.com/<you>/llama-paw
cd llama-paw

# NVIDIA (requires the CUDA toolkit)
cmake -B build -DGGML_CUDA=ON
# AMD / Intel / other (requires the Vulkan SDK, incl. glslc)
cmake -B build -DGGML_VULKAN=ON

cmake --build build --config Release -j
```

```sh
# interactive chat
./build/bin/llama-cli -m PAW-35B-A3B.gguf

# single-turn
./build/bin/llama-cli -m PAW-35B-A3B.gguf -st -p "your prompt"
```

GPU offload is automatic in GPU builds (no `-ngl` flag needed).

For the tuned server configuration — speculative drafter, KV quantization and
the codec kernel flags — see `SERVING.md` in the weights repo. Serving PAW
naively leaves about a third of its throughput on the table.

### Serving at 256k context - optimum settings

The PAW-27B-X3 3.5-bit weights plus the Q2_K DFlash2 drafter fit the full
`-c 262144` on one 24 GB RTX 3090 only with quantized KV on target and draft
and a capped draft ubatch. Measured optimum:

```sh
export GGML_PAW_X3_GEMV=2 GGML_PAW_MMQ_HEAD=1 GGML_PAW_GREEDY_IDS=1 GGML_PAW_DQ4=0

./build/bin/llama-server -m PAW-27B-X3-3.5bit.gguf \
  -md Qwen3.8-27B-DFlash2-Q2_K.gguf \
  --spec-type draft-dflash --spec-draft-n-max 5 \
  -fa on -ctk q4_0 -ctv q4_0 \
  -c 262144 -ub 2048 -b 8192 -ubd 256 \
  -ngl 99 -np 1 --no-warmup --reasoning off
```

- `-ctk/-ctv q4_0` on target and draft is the configuration that fits at
  262144 with speculative decoding; `-ubd 256` caps the draft ubatch.
- `-ub 2048 -b 8192`: measured 425.8 tok/s prefill at a 229k-token prompt in
  spec mode; AR decode ~16.8 tok/s at 242k after the FA GQA batching fix.
- At context depths up to ~160k, `-ub 4096` is faster than `-ub 2048` (930 vs
  903 tok/s PP8192 before the fp16-accumulate change) and passes the VRAM
  gate there; at 262144 it does not fit, so keep `-ub 2048` for the full
  context profile.
- The drafter's KV is ~50 MiB at 262144 (5 sliding-window-2048 layers).
- If you are tight on VRAM, lower `-ub` first - it is the footprint lever.
- At short context the same stack is verified at 100.45 tok/s median on an
  8k-token code workload with `-c 20480 -b 512 -ub 512`.
- A full-context speculative-decode tok/s figure is not yet measured and is
  not quoted; round arithmetic at 242k gives ~2.7-2.8 tokens/round.

## Notes

- **Requantization is not supported.** The weights are already packed code
  streams; `llama-quantize` refuses PAW checkpoints by design.
- **Serve with thinking disabled.** The published checkpoint has a documented
  runaway `<think>` loop. Use `--chat-template-kwargs '{"enable_thinking":false}'`.
- Kernel optimizations are opt-in via `GGML_PAW_*` environment variables and
  are verified byte-identical to the reference path.
