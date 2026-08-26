# llama-paw

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs **PAW**
models — checkpoints whose weights ship as packed trellis code streams and are
decoded on the fly inside the compute kernels.

Get the weights from [PAW-35B-A3B](https://huggingface.co/lackonendes/PAW-35B-A3B-GGUF).

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

## Notes

- **Requantization is not supported.** The weights are already packed code
  streams; `llama-quantize` refuses PAW checkpoints by design.
- **Serve with thinking disabled.** The published checkpoint has a documented
  runaway `<think>` loop. Use `--chat-template-kwargs '{"enable_thinking":false}'`.
- Kernel optimizations are opt-in via `GGML_PAW_*` environment variables and
  are verified byte-identical to the reference path.
