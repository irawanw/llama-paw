# PAW fork: structure, catchup, and kernel workflow

This fork runs PAW models: checkpoints whose weights ship as packed trellis
code streams and are decoded on the fly inside the compute kernels. It
descends from llama.cpp (and from llama.cpp-mach1 for the codec design).
This document explains where the custom code lives, how to catch up with
upstream, and how to change kernels without regressing.

## Branch layout

| branch        | purpose                                                          |
|---------------|------------------------------------------------------------------|
| `master`      | mainline. All new work lands here.                               |
| `main`        | mirror of `master` (kept in sync; `master` is the source of truth) |
| `paw/base`    | snapshot of the fork before the refactoring (tag `paw/pre-refactor-*`) |
| `paw/upstream`| the pinned upstream commit we are currently based on. Never edit. |
| `paw/<topic>` | short-lived feature branches, merged into `master`               |

The fork's history is rewritten (squashed import), so there is no shared
merge-base with upstream. That is why catchup works through the pin:
`paw/upstream` is the *only* thing that moves when upstream moves, and every
catchup is a plain merge of that pin into `master`.

## Catching up with upstream

```sh
scripts/paw/paw-catchup.sh            # merge latest upstream/master
scripts/paw/paw-catchup.sh <sha>      # or pin to a specific upstream commit
```

The script fetches upstream, advances `paw/upstream`, and merges it into the
current branch. On conflicts:

1. `scripts/paw/paw-custom-files.sh --stat` lists every file that differs
   between the pin and the tree. That list is the entire working set;
   anything not in it is pure upstream and should be taken as-is.
2. For each conflicted file, the question is always: "is this hunk a PAW
   customization or an upstream change?" PAW customizations are recognizable:
   `paw`/`Paw`/`PAW`/`mach1`/`GGML_PAW_*`/`dflash` symbols, or code in the
   file map below. Keep the PAW hunk, rebase it onto the upstream version.
3. `git add -A && git commit --no-edit`, then run the gate:
   `scripts/paw/paw-verify.sh`.

Do not rebase `master` onto upstream; merge only. Rebase would rewrite the
fork history again and defeat the pin.

## No-regression gate

```sh
scripts/paw/paw-verify.sh                 # build + codec + parity + bench
scripts/paw/paw-verify.sh --skip-build    # after an incremental build
PAW_VERIFY_MODEL=/path/model.gguf scripts/paw/paw-verify.sh
PAW_VERIFY_FIXTURE=/path/fixtures.gguf scripts/paw/paw-verify.sh
PAW_VERIFY_TG_MIN=40 scripts/paw/paw-verify.sh   # enforce a tok/s floor
```

- `test-paw-codec` is bit-exact and needs the exporter fixture
  (`PAW_VERIFY_FIXTURE`); it is skipped when unset.
- `llama-paw-parity` runs the shared AR protocol (same prompt IDs, warmup,
  argmax policy) and emits `PARITY_JSON:`; compare against the reference
  engine before/after a kernel change.
- `llama-bench` smoke: prefill 512 / decode 128, 2 runs. Set
  `PAW_VERIFY_TG_MIN` to fail on a throughput drop.

Run the gate after every catchup merge and after every kernel change, before
pushing.

## Where the custom code lives

New files (never conflict with upstream renames):

| path | what |
|------|------|
| `src/models/paw.cpp`, `src/models/paw-dense.cpp` | PAW and PAW-dense model arch (graph build, codec tensor wiring, greedy-ids sidecar) |
| `ggml/src/ggml-cuda/paw.cu`, `paw.cuh` | all PAW CUDA kernels (trellis decode, WS-mma apply, GEMV, head) |
| `ggml/src/ggml-cuda/fattn-dq4.cu/.cuh` | direct-q4 verify attention kernel (env-gated `GGML_PAW_DQ4`) |
| `ggml/src/ggml-vulkan/vulkan-shaders/paw_*.comp` | Vulkan codec shaders |
| `tools/paw-parity/paw-parity.cpp` | AR parity driver (JSON protocol) |
| `tests/test-paw-codec.cpp` | bit-exact codec gate |
| `tests/get-model.cpp/.h` | test model helper |

Modified upstream files (these are the ones that can conflict on catchup):

| path | PAW hooks |
|------|-----------|
| `src/models/models.h` | `LLM_ARCH_PAW`, `LLM_ARCH_PAW_DENSE`, `LLM_TENSOR_PAW_*`, `LLM_KV_PAW_RHT_BLOCK` |
| `src/llama-*.cpp` | model dispatch for PAW archs, greedy-ids buffer, buft overrides for packed sidecars |
| `common/speculative.cpp/.h` | DFlash2 draft type (`draft-dflash`), `GGML_PAW_SPEC_TIME` timing |
| `common/params.cpp/.h` | `--spec-draft-batch-size/-ubd` and related flags |
| `tools/server/*` | speculative-draft wiring, slot/KV quantization for 256k context |
| `ggml/src/ggml-cuda/*.cu` (non-paw) | dispatch widened to K=1/4, small-k top-k, graph-builder assert |

`vendor/`, `tools/ui/`, and `.github/` are never customized; take upstream
versions unmodified.

## Kernel change convention

1. One branch per kernel change: `paw/<kernel>-<what>`.
2. New behavior is env-gated (`GGML_PAW_*`) and defaults off, until measured
   positive; then flip the default in a separate commit.
3. Record the measurement in the commit message (workload, tok/s before/after,
   quality gate). A kernel that cannot be measured is not merged.
4. Before merging: `scripts/paw/paw-verify.sh` must pass, and `paw-parity`
   output must match the reference bit-for-bit (or the diff is explained).
5. Keep the kernel self-contained in `paw.cu` where possible; every touch of
   a shared upstream file (dispatch tables, asserts) is a future merge cost.

## Env flags

The full set lives in `ggml/src/ggml-cuda/paw.cu`, `fattn-dq4.cu`,
`src/` and `common/`; list them with:

```sh
grep -rhoE 'GGML_PAW_[A-Z0-9_]+|GGML_DFLASH2_[A-Z0-9_]+|PAW_DBG_RS' \
  ggml/src/ggml-cuda/paw.cu ggml/src/ggml-cuda/fattn-dq4.cu src/ common/ | sort -u
```

Other docs in this directory:

- `sglang-port-report.md` — session report from the PAW-35B SGLang port
  (weight conventions, GDN kernel semantics, open divergence at T>15).

Serving-relevant flags (see the measured recipe in `README.md`,
"Serving at 256k context"):

| flag | effect |
|------|--------|
| `GGML_PAW_X3_GEMV` | x3 small-m GEMV path (2 = EXL3 port) |
| `GGML_PAW_MMQ_HEAD` | tensor-core trellis GEMM for the head, nt >= 3 |
| `GGML_PAW_GREEDY_IDS` | model graph emits per-row argmax ids (verify path) |
| `GGML_PAW_DQ4` | direct-q4 verify attention kernel |
| `GGML_PAW_SPEC_TIME` | per-phase wall clock for the drafter lane |
| `GGML_DFLASH2_BLOCK_SIZE_OVERRIDE` | DFlash2 block size override |
| `PAW_DBG_RS` | GDN state I/O dump for batch diagnosis |
