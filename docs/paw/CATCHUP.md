# Catching llama-paw up with upstream llama.cpp

PAW is an overlay on llama.cpp. This is how to move it onto a newer upstream without losing
customizations, written 2026-09-18 while doing exactly that to pick up `qwen4exp`.

Companion tool: `scripts/paw/paw-overlay.sh`.

---

## 1. This repo has no shared history with upstream

    git merge-base HEAD upstream/master   ->   rc=1, nothing

- Our history is 71 commits rooted at `868134cc7 "llama-paw: initial import"` (2026-08-21).
- That import is a **squashed snapshot** of llama.cpp, not a fork point. Upstream's root is a
  different commit entirely.
- `paw/upstream` is a bare pin on upstream's history; **no PAW branch has ever merged it**.

So `git rebase upstream/master` cannot work, and `scripts/paw/paw-catchup.sh` (which merges
the pin) would be an unrelated-histories merge. Catchup here means: **start from upstream,
re-apply the overlay.**

A number worth keeping in perspective: `git rev-list --count HEAD..upstream/master` reported
11,028, but that counts all of upstream's history because the graphs are disjoint. Real drift
from our import point to 2026-09-17 was **461 commits**.

---

## 2. The overlay, and why the split matters

| class | files | patch lines | conflicts |
|-------|------:|------------:|-----------|
| **new files** - upstream has no such path | 21 | 19,530 | **never** |
| **edits into upstream files** | 35 | 3,452 | 63 hunks |

~85% of PAW by volume is new files (`ggml/src/ggml-cuda/paw-*.cu`, `paw.cuh`, `src/models/paw*.cpp`,
`tools/paw-parity`, the paw tests, the Vulkan `paw_*.comp` shaders). Those cost nothing to move.

**The 3,452 lines of edits into upstream files are the entire maintenance cost.** Driving that
number down is the only thing that makes future catchups cheaper. `paw-overlay.sh check` ranks
where that code lives:

    44  ggml/src/ggml.c
    42  ggml/src/ggml-cuda/ggml-cuda.cu
    35  ggml/src/ggml-cpu/ggml-cpu.c
    34  ggml/include/ggml.h
    24  ggml/src/ggml-cpu/ops.cpp

Much of the top of that list is irreducible - op enum entries and dispatch switches have to
live in upstream files. But it is measured now, and `check` will say if it starts growing.

---

## 3. Procedure

    # 0. back up uncommitted work first; this is destructive to dirty trees
    git diff > backup/uncommitted.patch
    for f in $(git ls-files --others --exclude-standard); do cp --parents "$f" backup/untracked/; done

    # 1. isolated worktree, so the working tree is never at risk
    git fetch upstream
    git worktree add -b paw-on-upstream /path/to/port upstream/master

    # 2. file list from CODE, not commits (see TRAP 1)
    scripts/paw/paw-overlay.sh list

    # 3. new files: copy them, never patch them
    git checkout <paw-branch> -- <new files>

    # 4. modified files: 3-way apply
    git diff <base>..<paw-branch> -- <modified files> > mod.patch
    git apply -3 mod.patch

    # 5. build with the RIGHT nvcc (see TRAP 2)
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc

---

## 4. Traps, all of which cost real time here

### TRAP 1: the commit diff is not the overlay

The initial import commit **already contains PAW code**, so `git diff <import>..HEAD` omits it
silently. Three things were lost this way, each found only by compiling:

- the eight `ggml_compute_forward_paw_*` declarations in `ggml/src/ggml-cpu/ops.h`
- `#include "ggml-cuda/paw.cuh"` in `ggml-cuda.cu`
- the entire Vulkan PAW backend (12 `paw_*.comp` shaders, `ggml-vulkan.cpp`, both CMakeLists)

Derive the file set from content instead - that is what `paw-overlay.sh list` does. It finds
**76 files** against 57 from the commit diff. Keep the pattern strict; a loose one (`m1_`)
matches unrelated code in the cann, riscv, hexagon, metal and opencl backends.

### TRAP 2: CMake picks the wrong nvcc

A configure without `-DCMAKE_CUDA_COMPILER` selected `/usr/bin/nvcc`, which on this box is
Ubuntu's distro package at **CUDA 11.5**, not the real toolkit at `/usr/local/cuda-12.6`.
`cudaKernel_t` does not exist in 11.5, so **1,690 errors** appeared in pure upstream files
(`argsort.cu`, `arange.cu`) and looked exactly like "upstream needs newer CUDA". It does not.

Signature: errors only in `*.cudafe1.stub.c`, never at a real source line, concentrated in
files with no local changes.

### TRAP 3: "keep both" duplicates what upstream also added

Most both-sides hunks resolve as keep-both. But where upstream independently added the *same*
thing PAW did, keep-both emits it twice and the compiler reports a redefinition far from the
merge. Here: the DFlash KV set (`LLM_KV_DFLASH_BLOCK_SIZE` + 4 siblings) in `llama-arch.h`,
its rows in `llama-arch.cpp`, and the `dflash_*` members in `llama-hparams.h` - **1,220
cascading errors from one header**, and the `llama-hparams.h` copy came from a hunk that
applied *cleanly*.

Check after any keep-both pass:

    grep -oE '^\s+(LLM_KV|LLM_ARCH|LLM_TENSOR|GGML_OP)_[A-Z0-9_]+' src/llama-arch.h \
      | tr -d ' ' | sort | uniq -d

Dedupe **only within one table**. The `LLM_KV_` name table has one row per key; the per-arch
tensor lists legitimately repeat `LLM_TENSOR_*`, and deduping those strips tensors from every
arch after the first.

### TRAP 4: never dedupe by line when merging code

Resolving a hunk by "keep both, minus lines the other side already has" **deletes bare `}`
lines**, because a closing brace appears verbatim on both sides. That silently removed block
terminators from `speculative.cpp` (+2 unbalanced), `server-context.cpp` (+8) and
`dflash.cpp` (+1), producing errors hundreds of lines from the damage.

Dedupe identifiers, never lines. And verify structurally after every merge pass:

    python3 -c "s=open(F).read(); print(s.count('{')-s.count('}'))"

Both sides balance at 0; any non-zero result is damage you introduced.

### TRAP 5: block order matters when a side ends mid-construct

If upstream's side of a hunk ends on an open construct - a struct body, a constructor
initializer list - appending PAW's side after it nests PAW's code inside upstream's. That
happened twice: `llama_model_paw` ended up declared *inside* `llama_model_qwen35moe`, and a
PAW constructor signature was left orphaned with no body. Check brace balance and
`grep -n '^struct '` after merging headers.

### Smaller gotchas

- `git apply -3 <whole patch>` fails **atomically** with `does not exist in index` for every
  new file and rolls the entire apply back. Split new files out first.
- `git apply ... | head -5; echo rc=$?` reports **head's** status. The apply had failed
  silently and the new files were simply missing.
- `paw.cu` appears in the cumulative diff but no longer exists in HEAD (split into `paw-*.cu`).
  Filter any file list through `git cat-file -e <branch>:<path>`.

---

## 5. Conflict taxonomy

Of 63 hunks in this catchup:

| shape | count | resolution |
|-------|------:|------------|
| `ours` empty | 20 | take theirs - a PAW insertion whose context shifted |
| `theirs` empty | 5 | take ours - upstream content our patch predates |
| both non-empty | 38 | usually keep both; watch TRAP 3 and TRAP 5 |

`GGML_OP_COUNT` conflicts on every catchup. Do not guess it - count the merged enum:

    sed -n '<enum start>,<GGML_OP_COUNT line>p' ggml/include/ggml.h \
      | grep -cE '^\s+GGML_OP_[A-Z0-9_]+,'

Upstream 101 + 13 PAW ops = 114.

---

## 6. Gate before trusting a catchup

Build, then:

    ./build/bin/test-paw-x3-moe                       # per-expert mixed-K MoE
    GGML_PAW_X3_MOE_FUSED_ROWS=0 ./build/bin/test-paw-x3-moe   # fallback path
    ./build/bin/test-paw-x3-mm-id                     # mm_id, K=1..4
    ./build/bin/test-paw-codec <fixtures.gguf>

Note what the first three do *not* prove: they run on random trellis and compare llama-paw
against llama-paw. Only `bonsai-pilot/scripts/flashnext_x3_phase0b_run.sh` compares llama-paw
against the EXL3 encoder on real archive bytes, which is what catches a Hadamard-convention or
suh/svh ordering mismatch.

---

## 7. State of the 2026-09-18 catchup

Upstream `972d2313b`. All 63 hunks resolved, plus the three TRAP 1 omissions.

**Carried across and building:** every PAW CUDA kernel, the ggml op set (13 ops),
`llama-arch`/`llama-model` registration, the Vulkan PAW backend, `tools/paw-parity`, all paw
tests.

**NOT carried across - open TODO:** `common/speculative.cpp` and `src/models/dflash.cpp` are
currently **upstream's versions**. Their PAW customizations (DFlash2 selector plumbing, the
`GGML_PAW_SPEC_TIME` phase instrumentation, `GGML_DFLASH2_BLOCK_SIZE_OVERRIDE`) were damaged by
the TRAP 4 dedupe and were reverted rather than shipped silently corrupted. The exact delta is
saved at `flashnext/rebase_backup/TODO_paw_dflash_spec.patch` and must be re-applied by hand,
hunk by hunk, with brace balance checked after each.

Upstream has meanwhile absorbed part of the DFlash work itself (the `LLM_KV_DFLASH_*` keys and
`dflash_*` hparams are upstream's now), so some of that patch is already redundant - which is
why it needs a person, not a script.

---

## 8. KNOWN REGRESSION: test-paw-x3-moe aborts on the ported tree

`./build/bin/test-paw-x3-moe` aborts before running any case:

    ggml-cuda.cu:115: GGML_ASSERT(device >= 0 && device < info.device_count) failed
    ggml_cuda_set_device -> ggml_cuda_op_paw_x3_moe+0x7d

What is established, so the next person does not repeat it:

- `test-paw-x3-mm-id` **passes all 14 cases** (K=1..4, both projection shapes), so the trellis
  kernels, per-expert K dispatch and the mm_id path are fine. The fault is specific to the
  fused `ggml_cuda_op_paw_x3_moe`.
- Instrumenting the op shows `ctx.device=0`, `curr_stream_no=0`, `device_count=1` at entry -
  all valid - and the abort happens on the very next statement, `ctx.stream()`.
- The device value reaching `ggml_cuda_get_physical_device` is garbage and **differs every
  run** (892764160, -762277888, -1123610624), i.e. uninitialized memory, not a stale constant.
- It is **not** an ODR/layout mismatch: `sizeof(ggml_backend_cuda_context)` is 4400 in both the
  paw-x3.cu and ggml-cuda.cu translation units, with `device` at offset 0 and
  `curr_stream_no` at 3248 in both.
- Not caused by device masking: fails identically under `CUDA_VISIBLE_DEVICES=0`, `0,2`, and
  unmasked.
- The dispatch site and `paw.cuh` signature match the pre-port versions.

That combination - valid `device`, valid `curr_stream_no`, matching layout, yet a garbage
argument arriving at `set_device` - is not explicable by reading the source, and print
bisection has been exhausted. Next step is gdb with a breakpoint on
`ggml_cuda_get_physical_device`, checking the actual call site and whether the `ctx` reference
is still the object the caller passed.

Note the op is dispatched from code that is part of the **uncommitted** PAW work
(`GGML_OP_PAW_X3_MOE` did not exist in any commit), so it has had far less exposure than the
committed paths. Upstream also added concurrent-stream support (`curr_stream_no`,
`stream_context`) in the 461-commit window, which is the most likely area of interaction.

Until this is fixed, the fused MoE path must be considered unverified on the ported tree. The
per-matrix path it falls back to (`paw_x3_mm_id`) is verified.
