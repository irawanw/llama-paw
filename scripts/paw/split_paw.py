#!/usr/bin/env python3
# Split ggml/src/ggml-cuda/paw.cu into per-op .cu files + a shared header.
# Pure line-moves: every original line lands in exactly one output file.
# Run from repo root. Reads paw.cu, writes paw-common.cuh + paw-<op>.cu,
# and deletes paw.cu. Verifies line coverage before deleting.
import os, sys

SRC = "ggml/src/ggml-cuda/paw.cu"
lines = open(SRC).read().splitlines()
N = len(lines)

# The 15 shared host functions all live in the prologue (lines 1..368) and are
# used by several op files, so they stay in the common header (defined once).
# The 9 rt-section host helpers (paw_launch_rt_apply_mma, paw_rt_walk_qtip_*,
# paw_rt_dense_decode_rate_launch*, paw_bank_cache_on, paw_rt_bank_get,
# paw_rt_idx_fp16_bank, paw_rt_idx_bytes) are used only by the rt section, so
# they stay in paw-rt.cu and need no exposure.
HEADER_END = 368   # lines 1..368 = shared prologue (includes the 15 shared fns)
FOOTER_START = 11808  # lines 11808..N = #undef block after the paw_x3 namespace

# Section slices (1-indexed, inclusive). Contiguous, no overlap.
SECTIONS = [
    ("paw-embed.cu",      369, 449),    # embed_gather
    ("paw-embed-rows.cu", 450, 1164),   # embed_rows (+ head bank cache + head kernels)
    ("paw-head.cu",       1165, 1321),  # head_mm
    ("paw-rt.cu",         1322, 5586),  # ne_mm + rt walk/apply/bank/quant + rt_mm + rt_mm_batch
    ("paw-exp.cu",        5587, 8705),  # exp_basis + exp walk/apply/slots + exp_mm_batch2 + exp_mm
    ("paw-misc.cu",       8706, 8871),  # v_reorder + dual_mm + supported
    ("paw-x3.cu",         8872, 11807), # moe_reduce + paw_x3 namespace + x3_mm
]

def emit_header():
    out = [
        "// Shared prologue for the PAW CUDA ops. Split from paw.cu; each",
        "// paw-<op>.cu includes this header. The 15 shared host helpers below",
        "// are defined once here and used by several op files.",
        "#pragma once",
        "#include \"common.cuh\"",
        "#include \"paw.cuh\"",
        "#include \"cp-async.cuh\"",
        "#include <cstring>",
        "#include <mma.h>",
        "#include <cooperative_groups.h>",
        "#include <cuda_pipeline.h>",
        "",
    ]
    out.extend(lines[0:HEADER_END])
    return out

def emit_section(s, e):
    out = [
        "// Split from paw.cu; see docs/paw/README.md for the file map.",
        "#include \"paw-common.cuh\"",
        "",
    ]
    out.extend(lines[s - 1:e])
    return out

outputs = {"paw-common.cuh": emit_header()}
for fname, s, e in SECTIONS:
    outputs[fname] = emit_section(s, e)

# Coverage check: prologue + all sections + footer must equal the whole file.
accounted = set(range(1, HEADER_END + 1))
for _, s, e in SECTIONS:
    accounted.update(range(s, e + 1))
accounted.update(range(FOOTER_START, N + 1))
missing = [l for l in range(1, N + 1) if l not in accounted]
# overlap check
all_lines = []
for _, s, e in SECTIONS:
    all_lines.extend(range(s, e + 1))
dupes = len(all_lines) - len(set(all_lines))

print(f"Total lines: {N}")
print(f"Accounted: {len(accounted)}  missing: {len(missing)}  section-overlap: {dupes}")
if missing:
    print("Missing sample:", [(l, lines[l-1][:50]) for l in missing[:10]])
if dupes:
    print("ERROR: sections overlap")
    sys.exit(1)
if missing:
    print("ERROR: lines not accounted for")
    sys.exit(1)

if "--write" in sys.argv:
    d = "ggml/src/ggml-cuda"
    for fname, content in outputs.items():
        open(os.path.join(d, fname), "w").write("\n".join(content) + "\n")
        print(f"  wrote {fname} ({len(content)} lines)")
    os.remove(SRC)
    print("removed paw.cu")
else:
    print("Dry run OK. Pass --write to write files.")
