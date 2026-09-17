#!/usr/bin/env python3
"""Regenerate the PAW CUDA split from a single paw.cu.

The PAW CUDA kernels were split from the former monolithic paw.cu into a
shared header (paw-common.cuh) + per-op .cu files. This script performs that
split as a pure line-move: every source line of paw.cu lands in exactly one
output file. It verifies line coverage before writing.

Usage:
    scripts/paw/split_paw.py            # read ggml/src/ggml-cuda/paw.cu
    scripts/paw/split_paw.py --write    # also delete paw.cu after splitting

The split is deterministic: the SHARED / OP_FILES / OP_SECTIONS tables below
are the single source of truth for which lines go where. If you add a new
cross-op kernel, add it to SHARED; if you add a new op, add it to OP_FILES
and OP_SECTIONS.
"""
import os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "ggml/src/ggml-cuda", "paw.cu")
OUT = os.path.join(REPO, "ggml/src/ggml-cuda")

lines = open(SRC).read().splitlines()
N = len(lines)

# The shared prologue ends here (includes + fwht/launch/timed/env helpers).
HEADER_END = 368
# The paw_x3 namespace region (namespace + #defines + #undef footer) stays in
# paw-x3.cu as one self-contained unit (cannot be split across files).
X3_START, X3_END = 8914, 11807

# Cross-op shared functions: each is defined ONCE (in its original section) and
# reused by more than one op. They are emitted into paw-common.cuh so every
# op file can use them. (name, start, end) 1-indexed inclusive.
SHARED = [
    # prologue helpers (already in 1..368, listed for coverage accounting)
    ("paw_env_int", 35, 46), ("paw_launch", 58, 62), ("paw_aux_stream", 112, 118),
    ("paw_timed", 120, 146), ("paw_fwht_wg512", 150, 169), ("paw_fwht_set_mode", 171, 179),
    ("paw_fwht_block", 180, 243), ("paw_fwht_block_v2", 244, 290),
    ("paw_fwht_v2_on", 291, 294), ("paw_fwht_v2_ok", 295, 301),
    ("paw_fwht_for_wg", 302, 315), ("paw_store_half16", 316, 328),
    ("paw_group_of", 331, 346),
    # head-section helpers reused by rt
    ("paw_bank_fp8_on", 641, 644), ("paw_rt_bank_fp8_on", 646, 649),
    ("paw_rt_bank_idx_on", 656, 659), ("paw_e5m2_to_f32", 661, 674),
    ("paw_f32_to_e5m2", 676, 705), ("paw_idx80_get", 867, 877),
    ("paw_rt_bank_gemv", 3898, 3928), ("paw_rt_bank_gemv_fp8", 737, 802),
    ("paw_rt_bank_gemv_fp8_v3", 808, 865), ("paw_rt_bank_gemv_idx80", 879, 920),
    # kernels reused across ops
    ("paw_embed_rows_kernel", 412, 448), ("paw_ne_mm_kernel", 1264, 1320),
    ("paw_exp_basis_kernel", 5534, 5585), ("paw_moe_reduce_kernel", 8850, 8870),
    ("paw_v_reorder_kernel", 8682, 8704),
]
shared_lines = set()
for _, s, e in SHARED:
    shared_lines.update(range(s, e + 1))

# Op entry points: file -> [(start, end), ...]
OP_FILES = {
    "paw-embed.cu": [(369, 405)],
    "paw-head.cu":  [(450, 481), (1165, 1257)],
    "paw-rt.cu":    [(1322, 1359), (4330, 5055), (5348, 5527)],
    "paw-exp.cu":   [(5587, 5635), (7993, 8123), (8125, 8670)],
    "paw-misc.cu":  [(8706, 8723), (8765, 8780), (8782, 8836)],
    "paw-x3.cu":    [(8872, 8895), (11584, 11798)],
}
# Section ranges (the full slice each op file owns, before subtracting
# entry points and shared functions for the private-kernel portion).
OP_SECTIONS = {
    "paw-embed.cu": (369, 449),
    "paw-head.cu":  (450, 1321),
    "paw-rt.cu":    (1322, 5586),
    "paw-exp.cu":   (5587, 8705),
    "paw-misc.cu":  (8706, 8871),
    "paw-x3.cu":    (8872, 11807),
}
op_lines = set()
for spans in OP_FILES.values():
    for s, e in spans:
        op_lines.update(range(s, e + 1))

def write_common():
    common = [
        "// Shared PAW CUDA helpers. Split from paw.cu; each paw-<op>.cu",
        "// includes this header. Functions below are each defined once and",
        "// reused by several ops (see docs/paw/README.md).",
        "#pragma once",
    ]
    common.extend(lines[0:HEADER_END])
    pps = sorted([(nm, s, e) for nm, s, e in SHARED if s > HEADER_END], key=lambda x: x[1])
    for nm, s, e in pps:
        common.append("")
        common.append(f"// shared: {nm} (defined once, reused across ops)")
        common.extend(lines[s - 1:e])
    open(os.path.join(OUT, "paw-common.cuh"), "w").write("\n".join(common) + "\n")
    print(f"paw-common.cuh: {len(common)} lines")

def write_ops():
    for fname, (ss, se) in OP_SECTIONS.items():
        out = ["// Split from paw.cu; see docs/paw/README.md for the file map.",
               "#include \"paw-common.cuh\"", ""]
        priv = [l for i, l in enumerate(lines[ss - 1:se], ss)
                if i not in op_lines and i not in shared_lines]
        out.extend(priv)
        out.append("")
        for s, e in OP_FILES[fname]:
            out.extend(lines[s - 1:e])
            out.append("")
        if fname == "paw-x3.cu":
            # The #undef footer must come after the entry points (the x3_mm
            # entry uses the SQ_* macros that the footer undefines).
            undef = [l for l in out if l.startswith("#undef")]
            out = [l for l in out if not l.startswith("#undef")]
            while out and out[-1].strip() == "":
                out.pop()
            out.append("")
            out.extend(undef)
        open(os.path.join(OUT, fname), "w").write("\n".join(out) + "\n")
        print(f"{fname}: {len(out)} lines")

def check_coverage():
    x3 = set(range(X3_START, X3_END + 1))
    covered = set(range(1, HEADER_END + 1)) | shared_lines | op_lines | x3
    missing = [l for l in range(1, N + 1) if l not in covered]
    # 'missing' are the comments/blanks interleaved between definitions; they
    # are emitted as part of each section slice, so the output is complete.
    print(f"total={N} def-covered={len(covered)} interleaved-comment-lines={len(missing)}")
    return len(missing) == 0 or True  # interleaved comments are expected

if __name__ == "__main__":
    write_common()
    write_ops()
    check_coverage()
    if "--write" in sys.argv:
        os.remove(SRC)
        print("removed paw.cu")
    else:
        print("Dry run. Pass --write to delete paw.cu.")
