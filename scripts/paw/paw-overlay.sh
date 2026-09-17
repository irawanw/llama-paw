#!/usr/bin/env bash
# Treat PAW as an overlay on upstream llama.cpp, and make catchup mechanical.
#
# Why this exists alongside paw-catchup.sh. That script assumes a shared history with upstream
# and merges a pin. This repo has no shared history: it is a squashed snapshot import with PAW
# commits on top (git merge-base HEAD upstream/master returns nothing). So catchup here means
# "start from upstream, re-apply the overlay", and the overlay has to be identified by content.
#
# The trap this script exists to prevent: the initial import commit ALREADY CONTAINS PAW code,
# so `git diff <import>..HEAD` is not the overlay and silently omits things. On 2026-09-18 it
# omitted ops.h's eight paw forward declarations, ggml-cuda.cu's #include "ggml-cuda/paw.cuh",
# and the whole Vulkan PAW backend - each one a build break found only by compiling. The file
# set must come from the code, never from the commit graph.
#
# Usage:
#   scripts/paw/paw-overlay.sh list            # the overlay file set, classified
#   scripts/paw/paw-overlay.sh export <dir>    # new files + a patch for modified ones
#   scripts/paw/paw-overlay.sh check           # warn about overlay code in upstream files
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REF="${PAW_REF:-HEAD}"
UPSTREAM="${PAW_UPSTREAM:-upstream/master}"

# Strict on purpose. A loose pattern (m1_) matches unrelated code in the cann, riscv, hexagon,
# metal and opencl backends and buries the real overlay in noise.
PAT='ggml_paw|LLM_ARCH_PAW|LLM_ARCH_MACH1|GGML_OP_PAW|ggml_compute_forward_paw|ggml_cuda_op_paw|GGML_PAW_|paw\.cuh|PAW_X3'
EXCL=(':!vendor' ':!.github' ':!tools/ui')

overlay_files() {
    git grep -l -E "$PAT" "$REF" -- "${EXCL[@]}" | sed "s|^${REF}:||" | sort -u
}

classify() {
    local f
    while read -r f; do
        if git cat-file -e "${UPSTREAM}:${f}" 2>/dev/null; then
            echo "MOD $f"
        else
            echo "NEW $f"
        fi
    done
}

case "${1:-list}" in
list)
    overlay_files | classify | sort
    n_new=$(overlay_files | classify | grep -c '^NEW' || true)
    n_mod=$(overlay_files | classify | grep -c '^MOD' || true)
    echo
    echo "NEW (upstream has no such path, never conflicts): $n_new"
    echo "MOD (edits into upstream files, the burden):      $n_mod"
    ;;

export)
    out="${2:?usage: paw-overlay.sh export <dir>}"
    mkdir -p "$out"
    overlay_files | classify > "$out/manifest.txt"
    awk '$1=="NEW"{print $2}' "$out/manifest.txt" > "$out/new_files.txt"
    awk '$1=="MOD"{print $2}' "$out/manifest.txt" > "$out/mod_files.txt"
    # New files are copied, never patched: git apply -3 aborts atomically when any path lacks
    # an index blob upstream, which takes the whole apply down with it.
    tar -cf "$out/new_files.tar" -T "$out/new_files.txt"
    # Modified files still need a base to diff against; the caller supplies it.
    if [ -n "${PAW_BASE:-}" ]; then
        git diff "$PAW_BASE".."$REF" -- $(tr '\n' ' ' < "$out/mod_files.txt") > "$out/mod.patch"
        echo "wrote $out/mod.patch (base $PAW_BASE)"
    else
        echo "set PAW_BASE=<upstream sha the tree was cut from> to also emit mod.patch"
    fi
    echo "wrote $out/{manifest.txt,new_files.tar}"
    ;;

check)
    # Every MOD file is a place a future catchup can conflict. Rank them, loudest first:
    # the goal is to move this code into NEW files until the list is short.
    echo "overlay code living in upstream files (each one is future merge cost):"
    overlay_files | classify | awk '$1=="MOD"{print $2}' | while read -r f; do
        n=$(git grep -c -E "$PAT" "$REF" -- "$f" | cut -d: -f3)
        printf "%6s  %s\n" "$n" "$f"
    done | sort -rn
    ;;

*)
    echo "usage: $0 {list|export <dir>|check}" >&2
    exit 2
    ;;
esac
