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
#   scripts/paw/paw-overlay.sh constants [ref] # overlay edits that carry no PAW identifier
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

constants)
    # The blind spot in every grep-based detector above: an overlay edit that mentions no PAW
    # identifier at all. PAW raises upstream's own GGML_MAX_SRC from 10 to 16 because the x3 MoE
    # op takes 15 sources. That line matches no pattern here, so a catchup takes upstream's value
    # without a conflict and without a warning - and the damage is silent. src[10..14] becomes
    # out of bounds, which is UB, so the compiler deleted the whole body of
    # ggml_cuda_op_paw_x3_moe (190 bytes for 228 lines) and ggml.c wrote past the end of
    # ggml_tensor. -Wno-array-bounds is in upstream's CUDA flags, so nothing was reported. See
    # docs/paw/CATCHUP.md TRAP 6.
    #
    # So: compare the VALUE of every plain #define in the shared files, and make each difference
    # an explicit decision.
    #   constants            - values PAW overrides upstream on; eyeball that each is intended
    #   constants <prev-ref> - after a catchup, report overrides that <prev-ref> had and we lost
    defines() {  # <ref> <file> -> "NAME VALUE" for simple numeric/plain object macros
        git show "${1}:${2}" 2>/dev/null \
            | grep -E '^[[:space:]]*#define[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]+[^([:space:]]' \
            | sed -E 's|//.*||; s|/\*.*||; s/^[[:space:]]*#define[[:space:]]+//; s/[[:space:]]+/ /g; s/ $//'
    }
    # rows where ref defines the same macro as upstream but with a different value
    diffs() {
        local ref="$1" f
        git grep -l -E "$PAT" "$ref" -- "${EXCL[@]}" | sed "s|^${ref}:||" | sort -u \
          | while read -r f; do
            git cat-file -e "${UPSTREAM}:${f}" 2>/dev/null || continue   # MOD files only
            comm -23 <(defines "$ref" "$f" | sort -u) <(defines "$UPSTREAM" "$f" | sort -u) \
              | while read -r name val; do
                    u=$(defines "$UPSTREAM" "$f" | awk -v n="$name" '$1==n{sub(/^[^ ]+ /,""); print; exit}')
                    [ -z "$u" ] && continue          # PAW-only macro, not an override
                    echo "$f|$name|$val|$u"
                done
        done
    }
    prev="${2:-}"
    if [ -z "$prev" ]; then
        echo "upstream constants PAW deliberately overrides (confirm each survives a catchup):"
        diffs "$REF" | awk -F'|' '{printf "  %-34s %-28s paw=%-8s upstream=%s\n", $1, $2, $3, $4}'
    else
        echo "overrides in $prev that $REF does not have. Each needs a decision, not a revert:"
        echo "  - a value PAW raised for its own reasons (GGML_MAX_SRC) must be restored"
        echo "  - a version upstream bumped (LLAMA_SESSION_VERSION) is correctly taken from upstream"
        lost=0
        diffs "$prev" > /tmp/paw_ov_prev.$$ || true
        diffs "$REF"  > /tmp/paw_ov_cur.$$  || true
        while IFS='|' read -r f name val u; do
            if ! grep -q "^${f}|${name}|" /tmp/paw_ov_cur.$$; then
                printf "  LOST %-30s %-26s was %s, now upstream's %s\n" "$f" "$name" "$val" "$u"
                lost=1
            fi
        done < /tmp/paw_ov_prev.$$
        rm -f /tmp/paw_ov_prev.$$ /tmp/paw_ov_cur.$$
        [ "$lost" = 0 ] && echo "  none - every override in $prev is still present"
    fi
    ;;

*)
    echo "usage: $0 {list|export <dir>|check|constants [prev-ref]}" >&2
    exit 2
    ;;
esac
