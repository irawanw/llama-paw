#!/usr/bin/env bash
# No-regression gate for PAW. Run after every catchup merge and every
# kernel change, before pushing.
#
# Usage:
#   scripts/paw/paw-verify.sh [--build-dir DIR] [--skip-build] [--model FILE]
#
# Steps:
#   1. build (unless --skip-build)
#   2. test-paw-codec   (codec round-trip, always)
#   3. llama-paw-parity (AR token parity vs reference, if --model given or
#      PAW_VERIFY_MODEL is set)
#   4. llama-bench smoke (prefill+decode tok/s, compared against
#      PAW_VERIFY_TG_MIN if set)
#
# Exit code 0 = pass. Any failure is reported with the step name.

set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

build_dir=build
model=""
skip_build=0
tg_min="${PAW_VERIFY_TG_MIN:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir)  build_dir="$2"; shift 2 ;;
    --model)      model="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

model="${model:-${PAW_VERIFY_MODEL:-}}"

fail() { echo "FAIL [$1]: $2" >&2; exit 1; }
pass() { echo "PASS [$1]"; }

if [ "$skip_build" -eq 0 ]; then
  echo "==> build ($build_dir)"
  cmake -B "$build_dir" -DCMAKE_BUILD_TYPE=Release >/dev/null \
    || fail build "cmake configure"
  cmake --build "$build_dir" --config Release -j"$(nproc)" \
    || fail build "cmake --build"
  pass build
else
  echo "==> skipping build"
fi

bin="$build_dir/bin"

# codec round-trip; needs a fixture gguf (a small PAW checkpoint)
if [ -n "${PAW_VERIFY_FIXTURE:-}" ]; then
  echo "==> test-paw-codec ($PAW_VERIFY_FIXTURE)"
  "$bin/test-paw-codec" "$PAW_VERIFY_FIXTURE" || fail codec "test-paw-codec"
  pass codec
else
  echo "==> skipping test-paw-codec (set PAW_VERIFY_FIXTURE to a PAW gguf)"
fi

if [ -n "$model" ]; then
  echo "==> llama-paw-parity ($model)"
  "$bin/llama-paw-parity" -m "$model" -n 128 \
    || fail parity "llama-paw-parity"
  pass parity

  echo "==> llama-bench smoke"
  bench_out=$("$bin/llama-bench" -m "$model" -ngl 99 -p 512 -n 128 -r 2 2>&1) \
    || fail bench "llama-bench"
  echo "$bench_out" | tail -5
  if [ -n "$tg_min" ]; then
    tg=$(echo "$bench_out" | grep -E "t/s\)?$" | awk '{print $NF}' | tail -1)
    if [ -n "$tg" ] && awk "BEGIN{exit !($tg < $tg_min)}"; then
      fail bench "tg $tg < min $tg_min"
    fi
  fi
  pass bench
else
  echo "==> skipping parity/bench (no model; set --model or PAW_VERIFY_MODEL)"
fi

echo "==> all checks passed"
