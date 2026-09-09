#!/usr/bin/env bash
# Print the exact set of files that differ between the pinned upstream
# (branch paw/upstream) and the current tree, excluding vendor/, tools/ui/
# and .github/. This is the working set for any catchup conflict resolution:
# every file listed here is either a PAW customization or an upstream change
# that must be reconciled.
#
# Usage:
#   scripts/paw/paw-custom-files.sh          # one column, path only
#   scripts/paw/paw-custom-files.sh --stat   # with per-file change stats

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PIN=paw/upstream

if [ "${1:-}" = "--stat" ]; then
  git diff --stat "${PIN}" HEAD | grep -vE "(^|/)(vendor|tools/ui|\.github)/"
else
  git diff --name-only "${PIN}" HEAD | grep -vE "(^|/)(vendor|tools/ui|\.github)/"
fi
