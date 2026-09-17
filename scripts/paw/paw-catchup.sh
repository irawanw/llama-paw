#!/usr/bin/env bash
# Advance the pinned upstream and merge it into the mainline.
#
# Usage:
#   scripts/paw/paw-catchup.sh [new-upstream-sha-or-ref]
#
# With no argument, upstream/master is used. The pin (branch paw/upstream)
# is advanced to the target, then merged into the current branch.
#
# After a successful merge, run scripts/paw/paw-verify.sh before pushing.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PIN=paw/upstream
UPSTREAM_REMOTE=upstream

target="${1:-master}"

echo "==> fetching ${UPSTREAM_REMOTE}"
git fetch "${UPSTREAM_REMOTE}"

resolved=$(git rev-parse --verify "${UPSTREAM_REMOTE}/${target}^{commit}")
echo "==> target upstream: ${resolved} ($(git log -1 --format=%s "${resolved}" | cut -c1-72))"

current=$(git rev-parse --abbrev-ref HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is dirty; commit or stash first" >&2
  exit 1
fi

echo "==> advancing pin ${PIN} -> ${resolved}"
git branch -f "${PIN}" "${resolved}"

echo "==> merging ${PIN} into ${current}"
if git merge "${PIN}" --no-edit -m "paw: catchup upstream to ${resolved:0:12}"; then
  echo "==> merge clean"
else
  echo "==> conflicts; resolve them, then:"
  echo "    git add -A && git commit --no-edit"
  echo "    scripts/paw/paw-verify.sh"
  exit 1
fi

echo "==> regenerating the custom-file manifest"
scripts/paw/paw-custom-files.sh

echo "==> done. Next: scripts/paw/paw-verify.sh, then push."
