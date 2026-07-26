#!/usr/bin/env bash
# Point git at the repo's committed hooks. Idempotent; safe to re-run.
#
# The hooks live in .githooks/ rather than .git/hooks/ so they are version
# controlled and shared. git never picks that directory up on its own, so
# every clone has to set core.hooksPath once — that is all this script does.
set -euo pipefail
cd "$(dirname "$0")/.."

git config core.hooksPath .githooks
chmod +x .githooks/*

echo "core.hooksPath -> .githooks"
echo "  pre-commit   ./scripts/ci_local.sh fast   fmt, clippy, wasm + forge-top builds"
echo "  pre-push     ./scripts/ci_local.sh full   the above + cargo test --release"
echo
echo "bypass either with --no-verify; undo with 'git config --unset core.hooksPath'"
