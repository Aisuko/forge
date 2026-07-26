#!/usr/bin/env bash
# Promote a trained char-level checkpoint into assets/shakespeare_char/ —
# the tracked artifact the website and TUI both load.
#
#   ./scripts/ship_char_model.sh [checkpoints/shakespeare_char.safetensors]
#
# Unlike models/ and checkpoints/ (both gitignored), this directory is in git:
# GitHub Pages serves the weights directly. Do NOT move it to Git LFS — Pages
# serves LFS pointer files as text, which would silently break the demo.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-checkpoints/shakespeare_char.safetensors}"
DEST="assets/shakespeare_char"
stem="$(basename "$SRC" .safetensors)"
dir="$(dirname "$SRC")"

for f in "$SRC" "$dir/$stem.config.json" "$dir/$stem.vocab.json"; do
  [[ -f "$f" ]] || { echo "missing $f — train it first:" >&2
                     echo "  cargo run --release --example train_shakespeare -- --backend wgpu" >&2
                     exit 1; }
done

mkdir -p "$DEST"
cp "$SRC" "$DEST/model.safetensors"
cp "$dir/$stem.config.json" "$DEST/config.json"
cp "$dir/$stem.vocab.json" "$DEST/vocab.json"

size=$(stat -c%s "$DEST/model.safetensors")
mb=$((size / 1000000))
echo "shipped $DEST"
ls -la "$DEST"

# 100 MB is GitHub's hard per-file limit; 50 MB triggers a push warning.
if (( size > 100000000 )); then
  echo "error: model.safetensors is ${mb} MB, over GitHub's 100 MB hard limit" >&2
  exit 1
elif (( size > 50000000 )); then
  echo "note: ${mb} MB is over GitHub's 50 MB soft limit — expect a push warning" >&2
fi
