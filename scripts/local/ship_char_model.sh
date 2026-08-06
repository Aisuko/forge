#!/usr/bin/env bash
# Promote a trained char-level checkpoint into assets/shakespeare_char/ —
# the tracked artifact the website and TUI both load.
#
#   ./scripts/local/ship_char_model.sh [checkpoints/shakespeare_char.safetensors]
#
# Training keeps f32 safetensors; shipping quantizes to `.fzm` q4. The site
# downloads what is in this directory, and 6.7 MB against 43 MB is the whole
# reason the format exists — the held-out loss measured below is measured on
# the quantized file, so the number beside the weights is the number a visitor
# actually gets.
#
# Unlike models/ and checkpoints/ (both gitignored), this directory is in git:
# GitHub Pages serves the weights directly. Do NOT move it to Git LFS — Pages
# serves LFS pointer files as text, which would silently break the demo.
set -euo pipefail
cd "$(dirname "$0")/../.."

SRC="${1:-checkpoints/shakespeare_char.safetensors}"
DEST="assets/shakespeare_char"
stem="$(basename "$SRC" .safetensors)"
dir="$(dirname "$SRC")"

for f in "$SRC" "$dir/$stem.config.json" "$dir/$stem.vocab.json"; do
  [[ -f "$f" ]] || { echo "missing $f — train it first:" >&2
                     echo "  cargo run --release --features train --example train_shakespeare -- --backend wgpu" >&2
                     exit 1; }
done

mkdir -p "$DEST"
cp "$dir/$stem.config.json" "$DEST/config.json"
cp "$dir/$stem.vocab.json" "$DEST/vocab.json"

echo "quantizing to .fzm q4..."
cargo run --release --quiet --example to_fzm -- \
  --config "$DEST/config.json" --in "$SRC" --out "$DEST/model.fzm"

# A previous ship left an f32 file here; leaving it behind would double the
# repo's weight and give checkpoint_in_dir() two answers.
rm -f "$DEST/model.safetensors"

# Measure what is being shipped, and record it beside the weights. Whatever
# loads this checkpoint gets the held-out loss with it — including
# tools/surprise/build.sh, which copies the whole directory into its dist/.
# Measured here rather than typed by
# anyone: a hard-coded quality figure is exactly what this file exists to
# replace, and the previous shipped model's loss was recorded nowhere at all.
TRAINER=target/release/examples/train_shakespeare
if [[ -x "$TRAINER" && -f data/tinyshakespeare.txt ]]; then
  echo "measuring held-out loss..."
  losses=$("$TRAINER" --backend "${BACKEND:-wgpu}" --eval-only --eval-windows 512 \
             --checkpoint "$DEST/model.fzm" 2>/dev/null | grep '^{') || true
  if [[ -n "${losses:-}" ]]; then
    val=$(sed 's/.*"val":\([0-9.]*\).*/\1/' <<<"$losses")
    train=$(sed 's/.*"train":\([0-9.]*\).*/\1/' <<<"$losses")
    cat > "$DEST/metrics.json" <<JSON
{
  "val_loss": $val,
  "train_loss": $train,
  "val_windows": 512,
  "measured": "$(date -u +%Y-%m-%d)",
  "format": "fzm-q4",
  "source": "$SRC",
  "reference": "nanoGPT train_shakespeare_char, val 1.4697"
}
JSON
    echo "  val $val, train $train — wrote $DEST/metrics.json"
  else
    echo "  note: could not measure; leaving metrics.json alone" >&2
  fi
else
  echo "  note: no release trainer built, skipping metrics.json" >&2
fi

size=$(stat -c%s "$DEST/model.fzm")
mb=$((size / 1000000))
echo "shipped $DEST"
ls -la "$DEST"

# 100 MB is GitHub's hard per-file limit; 50 MB triggers a push warning.
if (( size > 100000000 )); then
  echo "error: model.fzm is ${mb} MB, over GitHub's 100 MB hard limit" >&2
  exit 1
elif (( size > 50000000 )); then
  echo "note: ${mb} MB is over GitHub's 50 MB soft limit — expect a push warning" >&2
fi
