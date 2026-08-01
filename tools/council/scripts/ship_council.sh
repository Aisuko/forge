#!/usr/bin/env bash
# Promote the trained council into tools/council/assets/ — the tracked artifact
# the council page loads.
#
#   ./tools/council/scripts/ship_council.sh [checkpoints/council]
#
# Ships each expert's **best** checkpoint (best on its own quarter's held-out
# text), one shared config.json and one shared vocab.json. There is deliberately
# only one of each: the experts share a vocabulary and a shape, and shipping
# four copies would let them drift apart silently.
# Run from the repository root: the checkpoints and the corpus manifest live
# there, and only the destination belongs to this tool.
set -euo pipefail
cd "$(dirname "$0")/../../.."

SRC="${1:-checkpoints/council}"
DEST="tools/council/assets"
MANIFEST="data/council/manifest.json"

[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST — run ./tools/council/scripts/split_corpus.py" >&2; exit 1; }
N=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["n_experts"])' "$MANIFEST")

for f in "$SRC/expert0.config.json" "$SRC/expert0.vocab.json"; do
  [[ -f "$f" ]] || { echo "missing $f — run ./tools/council/scripts/train_council.sh" >&2; exit 1; }
done

rm -rf "$DEST"
mkdir -p "$DEST"
cp "$SRC/expert0.config.json" "$DEST/config.json"
cp "$SRC/expert0.vocab.json" "$DEST/vocab.json"

for ((k = 0; k < N; k++)); do
  best="$SRC/expert$k.best.safetensors"
  [[ -f "$best" ]] || { echo "missing $best — run ./tools/council/scripts/train_council.sh" >&2; exit 1; }
  cp "$best" "$DEST/expert$k.safetensors"
done

# The page's manifest: labels come from the corpus split, val losses from each
# expert's own training log. Measured, never typed — the same rule as
# ship_char_model.sh.
python3 - "$MANIFEST" "$SRC" "$DEST" "$N" <<'PY'
import json, os, sys

manifest, src, dest, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
labels = [e["label"] for e in json.load(open(manifest))["experts"]]

experts = []
for k in range(n):
    entry = {"file": f"expert{k}.safetensors", "label": labels[k]}
    log = os.path.join(src, f"expert{k}.metrics.jsonl")
    if os.path.exists(log):
        rows = [json.loads(l) for l in open(log) if l.strip()]
        if rows:
            entry["val_loss"] = round(min(r["val"] for r in rows), 4)
    entry["bytes"] = os.path.getsize(os.path.join(dest, entry["file"]))
    experts.append(entry)

out = {
    "n_experts": n,
    "experts": experts,
    "note": "Branched from one ancestor, fine-tuned with wte/wpe frozen so the "
            "experts share an embedding basis and a wte-tied decoder.",
}
json.dump(out, open(os.path.join(dest, "manifest.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
PY

echo
du -sh "$DEST"
echo "shipped $N experts to $DEST/"
