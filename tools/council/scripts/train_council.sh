#!/usr/bin/env bash
# Train the council: one shared ancestor, then four experts branched from it.
#
#   ./tools/council/scripts/train_council.sh           # everything
#   ./tools/council/scripts/train_council.sh --branch  # experts only (ancestor already trained)
#
# The ancestor sees the whole corpus. Each expert resumes from the ancestor's
# best checkpoint and is fine-tuned on one contiguous quarter with **wte and wpe
# frozen** — that freeze is the entire reason the four experts' hidden states
# can later be added together. Unfreeze it and the council page is a lie.
#
# Runs from the repository root whatever directory it is invoked from: the
# trainer is the runtime's own example, and the corpus and checkpoints live at
# the root, not under this tool.
set -euo pipefail
cd "$(dirname "$0")/../../.."

DIR=checkpoints/council
SHAPE=(--layers 4 --heads 4 --embd 128)
COMMON=(--backend wgpu --accum 16 --dropout 0.2 --sample-every 0 "${SHAPE[@]}")
BRANCH_ONLY=0
case "${1:-}" in
  --branch) BRANCH_ONLY=1 ;;
  "") ;;
  *) echo "unknown flag ${1} (use --branch)" >&2; exit 2 ;;
esac

mkdir -p "$DIR" logs
[[ -f data/council/manifest.json ]] || ./tools/council/scripts/split_corpus.py

if [[ $BRANCH_ONLY -eq 0 ]]; then
  echo "== ancestor (full corpus, 3000 steps)"
  cargo run --release -p forge-ml --features train --example train_shakespeare -- \
    "${COMMON[@]}" --steps 3000 --eval-every 250 \
    --checkpoint "$DIR/ancestor.safetensors" | tee logs/council_ancestor.log
fi

ANCESTOR="$DIR/ancestor.best.safetensors"
[[ -f "$ANCESTOR" ]] || { echo "missing $ANCESTOR — train the ancestor first" >&2; exit 1; }

N=$(python3 -c 'import json;print(json.load(open("data/council/manifest.json"))["n_experts"])')
for k in $(seq 0 $((N - 1))); do
  label=$(python3 -c "import json;print(json.load(open('data/council/manifest.json'))['experts'][$k]['label'])")
  echo "== expert $k — $label"
  # --resume reads the checkpoint it will also write, so branch by copying.
  cp "$ANCESTOR" "$DIR/expert$k.safetensors"
  cargo run --release -p forge-ml --features train --example train_shakespeare -- \
    "${COMMON[@]}" --resume --freeze-embeddings \
    --data "data/council/$k.txt" \
    --steps 800 --lr 3e-4 --no-cosine --warmup 50 --eval-every 100 \
    --checkpoint "$DIR/expert$k.safetensors" | tee "logs/council_expert$k.log"
done

echo
echo "done. next: cargo run --release -p forge-council --example council_demo"
