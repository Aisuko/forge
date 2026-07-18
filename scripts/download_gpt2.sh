#!/usr/bin/env bash
# Download GPT-2 (124M) weights + tokenizer files from the HuggingFace hub
# into models/gpt2/. Reads HF_TOKEN from .env if present (gpt2 is public, so
# the token is optional).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

mkdir -p models/gpt2
for f in model.safetensors vocab.json merges.txt config.json; do
  if [ ! -f "models/gpt2/$f" ]; then
    echo "downloading $f ..."
    curl -sL --fail ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
      -o "models/gpt2/$f" \
      "https://huggingface.co/openai-community/gpt2/resolve/main/$f"
  fi
done
ls -la models/gpt2/
