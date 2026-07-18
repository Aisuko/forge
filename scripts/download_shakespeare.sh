#!/usr/bin/env bash
# Fetch the Tiny Shakespeare corpus (~1.1 MB) for the Stage 10 training gate.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ -s data/tinyshakespeare.txt ]; then
    echo "data/tinyshakespeare.txt already present"
    exit 0
fi
URL=https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -fL "$URL" -o data/tinyshakespeare.txt
wc -c data/tinyshakespeare.txt
