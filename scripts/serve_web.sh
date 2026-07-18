#!/usr/bin/env bash
# Serve the browser demo (Stage 11) from the repo root so the page can fetch
# /models/gpt2/* over HTTP. Open http://localhost:8000/web/ in Chrome/Edge.
set -euo pipefail
cd "$(dirname "$0")/.."
port="${1:-8000}"
echo "http://localhost:${port}/web/"
python3 -m http.server "$port" --bind 0.0.0.0
