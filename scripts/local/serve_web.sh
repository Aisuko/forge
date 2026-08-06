#!/usr/bin/env bash
# Serve the built site locally, exactly as GitHub Pages will.
#
#   ./scripts/common/build_site.sh && ./scripts/local/serve_web.sh [PORT]
#
# Plain static files: Forge needs no COOP/COEP headers, because `rayon` is
# native-only and the browser path is single-threaded. Open in Chrome/Edge for
# WebGPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
port="${1:-8000}"
dist="docs/dist"
[[ -f "$dist/index.html" ]] || { echo "run ./scripts/common/build_site.sh first" >&2; exit 1; }
echo "http://localhost:${port}/"
python3 -m http.server "$port" --bind 0.0.0.0 -d "$dist"
