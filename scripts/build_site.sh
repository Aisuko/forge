#!/usr/bin/env bash
# Build the explainer site into docs/dist/.
#
#   ./scripts/build_site.sh              # full build, including the wasm bundle
#   ./scripts/build_site.sh --no-wasm    # skip wasm (fast; the demo will 404)
#
# Then:  python3 -m http.server -d docs/dist 8080
#
# The wasm bundle is built from src/ every time rather than copied from a
# committed artifact, so it cannot go stale. The same steps run in
# .github/workflows/pages.yml. Everything is relative to the artifact root
# because the site is served from /forge/, not /.
set -euo pipefail
cd "$(dirname "$0")/.."

TAILWIND_VERSION="4.3.3"
DIST="docs/dist"
WITH_WASM=1
case "${1:-}" in
  --no-wasm) WITH_WASM=0 ;;
  # Accepted for compatibility; building the wasm is now the default.
  --with-wasm|"") ;;
  *) echo "unknown flag ${1} (use --no-wasm)" >&2; exit 2 ;;
esac

# ── Tailwind standalone CLI: no Node, no package.json, matching a Rust repo.
TW="${TAILWIND_BIN:-.cache/tailwindcss-${TAILWIND_VERSION}}"
if [[ ! -x "$TW" ]]; then
  mkdir -p "$(dirname "$TW")"
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  asset="tailwindcss-linux-x64" ;;
    Linux-aarch64) asset="tailwindcss-linux-arm64" ;;
    Darwin-arm64)  asset="tailwindcss-macos-arm64" ;;
    Darwin-x86_64) asset="tailwindcss-macos-x64" ;;
    *) echo "unsupported platform for the Tailwind standalone CLI" >&2; exit 1 ;;
  esac
  echo "downloading tailwindcss v${TAILWIND_VERSION} ($asset)"
  # Pinned, never 'latest': an upstream release must not be able to break the site.
  curl -sSfL -o "$TW" \
    "https://github.com/tailwindlabs/tailwindcss/releases/download/v${TAILWIND_VERSION}/${asset}"
  chmod +x "$TW"
fi

rm -rf "$DIST"
mkdir -p "$DIST/assets"

echo "== css"
"$TW" -i docs/src/input.css -o "$DIST/assets/app.css" --minify

echo "== html + js"
cp docs/src/index.html docs/src/scene.js docs/src/demo.js docs/src/explainer.js "$DIST/"
cp -r docs/vendor "$DIST/vendor"
cp -r docs/static/. "$DIST/"

# The kernel inventory is generated from shaders/ so the page cannot drift
# from the actual kernel set.
python3 scripts/gen_kernels.py "$DIST/index.html"

if [[ $WITH_WASM -eq 1 ]]; then
  echo "== wasm"
  # Straight into dist/ — there is no committed copy to fall out of date.
  ./scripts/build_web.sh "$DIST/forge"
else
  echo "warning: --no-wasm, so the demo will 404 on ./forge/forge.js" >&2
fi

# Stage 11 gate fixture: the browser compares its own greedy tokens against
# these native-WGPU ids when the page is opened with ?gate.
if [[ -f tests/data/gate_expected.json ]]; then
  cp tests/data/gate_expected.json "$DIST/gate_expected.json"
fi

if [[ -d assets/shakespeare_char ]]; then
  mkdir -p "$DIST/model"
  cp assets/shakespeare_char/. "$DIST/model/" -r
else
  echo "warning: assets/shakespeare_char/ missing — train it with" >&2
  echo "         cargo run --release --example train_shakespeare -- --backend wgpu" >&2
fi

echo
python3 scripts/check_site.py "$DIST" || true
echo "serve: python3 -m http.server -d $DIST 8080"
