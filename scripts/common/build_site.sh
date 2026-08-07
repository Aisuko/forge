#!/usr/bin/env bash
# Compose the site into docs/dist/: the landing page and both tool pages, as
# one artifact. This is what .github/workflows/pages.yml deploys.
#
#   ./scripts/common/build_site.sh              # full build, including all three wasm bundles
#   ./scripts/common/build_site.sh --no-wasm    # skip wasm; the pages will 404
#   ./scripts/local/serve_web.sh                # then serve it
#
# Each page's source stays where it is owned — docs/src/ for the landing page,
# tools/*/web/ for the tools — and is copied here, never duplicated. Each tool
# ships its own bundle, because a #[wasm_bindgen] export is a GC root the
# linker cannot drop; the 6.7 MB checkpoint is the part worth sharing, and
# index.html and react.html read one copy of it.
#
# Everything is relative to the artifact root: the site is served from /forge/.
set -euo pipefail
cd "$(dirname "$0")/../.."

DIST="docs/dist"
WITH_WASM=1
case "${1:-}" in
  --no-wasm) WITH_WASM=0 ;;
  "") ;;
  *) echo "unknown flag ${1} (use --no-wasm)" >&2; exit 2 ;;
esac

rm -rf "$DIST"
mkdir -p "$DIST/assets"

echo "== css"
# shellcheck source=../tools/shared/tailwind.sh
source tools/shared/tailwind.sh
"$TW" -i tools/shared/input.css -o "$DIST/assets/app.css" --minify

echo "== pages"
cp docs/src/index.html docs/src/demo.js docs/src/attention.js docs/src/decision.js "$DIST/"
cp tools/council/web/council.html tools/council/web/council.js "$DIST/"
cp tools/surprise/web/react.html tools/surprise/web/react.js "$DIST/"
cp tools/shared/favicon.svg "$DIST/"
cp -r docs/static/. "$DIST/"

# The kernel inventory is generated from shaders/, so the page cannot drift
# from the actual kernel set.
python3 scripts/common/gen_kernels.py "$DIST/index.html"

if [[ $WITH_WASM -eq 1 ]]; then
  echo "== wasm"
  ./scripts/common/build_web.sh "$DIST/forge" forge-ml
  ./scripts/common/build_web.sh "$DIST/forge-council" forge-council
  ./scripts/common/build_web.sh "$DIST/forge-surprise" forge-surprise
else
  echo "warning: --no-wasm, so every page will 404 on its bundle" >&2
fi

echo "== weights"
# The browser gate fixture: index.html?gate compares its own greedy tokens
# against these native-WGPU ids.
[[ -f tests/data/gate_expected.json ]] && cp tests/data/gate_expected.json "$DIST/"

# index.html and react.html share one copy of the char model.
if [[ -d assets/shakespeare_char ]]; then
  mkdir -p "$DIST/model"
  cp -r assets/shakespeare_char/. "$DIST/model/"
else
  echo "warning: assets/shakespeare_char/ missing — train it with" >&2
  echo "         cargo run --release --features train --example train_shakespeare" >&2
fi

if [[ -d tools/council/assets ]]; then
  mkdir -p "$DIST/council"
  cp -r tools/council/assets/. "$DIST/council/"
else
  echo "warning: tools/council/assets/ missing — build it with" >&2
  echo "         ./tools/council/scripts/train_council.sh && ./tools/council/scripts/ship_council.sh" >&2
fi

echo
if [[ $WITH_WASM -eq 1 ]]; then
  # --strict: the deployed artifact ships all three pages, so a link between
  # them that does not resolve is an error here, though it is only a warning
  # for a standalone tool build.
  python3 tools/shared/check_site.py "$DIST" --strict
else
  echo "skipping check_site.py: --no-wasm leaves the module graph unresolvable" >&2
fi
echo "serve: ./scripts/local/serve_web.sh"
