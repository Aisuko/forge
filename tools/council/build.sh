#!/usr/bin/env bash
#   ./tools/council/build.sh              # full build, including the wasm bundle
#   ./tools/council/build.sh --no-wasm    # skip wasm
#   python3 -m http.server -d tools/council/dist 8080
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
here="$root/tools/council"
cd "$root"

DIST="$here/dist"
WITH_WASM=1
case "${1:-}" in
  --no-wasm) WITH_WASM=0 ;;
  "") ;;
  *) echo "unknown flag ${1} (use --no-wasm)" >&2; exit 2 ;;
esac

rm -rf "$DIST"
mkdir -p "$DIST/assets"

echo "== css"
# shellcheck source=../shared/tailwind.sh
source "$root/tools/shared/tailwind.sh"
"$TW" -i tools/shared/input.css -o "$DIST/assets/app.css" --minify

echo "== html + js"
cp "$here"/web/council.html "$here"/web/council.js "$DIST/"
cp "$root/tools/shared/favicon.svg" "$DIST/"

if [[ $WITH_WASM -eq 1 ]]; then
  echo "== wasm"
  cargo build --release --locked --target wasm32-unknown-unknown -p forge-council
  wasm-bindgen --target web --out-dir "$DIST/forge" \
      target/wasm32-unknown-unknown/release/forge_council.wasm
else
  echo "warning: --no-wasm, so the page will 404 on ./forge/forge_council.js" >&2
fi

if [[ -d "$here/assets" ]]; then
  mkdir -p "$DIST/council"
  cp -r "$here"/assets/. "$DIST/council/"
else
  echo "warning: tools/council/assets/ missing — build it with" >&2
  echo "         ./tools/council/scripts/train_council.sh && ./tools/council/scripts/ship_council.sh" >&2
fi

echo
python3 "$root/tools/shared/check_site.py" "$DIST"
echo "serve: python3 -m http.server -d $DIST 8080"
