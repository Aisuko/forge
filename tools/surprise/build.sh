#!/usr/bin/env bash
#   ./tools/surprise/build.sh              # full build, including the wasm bundle
#   ./tools/surprise/build.sh --no-wasm    # skip wasm
#   python3 -m http.server -d tools/surprise/dist 8081
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
here="$root/tools/surprise"
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
cp "$here"/web/react.html "$here"/web/react.js "$DIST/"
cp "$root/tools/shared/favicon.svg" "$DIST/"

if [[ $WITH_WASM -eq 1 ]]; then
  echo "== wasm"
  ./scripts/common/build_web.sh "$DIST/forge"
else
  echo "warning: --no-wasm, so the page will 404 on ./forge/forge.js" >&2
fi

if [[ -d assets/shakespeare_char ]]; then
  mkdir -p "$DIST/model"
  cp -r assets/shakespeare_char/. "$DIST/model/"
else
  echo "warning: assets/shakespeare_char/ missing — train it with" >&2
  echo "         cargo run --release --features train --example train_shakespeare -- --backend wgpu" >&2
fi

echo
if [[ $WITH_WASM -eq 1 ]]; then
  python3 "$root/tools/shared/check_site.py" "$DIST"
else
  echo "skipping check_site.py: --no-wasm leaves the module graph unresolvable" >&2
fi
echo "serve: python3 -m http.server -d $DIST 8081"
