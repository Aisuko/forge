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
  # forge-council/, not forge/: in the composed site forge/ is the core bundle.
  ./scripts/common/build_web.sh "$DIST/forge-council" forge-council
else
  echo "warning: --no-wasm, so the page will 404 on ./forge-council/forge_council.js" >&2
fi

if [[ -d "$here/assets" ]]; then
  mkdir -p "$DIST/council"
  cp -r "$here"/assets/. "$DIST/council/"
else
  echo "warning: tools/council/assets/ missing — build it with" >&2
  echo "         ./tools/council/scripts/train_council.sh && ./tools/council/scripts/ship_council.sh" >&2
fi

echo
if [[ $WITH_WASM -eq 1 ]]; then
  python3 "$root/tools/shared/check_site.py" "$DIST"
else
  echo "skipping check_site.py: --no-wasm leaves the module graph unresolvable" >&2
fi
echo "serve: python3 -m http.server -d $DIST 8080"
