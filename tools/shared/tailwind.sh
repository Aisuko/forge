#!/usr/bin/env bash
#   source tools/shared/tailwind.sh     # sets $TW to the executable
set -uo pipefail

TAILWIND_VERSION="${TAILWIND_VERSION:-4.3.3}"
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TW="${TAILWIND_BIN:-$_root/.cache/tailwindcss-${TAILWIND_VERSION}}"

if [[ ! -x "$TW" ]]; then
  mkdir -p "$(dirname "$TW")"
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  _asset="tailwindcss-linux-x64" ;;
    Linux-aarch64) _asset="tailwindcss-linux-arm64" ;;
    Darwin-arm64)  _asset="tailwindcss-macos-arm64" ;;
    Darwin-x86_64) _asset="tailwindcss-macos-x64" ;;
    *) echo "unsupported platform for the Tailwind standalone CLI" >&2; return 1 2>/dev/null || exit 1 ;;
  esac
  echo "downloading tailwindcss v${TAILWIND_VERSION} (${_asset})"
  curl -sSfL -o "$TW" \
    "https://github.com/tailwindlabs/tailwindcss/releases/download/v${TAILWIND_VERSION}/${_asset}"
  chmod +x "$TW"
fi

export TW
