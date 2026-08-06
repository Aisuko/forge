#!/usr/bin/env bash
# Install the wasm-bindgen CLI pinned to the wasm-bindgen in Cargo.lock.
# Idempotent; safe to re-run.
#
# The CLI and the crate must be the same version or the bindings it emits will
# not match the wasm it is reading, so the version is derived from the lockfile
# rather than written down anywhere. scripts/common/build_web.sh needs this,
# which means `make site` and the tail of `ci_local.sh fast` do too.
set -euo pipefail
cd "$(dirname "$0")/../.."

v=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | grep '^version' | cut -d'"' -f2)
[[ -n "$v" ]] || { echo "no wasm-bindgen in Cargo.lock" >&2; exit 1; }

if [[ "$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')" == "$v" ]]; then
  echo "wasm-bindgen ${v} already installed"
  exit 0
fi

# The prebuilt tarball is seconds; the cargo build is minutes. Only fall back
# when there is no release asset for this platform.
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  _asset="x86_64-unknown-linux-musl" ;;
  Linux-aarch64) _asset="aarch64-unknown-linux-gnu" ;;
  Darwin-arm64)  _asset="aarch64-apple-darwin" ;;
  Darwin-x86_64) _asset="x86_64-apple-darwin" ;;
  *)             _asset="" ;;
esac

_url="https://github.com/wasm-bindgen/wasm-bindgen/releases/download/${v}/wasm-bindgen-${v}-${_asset}.tar.gz"
if [[ -n "$_asset" ]] && curl -sSfL "$_url" \
     | sudo tar -xz --strip-components=1 -C /usr/local/bin \
         "wasm-bindgen-${v}-${_asset}/wasm-bindgen" 2>/dev/null; then
  echo "wasm-bindgen ${v} -> /usr/local/bin"
else
  echo "no prebuilt wasm-bindgen ${v} for this platform, building from source"
  cargo install wasm-bindgen-cli --version "$v" --locked
fi
