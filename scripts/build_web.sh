#!/usr/bin/env bash
# Build a wasm bundle: the wasm32 crate plus its JS bindings.
#
#   ./scripts/build_web.sh [OUT_DIR] [PACKAGE]   # default: target/web forge-ml
#
# The output is a build artifact, never committed, so it cannot go stale.
#
# Requires: rustup target add wasm32-unknown-unknown
#           cargo install wasm-bindgen-cli --version <the wasm-bindgen in Cargo.lock>
set -euo pipefail
cd "$(dirname "$0")/.."

out="${1:-target/web}"
pkg="${2:-forge-ml}"
# cargo names the artifact after [lib], not the package: forge-ml -> forge.wasm.
lib="${pkg//-/_}"
[[ "$pkg" == "forge-ml" ]] && lib="forge"

cargo build --release --locked --target wasm32-unknown-unknown -p "$pkg"
mkdir -p "$out"
wasm-bindgen --target web --out-dir "$out" \
    "target/wasm32-unknown-unknown/release/${lib}.wasm"
echo "   $pkg -> $out ($(du -sh "$out" | cut -f1))"
