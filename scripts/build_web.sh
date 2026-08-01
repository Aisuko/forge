#!/usr/bin/env bash
# Build the wasm bundle (Stage 11): wasm32 crate + JS bindings.
#
#   ./scripts/build_web.sh [OUT_DIR]     # default: target/web
#
# The output is a build artifact, not source: nothing is committed, so it cannot
# go stale relative to src/.
#
# Requires: rustup target add wasm32-unknown-unknown
#           cargo install wasm-bindgen-cli --version <wasm-bindgen crate version>
set -euo pipefail
cd "$(dirname "$0")/.."

out="${1:-target/web}"

cargo build --release --locked --target wasm32-unknown-unknown -p forge-ml
mkdir -p "$out"
wasm-bindgen --target web --out-dir "$out" \
    target/wasm32-unknown-unknown/release/forge.wasm
echo "$out updated:"
ls -la "$out"
