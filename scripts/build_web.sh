#!/usr/bin/env bash
# Build the browser demo (Stage 11): wasm32 crate + JS bindings into web/forge.
# Requires: rustup target add wasm32-unknown-unknown
#           cargo install wasm-bindgen-cli --version <wasm-bindgen crate version>
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/forge \
    target/wasm32-unknown-unknown/release/forge.wasm
echo "web/forge/ updated:"
ls -la web/forge/
