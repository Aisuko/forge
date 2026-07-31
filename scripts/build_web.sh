#!/usr/bin/env bash
# Build the wasm bundle (Stage 11): wasm32 crate + JS bindings.
#
#   ./scripts/build_web.sh [OUT_DIR]     # default: docs/dist/forge
#
# The output is a build artifact, not source: it goes straight into the site's
# dist/ instead of being committed, so it cannot go stale relative to src/.
# Normally you want ./scripts/build_site.sh, which calls this.
#
# Requires: rustup target add wasm32-unknown-unknown
#           cargo install wasm-bindgen-cli --version <wasm-bindgen crate version>
set -euo pipefail
cd "$(dirname "$0")/.."

out="${1:-docs/dist/forge}"

# `council` is off by default in Cargo.toml — it is research composed on top of
# the runtime, not part of it, so a crate that depends on forge-ml should not pay
# for it. The site is the one build that wants it: docs/src/council.html drives
# the WasmCouncil bindings, which only exist with this flag.
cargo build --release --target wasm32-unknown-unknown --features council
mkdir -p "$out"
wasm-bindgen --target web --out-dir "$out" \
    target/wasm32-unknown-unknown/release/forge.wasm
echo "$out updated:"
ls -la "$out"
