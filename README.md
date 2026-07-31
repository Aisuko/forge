# Forge

[![CI](https://github.com/Aisuko/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Aisuko/forge/actions/workflows/ci.yml)
[![Release](https://github.com/Aisuko/forge/actions/workflows/release.yml/badge.svg)](https://github.com/Aisuko/forge/actions/workflows/release.yml)
[![crates.io](https://img.shields.io/crates/v/forge-ml.svg)](https://crates.io/crates/forge-ml)
[![docs.rs](https://img.shields.io/docsrs/forge-ml)](https://docs.rs/forge-ml)
[![MSRV](https://img.shields.io/badge/MSRV-1.87-blue.svg)](Cargo.toml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--only-blue.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-WebGPU-8A2BE2)](https://aisuko.github.io/forge/)

## What is it

Forge is a runtime for training and running neural networks, aiming to be the efficient and portable one: a single Rust crate and one set of WGSL kernels, with no CUDA toolchain and no Python interpreter anywhere in the loop.

## Why

Running a transformer usually means Python, a CUDA toolchain, and a dependency stack tied to one vendor's hardware. Forge is a single Rust crate that trains and infers on any GPU wgpu reaches — Vulkan, Metal, D3D12, and WebGPU in a browser tab — with no CUDA, no Python interpreter, and no server round trip. That's not just a portability trick. Portable and efficient is where this is going: one runtime on every device you own, down to the edge, and eventually across them — the floor a vendor-independent runtime has to stand on before it can carry real research, and eventually local, safety-focused AI work, on hardware you control end to end.

## Demo

Try by using your own hardware at https://aisuko.github.io/forge/


https://github.com/user-attachments/assets/d3487d8f-40a1-4c84-9f98-d4a7c5ce1550



## Run it

```bash
# Generate — the 43 MB char-level Shakespeare model ships in the repo
cargo run --release --example generate -- --model assets/shakespeare_char --prompt "ROMEO:"

# Real GPT-2 124M weights — the parity run from tests/gpt2_e2e.rs
./scripts/download_gpt2.sh
cargo run --release --example generate -- --backend wgpu --prompt "Hello Forge!"

# Train from scratch
./scripts/download_shakespeare.sh
cargo run --release --example train_shakespeare -- --backend wgpu

# Terminal model browser + run dashboard
cargo run --release --features tui --bin forge-top -- --path models/

# Website + in-browser WebGPU demo
./scripts/build_site.sh && ./scripts/serve_web.sh

# Tests
cargo test --release
```

As a library:

```toml
# The crate publishes as `forge-ml` (the name `forge` was taken on crates.io in
# 2017); the library it builds is still `forge`, so imports read `use forge::…`.
[dependencies]
forge-ml = "0.1"
```

```rust
use forge::{AnyTokenizer, Device, Gpt2, Gpt2Config, Sampling};

// assets/shakespeare_char ships in the repo; swap in models/gpt2 after
// running download_gpt2.sh — AnyTokenizer picks char or BPE by what it finds.
let device = Device::wgpu()?; // or Device::Cpu
let config = Gpt2Config::from_json("assets/shakespeare_char/config.json")?;
let model = Gpt2::from_safetensors("assets/shakespeare_char/model.safetensors", config, &device)?;
let tok = AnyTokenizer::from_dir("assets/shakespeare_char")?;
let text = model.generate(&tok, "ROMEO:", 40, Sampling::Greedy)?;
```

On Linux, wgpu needs a Vulkan ICD — install `mesa-vulkan-drivers` for a
software fallback, or run [`scripts/setup_nvidia_vulkan.sh`](scripts/setup_nvidia_vulkan.sh)
for NVIDIA inside a container. Check with `cargo run --release --example wgpu_probe`.

## Acknowledgement

<a href="https://www.rmit.edu.au/about/schools-colleges/stem/research/race">
  <img src="docs/static/PB-RACE-BLUE-SQ-2.svg" alt="Powered by RMIT University RACE" height="90">
</a>

Supported by the RACE Merit Allocation Scheme (RMAS), RMIT University.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
