# Forge

[![CI](https://github.com/Aisuko/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Aisuko/forge/actions/workflows/ci.yml)

## What is it

A nanoGPT-class GPT-2 in Rust with two backends: **WebGPU** (Vulkan / Metal /
D3D12 / browser) and a **CPU** reference. Train a small Shakespeare model from
scratch, or load real OpenAI GPT-2 124M weights — same model code on either
device.

## Why

Running a transformer usually means Python, CUDA, and a stack of dependencies.
Forge is a single Rust crate that trains and infers on any GPU wgpu supports —
including a browser tab — with a CPU path that is mathematically identical, so
every kernel is verifiable against it.

## Run it

```bash
# Generate — the 43 MB char-level Shakespeare model ships in the repo
cargo run --release --example generate -- --model assets/shakespeare_char --prompt "ROMEO:"

# Or with real GPT-2 124M weights
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

```rust
use forge::{Device, Gpt2, Gpt2Config, Gpt2Tokenizer, Sampling};

let device = Device::wgpu()?; // or Device::Cpu
let config = Gpt2Config::from_json("models/gpt2/config.json")?;
let model = Gpt2::from_safetensors("models/gpt2/model.safetensors", config, &device)?;
let tok = Gpt2Tokenizer::from_dir("models/gpt2")?;
let text = model.generate(&tok, "Hello Forge!", 40, Sampling::Greedy)?;
```

On Linux, wgpu needs a Vulkan ICD — install `mesa-vulkan-drivers` for a
software fallback, or run [`scripts/setup_nvidia_vulkan.sh`](scripts/setup_nvidia_vulkan.sh)
for NVIDIA inside a container. Check with `cargo run --release --example wgpu_probe`.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

## More

[docs/Forge_Roadmap_V4.md](docs/Forge_Roadmap_V4.md) ·
[CONTRIBUTING.md](CONTRIBUTING.md) ·
[aisuko.github.io/forge](https://aisuko.github.io/forge/)
