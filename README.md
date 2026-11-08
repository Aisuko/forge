# Forge

[![Release](https://github.com/Aisuko/forge/actions/workflows/release.yml/badge.svg)](https://github.com/Aisuko/forge/actions/workflows/release.yml)
[![Pages](https://github.com/Aisuko/forge/actions/workflows/pages.yml/badge.svg)](https://aisuko.github.io/forge/)
[![crates.io](https://img.shields.io/crates/v/forge-ml.svg)](https://crates.io/crates/forge-ml)
[![docs.rs](https://img.shields.io/docsrs/forge-ml)](https://docs.rs/forge-ml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--only-blue.svg)](LICENSE)

Forge is a vendor-independent Rust neural network runtime, with WebGPU in the browser as a first-class target. No CUDA, no Python.

Demo: **[aisuko.github.io/forge](https://aisuko.github.io/forge/)**

## Run

```bash
cargo run --release --example generate -- --model assets/shakespeare_char --prompt "ROMEO:"
./scripts/local/download_gpt2.sh && cargo run --release --example generate -- --backend wgpu --prompt "Hello Forge!"
./scripts/local/download_shakespeare.sh && cargo run --release --features train --example train_shakespeare -- --backend wgpu
cargo run --release -p forge-top -- --path models/
cargo run --release -p forge-council --example council_demo -- --prompt "ROMEO:"
make site
cargo test --release --features train
```

On Linux, wgpu needs a Vulkan ICD (`mesa-vulkan-drivers`, or `scripts/devcontainer/setup_nvidia_vulkan.sh` for NVIDIA). Check with `cargo run --release --example wgpu_probe`.

## Library

```toml
forge-ml = "0.6"
```

```rust
use forge::{AnyTokenizer, Device, EncoderDecoder, EncoderDecoderConfig, Gpt2, Gpt2Config, Sampling};

let device = Device::wgpu()?;

let config = Gpt2Config::from_json("assets/shakespeare_char/config.json")?;
let weights = forge::serialization::checkpoint_in_dir("assets/shakespeare_char")?;
let gpt2 = Gpt2::from_checkpoint(weights, config, &device)?;
let tok = AnyTokenizer::from_dir("assets/shakespeare_char")?;
let text = gpt2.generate(&tok, "ROMEO:", 40, Sampling::Greedy)?;

let model = EncoderDecoder::from_checkpoint("model.fzm", None, &device)?;
let memory = model.encode(&src_ids, &[src_ids.len() as u32], src_ids.len())?;
let mut state = model.start(&memory, 0)?;
let logits = model.step(&[model.config().bos_id], &mut state)?;
let ids = model.generate_greedy(&src_ids, 96)?;
```

| Model | API |
|---|---|
| GPT-2 | `Gpt2`: causal decoder with a KV cache, streaming, and traces |
| Encoder–decoder | `EncoderDecoder`: masked encoder, causal decoder with cross-attention, greedy decoding |

The `train` feature adds autograd, AdamW and the backward kernels.

## Checkpoints

| Format | Content |
|---|---|
| `.safetensors` | f32/bf16 interchange |
| FZM v1 (`FZM1`) | Flat q4 tensors |
| FZM v2 (`FZM2`) | JSON manifest (architecture, config, quantization, provenance), named assets, named q4/f32/bf16 tensors |

`forge::serialization::fzm::load_fzm` dispatches on the version. On WebGPU, q4 weights stay packed and run through q4 kernels; the CPU backend dequantizes them to f32.

## Tools

| Tool | What it is |
|---|---|
| [`tools/council`](tools/council) | Four small GPT-2s merged in hidden space |
| [`tools/forge-top`](tools/forge-top) | Terminal model browser and run dashboard |
| [`tools/surprise`](tools/surprise) | Text resolving at the pace of model certainty |

## Acknowledgement

Supported by the RACE Merit Allocation Scheme (RMAS), RMIT University.

<a href="https://www.rmit.edu.au/about/schools-colleges/stem/research/race">
  <img src="docs/static/PB-RACE-BLUE-SQ-2.svg" alt="Powered by RMIT University RACE" height="90">
</a>
