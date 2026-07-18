# Forge

A WebGPU-native machine learning framework in Rust, intentionally scoped to
**GPT-2**. Inspired by [tracel-ai/burn](https://github.com/tracel-ai/burn),
but minimal: every operator, WGSL kernel, and module exists because GPT-2
needs it.

- **WGPU backend** — production path via [wgpu](https://github.com/gfx-rs/wgpu)
  (Vulkan / Metal / D3D12 / browser WebGPU)
- **CPU backend** — mathematically identical reference used for testing and
  verification

Model code is backend-agnostic: the same `Gpt2` runs on either device.

## Quick start

```bash
# 1. Fetch GPT-2 124M weights + tokenizer (reads HF_TOKEN from .env if set)
./scripts/download_gpt2.sh

# 2. Generate text
cargo run --release --example generate -- --backend wgpu --prompt "Hello Forge!"
cargo run --release --example generate -- --backend cpu  --prompt "Hello Forge!" --topk 40 --temp 0.8
```

```rust
use forge::{Device, Gpt2, Gpt2Config, Gpt2Tokenizer, Sampling};

let device = Device::wgpu()?; // or Device::Cpu
let config = Gpt2Config::from_json("models/gpt2/config.json")?;
let model = Gpt2::from_safetensors("models/gpt2/model.safetensors", config, &device)?;
let tok = Gpt2Tokenizer::from_dir("models/gpt2")?;
let text = model.generate(&tok, "Hello Forge!", 40, Sampling::Greedy)?;
```

## Verification

```bash
cargo test --release
```

- `tests/op_parity.rs` — every WGSL kernel vs. the CPU reference (≤ 1e-4)
- `tests/tokenizer.rs` — byte-level BPE vs. known GPT-2 encodings
- `tests/gpt2_e2e.rs` — CPU vs. WGPU logits (≤ 5e-3), identical greedy
  output, and a golden check against HF `transformers`
  (regenerate with `scripts/make_golden.py`)

Measured on this repo: CPU↔WGPU max logit diff **8.4e-5**; Forge↔HF
transformers **1.75e-4**; greedy continuations identical across CPU, WGPU,
and HF.

## GPU requirements (Linux)

wgpu needs a Vulkan ICD. In containers, `NVIDIA_DRIVER_CAPABILITIES` must
include `graphics` (see `.devcontainer/devcontainer.json`); otherwise install
`mesa-vulkan-drivers` for a software (llvmpipe) fallback.

## Layout

```
src/
  backend/cpu.rs      CPU reference ops
  backend/wgpu/       WebGPU context, buffers, pipeline cache, dispatch
  tensor.rs           device-agnostic tensor (f32/u32, contiguous)
  ops.rs              shape-checked op dispatch (both backends)
  nn/                 Linear, LayerNorm, Embedding (row-chunked wte)
  models/gpt2/        config, blocks, generation
  tokenizer/          byte-level BPE from vocab.json + merges.txt
  serialization/      safetensors loading
shaders/              WGSL compute kernels
```

## Roadmap

See [docs/Forge_Roadmap_V4.md](docs/Forge_Roadmap_V4.md). Current state is
**Stage 6 complete** (verified GPT-2 inference on CPU + WebGPU). Next:
KV-cache decode, autograd, training on Tiny Shakespeare, wasm/browser.
