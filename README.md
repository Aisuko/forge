# Forge

[![CI](https://github.com/Aisuko/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Aisuko/forge/actions/workflows/ci.yml)
[![Release](https://github.com/Aisuko/forge/actions/workflows/release.yml/badge.svg)](https://github.com/Aisuko/forge/actions/workflows/release.yml)
[![Pages](https://github.com/Aisuko/forge/actions/workflows/pages.yml/badge.svg)](https://aisuko.github.io/forge/)
[![crates.io](https://img.shields.io/crates/v/forge-ml.svg)](https://crates.io/crates/forge-ml)
[![docs.rs](https://img.shields.io/docsrs/forge-ml)](https://docs.rs/forge-ml)
[![MSRV](https://img.shields.io/badge/MSRV-1.87-blue.svg)](Cargo.toml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--only-blue.svg)](LICENSE)

## What is it

Forge is a runtime for training and running neural networks, aiming to be the efficient and portable one: a single Rust crate and one set of WGSL kernels, with no CUDA toolchain and no Python interpreter anywhere in the loop.

## Why

Research keeps climbing a ladder of abstraction: in 2016, researchers implemented and trained their own models; by 2018, they were downloading pretrained weights and fine-tuning them; today, most research happens through an API call to a model nobody in the loop trained, or even opened.

That climb isn't the problem by itself — every rung bought real productivity, the same way a compiler buys you freedom from hand-written assembly. But unlike a programming language or an operating system, these abstractions leak, and there is still fundamental research that requires tearing up the stack: reaching past the API, past the fine-tuning, into the model and the hardware underneath it. That kind of research needs people who understand the full stack, and understanding a stack this deep only really happens by building it. That's the premise this project runs on: understanding via building.

Running a transformer usually means Python, a CUDA toolchain, and a dependency stack tied to one vendor's hardware. Forge is a single Rust crate that trains and infers on any GPU wgpu reaches — Vulkan, Metal, D3D12, and WebGPU in a browser tab — with no CUDA, no Python interpreter, and no server round trip. That's not just a portability trick: portable and efficient is where this is going — one runtime on every device you own, down to the edge, and eventually across them — the floor a vendor-independent runtime has to stand on before it can carry real research, and eventually local, safety-focused AI work, on hardware you control end to end.

## Demo

**[aisuko.github.io/forge](https://aisuko.github.io/forge/)** — three pages that
run the runtime in a browser tab on your own GPU, with no server in the loop:

- **Watch it think** — one character at a time, showing the shortlist it chose from and the positions it attended to
- **[The council](https://aisuko.github.io/forge/council.html)** — four small models merging their hidden states into one character
- **[Surprise](https://aisuko.github.io/forge/react.html)** — select any text; the model tints it by how surprised it was to read it

Or build and serve the whole site locally:

```bash
make site
```


https://github.com/user-attachments/assets/d3487d8f-40a1-4c84-9f98-d4a7c5ce1550



## Run it

```bash
# Generate — the 43 MB char-level Shakespeare model ships in the repo
cargo run --release --example generate -- --model assets/shakespeare_char --prompt "ROMEO:"

# Real GPT-2 124M weights — the parity run from tests/gpt2_e2e.rs
./scripts/local/download_gpt2.sh
cargo run --release --example generate -- --backend wgpu --prompt "Hello Forge!"

# Train from scratch
./scripts/local/download_shakespeare.sh
cargo run --release --example train_shakespeare -- --backend wgpu

# Terminal model browser + run dashboard  (tools/forge-top)
cargo run --release -p forge-top -- --path models/

# A council of four small models, deciding one character together  (tools/council)
cargo run --release -p forge-council --example council_demo -- --prompt "ROMEO:"

# The site — all three pages, built and served locally
make site

# Tests
cargo test --release
```

As a library:

```toml
# The crate publishes as `forge-ml` (the name `forge` was taken on crates.io in
# 2017); the library it builds is still `forge`, so imports read `use forge::…`.
[dependencies]
forge-ml = "0.4"
```

```rust
use forge::{AnyTokenizer, Device, Gpt2, Gpt2Config, Sampling};

// assets/shakespeare_char ships in the repo; swap in models/gpt2 after
// running scripts/local/download_gpt2.sh — AnyTokenizer picks char or BPE by what it finds.
let device = Device::wgpu()?; // or Device::Cpu
let config = Gpt2Config::from_json("assets/shakespeare_char/config.json")?;
let model = Gpt2::from_safetensors("assets/shakespeare_char/model.safetensors", config, &device)?;
let tok = AnyTokenizer::from_dir("assets/shakespeare_char")?;
let text = model.generate(&tok, "ROMEO:", 40, Sampling::Greedy)?;
```

## The crate, and what is built on it

`forge-ml` is the runtime and nothing else: tensors, kernels, GPT-2, tokenizers,
serialization, and the browser bindings. One optional feature, off by default:

| Feature | What it adds | Why it's optional |
| --- | --- | --- |
| `train` | reverse-mode autograd, AdamW, the nine backward kernels, and `Gpt2::loss` / `loss_grads`. Needed by `examples/train_shakespeare.rs` and `make train`. | Forge is an inference runtime that also happens to train. `cargo add forge-ml` should not compile a tape you never record, and `src/wasm.rs` exports no training at all. Construction and serialization — `Gpt2::init_random`, `params`, `save_safetensors` — are *not* gated: they are not training, and the inference tests use them. |

```toml
forge-ml = { version = "0.4", features = ["train"] }
```

Everything else lives in [`tools/`](tools), downstream of the runtime — a crate
or a page that depends on `forge` and adds no kernel, no dtype and no device
work of its own:

| Tool | What it is |
| --- | --- |
| [`tools/council`](tools/council) | Four small GPT-2s run in parallel and merged in hidden space by an entropy router, plus the page that draws the vectors they exchange |
| [`tools/forge-top`](tools/forge-top) | A terminal model browser and run dashboard |
| [`tools/surprise`](tools/surprise) | A page that tints text by how surprised the model was to read it |

Through 0.1.0 the first two were `council` and `tui` features of this crate.
They are separate crates from 0.3.0 because the test of what belongs in the
runtime is whether a third-party crate could have written it against the public
API — and all three could. The primitives they stand on stayed:
`Gpt2::hidden_step`, `logits_from_hidden`, `wte_host`, `surprisal_async`,
`Sampler`, `top_probs`.

```bash
cargo run --release --features train --example train_shakespeare -- --backend wgpu
```

On Linux, wgpu needs a Vulkan ICD — install `mesa-vulkan-drivers` for a
software fallback, or run [`scripts/devcontainer/setup_nvidia_vulkan.sh`](scripts/devcontainer/setup_nvidia_vulkan.sh)
for NVIDIA inside a container. Check with `cargo run --release --example wgpu_probe`.

## Acknowledgement

<a href="https://www.rmit.edu.au/about/schools-colleges/stem/research/race">
  <img src="docs/static/PB-RACE-BLUE-SQ-2.svg" alt="Powered by RMIT University RACE" height="90">
</a>

Supported by the RACE Merit Allocation Scheme (RMAS), RMIT University.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
