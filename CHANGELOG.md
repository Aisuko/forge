# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-31

First public release. Forge is a single Rust crate that trains and runs GPT-2 on
any GPU `wgpu` reaches — Vulkan, Metal, D3D12, and WebGPU in a browser tab — with
no CUDA toolchain and no Python interpreter in the loop.

### Added

- **Tensors and ops** — f32/u32 tensors over a CPU backend and a WGPU backend
  driven by hand-written WGSL: matmul (batched, transposed, fused bias), gelu,
  softmax, layernorm, fused token+positional embedding, and their backward
  kernels.
- **GPT-2** — `Gpt2` / `Gpt2Config` loading HuggingFace `safetensors`, KV-cache
  incremental decoding, greedy and top-k sampling, and a `StepTrace` that reports
  per-layer attention for visualisation.
- **Tokenizers** — GPT-2 BPE and a char-level tokenizer behind `AnyTokenizer`,
  which picks between them from the contents of a model directory.
- **Training** — tape-based reverse-mode autograd and an AdamW optimiser, both
  running on the GPU backend; `examples/train_shakespeare.rs` trains from scratch.
- **Browser target** — `wasm32-unknown-unknown` build exposing inference to
  JavaScript over WebGPU, no server round trip. Live at
  <https://aisuko.github.io/forge/>.
- **`forge-top`** — an optional terminal model browser and run dashboard behind
  the `tui` feature, so its dependencies never reach the library's dependents.

### Verification

CPU↔WGPU parity to a max logit difference of 8.4e-5, and Forge↔HuggingFace
transformers to 1.75e-4, on GPT-2 124M weights on an NVIDIA RTX A5000.

[0.1.0]: https://github.com/Aisuko/forge/releases/tag/v0.1.0
