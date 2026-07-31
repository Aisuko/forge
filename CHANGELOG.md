# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.2.0

The inference release. v0.1.0 said "it runs anywhere"; this one is about how
fast it runs, and about saying plainly what Forge is.

### Added

- **Dispatch scopes** — `Device::dispatch_scope()` batches every kernel issued
  while the guard is alive into one command buffer and one `queue.submit`.
  A KV-cached decode step went from **100 submits to 1**.
- **A buffer pool** — `wgpu::Buffer`s are recycled through a free list keyed by
  exact size instead of allocated per op. Steady-state decode now allocates
  **no new buffers per token**, against 113 before.
- **Device counters** — `WgpuContext::stats()` reports dispatches, submits,
  buffers created and bytes allocated. Always compiled, so a regression is
  visible in any run rather than only under a benchmark build.
- **`examples/bench.rs`** — the inference benchmark. Reports ms/token and
  tokens/sec for prompt encode and cached decode separately, alongside the
  counters above, so a performance claim can be reproduced with one command.
- **`Gpt2::surprisal_async`** and **`WasmGpt2.surprisal`** — per-position
  surprisal in bits for text that already exists, plus what the model expected
  instead. A teacher-forced scoring pass, so a whole passage costs **one
  forward pass** rather than one decode step per character.
- **A new page**, [Surprise](https://aisuko.github.io/forge/react.html) — select
  any text and the model tints it by how surprised it was. No button: it reacts
  to reading. Selecting a phrase rescores it with only the selection for
  context, which is a visibly different answer from the same characters read in
  full context.
- **`tests/pool.rs`** — guards the invariant the pool depends on, that an op's
  result never depends on what was in the buffer it was handed.

### Changed

- **Training is behind the `train` feature, off by default.** `forge::autograd`,
  `forge::optim`, `Gpt2::loss`, `Gpt2::loss_grads` and thirteen `forge::ops`
  functions now require `features = ["train"]`. **Breaking.** Construction and
  serialization stay core: `Gpt2::init_random`, `params`, `params_mut`,
  `param_specs` and `save_safetensors` are not training and are used by the
  inference tests.
- `src/ops.rs` split into `src/ops/mod.rs` and `src/ops/train.rs`. Public paths
  are unchanged — `forge::ops::adamw` still resolves under the feature.
- The WGSL `SHADERS` table split by feature, which is what actually keeps the
  backward kernels' source out of a default build.
- `WgpuStorage.buf` is now `pub(crate)`; use `WgpuStorage::buffer()`.
  **Breaking**, though the field was public by accident.

### Fixed

- `unsplit_head` wrote one third of its output and relied on the rest being
  zero — true of a fresh `wgpu::Buffer`, false of a recycled one. It now
  allocates through a dedicated zeroed path. This silently corrupted `wte`
  gradients while the pool was being built; `tests/pool.rs` is the guard.

### Measured

| | before | after |
| --- | --- | --- |
| decode, ms/token | 6.53 | **4.26** (1.53×) |
| decode, tokens/sec | 153 | **235** |
| prompt encode, ms/token | 0.212 | 0.174 (1.22×) |
| submits per decoded token | 100 | **1** |
| buffers created per token | 113 | **0** |
| GPU bytes allocated, 128-token run | 392 MiB | **47.7 MiB** |
| `forge.wasm`, `train` gated out | — | 13,297 B smaller |

NVIDIA RTX A5000, Vulkan, `assets/shakespeare_char` (6 layers, n_embd 384).
Reproduce with `cargo run --release --example bench`.

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
- **Council** — `Council`, several small GPT-2s run on one prompt in parallel,
  exchanging hidden states rather than text and merged by an entropy router.
  Behind the optional `council` feature: it composes over the runtime rather than
  belonging to it, adding no dependency and no WGSL kernel. The runtime
  primitives it rests on — `Gpt2::hidden_step`, `Gpt2::logits_from_hidden` and
  `Gpt2::wte_host`, which split a model's body from its wte-tied head — are core
  and unconditional.

### Verification

CPU↔WGPU parity to a max logit difference of 8.4e-5, and Forge↔HuggingFace
transformers to 1.75e-4, on GPT-2 124M weights on an NVIDIA RTX A5000.

[0.1.0]: https://github.com/Aisuko/forge/releases/tag/v0.1.0
