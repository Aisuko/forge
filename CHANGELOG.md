# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`.fzm`, a hand-rolled q4 checkpoint format** (`src/serialization/fzm.rs`):
  flat, magic-prefixed, per-group affine 4-bit with `GROUP_SIZE = 64`. No GGUF
  and no candle bindings — the same rule the kernels follow. `save_fzm_q4` /
  `load_fzm_q4` at the host f32 boundary, plus `read_fzm_header` for listing a
  checkpoint without dequantizing it.
- `Gpt2::from_checkpoint` / `from_checkpoint_bytes` pick the codec by sniffing
  the leading bytes, not the extension, so the wasm bindings take either format
  and the JS never learns which. `Gpt2::save_checkpoint` is the write-side
  counterpart, dispatching on the extension.
  `forge::serialization::checkpoint_in_dir` resolves a model directory to
  whichever of `model.fzm` / `model.safetensors` is there.
- `examples/to_fzm.rs` — the one-shot converter both ship scripts now call.

### Changed

- **Every shipped checkpoint is `.fzm` q4.** `assets/shakespeare_char/` is
  6.7 MB instead of 43 MB, and each of the four council experts 0.51 MB instead
  of 3.3 MB. The composed site is 13.5 MB, down from 61 MB. Held-out loss on the
  char model, measured on the quantized file the site actually serves: val
  1.4592 against 1.4542 for the f32 original (+0.0050, 0.3% relative), train
  1.0340 against 1.0282. The council's experts still share a byte-identical
  `wte` after quantization, which is what makes the merge meaningful.
- Dequant happens on load: the weights reach the GPU as f32 and every kernel is
  untouched. The saving is download size and repo size, not VRAM — a packed q4
  WGSL matmul is a separate, later step.
- Training still writes f32 safetensors. Quantization is a shipping step, so
  `ship_char_model.sh` and `ship_council.sh` convert on the way into the tracked
  asset directories, and `ship_char_model.sh` measures the loss it records
  against the `.fzm` file rather than its f32 source.
- `tests/data/gate_expected.json` regenerated: greedy decoding diverges from the
  f32 model after ~24 tokens, so the browser gate is now the q4 reference. It
  passes — browser and native WGPU agree token-for-token on the quantized
  weights.
- `forge-top` discovers and runs `.fzm` as well as `.safetensors`, reading only
  the header for the listing.

## [0.4.0] — 2026-08-02

The site, which 0.3.0 deferred, and the test-suite hang, diagnosed. No breaking
API change: `Device::dispatch_scope()` and the 0.2.0 decode rate both stay.

### Added

- **The site is back, and it is one artifact.** `scripts/build_site.sh`
  composes `docs/dist/` from the landing page in `docs/src/` and both tool pages
  in `tools/*/web/`, sharing one core wasm bundle, one `app.css` and one copy of
  the 43 MB checkpoint — 61 MB, against 101 MB if the three pages were
  assembled from three self-contained artifacts. `.github/workflows/pages.yml`
  deploys it. Each tool page still builds standalone.
- **`make site-verify`** (`scripts/check_pages.py`) drives all three pages in
  headless Chromium on a real GPU: press Run, wait for output, assert on the
  DOM, fail on any console error or 4xx. `check_site.py` proves an artifact
  resolves; this proves it runs. Deliberately not in CI — hosted runners have no
  GPU, so `requestAdapter()` returns null and every page fails there for a
  reason no code change can fix.
- `scripts/ci_local.sh fast` now builds the site (~0.5 s, both wasm crates
  already compiled by the steps above it), so a page naming a moved asset fails
  at commit time. That failure was live in this repository until now.

### Changed

- Both tools depend on `forge-ml = { path = "../..", version = "0.4" }`.
  `path` is what the workspace resolves; `version` is only sayable because
  the crate is published, and makes each manifest a publishable shape. They stay
  `publish = false`: they demonstrate the runtime, they are not libraries to
  depend on.
- `check_site.py` separates errors from warnings. A missing asset or an
  unresolvable module specifier is an error — the page is broken before a line
  runs. A hyperlink to a page the artifact does not ship is a warning, which is
  the normal state of a standalone tool build; the composed site is built with
  `--strict`, where it is an error. The size budget is now per page, not per
  artifact.
- `scripts/build_web.sh` takes a package name, so one script builds both the
  core and the council bundles. The council's bundle lands in `forge-council/`,
  because in a shared artifact `forge/` is the core's.
- Comments throughout state one fact in the shortest sentence that carries it.
  The `roadmap v4, Stage N` and `pitfall N` references are gone: they pointed at
  a directory that is not in the repository. The Vulkan-in-a-container
  diagnosis moved from a 42-line script header into `CONTRIBUTING.md`, where a
  reader will find it.

### Fixed

- **The test suite hung for minutes at a time, and dispatch scopes were not
  why.** Creating more than one `Device` concurrently wedges inside the driver:
  eight parallel `Device::wgpu()` calls, doing no compute at all, hung 25 runs
  out of 25. `tests/pool.rs` did exactly that — libtest runs its tests on
  parallel threads, and each built its own device — so a full `cargo test`
  hung roughly one run in four.

  `WgpuContext::new_async` now serializes creation behind a process-global
  gate: 0 hangs out of 25 on the same reproducer, and 0 out of 8 full-suite
  runs. Every path is covered — the sync facade holds no lock of its own, so a
  thread on `Device::wgpu()` and a thread on `Device::wgpu_async()` serialize
  against each other rather than racing past two separate locks. The gate is a
  40-line async mutex (`src/backend/wgpu/serial.rs`) rather than
  `std::sync::Mutex`, because a `std` guard held across the `.await` in
  `new_async` would make that future `!Send` and stop dependents spawning it on
  a runtime; and rather than `tokio::sync::Mutex`, because that is a runtime
  dependency — on the wasm build too — to hold a bool.

  The tests keep a device each — the pool and the scope depth are per-context,
  so sharing one would have let a neighbouring test take the dirtied buffer or
  batch the arm the scope test calls "unscoped", passing while checking less.
  `tests/device_concurrency.rs` is the reproducer, kept: eight threads, four
  rounds of create-and-drop each, on the async path and on a mix of both. With
  the gate removed it wedges; with it, 3 runs of 3 clean. It joins through
  `recv_timeout`, so a regression fails in two minutes instead of hanging the
  suite with no message.

  Scopes were removed in the run-up to this release on the theory that they
  caused the hang. They did not — it reproduces with them gone, in a test that
  never dispatches — so they are restored, and with them the batching 0.2.0
  shipped. Measured back to back on one machine, RTX A5000 / Vulkan /
  `assets/shakespeare_char`, `cargo run --release --example bench`:

  | | with scopes | without |
  | --- | --- | --- |
  | decode, ms/token | **4.197** | 6.581 (1.57× slower) |
  | decode, tokens/sec | **238.3** | 152.0 |
  | submits per decoded token | **1.0** | 100.0 |
  | submits, 128-token session | **798** | 79,800 |
- **The council's projection panel drew the wrong thing.** Its extent was fitted
  to a rolling window of the merged vector's path, which travels ~2 units per
  character while the four experts differ by 2.45 in total — so the five dots
  the panel is named after occupied 17% of it, measured on the built page, while
  the trail sprawled across the rest. The merge is now pinned to the centre and
  the scale is the disagreement around it, which puts the furthest expert at 80%
  of the panel radius. The path cannot survive that rescale — a character of
  travel is more than a radius — so only its bearing is drawn.
- `build_site.sh` ended in `check_site.py … || true`, so it exited 0 while
  reporting three missing assets. The site build had been broken since
  `assets/council/` moved, and nothing said so.

### Removed

- The duplicate copies of `council.{html,js}`, `react.{html,js}`, `input.css`
  and `check_site.py` under `docs/`. Each page and each piece of furniture now
  has exactly one definition, in the tool that owns it.
- The dead `Cargo.lock` line in `.gitignore`. The lockfile is tracked, and
  every gate passes `--locked`.
- `docs/static/.nojekyll`. The Pages artifact is served as uploaded and Jekyll
  never runs on it, so the file did nothing — and
  `upload-pages-artifact` v4 onwards excludes dotfiles anyway, so it would not
  have survived the upload.

## [0.3.0] — 2026-08-01

Forge is *the most efficient, portable runtime for neural networks*. Through
0.2.0 the crate was that plus whatever had been demonstrated with it. This
release separates the two: `forge-ml` is the runtime, and everything built on it
sits beside it in `tools/`, depending on it the way any other crate would.

The test applied to each piece was whether a third-party crate could have
written it against the public API. All three could.

### Changed

- **The council and `forge-top` are downstream crates in `tools/`, and the
  `council` and `tui` features are gone.** **Breaking.** `forge::Council` is now
  `forge_council::Council` (`cargo run -p forge-council --example
  council_demo`), and `forge-top` is `cargo run -p forge-top`. The repository is
  a cargo workspace; `forge-ml` is still the only package that publishes, and
  each tool carries a path dependency with `publish = false` until 0.3.0 is on
  crates.io.
- The runtime primitives the tools stand on are unchanged and unconditional:
  `Gpt2::hidden_step`, `logits_from_hidden`, `wte_host`, `surprisal_async`.

### Added

- **`Sampler` and `top_probs`** are public API. Drawing a token from logits and
  ranking a distribution are runtime primitives, and `tools/council` needs both
  from outside the crate. `Sampler` owns its RNG, so no caller has to name
  `rand`'s types.
- **`tools/`** — `council` (the four experts, their page, and the training
  scripts), `forge-top` (the terminal dashboard), `surprise` (a page over
  `WasmGpt2.surprisal`, no Rust of its own), and `shared` for the page furniture
  the two web tools have in common.

### Removed

- **The `council` and `tui` features**, with the five TUI dependencies —
  ratatui, crossterm, sysinfo, nvml-wrapper, memmap2. They were optional so they
  could never reach a dependent's tree or a wasm bundle; living in
  `tools/forge-top`'s own manifest is the same promise with no flag left to get
  wrong.
- **`WasmCouncil` from the runtime's wasm bundle.** `#[wasm_bindgen]` exports
  are GC roots the linker cannot eliminate, so it now lives in the council
  crate's own cdylib. `WasmGpt2.surprisal` stayed: it is marshalling over
  `Gpt2::surprisal_async`, a scoring pass in the way `generate` is a decoding
  one.
- `assets/council/` moved to `tools/council/assets/`.

### Not in this release

`docs/` and the Pages site are untouched and still describe the 0.2.0 layout;
`docs/src/council.html` will not run against a 0.3.0 bundle, because the council
bindings are no longer in it. The site follows in a separate change, after this
release reaches crates.io and the tools can depend on a published version. (It
did, the same day — see Unreleased.)

The four items deferred from 0.2.0 — GPU-side sampling, int8, kernel fusion,
multi-sequence batching — are still deferred and still ranked in that order.
This release was scoped to the architecture.

## [0.2.0] — 2026-07-31

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

### Deferred, and why

This release was scoped to include int8 quantisation, kernel fusion and
multi-sequence batching. The benchmark above is the reason none of them landed:
the remaining 4.26 ms of decode is **1.49 ms host readback, ~1.0 ms CPU-side
recording, ~1.8 ms GPU execution**. The arithmetic is 41% of decode, so
quantisation and fusion compete for the smallest share.

- **Multi-sequence batching — dropped.** Nothing consumes it; the site runs one
  sequence and the council runs its models sequentially.
- **Kernel fusion — deferred.** ~18 of 99 dispatches per token, ~10% of decode,
  against a new WGSL kernel plus its CPU reference and parity cases in
  perpetuity.
- **int8 — deferred to 0.2.1, for the download rather than the arithmetic.** The
  shipped char model is 43 MB of f32 that a visitor downloads before the page
  renders. Shipping int8 and dequantising at load makes that ~11 MB with no new
  kernel.
- **GPU-side sampling — 0.2.1, and ranked above all of the above.** Removing the
  per-token host round trip takes 1.49 ms out of 4.26. It was not on the
  original list; the benchmark found it.

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

[0.4.0]: https://github.com/Aisuko/forge/releases/tag/v0.4.0
[0.3.0]: https://github.com/Aisuko/forge/releases/tag/v0.3.0
[0.2.0]: https://github.com/Aisuko/forge/releases/tag/v0.2.0
[0.1.0]: https://github.com/Aisuko/forge/releases/tag/v0.1.0
