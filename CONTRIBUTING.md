# Contributing to Forge

Thanks for your interest in Forge. This document orients new contributors:
what the project is trying to be, how the code is organized, and how to
verify a change before sending it out.

## Core idea

Forge is a WebGPU-native machine learning framework in Rust, **intentionally
scoped to GPT-2**. It is not a general-purpose tensor library — every
operator, WGSL shader, and module exists because GPT-2 inference/training
needs it, and nothing else. This scoping is deliberate: it keeps the
surface area small enough that a single contributor can hold the whole
system in their head, and it gives every kernel a concrete, checkable
correctness target (does it reproduce GPT-2's numbers?).

Two backends implement the same op set:

- **WGPU backend** (`src/backend/wgpu/`) — the production path, running on
  Vulkan / Metal / D3D12 natively or WebGPU in the browser, via
  [wgpu](https://github.com/gfx-rs/wgpu).
- **CPU backend** (`src/backend/cpu.rs`) — a mathematically identical
  reference implementation used to verify the GPU backend and to run
  without a GPU.

Model code (`src/models/gpt2/`) is written against a backend-agnostic
`Tensor`/`Device`, so the same `Gpt2` struct runs unchanged on either
backend. When you add or change an op, you almost always touch it in three
places: the CPU reference, the WGSL kernel, and (if the op is new) the
shape-checked dispatcher in `src/ops.rs`.

See the roadmap docs under `.claude/commands/Forge_Roadmap_V*.md` for the
staged plan (inference → KV-cache → autograd → training → wasm/browser) and
the current stage.

## Project structure

```
src/
  tensor.rs           device-agnostic tensor (f32/u32, contiguous)
  shape.rs            shape/stride helpers
  dtype.rs            dtype definitions
  device.rs           Device enum (Cpu / Wgpu) and selection
  ops.rs              shape-checked op dispatch, shared across both backends
  backend/
    cpu.rs            CPU reference implementation of every op
    wgpu/             WebGPU context, buffer management, pipeline cache,
                       kernel dispatch
  nn/                 Linear, LayerNorm, Embedding — exactly what GPT-2 needs
  models/gpt2/        config, transformer blocks, generation/sampling
  autograd/           reverse-mode autodiff over the op graph
  optim/              optimizer(s) (AdamW)
  tokenizer/          byte-level BPE (vocab.json + merges.txt)
  serialization/      safetensors checkpoint loading/saving
  wasm.rs             wasm32 bindings for the browser demo (cfg-gated)

shaders/              WGSL compute kernels, one file per op (matmul.wgsl,
                       layernorm.wgsl, gelu.wgsl, adamw.wgsl, ...) — each
                       has a CPU counterpart in backend/cpu.rs

tests/                integration tests; see "Testing" below
examples/             runnable binaries (generate, train_shakespeare, ...) —
                       gitignored (local/generated), run via `cargo run --example`
scripts/              setup/data-fetch/build helpers (see below)
docs/                 roadmap + website source (built into docs/dist/)
models/, data/,
checkpoints/          local, gitignored — model weights, datasets, and
                       training checkpoints fetched/produced by scripts
```

Key invariant to preserve: **CPU and WGPU must stay numerically equivalent**
for every op, within the tolerances enforced by `tests/op_parity.rs`. If you
add a new op, add it to both backends and add a parity case.

## Environment setup

```bash
# One-time: activate the local CI hooks (see "The local CI gate" below).
# Already run for you in the devcontainer.
./scripts/install_hooks.sh

# Fetch GPT-2 124M weights + tokenizer into models/gpt2/
# (reads HF_TOKEN from .env if set; gpt2 is public so this is optional)
./scripts/download_gpt2.sh

# Fetch the Tiny Shakespeare corpus into data/ (used by training tests/examples)
./scripts/download_shakespeare.sh
```

On Linux, the WGPU backend needs a working Vulkan ICD. See
`.devcontainer/devcontainer.json` and `scripts/setup_nvidia_vulkan.sh` if
you're running in a container with an NVIDIA GPU and `wgpu::Device`
initialization fails to find a hardware adapter (falls back to software
rendering via Mesa's llvmpipe otherwise, which works but is slow).

## The local CI gate

Verification runs on your machine, in git hooks, rather than on GitHub. The
only remaining workflow that runs automatically is the Pages deploy
(`.github/workflows/pages.yml`); `.github/workflows/ci.yml` is dispatch-only,
kept for pull requests from forks where nobody's hooks ran.

The reason is that the GitHub runners were verifying strictly less than a
developer machine can. They have no GPU, so every WGPU test ran against Mesa's
software Vulkan driver, and they don't have the gitignored 548 MB
`models/gpt2/`, so `gpt2_e2e` and `kv_cache` — the suites that check real
GPT-2 numerics against HF `transformers` — were skipped entirely. The pre-push
hook runs both, on real hardware.

`scripts/ci_local.sh` is the single source of truth for what "green" means,
and both hooks are thin wrappers around it:

| stage | checks | cost (warm `target/`) | hook |
| --- | --- | --- | --- |
| `fast` | `cargo fmt --check`, `cargo clippy -D warnings`, wasm32 build, `forge-top` build, TUI dependency-leak assert | ~6s | `pre-commit` |
| `full` | everything in `fast`, plus `cargo test --release --locked` (all suites) | ~1m10s | `pre-push` |

Stages are ordered cheapest-first and stop at the first failure, so a
formatting slip doesn't cost you a minute of GPU tests. Run either by hand:

```bash
./scripts/ci_local.sh fast
./scripts/ci_local.sh full
```

`full` deliberately repeats the `fast` checks — a push can carry commits made
with `--no-verify`, or fetched from another machine.

Two things worth knowing:

- **Activation is per clone.** git ignores `.githooks/` until
  `core.hooksPath` points at it, which is what `./scripts/install_hooks.sh`
  does. Undo with `git config --unset core.hooksPath`.
- **`pre-commit` checks the working tree, not the index.** With a partially
  staged change it verifies a different state than the one being committed,
  and warns when it notices unstaged changes. `pre-push` has no such gap.

To commit or push a knowingly-broken WIP state, bypass with
`git commit --no-verify` / `git push --no-verify`.

## Testing

Run the full suite before sending a change:

```bash
cargo test --release
```

(`--release` matters: several tests run real GPT-2 forward/backward passes
on both backends and are impractically slow in debug builds.)

What each test file checks:

- `tests/op_parity.rs` — every WGSL kernel vs. the CPU reference, including
  non-square and non-power-of-two shapes (tolerance ≤ 1e-4 abs)
- `tests/tokenizer.rs` — byte-level BPE vs. known GPT-2 encodings
  (requires `models/gpt2/vocab.json` + `merges.txt`; skipped when absent),
  plus the character-level vocab round-tripping all of Tiny Shakespeare
- `tests/streaming.rs` — `generate_streaming` is byte-identical to `generate`
  and its callback fires exactly once per generated token (no weights needed)
- `tests/char_model.rs` — the shipped `assets/shakespeare_char/` checkpoint:
  config/vocab agreement, and identical greedy output on CPU and WGPU
- `tests/gpt2_e2e.rs` — CPU vs. WGPU last-position logits (≤ 5e-3 abs),
  identical greedy continuations on both backends, and (when
  `tests/data/hf_golden.json` exists) a golden check against HF
  `transformers` — regenerate that golden file with `scripts/make_golden.py`
  (requires `models/gpt2/`; skipped when absent)
- `tests/kv_cache.rs` — KV-cache decode is token-identical to the no-cache
  path for ≥ 64 generated tokens, on both backends (requires `models/gpt2/`)
- `tests/autograd.rs` — analytic gradients match central-difference
  numerical gradients on a small random-init model (CPU), and CPU vs. WGPU
  gradients agree per parameter
- `tests/train_ops.rs` — backward kernels and training modules match the
  CPU reference (≤ 1e-3 abs), dropout masks are identical across backends,
  AdamW matches a hand-computed step
- `tests/training.rs` — a scaled-down random-init model overfits a fixed
  sequence (loss drops sharply) on both backends, and checkpoints
  round-trip bit-identically

Tests that need `models/gpt2/` or `data/` are skipped (not failed) when the
corresponding fetch script hasn't been run — run the scripts above first if
you want full coverage.

To run a single test file or test:

```bash
cargo test --release --test op_parity
cargo test --release --test gpt2_e2e -- golden
```

### Manual verification

For interactive sanity checks, generate text with each backend and compare:

```bash
cargo run --release --example generate -- --backend cpu  --prompt "Hello Forge!"
cargo run --release --example generate -- --backend wgpu --prompt "Hello Forge!"
```

If you're touching the browser/wasm path, build and serve the demo:

```bash
./scripts/build_site.sh  # requires rustup target add wasm32-unknown-unknown
                         # and wasm-bindgen-cli matching the wasm-bindgen crate version
./scripts/serve_web.sh   # serves the built site at http://localhost:8000/
```

## Making a change

1. Check the current stage in the roadmap
   (`.claude/commands/Forge_Roadmap_V4.md`) so new work fits the intended
   sequencing (e.g. don't build training features before the autograd gate
   is green).
2. If you add or modify an op: implement it in `backend/cpu.rs`, add/update
   the matching WGSL kernel in `shaders/`, wire it through `ops.rs`, and add
   a parity case in `tests/op_parity.rs`.
3. Run `./scripts/ci_local.sh full` (or just let the `pre-push` hook do it)
   and, if relevant, the manual CPU/WGPU generation comparison above.
4. Keep changes scoped to what GPT-2 needs — this project deliberately
   avoids generality for its own sake.
