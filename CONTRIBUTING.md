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
  serialization/      checkpoint I/O — safetensors, and .fzm q4 (fzm.rs)
  wasm.rs             wasm32 bindings — the browser is a runtime target

shaders/              WGSL compute kernels, one file per op (matmul.wgsl,
                       layernorm.wgsl, gelu.wgsl, adamw.wgsl, ...) — each
                       has a CPU counterpart in backend/cpu.rs

tests/                integration tests; see "Testing" below
examples/             runnable binaries (generate, train_shakespeare, ...) —
                       gitignored (local/generated), run via `cargo run --example`
scripts/              setup/data-fetch/build helpers (see below)
tools/                downstream of the runtime: crates and pages that depend
                       on forge and add nothing to it — the council, forge-top,
                       the Surprise page. See tools/README.md
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
./scripts/common/install_hooks.sh

# Fetch GPT-2 124M weights + tokenizer into models/gpt2/
# (reads HF_TOKEN from .env if set; gpt2 is public so this is optional)
./scripts/local/download_gpt2.sh

# Fetch the Tiny Shakespeare corpus into data/ (used by training tests/examples)
./scripts/local/download_shakespeare.sh
```

On Linux, the WGPU backend needs a working Vulkan ICD. See
`.devcontainer/devcontainer.json` and `scripts/devcontainer/setup_nvidia_vulkan.sh` if
you're running in a container with an NVIDIA GPU and `wgpu::Device`
initialization fails to find a hardware adapter (falls back to software
rendering via Mesa's llvmpipe otherwise, which works but is slow).

### Vulkan in the devcontainer

`scripts/devcontainer/setup_nvidia_vulkan.sh` fixes three separate causes of the same
`Found no drivers!` message on driver 580.x with an RTX A5000. Each was
diagnosed independently; any one of them alone produces that error, so fixing
only the obvious one leaves the symptom unchanged.

1. **DRM node permissions.** `wgpu` needs `/dev/dri/{card*,renderD*}` for the
   NVIDIA GPU, openable under the container's device cgroup — the
   `--device-cgroup-rule` entries for majors 226 and 195 are already in
   `devcontainer.json`. But `--device=/dev/dri` bind-mounts the *host's* nodes
   verbatim (`root:video` / `root:<host render gid>`, mode 660), and the
   container user is in neither group, so every `open()` fails `EACCES` even
   though the cgroup rule permits it. The script `chmod 666`s the
   container-local nodes; the host's are untouched.
2. **The Vulkan ICD manifest.** Without
   `/usr/share/vulkan/icd.d/nvidia_icd.json` pointing at `libGLX_nvidia.so.0`,
   the loader finds only Mesa's llvmpipe/lvp ICDs.
3. **The GLVND EGL vendor manifest.** This is the one that is easy to miss:
   only `50_mesa.json` exists, and `10_nvidia.json` pointing at
   `libEGL_nvidia.so.0` has to be added. `libGLX_nvidia.so.0` is one library
   behind the GLX, EGL and Vulkan entry points, and without the EGL manifest
   its internal bring-up never completes. The loader then loads it
   successfully and calls it, but `vk_icdNegotiateLoaderICDInterfaceVersion`
   returns `VK_ERROR_INITIALIZATION_FAILED` (-3) with `vkCreateInstance` still
   NULL — *before* the driver ever touches `/dev/nvidiactl` or the DRM nodes,
   and identically as the container user or as root. So it is not a
   permissions symptom, however much "Found no drivers!" reads like one.
   Adding the manifest makes negotiate return 0 and `vkCreateInstance` resolve
   immediately, which is the check the script ends with.

## The local CI gate

Verification runs on your machine, in git hooks, rather than on GitHub. The
only remaining workflow that runs automatically is the Pages deploy
(`.github/workflows/pages.yml`); `.github/workflows/ci.yml` is dispatch-only,
kept for pull requests from forks where nobody's hooks ran, and it invokes
`scripts/local/ci_local.sh fast` rather than restating the stages. The release
workflow checks only what the local gate cannot: the tag matching the manifest,
the docs.rs render, and the `cargo package` tarball.

The reason is that the GitHub runners were verifying strictly less than a
developer machine can. They have no GPU, so every WGPU test ran against Mesa's
software Vulkan driver, and they don't have the gitignored 548 MB
`models/gpt2/`, so `gpt2_e2e` and `kv_cache` — the suites that check real
GPT-2 numerics against HF `transformers` — were skipped entirely. The pre-push
hook runs both, on real hardware.

`scripts/local/ci_local.sh` is the single source of truth for what "green" means,
and both hooks are thin wrappers around it:

| stage | checks | cost (warm `target/`) | hook |
| --- | --- | --- | --- |
| `fast` | `cargo fmt --check`, clippy on the crate and both tools, the two wasm32 bundles, the `forge-top` build, the TUI dependency assert, the site build | ~9s | `pre-commit` |
| `full` | everything in `fast`, plus the release test suites for `forge-ml` and `forge-council` | ~1m10s | `pre-push` |

Stages are ordered cheapest-first and stop at the first failure, so a
formatting slip doesn't cost you a minute of GPU tests. Run either by hand:

```bash
./scripts/local/ci_local.sh fast
./scripts/local/ci_local.sh full
```

`full` deliberately repeats the `fast` checks — a push can carry commits made
with `--no-verify`, or fetched from another machine.

Two things worth knowing:

- **Activation is per clone.** git ignores `.githooks/` until
  `core.hooksPath` points at it, which is what `./scripts/common/install_hooks.sh`
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
  `transformers` — regenerate that golden file with `scripts/local/make_golden.py`
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

If you're touching the browser/wasm path, build and run the pages. This needs
`rustup target add wasm32-unknown-unknown` and a `wasm-bindgen-cli` matching
the `wasm-bindgen` crate version in `Cargo.lock`.

```bash
make site          # build the whole site and serve it on :8080
make site-verify   # build it, then drive all three pages on a real GPU
```

`scripts/common/build_site.sh` composes `docs/dist/` from the landing page in
`docs/src/` and both tool pages in `tools/*/web/`, sharing one core wasm bundle
and one copy of the 6.7 MB checkpoint. It is what `.github/workflows/pages.yml`
deploys, and it runs in the `fast` gate, so a page that names a moved asset
fails at commit time.

`make site-verify` (`scripts/local/check_pages.py`) is the check that matters:
`check_site.py` proves the artifact resolves, but a page that builds and then
fails inside `WasmGpt2.load` looks identical over HTTP. It drives each page in
headless Chromium with WebGPU — press Run, wait for output, assert on the DOM —
and fails on any console error, page error or 4xx. It needs `pip install
playwright && playwright install chromium`, and it is deliberately not in CI:
hosted runners have no GPU, so every page would fail there for a reason no code
change can fix.

Each tool page also builds standalone, without the rest of the site:

```bash
./tools/surprise/build.sh   # drives WasmGpt2 — the runtime's own bundle
./tools/council/build.sh    # drives WasmCouncil — the council crate's cdylib
```

A standalone artifact links to pages it does not ship, which `check_site.py`
reports as warnings; the composed site is built with `--strict`, where those
would be errors.

## Making a change

1. If you add or modify an op: implement it in `backend/cpu.rs`, add/update
   the matching WGSL kernel in `shaders/`, wire it through `ops.rs`, and add
   a parity case in `tests/op_parity.rs`.
2. Run `./scripts/local/ci_local.sh full` (or just let the `pre-push` hook do it)
   and, if relevant, the manual CPU/WGPU generation comparison above.
3. Keep changes scoped to what GPT-2 needs — this project deliberately
   avoids generality for its own sake.
4. Ask where the change belongs. `src/` is the runtime: tensors, kernels,
   GPT-2, tokenizers, serialization, the browser bindings. If a third-party
   crate could have written your change against the public API, it is a tool
   and belongs in `tools/` — and if writing it there needs a primitive the
   runtime does not expose yet, expose the primitive rather than moving the
   composition inward.
