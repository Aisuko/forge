# Plan 01 — Repo Hygiene: Dead Code, `.gitignore` Repair, CI

**Goal.** Remove genuinely unreferenced code, fix the `.gitignore` rules that
leave required files untracked, and add CI. **No capability is removed.**

**Non-goals.** Do not delete autograd, optimizer, training, KV-cache, or wasm.
Do not "de-GPT-2-ify" the model code. Do not add a CUDA backend.

**Invariant for every task: `cargo test --release` stays at 30/30 passing.**

---

## Baseline (measure before touching anything)

```bash
cd /workspaces/forge
cargo test --release 2>&1 | grep -E "^test result"   # expect 8 lines, 0 failed
cargo clippy --all-targets --release 2>&1 | grep -c "^warning"
```

Record both numbers. Test count must not drop. Clippy warning count must not rise.

---

## Task 1 — Delete verified-unreferenced code

Each item below was checked by grepping `src/ tests/ examples/` for the
identifier and confirming the only hit is the definition itself.

### 1a. `Device::is_cpu`

**File:** `src/device.rs:29-31`

```rust
pub fn is_cpu(&self) -> bool {
    matches!(self, Device::Cpu)
}
```

Zero references anywhere. **Delete it.**

### 1b. `serialization::load_safetensors` and `load_safetensors_bytes`

**File:** `src/serialization/mod.rs:15-40`

Both are unreferenced. This is not obvious, so here is the evidence:
`Gpt2::from_safetensors` (`src/models/gpt2/mod.rs:135`) reads the file itself
and delegates to `Gpt2::from_safetensors_bytes` (`:148`), which calls
`safetensors::SafeTensors::deserialize` **directly** at `:152` — it never goes
through the `serialization` module's loaders. The generic
`HashMap<String, Tensor>` loaders have no remaining caller.

**Delete both functions.** Keep `save_safetensors` (`:44`) — it *is* used, by
`Gpt2::save_safetensors` at `src/models/gpt2/mod.rs:595`.

After deletion the module keeps only `save_safetensors`; prune the now-unused
imports (`std::collections::HashMap`, `Dtype` is still needed by `save`,
`crate::tensor::Tensor` and `crate::device::Device` likely become unused).
Let `cargo build` tell you exactly which.

> **If implementing Plan 02 first:** the TUI does **not** need these loaders. It
> reads safetensors *headers only* (see Plan 02 Task 2) and must never
> materialize a 548 MB file into tensors just to display shapes.

### 1c. DO NOT delete — verified false positives

A naive "grep for identifier, count == 1" sweep flags these. All three must be **kept**:

| Item | Why it looks dead | Why it is not |
|------|-------------------|---------------|
| `wasm::device_info` (`src/wasm.rs:50`) | no Rust caller | `#[wasm_bindgen]` export, called from JavaScript in `web/index.html` |
| `getrandom` dep (`Cargo.toml`) | 0 direct references | Feature-enabler for `rand` on wasm32; required by the `getrandom_backend="wasm_js"` rustflag in `.cargo/config.toml`. Removing it breaks the wasm build. |
| `Device::wgpu_async`, `logits_step_async`, `generate_async` | thin native usage | The browser/wasm path depends on them |

**Acceptance:** `cargo build --release && cargo test --release` → 30/30.

---

## Task 2 — Audit dependencies with a real tool, not grep

Grep counts are unreliable for build-time/feature-only deps (see 1c). Use:

```bash
cargo install cargo-udeps --locked   # if not present
cargo +nightly udeps --all-targets
```

Direct-reference counts observed this session, as a *hint only*:

| Dep | Direct refs | Note |
|-----|-------------|------|
| `wgpu` | 103 | core |
| `safetensors` | 24 | core |
| `bytemuck` | 13 | core |
| `rand` | 8 | core |
| `serde_json` | 8 | core |
| `serde` | 7 | core |
| `wasm-bindgen` | 4 | wasm |
| `pollster` | 3 | native sync facade |
| `fancy-regex` | 2 | tokenizer BPE split |
| `rayon` | 2 | `src/backend/cpu.rs:5,40` |
| `js-sys` | 1 | `src/wasm.rs:64` |
| `console_error_panic_hook` | 1 | wasm |
| **`wasm-bindgen-futures`** | **0** | **investigate — likely still required** |
| **`getrandom`** | **0** | **KEEP — see 1c** |

`wasm-bindgen-futures` is the only real candidate. Before removing it, verify
the wasm target still builds, because `#[wasm_bindgen]` can generate code
referencing it even with no source-level mention:

```bash
cargo build --release --target wasm32-unknown-unknown
```

If that fails after removal, put it back and add a comment saying why.

**Acceptance:** both native and wasm32 targets build.

---

## Task 3 — Fix the 2 clippy warnings

```bash
cargo clippy --all-targets --release
```

Two `this operation has no effect` warnings in `tests/op_parity.rs`. Apply:

```bash
cargo clippy --fix --test op_parity -p forge
```

Then re-run `cargo clippy --all-targets --release` and confirm **zero** warnings.

**Acceptance:** clippy clean; `tests/op_parity.rs` still 11/11.

---

## Task 4 — Repair `.gitignore` (this is a real bug, not cosmetics)

`.gitignore` currently makes required files untracked. Verified with
`git check-ignore -v`:

| Path | Ignored by | Consequence |
|------|-----------|-------------|
| `examples/generate.rs` | `.gitignore:36` → `examples/` | **`README.md` tells users to run `--example generate`, and it is not in the repo.** All 4 examples are untracked. |
| `tests/data/hf_golden.json` | `.gitignore:35` → `data/` | The pattern has no leading slash, so it matches `tests/data/` too. The HF golden fixture used by `tests/gpt2_e2e.rs:101` is untracked. |
| `Cargo.lock` | `.gitignore:28` | Non-reproducible CI builds |
| `plan/` | `.gitignore:38` | These plan files are untracked |

`web/` is **tracked** (6 files) — including the built binary `web/forge/forge_bg.wasm`.
That is a build artifact in version control; flag it to the user but do not
remove it unilaterally, since `scripts/build_web.sh` regenerates it and Plan 03
does not depend on it.

### Changes

1. Anchor the dataset ignore so it stops matching `tests/data/`:
   ```diff
   -data/
   +/data/
   ```
2. Remove `examples/` entirely — examples are source, they belong in the repo.
3. Remove `Cargo.lock` — keep the lockfile for reproducible CI. (The crate is a
   `cdylib`/binary-producing project, so committing it is correct.)
4. Decide on `plan/`: remove the line if these plans should be tracked.
5. Leave `models/`, `checkpoints/`, and `.claude/` ignored — correct as-is.

Then actually add the files:

```bash
git add -f examples/ tests/data/hf_golden.json Cargo.lock
git status --short
```

### Also fix the broken roadmap link

`README.md` links `docs/Forge_Roadmap_V4.md`, but `docs/` is **empty** — the
file lives at `.claude/commands/Forge_Roadmap_V4.md`, and `.claude/` is
gitignored. Either copy the roadmap to `docs/Forge_Roadmap_V4.md` and track it
(**recommended** — Plan 03's website links to it), or correct the README link.

`book/` and `benches/` are also empty. Remove them or add a `.gitkeep` with a
note; do not leave bare empty directories.

**Acceptance:** `git ls-files | wc -l` increases by at least 6 (4 examples +
golden fixture + `Cargo.lock`). A fresh `git clone` into a temp dir can run
`cargo build --release --examples`.

---

## Task 5 — Add CI (GitHub Actions)

**Constraint: GitHub-hosted runners have no NVIDIA GPU.** The suite must
therefore be split. Do not try to run the WGPU tests against a real GPU in CI.

Create `.github/workflows/ci.yml`:

- **Trigger:** `push` to `main`, `pull_request`.
- **Job `check`** on `ubuntu-latest`:
  - `actions/checkout@v7` (v7.0.1 is current as of 2026-07-26)
  - `dtolnay/rust-toolchain@stable` with `components: rustfmt, clippy`
  - `Swatinem/rust-cache@v2` (v2.9.1 current)
  - `cargo fmt --all --check`
  - `cargo clippy --all-targets -- -D warnings`
- **Job `test-cpu`** on `ubuntu-latest`:
  - Install a software Vulkan ICD so WGPU-backed tests can still run on
    llvmpipe: `sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers`
  - `cargo test --release`
  - **These tests need weights that are not in the repo** (`models/gpt2/` is
    correctly gitignored, 548 MB). `tests/gpt2_e2e.rs` and `tests/kv_cache.rs`
    require them. Either run `scripts/download_gpt2.sh` (slow, needs `HF_TOKEN`)
    or filter them out:
    ```bash
    cargo test --release --test op_parity --test tokenizer \
                         --test train_ops --test training --test autograd
    ```
    Prefer the filter for PR CI; the weight-dependent suites stay a local/GPU
    concern. **Verify the chosen subset actually passes on llvmpipe before
    merging** — `tests/training.rs::overfit_fixed_sequence_wgpu` is slow on a
    software rasterizer and may need `--ignored` gating or a longer timeout.
- **Job `build-wasm`** on `ubuntu-latest`:
  - add target `wasm32-unknown-unknown`, then `cargo build --release --target wasm32-unknown-unknown`

**Acceptance:** all jobs green on a PR. Record in the workflow file a comment
stating that GPU-backed parity is verified locally, not in CI.

---

## Task 6 — Update `README.md` to match reality

### Reposition as nanoGPT-class (framing only — no code changes)

Decision taken this session: Forge leads with the small char-level model, with
GPT-2 124M as the compatibility proof point. **This is a docs change only.**
Type names stay (`Gpt2`, `Gpt2Config`, `Gpt2Tokenizer`), no `bias` field is
added, no `block_size` alias, no `.pt` importer.

The justification, worth stating in the README so nobody "fixes" it later:
**nanoGPT is the GPT-2 architecture** — pre-LN blocks, causal self-attention,
4× GELU MLP, learned positional embeddings, weight tying. Its own headline
feature is loading OpenAI GPT-2 weights. Forge already implements it. So
repositioning loses nothing: the same code runs a 43 MB char model you train in
minutes *and* GPT-2 124M.

Current opening line is "A WebGPU-native machine learning framework in Rust,
intentionally scoped to **GPT-2**." Replace with framing along the lines of: a
nanoGPT-class GPT in Rust, WebGPU-native; train a 43 MB Shakespeare model in
minutes, and load real GPT-2 124M weights with the same code.

One honest caveat to keep: Forge stores `Linear` weights `[in, out]`
(HF Conv1D convention, `src/nn/mod.rs:7-10`), whereas nanoGPT uses `nn.Linear`
(`[out, in]`). Checkpoints are therefore **not** binary-interchangeable with
upstream nanoGPT `.pt` files without transposing four keys per block. Say so
rather than implying drop-in compatibility.

### Other README corrections

- Roadmap status line currently says "**Stage 6 complete** … Next: KV-cache
  decode, autograd, training on Tiny Shakespeare, wasm/browser." **All of those
  are now implemented** (`KvCache` is exported from `src/lib.rs`;
  `src/autograd/`, `src/optim/`, `src/wasm.rs`, and `web/` all exist and are
  tested). Update to reflect Stages 1–11.
- Fix the `docs/Forge_Roadmap_V4.md` link (Task 4).
- Add a short "GPU in containers" pointer to `scripts/setup_nvidia_vulkan.sh`
  and note the harmless `XDG_RUNTIME_DIR` stderr line.
- Add CI badge once Task 5 lands.

**Acceptance:** every command in `README.md` runs successfully from a fresh clone
(after `scripts/download_gpt2.sh`).

---

## Definition of done

- [ ] `cargo test --release` → 30/30, unchanged
- [ ] `cargo clippy --all-targets --release` → 0 warnings
- [ ] `cargo build --release --target wasm32-unknown-unknown` → succeeds
- [ ] Fresh clone can build examples without `-f` overrides
- [ ] CI green on a PR
- [ ] `README.md` contains no false claims and no broken links
