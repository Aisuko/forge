# Forge — Implementation Plans (2026-07-26)

Index for the work agreed in the `26_jul` session. Read this file first, then
the numbered plan for the workstream you are implementing.

| Plan | Workstream | Depends on | Status |
|------|-----------|------------|--------|
| [01-repo-hygiene.md](01-repo-hygiene.md) | Dead-code pass, `.gitignore` repair, CI | — | not started |
| [04-char-shakespeare.md](04-char-shakespeare.md) | Char-level nanoGPT Shakespeare model (43 MB) | — | not started |
| [02-tui-forge-top.md](02-tui-forge-top.md) | `forge-top` terminal model browser + live dashboard | 01, its own Task 1 | not started |
| [03-website.md](03-website.md) | Tailwind + three.js site **+ live in-browser demo** | 01, **04** | not started |

**Order: 01 → 04 → (02 ‖ 03).** Plan 04 is numbered last but should be built
early — it is a hard dependency of Plan 03's demo, and its `Tokenizer` trait
touches the same `generate` signatures that Plan 02 Task 1 modifies.

> **Coordination warning.** Plan 02 Task 1 adds `Gpt2::generate_streaming` and
> Plan 04 Task 1 makes the generation methods generic over a `Tokenizer` trait.
> Both edit the same functions in `src/models/gpt2/mod.rs`. Do **04 first**, then
> 02, or expect a merge conflict.

---

## Environment findings (verified 2026-07-26, do not re-litigate)

These were open questions at the start of the session. All are now settled by
direct measurement on this machine. **Do not plan work premised on the
opposite.**

### WebGPU works. On the NVIDIA GPU. Without root.

`cargo run --release --example wgpu_probe` output:

```
== wgpu adapters ==
  Vulkan | NVIDIA RTX A5000 | DiscreteGpu
  Vulkan | llvmpipe (LLVM 19.1.7, 256 bits) | Cpu
  Gl | NVIDIA RTX A5000/PCIe/SSE2 | Other
selected: Vulkan | NVIDIA RTX A5000
compute result: [2.0, 4.0, 6.0, 8.0] (expected [2, 4, 6, 8])
WebGPU OK
```

- Running as uid 1000 (`vscode`), **not** root.
- `/dev/nvidia*` and `/dev/dri/*` are mode `666` — no group membership needed.
- `sudo` is passwordless here anyway, so `scripts/setup_nvidia_vulkan.sh` runs fine.
- [`scripts/setup_nvidia_vulkan.sh`](../scripts/setup_nvidia_vulkan.sh) already
  solves the three real blockers (NVIDIA Vulkan ICD manifest, GLVND EGL vendor
  manifest, DRM node permissions) and
  [`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json)
  re-runs it via `postStartCommand` on every container start.

### "WebGPU" and "use the NVIDIA GPU" are the same code path

`wgpu` reaches the NVIDIA GPU **through Vulkan**. There is no separate thing to
build. A native CUDA backend would duplicate the existing WGSL kernel surface
for zero new capability and is explicitly **out of scope**.

### Both backends pass, so CPU-only testing is not a fallback we need

Full suite as of this session: **30/30 passing**.

| Suite | Tests |
|-------|-------|
| `tests/autograd.rs` | 2 |
| `tests/gpt2_e2e.rs` | 1 |
| `tests/kv_cache.rs` | 1 |
| `tests/op_parity.rs` | 11 |
| `tests/tokenizer.rs` | 3 |
| `tests/train_ops.rs` | 9 |
| `tests/training.rs` | 3 |

CPU is still the correctness *reference* (roadmap v4 principle), and CI has no
GPU — see Plan 01 Task 5 for how the split is handled. That is a CI constraint,
not a capability limit.

### Harmless stderr noise

`error: XDG_RUNTIME_DIR is invalid or not set in the environment.` is printed by
the Vulkan loader on every run. It is **not** a failure. Do not "fix" it by
changing backend selection logic.

---

## Scope decisions taken by the user this session

1. **Code trim = dead-code pass only.** Do *not* delete autograd, optimizer, or
   training. Roadmap v4 (inference **and** training) stays intact. All 30 tests
   must still pass afterward. Plan 01 implements exactly this.
2. **TUI = model browser *and* live run dashboard**, in one app. Plan 02.
3. **Website = explainer + three.js + a live in-browser demo.** Plan 03.
4. **Train a char-level Shakespeare model, nanoGPT-style.** Plan 04.
5. **Reposition as nanoGPT-class — framing and docs only, no code churn.**
   Plan 01 Task 6 and Plan 03 Task 2.

### On "migrating to nanoGPT" (asked and settled this session)

**There is nothing to migrate: nanoGPT *is* the GPT-2 architecture** — pre-LN
blocks, causal self-attention, 4× GELU MLP, learned positional embeddings,
weight tying. Its headline feature is loading OpenAI GPT-2 weights. Forge
already implements it.

Verified differences against upstream nanoGPT, and what each costs:

| | nanoGPT | Forge today | Action |
|---|---|---|---|
| Tokenizer | tiktoken BPE **or** char | BPE only | **Plan 04** — the real gap |
| Train from scratch | primary path | `init_random` + `train_shakespeare.rs` | already works |
| `bias` flag | configurable, default `True` | always `Some(..)` | none needed — `shakespeare_char` uses the default |
| `Linear` layout | `nn.Linear` → `[out, in]` | Conv1D/HF → `[in, out]` (`src/nn/mod.rs:7-10`) | none — Forge trains and saves its own layout |
| Checkpoint | PyTorch `.pt` (pickle) | safetensors | **declined** — no importer |

**Explicitly rejected:** renaming `Gpt2`/`Gpt2Config`/`Gpt2Tokenizer`, adding a
`bias` config field, adding a `block_size` alias, and writing a `.pt` converter.
All churn, no functional gain.

**Do not claim `.pt` interchangeability with upstream nanoGPT.** The `Linear`
weight-layout difference means four keys per block would need transposing
(`attn.c_attn.weight`, `attn.c_proj.weight`, `mlp.c_fc.weight`,
`mlp.c_proj.weight`).

### Decision 3 was revised mid-session — here is why

The demo was initially cut because GPT-2 124M's weights are 548 MB, over
GitHub's 100 MB per-file limit. Investigating smaller models overturned that:

- The existing `checkpoints/shakespeare.safetensors` is **84% dead weight** — a
  50257-row BPE embedding table on a 338k-token corpus, so >90% of it never left
  random init. Verified by sampling: it emits `Vanderbilt`, `encyclopedia`,
  `constitutional`. It loads in 0.09 s and runs at 164 tok/s, so the machinery is
  fine; the vocabulary is the problem.
- Tiny Shakespeare has exactly **65 distinct characters** — the same vocab size
  nanoGPT uses. At vocab 65 the embedding table drops from 9.65M params to
  24,960.
- The resulting 6L/6H/384d model is **43.1 MB**, comfortably under the 100 MB
  limit.

So the demo is self-hosted on Pages: ~45 MB total (2.0 MB wasm + 43.1 MB
weights), **zero cross-origin requests**, no HuggingFace CDN. GPT-2 124M stays
a local-only artifact via `scripts/serve_web.sh`.

Two facts that make this viable and are easy to lose:
- `WasmGpt2::load()` (`src/wasm.rs:33`) already takes weight **bytes** from JS,
  so the weight source is pluggable — `web/index.html:83` just hardcodes a path.
- Forge needs **no `SharedArrayBuffer`**, because `rayon` is native-only. That
  matters because GitHub Pages cannot set the COOP/COEP headers such a
  dependency would require. Do not add wasm threads.

---

## Repo facts an implementing agent will need

- Remote: `https://github.com/Aisuko/forge.git` → Pages URL `https://aisuko.github.io/forge/`.
- Rust edition 2024. Crate is `rlib` + `cdylib`.
- 5,862 LOC Rust, 23 WGSL kernels in `shaders/`.
- Roadmap: `.claude/commands/Forge_Roadmap_V4.md` (note: `README.md` links it as
  `docs/Forge_Roadmap_V4.md`, but `docs/` is **empty** — a broken link, fixed in
  Plan 01 Task 4).
- Published parity numbers (from `README.md`, quoted by the website):
  CPU↔WGPU max logit diff **8.4e-5**; Forge↔HF transformers **1.75e-4**;
  greedy continuations identical across CPU, WGPU, and HF.

### Local weights inventory (measured)

| File | Size | Params | Layers | vocab × d_model |
|---|---|---|---|---|
| `models/gpt2/model.safetensors` | 548.1 MB | 137.0M | 12 | 50257 × 768 |
| `checkpoints/shakespeare.safetensors` | 45.9 MB | 11.5M | 4 | 50257 × 192 |
| `checkpoints/shakespeare_wgpu_smoke.safetensors` | 27.4 MB | 6.8M | 2 | 50257 × 128 |

All f32 — which matters, because the loader **rejects non-f32 tensors**
(`src/models/gpt2/mod.rs:160`). Any f16/bf16 checkpoint needs conversion work
before Forge can read it.

## Caveat on this `plan/` directory

`plan/` is currently listed in `.gitignore` (line 38), so these files are **not
tracked by git**. Plan 01 Task 4 decides whether to keep it that way. If you
want these plans in version control, remove that line first.
