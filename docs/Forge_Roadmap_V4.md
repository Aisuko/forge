# Forge Roadmap v4
## A WebGPU-Native Machine Learning Framework for Rust

**Roadmap version:** 4.0
**Product target:** Forge 1.0 (GPT-2 inference + training on CPU & WebGPU)

> Roadmap v4 supersedes v3. Changes are listed at the end of this document.

## Vision

Forge is a WebGPU-native machine learning framework written entirely in Rust,
inspired by [tracel-ai/burn](https://github.com/tracel-ai/burn) but intentionally
minimal.

Forge 1.0 is scoped around a single target model: **GPT-2** (124M).
Every tensor operator, kernel, neural-network layer, autograd rule, and
serialization component exists because GPT-2 requires it.

The CPU backend is a mathematically correct reference implementation used for
testing and verification. Production execution targets WebGPU through the
`wgpu` ecosystem, enabling the same model code to run on Linux, Windows,
macOS, and modern browsers.

---

# Core Principles

- WebGPU-first architecture.
- GPT-2 defines the framework scope.
- Minimality before completeness.
- Correctness before optimization — every op has a CPU reference and a
  defined numerical tolerance.
- Backend-agnostic model code.
- Inference before training — the forward path is verified against a golden
  reference before autograd is built on top of it.
- **Backward from forward (new in v4):** autograd composes backward passes
  out of already-verified forward ops wherever possible; dedicated backward
  kernels exist only where composition is impossible or wasteful
  (GELU', LayerNorm dx, softmax dx, scatter-add). Both backends therefore get
  training from one autograd implementation.
- One API for desktop and browser (async-friendly by design; see Pitfalls).

---

# Architecture

```
                GPT-2
                  │
        Transformer Modules (nn)      Autograd Tape
                  │                        │
          Tensor Operations ───────────────┘
                  │
         Backend Abstraction
          ┌──────────────┐
          │              │
      WGPU Backend   CPU Reference
          │
     Native + Browser
```

Dtype policy (explicit): **f32** for all compute, **u32** for token indices.
Contiguous row-major layout only in 1.0. No mixed precision until Forge 2.x.

Training batch policy (new in v4): the op surface stays **single-sequence**
(`[t, c]` activations). Batched training is achieved by **gradient
accumulation** over sequences, not by adding a batch dimension to every op.
This keeps kernels simple; a true batch dim is a Forge 2.x concern.

---

# Repository Layout

```text
forge/
├── src/
│   ├── error.rs
│   ├── dtype.rs
│   ├── shape.rs
│   ├── device.rs
│   ├── tensor.rs
│   ├── ops.rs              # shape-checked dispatch (fwd + bwd primitives)
│   ├── backend/
│   │   ├── cpu.rs          # reference implementation
│   │   └── wgpu/           # context, buffers, pipeline cache, dispatch
│   ├── autograd/           # tape, Var, backward rules, gradient checks
│   ├── nn/                 # Embedding, Linear, LayerNorm, Dropout
│   ├── optim/              # AdamW (+ grad clip)
│   ├── models/gpt2/        # config, blocks, KV cache, generation, training fwd
│   ├── tokenizer/          # byte-level BPE
│   └── serialization/      # safetensors load + save (checkpoints)
├── shaders/                # WGSL, embedded via include_str!
├── tests/
├── benches/                # populated in the Optimization stage
├── examples/               # generate, train_shakespeare, wgpu_probe
├── docs/                   # roadmap + website source (Stage 11 demo)
└── scripts/                # weight/tokenizer/corpus download from HF hub
```

---

# Backend Strategy

## Primary Backend: `wgpu`

- Linux (Vulkan), Windows (D3D12), macOS (Metal), Browser (WebGPU)
- NVIDIA / AMD / Intel / Apple Silicon

No model code changes are required between platforms.

**Runtime requirement:** on Linux, wgpu needs a Vulkan ICD. Containers
exposing only the `compute,utility` NVIDIA capabilities do **not** have one —
`NVIDIA_DRIVER_CAPABILITIES` must include `graphics` (or Mesa's
lavapipe/llvmpipe must be installed as a software fallback for CI).

## Reference Backend: CPU

Identical semantics, plain Rust. Used for unit testing, numerical validation,
CI, and gradient checking.

---

# GPT-2 Scope

Inference operator surface (verified in Stages 1–6):

- Embedding gather (token + positional)
- Linear (y = x·W + b, W stored `[in, out]` — HF Conv1D convention)
- LayerNorm (last dim, eps 1e-5)
- GELU (**tanh approximation** — GPT-2 uses `gelu_new`, not erf)
- Batched MatMul with optional B-transpose (QKᵀ)
- Causal-masked Softmax (fused mask, max-subtraction for stability;
  `off = key_len − query_len` parameter already carries the KV-cache case)
- Head split / merge (`[T, 3C] → 3×[H, T, hd]`, `[H, T, hd] → [T, C]`)
- Residual add, broadcast bias add, scalar scale
- Argmax / top-k sampling (decode side)

Training operator surface (complete list — new in v4; v3 only said
"Dropout, CrossEntropy, AdamW"):

- MatMul **A-transpose** variant (dB = Aᵀ·dY; v3's list had trans_b only)
- Elementwise mul (Hadamard, for backward chains)
- Row-sum reduction `[rows, cols] → [cols]` (bias/γ/β gradients)
- GELU backward, LayerNorm backward (dx + dγ/dβ), causal-Softmax backward
- Embedding backward (scatter-add into wte/wpe; see Pitfalls #9)
- Fused CrossEntropy forward (NLL gather) + backward (softmax − one-hot)
- Dropout (hash-counter RNG, identical mask on both backends; see Pitfalls #12)
- AdamW update (in-place, decoupled weight decay) + global grad-norm clip

Excluded: CNN, RNN, vision, audio, ONNX, quantization, distributed training,
mixed precision, MoE, batch dimension on activations (grad accumulation
instead).

---

# Milestones

Stages 1–6 are complete and gated (per-op parity ≤ 1e-4; CPU↔WGPU logits
≤ 5e-3; Forge↔HF ≤ 1e-2; identical greedy tokens). Stage 7 onward is the
remaining work. **v4 renumbers the tail: KV-cache decode is promoted from a
Stage-6 sub-bullet to its own Stage 7** — it was never implemented under v3
and is a prerequisite for usable decode speed everywhere (especially wasm).

## Stage 1 — Core Types ✅
Shape, DType, Device, Error, Tensor (Arc'd storage), contiguous layout.

## Stage 2 — CPU Reference Ops ✅
All inference operators, plain Rust, unit-tested against hand-computed values.

## Stage 3 — WebGPU Runtime ✅
Instance, adapter, device, queue, shader loader, pipeline cache. Elevated
limits requested at device creation. Sync readback facade on native.

## Stage 4 — WGSL Kernels ✅
add, gelu, matmul (batched, opt. transpose-B, column-chunked output),
causal-softmax, layernorm, embedding-gather (row-chunked), split/merge heads.
**Gate (met):** every kernel matches the CPU reference within 1e-4 abs.

## Stage 5 — Tokenizer & Serialization ✅
Byte-level BPE, safetensors loading, HF weight-name mapping, download script.
**Gate (met):** round-trip + known encodings match HF.

## Stage 6 — GPT-2 Inference ✅ (Forge 0.1)
Full-context recompute, greedy + top-k sampling, weight-tied chunked LM head.
**Gate (met):** logits vs HF ≤ 1e-2 abs; identical greedy CPU/WGPU/HF.

## Stage 7 — KV-Cache Decode
Per-layer K/V cache tensors `[n_head, n_ctx, head_dim]` preallocated on
device with an append kernel; single-token decode path (q_len = 1,
softmax `off = cached_len`); positional embedding offset already supported.
Strided B-view matmul so attention reads only the first `kv_len` cache rows.
**Gate:** greedy continuations token-identical to the no-cache path for
≥ 64 generated tokens, on CPU and WGPU; decode step cost no longer grows
with a full-context forward.

## Stage 8 — Autograd
Tape-based reverse mode over the existing ops layer (design fixed in v4):

- `Tape` records nodes during a training forward; `Var` = tensor + node id.
- Backward rules composed from forward ops (matmul with trans_a/trans_b,
  add, mul, row-sum) plus dedicated kernels only for GELU', LayerNorm dx,
  causal-softmax dx, and embedding scatter-add.
- Gradient accumulation: `backward()` adds into existing grads; the tape is
  rebuilt each step (define-by-run).
- Saved-for-backward: softmax saves its output; layernorm recomputes row
  stats in backward (cheaper than storing them for [t,768] rows).

**Gate:** analytic gradients match central-difference numerical gradients on
every op and on a 2-layer random-init model (CPU reference), relative error
≤ 1e-2 at f32; CPU↔WGPU gradient parity ≤ 1e-3 abs per op.

## Stage 9 — Training Modules
Dropout (deterministic counter-based RNG, identical across backends),
fused CrossEntropy (loss readback is `[t]` NLL values, never the
`[t, vocab]` probability matrix), AdamW with decoupled weight decay and
global gradient-norm clipping.
**Gate:** CE forward/backward matches a hand-computed reference; AdamW step
matches a scalar PyTorch-semantics reference; dropout masks identical on
CPU and WGPU for a fixed seed.

## Stage 10 — GPT-2 Training (Forge 0.2)
Tiny Shakespeare (BPE-tokenized, cached to disk), random-init training with
gradient accumulation, checkpoint save/load via safetensors (round-trip
gate), loss logging.
**Gate:** smoothed loss decreases from ~10.8 (= ln 50257, random init) to
< 4.0 on Tiny Shakespeare on the WGPU backend; a saved checkpoint reloads
and resumes/evaluates at the same loss; a scaled-down config trains on CPU
in CI.

## Stage 11 — Browser Deployment (Forge 0.3)
wasm32-unknown-unknown, WebGPU in Chrome/Edge. Requirements fixed in v4:

- All device creation and readback paths **async** on wasm (`pollster` and
  `device.poll(Wait)` are native-only; the sync API is a cfg'd facade).
- `Device::wgpu_async()` + async generate/decode entry points; per-token
  callback so the page can stream text.
- wte row-chunking already keeps bindings under the browser's 128 MiB cap;
  also respect `max_buffer_size`.
- Weights fetched over HTTP (served locally for the demo); inference only —
  training is out of browser scope for 1.0.

**Gate:** the crate compiles for wasm32-unknown-unknown; the demo page
generates identical greedy tokens to native WGPU for the same prompt.

## Stage 12 — Optimization (post-1.0)
Kernel fusion, buffer pooling/reuse (today every op allocates a fresh
buffer and every dispatch a fresh uniform buffer), persistent bind groups,
matmul tiling beyond 16×16, `benches/` populated, decode tokens/sec and
training step-time tracked vs. baseline. Explicitly **not** a 1.0 gate.

---

# Known Pitfalls

1. **HF Conv1D layout.** GPT-2 checkpoints store attention/MLP weights as
   `[in_features, out_features]` (Conv1D), *transposed* relative to
   `nn.Linear`. Forge's Linear adopts `y = x·W + b` with `W: [in, out]` so HF
   weights load without transposition. Getting this wrong fails silently.
2. **wgpu default limits.** `max_storage_buffer_binding_size` defaults to
   128 MiB; GPT-2's token embedding is 50257×768×4 ≈ 147 MiB. Request the
   adapter's actual limits at device creation; row-chunk the wte. Browsers
   cap at 128 MiB regardless. Also check `max_buffer_size`, not just the
   binding size.
3. **Tokenizer regex.** GPT-2's split pattern uses negative lookahead
   (`\s+(?!\S)`), unsupported by the `regex` crate. Use `fancy-regex` or a
   hand-written scanner.
4. **GELU flavor.** GPT-2 uses the tanh approximation. Using the erf version
   introduces ~1e-3 drift that breaks logit-parity gates.
5. **Softmax stability.** Subtract the row max before exp, and apply the
   causal mask *before* max-subtraction (mask with -inf, guard fully-masked
   rows).
6. **Weight tying.** `lm_head` shares `wte`; the checkpoint has no separate
   lm_head tensor. **Training corollary (new in v4):** wte's gradient has
   two contributors — embedding scatter-add *and* the LM-head matmul
   (dwte += dlogitsᵀ·h). Missing either fails silently as slow convergence.
7. **wgpu buffer alignment.** Storage/uniform offsets need 256-byte
   alignment; `COPY_DST`/`MAP_READ` usage flags must be planned per buffer
   role.
8. **Non-uniform workgroup tails.** Guard every kernel against
   out-of-bounds threads (`if idx >= n { return; }`).
9. **No f32 atomics in WGSL (new in v4).** `atomicAdd` exists only for
   u32/i32. Embedding scatter-add (repeated token ids!) uses a
   compare-exchange loop over `bitcast<u32>(f32)`; a plain non-atomic add
   loses gradient mass silently. Requires a single-chunk wte on the training
   device (native limits allow it; browser training is out of scope).
10. **Backward needs trans_a (new in v4).** dB = Aᵀ·dY. v3's kernel list
    had only trans_b; without an A-transpose matmul path every weight
    gradient materializes an explicit transpose.
11. **Causal softmax backward (new in v4).** dx = y⊙(dy − Σ_visible dy⊙y);
    masked entries must produce exactly 0 gradient. Reusing an unmasked
    softmax-backward silently corrupts attention gradients.
12. **Dropout RNG parity (new in v4).** Sampling masks host-side and
    uploading them is slow; per-element hash-counter RNG (PCG hash of
    (seed, index)) implemented identically in Rust and WGSL gives
    reproducible, backend-identical masks.
13. **f32 gradient checking (new in v4).** Central differences at f32 need
    per-parameter step sizes (~1e-2·max(1,|θ|)) and *relative* tolerance;
    absolute 1e-6-style tolerances produce false failures.
14. **No sync on wasm (new in v4).** `pollster::block_on` and
    `device.poll(Wait)` cannot exist on wasm32 — every readback and device
    request must be `await`ed on the JS event loop. Design entry points
    async-first; the native sync API is the facade, not the other way round.

---

# Acceptance Criteria

| Check | Tolerance |
|---|---|
| Per-op CPU ↔ WGPU parity (fwd) | ≤ 1e-4 abs |
| Per-op CPU ↔ WGPU parity (bwd) | ≤ 1e-3 abs |
| End-to-end logits CPU ↔ WGPU | ≤ 5e-3 abs |
| Logits vs. HF transformers (f32) | ≤ 1e-2 abs |
| Greedy tokens CPU vs. WGPU | identical |
| KV-cache vs. no-cache greedy tokens (≥64 steps) | identical |
| Tokenizer vs. HF on test corpus | identical ids |
| Analytic vs. numerical gradients (CPU) | rel ≤ 1e-2 |
| Dropout mask CPU vs. WGPU (fixed seed) | identical |
| Checkpoint save→load round trip | bit-identical params |
| Tiny Shakespeare smoothed loss (WGPU) | 10.8 → < 4.0 |
| wasm32 build + browser greedy tokens vs. native | compiles / identical |

---

# Version 1.0 Definition

Forge 1.0 is complete when the following executes unchanged on desktop
(sync facade) and browser (async form):

```rust
let device = Device::wgpu()?;   // or Device::Cpu; wasm: Device::wgpu_async().await?

let model = Gpt2::from_safetensors("gpt2/model.safetensors", config, &device)?;
let tok   = Gpt2Tokenizer::from_files("gpt2/vocab.json", "gpt2/merges.txt")?;

let output = model.generate(&tok, "Hello Forge!", 40, Sampling::Greedy)?;
```

and GPT-2 training reduces loss on Tiny Shakespeare (Stage 10 gate).

Intermediate releases: **0.1** = inference gate (Stage 6) ✅, **0.2** =
training gate (Stage 10), **0.3** = browser (Stage 11).

---

# Future Versions

- Forge 2.x — GPT-Neo, GPT-J, batch dimension, mixed precision groundwork
- Forge 3.x — Llama, Mistral (RoPE, RMSNorm, SwiGLU enter scope here)
- Forge 4.x — MoE, Flash Attention

The tensor engine API should remain stable across these versions.

---

# Changes from v3

1. **KV-cache promoted to Stage 7.** In v3 it was a sub-bullet of Stage 6
   ("optimization sub-stage") yet Stage 6 shipped without it, leaving it
   unowned. It now has its own stage, gate (token-identity vs. no-cache for
   ≥64 steps on both backends), and design (preallocated per-layer cache +
   append kernel + strided B-view matmul). Later stages renumber:
   autograd 8, training modules 9, GPT-2 training 10, browser 11,
   optimization 12.
2. **Autograd design specified.** v3 said "reverse mode" with no integration
   plan. v4 fixes tape-based define-by-run over the backend-agnostic ops
   layer, backward-from-forward composition, and the exact set of dedicated
   backward kernels — so one autograd serves both backends.
3. **Training operator surface enumerated.** v3 listed only
   "Dropout, CrossEntropy, AdamW", omitting trans_a matmul, elementwise mul,
   row-sum reductions, the per-op backward kernels, scatter-add, and grad
   clipping — most of the actual work.
4. **Batching policy stated.** Gradient accumulation over single sequences;
   no batch dimension on the 1.0 op surface. v3 was silent, implying batched
   variants of every op/kernel.
5. **Six new pitfalls (9–14)** covering training and wasm: no f32 atomics in
   WGSL, trans_a requirement, causal softmax backward masking, dropout RNG
   parity, f32 gradcheck methodology, async-only wasm APIs. Pitfall 6
   gains the tied-weight *gradient* corollary.
6. **Browser stage made concrete.** Async-first API requirement
   (`Device::wgpu_async()`, async readback/generate with streaming
   callback), HTTP weight fetching, inference-only scope, and a
   native-vs-browser token-identity gate. v3 only said "no pollster on wasm".
7. **Optimization moved post-1.0.** v3's Stage 11 sat inside the 1.0 stage
   list with no gate; v4 makes explicit that 1.0 does not depend on it, and
   scopes it (buffer pooling, uniform-buffer reuse, fusion, benches/).
8. **Acceptance criteria extended** with backward parity, KV-cache identity,
   gradcheck, dropout determinism, checkpoint round-trip, the training loss
   gate, and the wasm gate. Checkpoint *saving* added to serialization scope
   (v3 had load-only).
9. **Stage 9 training gate clarified** — ~10.8 is ln(50257): random init
   over the full BPE vocab, not a fine-tune; and the loss gate is pinned to
   the WGPU backend with a scaled-down CPU CI config.
