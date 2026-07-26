# Plan 04 — Char-Level Shakespeare Model (nanoGPT-style)

**Goal.** Train a small character-level GPT on Tiny Shakespeare so Forge has a
weights artifact small enough to ship on GitHub Pages and load instantly in the
TUI. Target config is Karpathy's nanoGPT Shakespeare-char reference.

**Target:** `n_layer=6, n_head=6, n_embd=384, n_ctx=256, vocab_size=65,
dropout=0.2` → **10.77M params ≈ 43.1 MB f32** (13× smaller than GPT-2 124M,
and under GitHub's 100 MB per-file limit).

---

## Why: the existing checkpoint is 84% dead weight

`checkpoints/shakespeare.safetensors` is already nanoGPT-shaped (4 layers,
192-dim) but uses the **full 50257-entry GPT-2 BPE vocab**:

```
wte (embedding table):  9.65M  (84% of model)
transformer blocks   :  1.78M  (16%)
total                : 11.48M   -> 45.9 MB
```

Tiny Shakespeare is 1,115,394 chars ≈ 338k BPE tokens, so the overwhelming
majority of those 50257 embedding rows never leave random init. Sampling from it
(verified this session, top-k 40, T=0.8, prompt `"ROMEO:"`) produces:

```
ROMEO:Each , grateful Visorst,shield calendars Doct False, 396
: Explorerilaterally encyclopediabalanced ,constitutional, host FoundersUT
```

Those are random embeddings, not undertraining you can fix with more steps. The
model loads in 0.09 s and runs at 164 tok/s on the A5000 — the machinery is
fine, the vocabulary is the problem.

**Tiny Shakespeare has exactly 65 distinct characters** (verified), which is the
same vocab size nanoGPT uses:

```
"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
```

At vocab 65 the embedding table drops from 9.65M params to 24,960 — essentially
every parameter does real work.

### Size comparison (computed, not estimated)

| config | vocab | params | f32 size | vs GPT-2 |
|---|---|---|---|---|
| current ckpt 4L/6H/192d | 50257 | 11.48M | 45.9 MB | 12× |
| char tiny 4L/4H/128d | 65 | 0.83M | 3.3 MB | 164× |
| char small 4L/6H/192d | 65 | 1.84M | 7.4 MB | 74× |
| **char nanoGPT 6L/6H/384d** | **65** | **10.77M** | **43.1 MB** | **13×** |

---

## Task 1 — `Tokenizer` trait

`Gpt2::generate` (`src/models/gpt2/mod.rs:726`) and `generate_async` (`:761`)
take `&Gpt2Tokenizer` **concretely**, so a second tokenizer cannot be used
without a small refactor.

Introduce a trait in `src/tokenizer/mod.rs` covering exactly the three methods
the generation path uses:

```rust
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> String;
    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8>;
    fn vocab_size(&self) -> usize;
}
```

`Gpt2Tokenizer` already has all four as inherent methods
(`:92`, `:109`, `:118`, `:126`) — implement the trait by delegating, and keep
the inherent methods so existing callers (`tests/tokenizer.rs`,
`examples/generate.rs`) compile unchanged.

Then make the generation methods generic: `tokenizer: &impl Tokenizer` (or
`&dyn Tokenizer`). Prefer `&impl` — no vtable, and there is no need for
heterogeneous collections here.

**Do not** change `Gpt2Tokenizer`'s public API. This must be additive.

**Acceptance:** 30/30 tests still pass with no changes to test files.

---

## Task 2 — `CharTokenizer`

New file `src/tokenizer/char.rs`, exported from `src/tokenizer/mod.rs`.

```rust
pub struct CharTokenizer { itos: Vec<char>, stoi: HashMap<char, u32> }

impl CharTokenizer {
    /// Build the vocab from the sorted unique chars of a corpus (nanoGPT's rule).
    pub fn from_corpus(text: &str) -> Self;
    /// Persist/restore the vocab so inference matches training exactly.
    pub fn to_json(&self) -> String;
    pub fn from_json(s: &str) -> Result<Self>;
}
```

Semantics, matching nanoGPT:
- vocab = `sorted(set(text))`, id = index in that sorted order
- `encode` maps char → id; **unknown chars must not panic.** Decide one policy
  and document it: either skip them or return `ForgeError::Tokenizer`. Prompts
  typed into the web demo *will* contain characters outside the 65.
- `decode` maps ids → `String`
- `decode_bytes` returns `decode(ids).into_bytes()` — for a char vocab there is
  no partial-UTF-8 problem, but the trait method must still exist because
  `generate_async`'s `emit_valid_prefix` streaming logic calls it

**The vocab file must ship with the checkpoint.** Sorted-unique-chars is stable
for a fixed corpus, but silently rederiving it at inference time from a
different text would misalign every token id. Write `vocab.json` next to the
`.safetensors` and load it explicitly.

**Acceptance:** round-trip test `decode(encode(text)) == text` over the whole of
`data/tinyshakespeare.txt`; `vocab_size() == 65`.

---

## Task 3 — Wire up `train_shakespeare.rs`

`examples/train_shakespeare.rs` already has the full training loop, arg parsing
(`--layers/--heads/--embd/--steps/--lr/--warmup/--cosine/--dropout/--seed/--checkpoint/--resume/--sample-every`),
and a `load_tokens` cache at `data/tinyshakespeare.ids`.

Changes needed:

1. Add `--tokenizer char|bpe` (default `char`).
2. In char mode, build `CharTokenizer::from_corpus` from
   `data/tinyshakespeare.txt` and set `vocab_size` from it rather than
   hardcoding 50257 (currently `:115`).
3. **Invalidate the token cache per tokenizer.** `load_tokens` caches to
   `data/tinyshakespeare.ids` with no tokenizer tag; reusing BPE ids for a
   char model would train on garbage. Use `tinyshakespeare.char.ids` /
   `tinyshakespeare.bpe.ids`, or embed the vocab hash in the filename.
4. Save the tokenizer vocab alongside the checkpoint.
5. Default the char run to the nanoGPT config: `n_layer=6, n_head=6,
   n_embd=384, n_ctx=256, dropout=0.2`.

Note `dropout` is already an arg and `shaders/dropout.wgsl` exists with a
cross-backend determinism test, so 0.2 is safe to use.

---

## Task 4 — Train

nanoGPT's `config/train_shakespeare_char.py` (verified upstream) sets:

```
n_layer 6   n_head 6   n_embd 384   block_size 256   dropout 0.2
batch_size 64   learning_rate 1e-3   max_iters 5000
lr_decay_iters 5000   warmup_iters 100   beta2 0.99
```

`bias` is **not set**, so it takes `GPTConfig`'s default `bias=True` — which
matches Forge, where `Linear.b` is always `Some(..)` at load
(`src/models/gpt2/mod.rs:196`). **No bias work is required for this model.**

```bash
cargo run --release --example train_shakespeare -- \
  --backend wgpu --tokenizer char \
  --layers 6 --heads 6 --embd 384 --seq-len 256 \
  --dropout 0.2 --steps 5000 --lr 1e-3 --warmup 100 --cosine \
  --accum 64 --sample-every 500 \
  --checkpoint checkpoints/shakespeare_char.safetensors
```

### `--accum 64` is not optional — do not leave it at the default

nanoGPT uses `batch_size=64`. Roadmap v4's training policy keeps the op surface
**single-sequence** (`[t, c]`) and reaches batching through **gradient
accumulation**, so Forge's equivalent of nanoGPT's batch 64 is `--accum 64`.
The example currently defaults to `--accum 2`
(`examples/train_shakespeare.rs:23`). Training at effective batch 2 with
`lr=1e-3` — a learning rate tuned for batch 64 — will be unstable and will not
reach the target loss. If it diverges, lower the LR rather than assuming the
model is broken.

### `beta2` also differs

nanoGPT sets `beta2=0.99` for this run. `AdamWOpts` already exposes `beta1`/
`beta2` (`src/optim/mod.rs:12-13`) but defaults to **`beta1=0.9, beta2=0.95`**
(`:24-25`) — so Forge's default is *not* nanoGPT's value. `train_shakespeare.rs`
has no `--beta2` flag. Add one (defaulting to 0.99 for the char run), or set it
explicitly in the example. Do not silently train at 0.95 and then compare the
loss curve against nanoGPT's published 1.48.

nanoGPT's reference run reaches **val loss ≈ 1.48** on this config. Treat that
as the target; a run that plateaus much above ~1.6 indicates a bug, not just a
short run.

`--sample-every 500` prints intermediate samples — use them to catch divergence
early rather than after the whole run.

**Hold out a validation split** (nanoGPT uses the last 10% of the text) and
report train and val loss separately. Without it you cannot tell 1.48 from
memorization.

**Verification gates:**
- Loss decreases monotonically in trend; no NaN
- Final sample contains recognizable structure: capitalized speaker names
  followed by `:`, newlines, and mostly real English words
- Reload the checkpoint and confirm greedy sampling is deterministic and matches
  the in-training sample (`tests/training.rs::checkpoint_roundtrip` covers the
  save/load path already)
- Confirm CPU and WGPU produce the same greedy continuation from the trained
  checkpoint, consistent with the existing parity gates

---

## Task 5 — Ship the artifact

Unlike `models/` and `checkpoints/` (both correctly gitignored), this checkpoint
**must be tracked** — the website and TUI depend on it.

- Path: `assets/shakespeare_char/` containing `model.safetensors` (43.1 MB),
  `config.json`, `vocab.json`
- Add a `.gitignore` negation or place it outside the ignored dirs
- 43.1 MB is under GitHub's 100 MB hard limit but over the **50 MB soft warning**
  — expect a push warning, and consider whether the 7.4 MB `char small` variant
  is preferable if repo size becomes a concern
- Do **not** use Git LFS: GitHub Pages serves LFS pointer files as text, which
  would silently break the web demo

**Acceptance:** a fresh clone has the weights; `cargo run --release --example
generate -- --backend cpu` (pointed at the char model) produces Shakespeare.

---

## Definition of done

- [ ] `CharTokenizer` round-trips all of `data/tinyshakespeare.txt`, vocab 65
- [ ] `Tokenizer` trait added; 30/30 existing tests pass unmodified
- [ ] Trained checkpoint reaches val loss ≈ 1.48, no NaN
- [ ] Samples show speaker-name/dialogue structure and real English words
- [ ] CPU and WGPU agree on greedy continuation from the checkpoint
- [ ] Artifact tracked in git, 43.1 MB, no LFS
- [ ] Loads in the browser (Plan 03) and the TUI (Plan 02)
