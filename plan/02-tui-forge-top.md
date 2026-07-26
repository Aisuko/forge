# Plan 02 — `forge-top`: Terminal Model Browser + Live Run Dashboard

**Goal.** One Rust TUI that (a) scans a path for local model weights and shows
their structure, and (b) runs generation on a selected model while charting
tokens/s, VRAM, host RAM, and GPU utilization live.

**Reference implementations to study:** [nviwatch](https://crates.io/crates/nviwatch)
(ratatui + NVML, GPU dashboard) and [llamamon](https://llamamon.com/)
(ratatui, LLM throughput). Neither is a dependency — they are prior art for layout.

---

## Target layout

```
┌ models ───────────┬ forge ──────────────┐
│> gpt2      548MB  │ tok/s  32.4         │
│  shakespeare 45MB │ ▁▃▅▇█▇▅▃▁▃▅▇        │
│  smoke       27MB │                     │
├───────────────────┤ VRAM  1.8/24.0 GB   │
│ layers   12       │ ███░░░░░░░░░    7%  │
│ heads    12       │ RAM   3.1/64.0 GB   │
│ d_model  768      │ GPU util       87%  │
│ params   124M     │ 62C   118W          │
└───────────────────┴─────────────────────┘
[↑↓] nav  [enter] run  [q] quit
```

---

## Task 1 — Prerequisite: add a sync per-token streaming API

**This is a required addition to the library.** The TUI cannot get accurate
tokens/s with today's public API:

- `Gpt2::generate` (`src/models/gpt2/mod.rs:724`) is sync but returns only the
  final `String` — no per-token hook.
- `Gpt2::generate_async` (`:761`) has an `on_text: impl FnMut(&str)` hook, but
  it is `async`, and its callback fires on *decoded text deltas*, not tokens.
  `emit_valid_prefix` (`:803`) deliberately withholds partial UTF-8 sequences,
  so callback invocations do **not** map 1:1 to tokens. Counting them would
  under-report tokens/s.
- `sample` (`:837`) is a private free function, so the TUI cannot correctly
  drive its own decode loop via `logits_step` + `new_cache` without
  reimplementing sampling.

**Add** to `impl Gpt2`, mirroring `generate_async` but sync:

```rust
/// Sync streaming generation with KV-cache decode. `on_token` fires exactly
/// once per generated token with the token id and the newly decodable text
/// delta (which may be empty when a multi-byte character spans BPE tokens).
pub fn generate_streaming(
    &self,
    tokenizer: &Gpt2Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: Sampling,
    mut on_token: impl FnMut(u32, &str),
) -> Result<String>
```

Implement by copying the body of `generate_async` and substituting
`logits_step` for `logits_step_async` (drop the `.await`s). Reuse the existing
`emit_valid_prefix` helper; capture the emitted delta into a small buffer so it
can be passed alongside the token id.

Then refactor `generate` to delegate to `generate_streaming` with a no-op
closure, so sampling/EOS/`n_ctx` logic exists in exactly one place. Keep
`generate`'s signature unchanged — `examples/generate.rs`, `tests/kv_cache.rs`,
and `tests/gpt2_e2e.rs` all call it.

**Acceptance:** 30/30 tests still pass; `generate` and `generate_streaming`
produce byte-identical output for the same seed and prompt. Add a test asserting
that the `on_token` callback count equals the number of new tokens.

---

## Task 2 — Model discovery without reading 548 MB

Walk the target directory to a depth limit of 4, collecting every
`*.safetensors`.

**Default `--path` to the current directory, not `models/`.** This repo keeps
weights in two places — `models/gpt2/model.safetensors` and three checkpoints in
`checkpoints/` — so defaulting to `models/` would find only one of the four and
make the browser look broken. Accept repeated `--path` flags so a user can scan
several roots.

### The gotcha that will bite you

`safetensors::SafeTensors::read_metadata(buffer)` looks like it accepts just the
header, **but it does not**. At `safetensors-0.5.3/src/tensor.rs:320-322` it
asserts:

```rust
if buffer_end + 8 + n != buffer_len {
    return Err(SafeTensorError::MetadataIncompleteBuffer);
}
```

Passing only the first `8 + n` bytes fails. Two valid approaches:

1. **Memory-map the file** (`memmap2`) and pass the full mapped slice. The OS
   pages in only the header, so a 548 MB file costs a few KB resident. This is
   the pattern in the crate's own doc example. **Recommended.**
2. Parse the header yourself: read 8 bytes → `u64::from_le_bytes` → `n`, read
   `n` more bytes → `serde_json::from_slice`. Avoids the `memmap2` dep.

Either way, **never call `SafeTensors::deserialize` here** and never construct
`Tensor`s — that would load the whole model just to draw a list.

### Data to extract per model

From `Metadata::tensors() -> HashMap<String, &TensorInfo>`, where
`TensorInfo { dtype: Dtype, shape: Vec<usize>, data_offsets: (usize, usize) }`:

- file size on disk, tensor count
- total parameter count (`shape.iter().product()` summed)
- per-tensor name / shape / dtype (for the expandable detail pane)
- sibling files present: `config.json`, `vocab.json`, `merges.txt` — show a
  ✓/✗ so the user knows whether the model is actually runnable
- if `config.json` parses via `Gpt2Config::from_json`, show `n_layer`,
  `n_head`, `n_embd`, `n_ctx`

Scanning must be non-blocking: do it on a worker thread and stream results, so
a slow network mount cannot freeze the UI.

---

## Task 3 — Metrics sampling

### GPU (`nvml-wrapper` 0.12.1) — optional, must degrade gracefully

**Verified container gotcha:** this machine has only
`/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`. There is **no unversioned
`libnvidia-ml.so`**, which is what `Nvml::init()` looks for by default. Use:

```rust
use nvml_wrapper::Nvml;
use std::ffi::OsStr;

let nvml = Nvml::builder()
    .lib_path(OsStr::new("libnvidia-ml.so.1"))
    .init()
    .ok();   // None on CPU-only machines — the TUI must still work
```

Per sample, from `nvml.device_by_index(0)?`:

| Metric | Call |
|--------|------|
| VRAM used/total | `device.memory_info()` |
| GPU utilization % | `device.utilization_rates()` |
| Temperature | `device.temperature(TemperatureSensor::Gpu)` |
| Power draw | `device.power_usage()` (mW) |

**Second verified gotcha:** per-process VRAM attribution is unreliable in
containers. `nvidia-smi --query-compute-apps=pid,used_memory` returns no rows
here because of PID namespacing. Do **not** build the VRAM gauge on
`running_compute_processes()`. Show device-wide `memory_info()` and label it
"device VRAM", not "forge VRAM".

Every NVML pane must render an "n/a" state when `nvml` is `None`.

### Host (`sysinfo` 0.39.6)

`System::new_all()` once, then per tick `refresh_memory()` and
`refresh_cpu_usage()` (both confirmed present on `System` in 0.39.6). Use
`total_memory()` / `used_memory()` — these return **bytes**, not KB.

Two documented caveats:
- Respect `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`; refreshing CPU faster than
  that yields garbage.
- `refresh_cpu_usage()` is **inaccurate on its first call** by design — it needs
  two samples to compute a delta. Discard the first reading rather than
  rendering a bogus 0% or 100%.

### Throughput

Computed in the app from `generate_streaming`'s `on_token` callback, not from
any system API. Keep a ring buffer of the last N inter-token durations; display
both instantaneous (EMA over ~8 tokens) and session-average tokens/s. Track
prefill separately — the first `logits_step` covers the whole prompt and is much
slower than a decode step, so folding it into tokens/s makes the number
meaningless. Report **TTFT** (time to first token) as its own field.

---

## Task 4 — Application structure

Multi-file binary target (cargo supports a directory with `main.rs`):

```
src/bin/forge-top/
  main.rs      arg parsing, terminal init/restore, event loop
  scan.rs      Task 2 — discovery + header parsing
  metrics.rs   Task 3 — NVML + sysinfo sampling
  run.rs       worker thread driving generate_streaming
  ui.rs        ratatui rendering (pure fn of &AppState)
```

### Cargo wiring — keep the library's dep tree clean

The TUI deps must **never** enter the wasm build or the library's dependents.
Use an optional feature, not unconditional deps:

```toml
[features]
tui = ["dep:ratatui", "dep:crossterm", "dep:sysinfo", "dep:nvml-wrapper", "dep:memmap2"]

[[bin]]
name = "forge-top"
required-features = ["tui"]

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
ratatui      = { version = "0.30", optional = true }
crossterm    = { version = "0.29", optional = true }
sysinfo      = { version = "0.39", optional = true }
nvml-wrapper = { version = "0.12", optional = true }
memmap2      = { version = "0.9",  optional = true }
```

Build/run: `cargo run --release --features tui --bin forge-top -- --path models/`

Note `ratatui` 0.30 already depends on `crossterm`; declare it explicitly only
if you construct the backend by hand rather than using `ratatui::init()`.

### Threading

Three threads, communicating over `std::sync::mpsc`:

1. **main** — owns the terminal, polls `crossterm` events with a timeout, redraws at ~15 FPS
2. **scanner** — Task 2, sends `Event::ModelFound(ModelInfo)`
3. **runner** — Task 1's `generate_streaming`, sends `Event::Token { id, text, at: Instant }`

**Never call inference on the main thread.** A single `logits_step` on CPU takes
long enough to visibly freeze input handling.

Metrics sampling can live on the main thread (NVML calls are sub-millisecond),
but throttle it to ~4 Hz rather than once per frame.

### Terminal safety — do this before writing any UI code

A panic inside raw mode leaves the user's terminal unusable.

`ratatui` 0.30.2 exports `init()`, `try_init()`, `restore()`, `try_restore()`,
and **`run()`**. Use `run()` — its docs are explicit that it "handles terminal
initialization, restoration, and panic hooks automatically", which is the only
one of these that documents the panic hook:

```rust
ratatui::run(|terminal| run_app(terminal))?
```

Do **not** assume `init()` alone installs a panic hook — the 0.30 docs do not
say it does. If you need `init()`/`restore()` for finer control, install a hook
yourself:

```rust
let prev = std::panic::take_hook();
std::panic::set_hook(Box::new(move |info| {
    let _ = ratatui::try_restore();
    prev(info);
}));
```

Never hand-roll `enable_raw_mode()` / `EnterAlternateScreen` without also
hand-rolling the panic hook.

---

## Task 5 — Keybindings and UX

| Key | Action |
|-----|--------|
| `↑` / `↓`, `k` / `j` | move model selection |
| `enter` | run generation on the selected model |
| `esc` | cancel a running generation |
| `tab` | toggle model-detail pane between summary and per-tensor list |
| `q`, `ctrl-c` | quit (must also stop the runner thread) |

Cancellation needs an `AtomicBool` the runner checks between tokens —
`generate_streaming`'s callback is the natural checkpoint. Since the callback
cannot abort the loop by itself, either have the callback set a flag the loop
checks (requires a small `Result`-returning callback variant) or accept that
cancellation takes effect after the current token. **Prefer the simple version:
cancel after the current token, and say so in the status line.**

---

## Task 6 — Graceful degradation matrix

The app must start and remain usable in all of these:

| Condition | Required behavior |
|-----------|-------------------|
| No NVIDIA GPU / NVML load fails | GPU panes show "n/a"; CPU backend still selectable |
| No models found at `--path` | Empty-state message naming the searched path, not a crash |
| `config.json` missing or unparseable | Model still listed; config fields show "—"; running it is disabled with a reason |
| Terminal narrower than ~80 cols | Collapse to a single column; never panic on a zero-width `Rect` |
| Weights present but `vocab.json`/`merges.txt` missing | Listed, marked not-runnable |

---

## Definition of done

- [ ] `cargo run --release --features tui --bin forge-top` run from the repo
      root lists all 4 local `.safetensors` files (`models/gpt2/model.safetensors`
      plus the 3 in `checkpoints/`) in under 1 s and with < 50 MB RSS — the RSS
      figure is the real assertion here: it proves headers were parsed rather
      than the 548 MB file being read
- [ ] Selecting `models/gpt2` and pressing enter streams text with a live
      tokens/s figure that matches `examples/generate.rs`'s reported rate ±10%
- [ ] VRAM gauge tracks the RTX A5000's actual usage during a WGPU run
- [ ] `cargo build --release` (no `--features tui`) does **not** compile ratatui,
      NVML, or sysinfo — verify with `cargo tree -e normal | grep -c ratatui` → 0
- [ ] `cargo build --release --target wasm32-unknown-unknown` still succeeds
- [ ] Killing the app with `q`, `ctrl-c`, or a forced panic always leaves a
      working terminal
- [ ] 30/30 library tests still pass, plus the new `generate_streaming` test
