# forge-top

A terminal model browser and live run dashboard for the Forge runtime.

```bash
cargo run --release -p forge-top -- --path models/ --path checkpoints/
```

It scans directories for models, reports what each one is from its safetensors
header without loading it onto a device, and runs generation against the one you
pick — tokens/sec, GPU utilisation and memory, live.

Three threads over one `mpsc` channel: main owns the terminal and redraws at
~15 FPS, the scanner streams discovered models, the runner drives generation.
Inference never touches the main thread — a single CPU `logits_step` is long
enough to visibly freeze input handling.

## Why it is a crate and not a feature

Through 0.2.0 this was a `tui` feature of `forge-ml` with five optional
dependencies — ratatui, crossterm, sysinfo, nvml-wrapper, memmap2 — kept
optional so they never reached the library's dependents or a wasm bundle. That
promise now holds structurally: the dependencies are in *this* manifest, and
there is no feature flag left that could turn them on in someone else's build.

Not published to crates.io: see [`../README.md`](../README.md).
