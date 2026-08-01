# forge-council

Several small GPT-2s that run on the same prompt in parallel, exchange **hidden
states rather than text**, and produce one character together.

Every expert branched from one ancestor checkpoint and was fine-tuned with
`wte`/`wpe` frozen, so all four read the same token ids into the same basis and
all four are decoded by the same wte-tied head. That freeze is the whole trick:
it makes `Σ wᵢ·hᵢ` a vector the head can still read. The router does no
learning — the expert whose own distribution has the lowest entropy gets the
most weight, with `beta` setting how sharply.

## Run it

```bash
# the four experts, in the terminal, with their disagreement rate
cargo run --release -p forge-council --example council_demo -- --prompt "ROMEO:" --chars 200

# the page: the vectors they exchanged, the weight each earned, the decision
./tools/council/build.sh
python3 -m http.server -d tools/council/dist 8080

# the two invariants the merge rests on
cargo test -p forge-council --release
```

The shipped experts are in [`assets/`](assets) — four checkpoints of 3.3 MB, one
shared `config.json`, one shared `vocab.json`. To retrain them:

```bash
./tools/council/scripts/train_council.sh     # ancestor, then four branches
./tools/council/scripts/ship_council.sh      # promote the best of each into assets/
```

## Where the line falls

This crate is written entirely against `forge`'s public API —
`Gpt2::hidden_step`, `Gpt2::logits_from_hidden`, `Gpt2::wte_host`, `Sampler`,
`top_probs` — and adds no kernel and no device work. It shipped as a `council`
feature of `forge-ml` through 0.2.0; a separate crate says the same thing
without asking the runtime to carry it.

`src/wasm.rs` is this crate's own `cdylib`, so the page loads
`./forge/forge_council.js` rather than the runtime's `forge.js`. wasm-bindgen
collects its custom section from every linked rlib, so that one bundle carries
`WasmCouncil` and the runtime's `WasmGpt2` both — and it is also why nothing
here declares a second `#[wasm_bindgen(start)]`.

Not published to crates.io: see [`../README.md`](../README.md).
