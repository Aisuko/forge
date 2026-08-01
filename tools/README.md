# tools

Everything here is **downstream of the runtime**. Each tool depends on `forge`
and adds nothing to it: no kernel, no dtype, no device work of its own. The test
of whether something belongs here rather than in `src/` is whether a
third-party crate could have written it against the public API. If it could, it
does.

| Tool | What it is | Build |
| --- | --- | --- |
| [`council/`](council) | Four small GPT-2s run in parallel and merged in hidden space, plus the page that draws the vectors they exchange | `cargo run -p forge-council --example council_demo` / `./tools/council/build.sh` |
| [`forge-top/`](forge-top) | A terminal model browser and run dashboard | `cargo run --release -p forge-top -- --path models/` |
| [`surprise/`](surprise) | A page that tints text by how surprised the model was to read it. No Rust — it drives `WasmGpt2.surprisal`, which is runtime | `./tools/surprise/build.sh` |

`shared/` is not a tool: it is the page furniture the two web tools have in
common — one Tailwind entry point so they look like one project, the standalone
Tailwind fetcher, the favicon, and `check_site.py`, which verifies a built
artifact resolves every asset and import it names.

## Why these are not features of `forge-ml`

Through 0.1.0 the council was a `council` Cargo feature and `forge-top` a `tui`
one. Both compiled inside the crate that calls itself *the most efficient,
portable runtime for neural networks*, which made the crate the sum of the
runtime and whatever had been demonstrated with it. A separate crate is the
stronger form of the same promise the feature flags were making: there is no
flag left that could pull ratatui into a dependent's tree, and no
`#[wasm_bindgen]` export of a council in a bundle built by someone who wanted a
runtime.

The runtime kept what the tools stand on, because those are primitives and not
demonstrations: `Gpt2::hidden_step`, `logits_from_hidden` and `wte_host` (split
a model's body from its wte-tied head), `Gpt2::surprisal_async` (score text that
already exists), `Sampler` and `top_probs`.

## The dependency is a path, on purpose

```toml
forge-ml = { path = "../.." }
```

`forge-ml` 0.3.0 is not on crates.io yet, so no tool here can name a version —
and every tool sets `publish = false`, because a crate carrying a path
dependency is unpublishable regardless. When 0.3.x ships, each becomes
`forge-ml = "0.3"` and the `publish` line comes off. That is the whole
migration; nothing else about these crates assumes it lives in this repository.

## The pages are not deployed

`council/` and `surprise/` build into their own `dist/`, which is gitignored.
Build one and serve it locally — each `build.sh` prints the command.

The Pages workflow is gone, so nothing here publishes. `docs/` and the site
still deployed at aisuko.github.io/forge are unchanged and describe the 0.2.0
layout; both are updated in a separate change, after 0.3.0 reaches crates.io and
these crates can depend on a published version.
