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

`shared/` is not a tool: it is the page furniture **every** Forge page uses, the
landing page in `docs/src/` included — one Tailwind entry point so all three
look like one project, the standalone Tailwind fetcher, the favicon, and
`check_site.py`, which verifies a built artifact resolves every asset and import
it names.

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

## The dependency names a path and a version

```toml
forge-ml = { path = "../..", version = "0.4" }
```

`path` is what a workspace build resolves, so a tool always compiles against the
`src/` beside it. `version` is what a registry would resolve, and it is only
sayable because `forge-ml` is on crates.io — a manifest in this shape is
publishable, so nothing here assumes it lives in this repository.

They stay `publish = false` anyway. These demonstrate the runtime; they are not
libraries anyone should take a dependency on.

## Where the pages go

Each web tool builds a self-contained artifact into its own gitignored `dist/`:

```bash
./tools/council/build.sh     # then serve tools/council/dist
./tools/surprise/build.sh    # then serve tools/surprise/dist
```

The deployed site at [aisuko.github.io/forge](https://aisuko.github.io/forge/)
is a different artifact: `scripts/common/build_site.sh` composes the landing page in
`docs/src/` with both pages here, sharing one core wasm bundle and one copy of
the 6.7 MB checkpoint. The page source is not copied into `docs/` to do it —
`web/` here stays the only definition of each page.

A standalone `dist/` therefore has nav links to pages it does not ship, and
`check_site.py` reports them as warnings. The composed site is built with
`--strict`, where they would be errors.
