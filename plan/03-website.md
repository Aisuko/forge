# Plan 03 — Explainer Website on GitHub Pages (Tailwind + three.js)

**Goal.** A static site that explains what Forge is and how it works, styled
with Tailwind, with one interactive three.js visualization of the GPT-2
transformer stack **and a live in-browser inference demo** running the
char-level Shakespeare model on the visitor's own GPU.

**Live URL (from `git remote`):** `https://aisuko.github.io/forge/`

**Depends on [Plan 04](04-char-shakespeare.md)** for the 43.1 MB char-level
checkpoint. Build the explainer first; the demo section can land second.

### Scope change — read this if you saw an earlier version of this plan

This plan originally excluded the demo, because GPT-2 124M's weights are 548 MB
— over GitHub's 100 MB per-file limit. **Plan 04 removes that constraint.** The
char-level Shakespeare model is 43.1 MB, so the entire demo is self-hosted on
Pages:

| Asset | Size |
|---|---|
| `forge_bg.wasm` | 2.0 MB |
| `model.safetensors` (char, 6L/6H/384d) | 43.1 MB |
| `vocab.json` (65 chars) | < 1 KB |
| **total first load** | **~45 MB** |

versus 548 MB + a HuggingFace CDN dependency. **No cross-origin requests, no
CORS risk, no external CDN.**

GPT-2 124M itself remains out of scope for the site — link to
`scripts/serve_web.sh` for running the full model locally.

---

## Toolchain (versions verified 2026-07-26)

| Tool | Version | Why |
|------|---------|-----|
| Tailwind CSS | 4.3.3 | **standalone CLI** — no Node.js, no `package.json`, matches a Rust repo |
| three.js | r185 | vendored ES module, no CDN |
| `actions/checkout` | v7 | v7.0.1 |
| `actions/configure-pages` | v6 | v6.0.0 |
| `actions/deploy-pages` | v5 | v5.0.0 |
| `actions/upload-pages-artifact` | v5 | v5.0.0 |

Using the [Tailwind standalone CLI](https://tailwindcss.com/blog/standalone-cli)
keeps the repo Node-free. Download it in CI rather than committing a binary.

---

## Task 1 — Site source layout

```
site/
  src/
    index.html          the single page
    input.css           @import "tailwindcss";  + any @theme tokens
    scene.js            three.js transformer visualization
    demo.js             wasm loader + generation UI (Task 6)
  vendor/
    three.module.min.js  three.js r185, pinned + committed
  static/
    .nojekyll            REQUIRED — see Task 4
    favicon.svg
```

The wasm bundle (`web/forge/`) and the char checkpoint are copied into
`site/dist/` by CI rather than duplicated in `site/` — see Task 4.

Build output goes to `site/dist/` (gitignored). CI uploads that directory.

**Tailwind v4 note:** v4 has no `tailwind.config.js` by default — configuration
lives in CSS via `@theme`. `input.css` starts with `@import "tailwindcss";`.
Content detection is automatic in v4; if it misses `scene.js`-injected classes,
add explicit `@source` directives rather than reintroducing a JS config.

Build command:

```bash
tailwindcss -i site/src/input.css -o site/dist/assets/app.css --minify
```

---

## Task 2 — Page content

Source everything from the repo so the site cannot drift into fiction. Do not
invent benchmark numbers.

1. **Hero.** Lead with **nanoGPT-class**, not "scoped to GPT-2" — a
   WebGPU-native GPT in Rust; train a 43 MB Shakespeare model in minutes, and
   load real GPT-2 124M weights with the same code. Take the exact wording from
   the README once Plan 01 Task 6 lands, so site and README cannot drift.
   Buttons: GitHub repo, quick start, roadmap.
   
   Worth one line on the page: nanoGPT *is* the GPT-2 architecture, so this is
   one framework, not two modes. Do **not** claim `.pt` compatibility with
   upstream nanoGPT — the `Linear` weight layout differs (see Plan 01 Task 6).
2. **What it is.** From `README.md`: WGPU production backend
   (Vulkan/Metal/D3D12/browser WebGPU) + a mathematically identical CPU
   reference. Backend-agnostic model code — the same `Gpt2` runs on either.
3. **Three.js visualization.** Task 3.
4. **Verification / parity.** The real measured numbers:
   | Comparison | Max difference |
   |---|---|
   | CPU ↔ WGPU logits | 8.4e-5 |
   | Forge ↔ HF `transformers` | 1.75e-4 |
   
   Plus: greedy continuations identical across CPU, WGPU, and HF; 30 tests
   across 7 suites.
5. **Kernel inventory.** All 23 WGSL kernels from `shaders/`, grouped
   forward / backward / optimizer. Generate this list from the directory at
   build time if practical, so it cannot go stale.
6. **Quick start.** The two commands from `README.md`, in a copy-button code
   block.
7. **Architecture.** The stack diagram from the roadmap
   (GPT-2 → nn modules + autograd tape → tensor ops → backend abstraction →
   WGPU / CPU).
8. **Roadmap.** Stages 1–11 with real status. **Check `README.md` against
   reality first** — it currently claims "Stage 6 complete" while KV-cache,
   autograd, training, and wasm are all implemented and tested. Plan 01 Task 6
   fixes the README; take the corrected version.
9. **Live demo.** Task 6. Place it high on the page — directly after the hero is
   better than burying it below the explainer, since it is the strongest proof
   the framework works.
10. **Footer.** License (Apache-2.0, per `LICENSE`), repo link.

---

## Task 3 — The three.js visualization

**Concept.** A vertical stack of 12 translucent slabs = GPT-2's 12 transformer
blocks. Slow auto-rotate. Hover highlights a block; click expands it to show its
sub-layers (LayerNorm → causal self-attention with 12 heads → LayerNorm → MLP
4×) with real tensor shapes from `Gpt2Config::gpt2()`
(`n_layer=12, n_head=12, n_embd=768, n_ctx=1024`).

**Implementation constraints:**

- Vendor `three.module.min.js` into `site/vendor/` and load it with an
  `importmap`. No CDN — keeps the site self-contained and immune to CDN drift.
- Only import what is used (`WebGLRenderer`, `Scene`, `PerspectiveCamera`,
  `BoxGeometry`, `MeshStandardMaterial`, lights). Skip `OrbitControls` unless
  hand-drag is genuinely wanted; a simple pointer-delta rotation is ~15 lines
  and avoids the extra addon file.
- `renderer.setPixelRatio(Math.min(devicePixelRatio, 2))` — uncapped DPR on a
  4K screen tanks the framerate.
- **Stop the render loop when off-screen.** Use an `IntersectionObserver` to
  `cancelAnimationFrame` when the canvas scrolls out of view; a permanently
  spinning scene drains laptop batteries.
- **`prefers-reduced-motion`:** disable auto-rotation and render a single static
  frame. This is an accessibility requirement, not a nicety.
- **No-WebGL fallback:** if `WebGLRenderer` construction throws, replace the
  canvas with the static architecture diagram from Task 2 item 7. The page must
  be fully informative without WebGL.
- Give the canvas a fixed aspect ratio via CSS so the page does not reflow while
  three.js initializes.

---

## Task 4 — GitHub Actions deployment

Create `.github/workflows/pages.yml`.

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'site/**'
      - 'src/**'                      # wasm is rebuilt from source
      - 'assets/shakespeare_char/**'  # Plan 04 checkpoint
      - '.github/workflows/pages.yml'
      - 'shaders/**'
      - 'README.md'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false
```

Two jobs:

**`build`** on `ubuntu-latest`:
1. `actions/checkout@v7`
2. Download the Tailwind standalone CLI, pinned to v4.3.3:
   ```bash
   curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/download/v4.3.3/tailwindcss-linux-x64
   chmod +x tailwindcss-linux-x64
   ```
   Pin the version — do not fetch `latest`, or the site breaks on an upstream release.
3. `./tailwindcss-linux-x64 -i site/src/input.css -o site/dist/assets/app.css --minify`
4. **Build the wasm bundle** — do not deploy the committed `web/forge/`
   artifact, which can silently go stale relative to `src/`:
   ```bash
   rustup target add wasm32-unknown-unknown
   cargo install wasm-bindgen-cli --version <match the wasm-bindgen crate version in Cargo.lock>
   ./scripts/build_web.sh
   ```
   The `wasm-bindgen-cli` version **must match** the `wasm-bindgen` crate version
   or the CLI errors out on a schema mismatch. Read it from `Cargo.lock` rather
   than hardcoding it.
5. Copy into `site/dist/`: `site/src/index.html`, `site/src/scene.js`,
   `site/src/demo.js`, `site/vendor/`, `site/static/` (including `.nojekyll`),
   `web/forge/` → `dist/forge/`, and `assets/shakespeare_char/` → `dist/model/`
6. `actions/configure-pages@v6`
7. `actions/upload-pages-artifact@v5` with `path: site/dist`

**Artifact size:** the Pages artifact will be ~45 MB. That is well within the
limits (Pages allows 1 GB sites), but it makes each deploy slower — another
reason for the `paths:` filter above.

**`deploy`** — `needs: build`, `environment: github-pages`, runs
`actions/deploy-pages@v5`.

### Two things that will silently break this

1. **`.nojekyll` is mandatory.** Without it, GitHub Pages runs Jekyll, which
   **ignores every file and directory starting with an underscore**. It also
   adds needless build latency. Ship an empty `.nojekyll` at the artifact root.
2. **Pages must be set to "GitHub Actions" as its source** in
   *Settings → Pages → Build and deployment → Source*. This is a one-time
   **manual** repo setting the workflow cannot do for itself. If it is left on
   "Deploy from a branch", `deploy-pages` fails with a permissions-shaped error
   that does not name the real cause. **Tell the user to set this.**

### Path-prefix trap

The site is served from `/forge/`, not `/`. Every asset reference must be
relative (`./assets/app.css`, `./vendor/three.module.min.js`) or absolute with
the `/forge/` prefix. Root-absolute paths like `/assets/app.css` will 404 in
production while working perfectly in local preview — test with:

```bash
python3 -m http.server -d site/dist 8080
# then browse http://localhost:8080/ and confirm no 404s in devtools
```

---

## Task 5 — Quality bar

- **Dark mode.** Use Tailwind's `dark:` variants driven by
  `prefers-color-scheme`. The three.js scene background must follow the theme.
- **Responsive.** Single-column below `md`. The canvas gets a shorter fixed
  height on mobile.
- **Performance budget.** HTML + CSS + JS excluding three.js under 100 KB.
  three.js r185 minified is ~600 KB — load `scene.js` and the three.js module
  with `type="module"` so they do not block first paint, and only initialize
  the scene when the canvas first scrolls into view.
- **Semantics and a11y.** Real `<h1>`/`<h2>` hierarchy, `<nav>`, `<main>`,
  visible focus rings, and `aria-hidden="true"` on the decorative canvas.
- **No external network requests at runtime.** No CDN fonts, no analytics.
  Verify with a devtools network panel showing only same-origin requests.

---

## Task 6 — The live in-browser demo

Runs the Plan 04 char-level model on the visitor's GPU via WebGPU. **Everything
is same-origin** — no CDN, no HuggingFace fetch.

### Why this works on Pages, and the trap it avoids

The usual killer for wasm on GitHub Pages is `SharedArrayBuffer`, which requires
COOP/COEP response headers that **Pages cannot set**. Forge sidesteps this:
`rayon` is gated to `cfg(not(target_arch = "wasm32"))` in `Cargo.toml`, so the
browser path is single-threaded and needs no cross-origin isolation.
`scripts/serve_web.sh` is a bare `python3 -m http.server` with no special
headers, which confirms it. **Do not introduce a wasm-threads dependency** — it
would make the site undeployable on Pages.

### Wiring

`WasmGpt2::load()` (`src/wasm.rs:33`) already takes weight bytes from
JavaScript:

```rust
pub async fn load(model_bytes: Vec<u8>, config_json: &str,
                  vocab_json: &str, merges: &str) -> Result<WasmGpt2, JsValue>
```

`web/index.html:83` hardcodes `const base = "../models/gpt2"`. For the site,
point it at `./model/`. The existing page also gives you `fetchBytes` with
progress reporting (`:53`) and streaming generation via `on_text` — reuse both.

**One signature problem:** `load` takes `merges: &str`, which is BPE-specific.
The char model has a `vocab.json` and no merges. Plan 04 introduces the
`Tokenizer` trait; extend the wasm facade accordingly — either add a
`WasmGpt2::load_char(model_bytes, config_json, vocab_json)` constructor or make
`merges` optional. Decide this while implementing Plan 04 Task 1, not here.

### UI requirements

- Prompt textarea (default something Shakespearean, e.g. `"ROMEO:"`), token
  count, greedy/top-k toggle — mirror the controls already in `web/index.html`
- Download progress bar for the 43 MB fetch; state clearly that the browser
  caches it after first load
- Show `device_info()` (`src/wasm.rs:50`) so visitors see which adapter ran it
- Stream tokens as they arrive; display tokens/s
- **Character-set warning:** the char tokenizer knows only 65 characters. Show
  which characters of the prompt will be dropped or rejected, per whatever
  policy Plan 04 Task 2 settles on. Do not let a stray `é` produce a silent
  wrong answer or an unhandled panic.

### Failure modes — all must be handled, none may blank the page

| Condition | Behavior |
|---|---|
| No WebGPU (`navigator.gpu` undefined) | Replace the demo with an explanatory card naming supported browsers; the rest of the page stays fully functional |
| Adapter request returns null | Same, but say the browser has WebGPU disabled or no suitable GPU |
| Weight fetch fails / offline | Retry button, clear error — never a blank panel |
| Generation throws | Surface the message; `console_error_panic_hook` is already installed by `wasm::start` (`src/wasm.rs:11`) |

The demo is **progressive enhancement**. The page must be complete and
informative with JavaScript disabled entirely.

### Performance

- Lazy-load: do **not** fetch 43 MB on page load. Fetch only when the visitor
  clicks "Run" or the demo scrolls into view. A visitor reading the explainer
  must not pay 43 MB.
- Run generation off the main thread if it janks the UI — but note a Web Worker
  needs its own WebGPU device, so measure before adding that complexity.

---

## Definition of done

- [ ] `site/dist/` builds locally with the standalone CLI and serves correctly
      from `python3 -m http.server` with zero 404s
- [ ] Workflow deploys and `https://aisuko.github.io/forge/` renders
- [ ] Dark mode works; `prefers-reduced-motion` stops the animation
- [ ] Page is fully readable with JavaScript disabled and with WebGL unavailable
- [ ] Zero cross-origin requests at runtime
- [ ] **Demo generates Shakespeare in Chrome/Edge on a real GPU**, streaming
      tokens with a live tokens/s figure
- [ ] Demo degrades to an explanatory card — never a blank panel — with WebGPU
      absent or disabled
- [ ] The 43 MB weight fetch is lazy: loading the page and reading the explainer
      transfers well under 5 MB (verify in the devtools network panel)
- [ ] wasm is rebuilt from `src/` in CI, not copied from the committed
      `web/forge/` artifact
- [ ] Every number and claim on the page traces to `README.md`, `shaders/`, or
      the roadmap — nothing invented
- [ ] Lighthouse ≥ 90 on Performance and ≥ 95 on Accessibility
