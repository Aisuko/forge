// The live demo: the char-level Shakespeare model, run through WebGPU on the
// visitor's own GPU. Everything is same-origin — no CDN, no HuggingFace fetch.
//
// Progressive enhancement throughout: every failure below replaces the panel
// with an explanation, and none of them blanks it or breaks the rest of the
// page. The three moving parts fail independently on purpose — WebGPU (the
// model), the 2D attention grid, and the plain HTML readouts — and any one of
// them missing still leaves a complete page.

const $ = (id) => document.getElementById(id);
const panel = $("demo-panel");
const app = $("demo-app");

// Candidates the trace carries per step, out of a 65-character vocabulary.
// decision.js draws the top eight of them and accounts for the rest in a line;
// the extra depth is what keeps the sampled character findable when top-k is
// raised above eight, since only a candidate present in this list can be marked
// as the one that was chosen.
const TOP_N = 24;

// ── The grid cover ────────────────────────────────────────────────────────
// The grid is empty until a trace arrives, and the control that produces one
// is a button in the card underneath it. Anyone who reads "Watch it think"
// without scrolling past it sees a blank box and no explanation, so the cover
// carries the section's state — idle, loading, failed — and starts the run
// itself. It is looked up on each call because startStage() removes the whole
// grid if the canvas cannot be created.

/** True once the grid has numbers of its own; the cover never returns after. */
let stageLive = false;

/** Show the cover with `text`, offering the Run button only when it can help. */
function stageState(text, offerRun) {
  const idle = $("stage-idle");
  if (!idle || stageLive) return;
  idle.hidden = false;
  $("stage-idle-text").textContent = text;
  $("stage-idle-run").hidden = !offerRun;
}

/** Uncover the stage for good — called on the first trace it can draw. */
function stageGoLive() {
  stageLive = true;
  const idle = $("stage-idle");
  if (idle) idle.hidden = true;
}

/** Replace the demo with an explanatory card. Never leaves an empty box. */
function explain(title, body, retry) {
  panel.innerHTML = "";
  const h = document.createElement("h3");
  h.className = "font-semibold";
  h.textContent = title;
  const p = document.createElement("p");
  p.className = "mt-2 text-sm text-ink-500 dark:text-ink-300";
  p.textContent = body;
  panel.append(h, p);
  if (retry) {
    const b = document.createElement("button");
    b.className = "btn-primary mt-4";
    b.type = "button";
    b.textContent = "Try again";
    b.addEventListener("click", retry);
    panel.append(b);
  }
}

// ── HTML readouts ─────────────────────────────────────────────────────────
// Plain DOM, no three.js: these are what survives when WebGL does not, and a
// ranked list gains nothing from being drawn in 3D.

const glyph = (s) => (s ?? "").replace(/\n/g, "↵").replace(/ /g, "␣") || "·";

/** A number input's value, or `fallback` when it is empty or nonsense. */
function clampNum(raw, lo, hi, fallback) {
  const v = Number(raw);
  if (!Number.isFinite(v)) return fallback;
  return Math.min(hi, Math.max(lo, v));
}

/** The token columns, for the no-WebGL panel. */
function renderTokens(el, tokens) {
  if (!el) return;
  el.textContent = tokens.length
    ? `positions: ${tokens.map(glyph).join(" ")}`
    : "";
}

// ── The attention grid ────────────────────────────────────────────────────
// Separate from the WebGPU check on purpose: a missing 2D context and a
// missing WebGPU adapter fail independently, and either one must still leave
// a complete page.

let stagePromise = null;

/** Start the grid at most once; resolves to the controller or null. */
function ensureStage() {
  stagePromise = stagePromise || startStage();
  return stagePromise;
}

async function startStage() {
  const canvas = $("heat");
  if (!canvas) return null;
  try {
    const { createAttention } = await import("./attention.js");
    const heat = createAttention({
      canvas,
      readout: $("heat-readout"),
      strip: $("heat-strip"),
      caption: $("heat-caption"),
      onSelect: showPickers,
    });
    wirePickers(heat);
    // A canvas inside a closed <details> has clientWidth 0, which draw()
    // treats as "not laid out yet" and skips — so without this the matrix
    // opens blank the first time and stays blank until the next trace.
    $("heat-details")?.addEventListener("toggle", (e) => {
      if (e.target.open) heat.redraw();
    });
    return heat;
  } catch (e) {
    // No 2D context, or the module itself failed to load. Drop both views it
    // owns — an empty rectangle is worse than no rectangle — and show the panel
    // that says the same thing in words. Generation is untouched, and so is
    // "What it chose": decision.js is a separate import over separate data, so
    // the section keeps its lead panel and loses only the attention half.
    console.warn("attention views unavailable:", e);
    $("heat-details")?.remove();
    $("look-card")?.remove();
    const fallback = $("stage-fallback");
    if (fallback) fallback.hidden = false;
    return null;
  }
}

// ── the decision panel ────────────────────────────────────────────────────
// Independent of both WebGPU and the canvas: plain DOM over `trace.top`, which
// Rust has been sending all along.

let decisionPromise = null;

function ensureDecision() {
  decisionPromise = decisionPromise || startDecision();
  return decisionPromise;
}

async function startDecision() {
  const bars = $("dec-bars");
  if (!bars) return null;
  try {
    const { createDecision } = await import("./decision.js");
    return createDecision({ bars, context: $("dec-context"), note: $("dec-note") });
  } catch (e) {
    console.warn("decision panel unavailable:", e);
    return null;
  }
}

// ── block and head pickers ────────────────────────────────────────────────
// Built from the model's own config rather than hard-coded to 6 x 6: the
// section has to describe the model that is running.

let heatRef = null;

function wirePickers(heat) {
  heatRef = heat;
  for (const [id, pick] of [
    ["pick-block", (i) => heat.select(i, heat.selection().head)],
    ["pick-head", (i) => heat.select(heat.selection().block, i)],
  ]) {
    $(id)?.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-i]");
      if (b) pick(Number(b.dataset.i));
    });
  }
}

/** Redraw both rows of buttons; also the first call that creates them. */
function showPickers(block, head, cfg) {
  fillPicker($("pick-block"), cfg.nLayer, block);
  fillPicker($("pick-head"), cfg.nHead, head);
}

function fillPicker(el, count, active) {
  if (!el) return;
  if (el.children.length !== count) {
    el.textContent = "";
    for (let i = 0; i < count; i++) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "pick";
      b.dataset.i = String(i);
      b.textContent = String(i + 1);
      el.append(b);
    }
  }
  for (const b of el.children) {
    const on = Number(b.dataset.i) === active;
    b.classList.toggle("is-on", on);
    b.setAttribute("aria-pressed", String(on));
  }
}

const section = $("demo");
if (section && "IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (entries, obs) => {
      if (entries.some((e) => e.isIntersecting)) {
        obs.disconnect();
        ensureStage();
        ensureDecision();
      }
    },
    { rootMargin: "200px" },
  );
  io.observe(section);
} else {
  ensureStage();
  ensureDecision();
}

// ── The demo itself ───────────────────────────────────────────────────────

if (!("gpu" in navigator)) {
  explain(
    "WebGPU is not available in this browser",
    "The demo needs WebGPU: Chrome or Edge 113+, or Safari 26+. Everything " +
      "else on this page works without it, and the model runs natively with " +
      "cargo run --release --example generate.",
  );
  // Say it on the stage too. Left as it was, the scaffold would sit there
  // empty and unexplained for exactly the visitors who can never fill it.
  stageState(
    "This browser has no WebGPU, so the model cannot run here and the " +
      "stage stays empty. Chrome or Edge 113+, or Safari 26+, will fill it.",
    false,
  );
} else {
  app.hidden = false;
  wire();
}

/**
 * The shipped model's measured quality, from `model/metrics.json` — written by
 * scripts/ship_char_model.sh, not typed into this file.
 *
 * Silent when the file is missing or malformed: the page must stay correct
 * when it is served against a model directory that predates the trainer
 * change, and a stale hard-coded figure is exactly what this replaces.
 */
async function showQuality() {
  try {
    const m = await fetch("./model/metrics.json").then((r) =>
      r.ok ? r.json() : null,
    );
    if (!m || typeof m.val_loss !== "number") return;
    const val = m.val_loss.toFixed(3);

    // The model card quotes the same measured figure, and one source beats two
    // places to edit. Filled first and unconditionally: it describes the model,
    // not the browser, so it must survive a machine with no WebGPU — and
    // explain() may already have replaced the panel holding #demo-quality.
    const card = $("model-val");
    if (card) card.textContent = val;

    const el = $("demo-quality");
    if (!el) return;
    const when = m.measured ? ` on ${m.measured}` : "";
    el.textContent =
      `Held-out validation loss ${val} on Tiny Shakespeare, ` +
      `measured${when} over ${m.val_windows ?? "?"} windows of the 10% this ` +
      `model never trained on. nanoGPT's published reference for this ` +
      `configuration is 1.4697.`;
    el.hidden = false;
  } catch {
    // No metrics file, or it is not JSON. Say nothing.
  }
}

// Not inside wire(): the card's number is a fact about the shipped weights, and
// a visitor without WebGPU is still entitled to it.
showQuality();

function wire() {
  const status = (t) => {
    // explain() can replace the whole panel, and this element with it.
    const el = $("demo-status");
    if (el) el.textContent = t;
    // Same message on the stage until the stage has numbers to show instead.
    stageState(t, false);
  };
  const progress = (frac) => {
    const wrap = $("demo-progress-wrap");
    wrap.hidden = frac === null;
    if (frac !== null) $("demo-progress").style.width = `${(frac * 100).toFixed(1)}%`;
  };

  let model = null;
  let loading = null;
  let stop = false;
  let weightBytes = 0;

  /** Fetch with a progress callback — even 6.7 MB deserves a bar. */
  async function fetchBytes(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get("Content-Length")) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (total && onProgress) onProgress(got / total);
    }
    const out = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.length;
    }
    return out;
  }

  // Lazy by design: a visitor reading the explainer must not pay for weights
  // they never run. The fetch starts on the first Run, not on page load.
  async function load() {
    const { default: init, WasmGpt2 } = await import("./forge/forge.js");
    await init();

    status("fetching weights (6.7 MB, cached by your browser after this)…");
    progress(0);
    const base = "./model";
    const [bytes, config, vocab] = await Promise.all([
      fetchBytes(`${base}/model.fzm`, progress),
      fetch(`${base}/config.json`).then((r) => r.text()),
      fetch(`${base}/vocab.json`).then((r) => r.text()),
    ]);
    progress(null);
    weightBytes = bytes.length;

    status("requesting a WebGPU adapter and uploading weights…");
    const m = await WasmGpt2.load_char(bytes, config, vocab);
    status(`ready — ${m.device_info()} · ${m.vocab_size()}-token ${m.tokenizer_kind()} vocab`);
    return m;
  }

  /**
   * Parameter count from the config the model reports, not a constant typed
   * into this file. Weight-tied LM head, so wte is counted once.
   */
  function paramCount(m) {
    const c = m.n_embd();
    // Per block: 12c² of weights (3c² qkv, c² proj, 4c² fc, 4c² mlp proj) and
    // 13c of biases and LayerNorm parameters.
    const perBlock = 12 * c * c + 13 * c;
    return m.vocab_size() * c + m.n_ctx() * c + m.n_layer() * perBlock + 2 * c;
  }

  /** The efficiency tiles. Measured values or an em dash — never a guess. */
  function fillTiles(m, tokPerSec) {
    $("eff-gpu").textContent = m.device_info();
    $("eff-tps").textContent = `${tokPerSec.toFixed(1)} tok/s`;
    $("eff-size").textContent = `${(weightBytes / 1e6).toFixed(1)} MB`;
    $("eff-params").textContent = `${(paramCount(m) / 1e6).toFixed(2)} M`;
    $("eff-note").textContent =
      "Measured on your machine, in the run above — not on ours.";
  }

  /** The char vocab knows 65 characters; say so before generating, not after. */
  function checkCharset() {
    const el = $("demo-charset");
    if (!model) {
      el.hidden = true;
      return true;
    }
    const bad = model.unsupported_chars($("demo-prompt").value);
    if (!bad) {
      el.hidden = true;
      return true;
    }
    el.hidden = false;
    el.textContent =
      `This model's ${model.vocab_size()}-character vocabulary has no ` +
      `${[...bad].map((c) => JSON.stringify(c)).join(", ")}. ` +
      `Remove ${bad.length > 1 ? "them" : "it"} to generate.`;
    return false;
  }

  $("demo-prompt").addEventListener("input", checkCharset);

  // Live value beside the slider — a bare range input tells you nothing.
  const temp = $("demo-temp");
  const tempOut = $("demo-temp-out");
  const showTemp = () => (tempOut.textContent = Number(temp.value).toFixed(2));
  temp.addEventListener("input", showTemp);
  showTemp();

  // One run path, two buttons: the cover's button drives the real one so the
  // prompt, token count and sampling below always apply.
  $("stage-idle-run")?.addEventListener("click", () => $("demo-run").click());

  $("demo-stop").addEventListener("click", () => {
    stop = true;
    status("stopping…");
  });

  $("demo-run").addEventListener("click", async () => {
    const run = $("demo-run");
    run.disabled = true;
    // Pressing Run twice must not offer a third press from the cover.
    stageState("starting…", false);
    try {
      if (!model) {
        // One in-flight load, however many times Run is pressed.
        loading = loading || load();
        model = await loading;
      }
      if (!checkCharset()) {
        // Nothing will run, so the cover has to hand the visitor back a way
        // to try again once they have fixed the prompt.
        stageState($("demo-charset").textContent, true);
        return;
      }

      // The visualisation must describe the model that is running, not the
      // defaults it was built with.
      const stage = await ensureStage();
      stage?.setConfig({
        nLayer: model.n_layer(),
        nHead: model.n_head(),
        nCtx: model.n_ctx(),
      });
      stage?.reset();

      const dec = await ensureDecision();
      dec?.setVocab(model.vocab_size());
      dec?.reset();

      const prompt = $("demo-prompt").value;
      const tokens = Array.from(model.tokenize_display(prompt));
      stage?.setTokens(tokens);
      renderTokens($("stage-fallback-tokens"), tokens);

      $("demo-output").textContent = "";
      $("demo-stop").hidden = false;
      stop = false;

      // n_ctx is 256; the prompt takes the rest, so cap well below it.
      const n = Math.max(1, Math.min(240, Number($("demo-tokens").value) || 200));
      const topk = clampNum($("demo-topk").value, 0, model.vocab_size(), 12);
      const temp = clampNum($("demo-temp").value, 0.1, 1.5, 0.8);
      const t0 = performance.now();
      let count = 0;
      let decodeStart = null;

      const onText = (s) => {
        // Returning false stops generation after the current token.
        if (stop) return false;
        $("demo-output").textContent += s;
        // The first delta is the prompt itself, echoed once the prefill lands.
        // It is not a generated token and must not be counted as one.
        if (decodeStart === null) {
          decodeStart = performance.now();
          dec?.setContext(s);
          return;
        }
        count += 1;
        stage?.pushToken(s);
        // The character that settles the bars drawn by the trace before it —
        // see onTrace. Nothing is marked when it is not among them.
        dec?.chose(s);
        const dt = (performance.now() - decodeStart) / 1000;
        if (dt > 0) status(`generating — ${(count / dt).toFixed(1)} tok/s`);
      };

      const onTrace = (trace) => {
        // A failing visualization must never take generation down with it.
        try {
          // The first trace is the moment the panels stop being a scaffold.
          if (stage || dec) stageGoLive();
          stage?.pushTrace(trace);
          // The shortlist for the character that has not been sampled yet: the
          // model computes a step, this fires, and only then does generation
          // pick from it and emit. So the bars always land one callback before
          // the onText that marks the winner.
          dec?.pushTop(trace.top);
        } catch (e) {
          console.warn("visualization step failed:", e);
        }
      };

      // Text generation never depends on the views: onTrace swallows its own
      // failures, so a broken grid costs the picture and nothing else.
      //
      // detail_layers is 0 and now always will be: `attn` carries every
      // block's probabilities on its own, and the per-layer tensor capture
      // that fed the 3D tiles cost megabytes of readback per prefill for a
      // picture nothing draws any more.
      await model.generate_with_trace(
        prompt,
        n,
        topk,
        temp,
        BigInt(Date.now() % 100000),
        onText,
        onTrace,
        0,
        TOP_N,
      );

      const dt = (performance.now() - t0) / 1000;
      const decode =
        decodeStart === null ? dt : (performance.now() - decodeStart) / 1000;
      const rate = count / Math.max(decode, 1e-3);
      status(
        `${count} tokens in ${dt.toFixed(1)}s · ${rate.toFixed(1)} tok/s · ` +
          `${model.device_info()}`,
      );
      dec?.end();
      if (count > 0) fillTiles(model, rate);
    } catch (e) {
      // console_error_panic_hook is installed by wasm::start, so a Rust panic
      // arrives here with a real message rather than "unreachable".
      progress(null);
      const msg = String((e && e.message) || e);
      if (!model) {
        explain(
          "Could not start the demo",
          `${msg} — this is usually a failed weight download or a browser with ` +
            "WebGPU disabled at chrome://flags.",
          () => location.reload(),
        );
        // explain() just removed the Run button this cover points at, so the
        // reload is the only offer left to make.
        stageState(`The model could not start — ${msg}`, false);
      } else {
        status(`error: ${msg}`);
      }
    } finally {
      // explain() may have replaced the panel, taking both buttons with it.
      const runBtn = $("demo-run");
      if (runBtn) runBtn.disabled = false;
      const stopBtn = $("demo-stop");
      if (stopBtn) stopBtn.hidden = true;
    }
  });

  // ── Stage 11 gate (roadmap v4) ────────────────────────────────────────
  // Browser greedy tokens must be identical to native WGPU for the same
  // prompt. Opening the page with ?gate runs the check and reports through
  // document.title so a headless driver can poll it. There is deliberately no
  // button: this is a verification hook, not a feature.
  if (new URLSearchParams(location.search).has("gate")) {
    runGate().catch((e) => {
      document.title = `GATE ERROR: ${e}`;
    });
  }

  async function runGate() {
    status("running the Stage 11 gate…");
    const expected = await fetch("./gate_expected.json").then((r) => {
      if (!r.ok) {
        throw new Error(
          "gate_expected.json missing — run `cargo run --release --example gate_tokens`",
        );
      }
      return r.json();
    });
    model = model || (await (loading = loading || load()));

    const got = Array.from(
      await model.greedy_ids(expected.prompt, expected.max_new_tokens),
    );
    const pass =
      got.length === expected.ids.length &&
      got.every((v, i) => v === expected.ids[i]);
    const where = got.findIndex((v, i) => v !== expected.ids[i]);

    document.title = pass ? "GATE PASS" : "GATE FAIL";
    status(
      pass
        ? `GATE PASS — ${got.length} greedy tokens identical to native WGPU`
        : `GATE FAIL — first mismatch at index ${where}`,
    );
    $("demo-output").textContent =
      `model:    ${expected.model}\nprompt:   ${expected.prompt}\n` +
      `expected: ${expected.ids.length} ids from ${expected.backend}\n` +
      `got:      ${got.length} ids from ${model.device_info()}`;
    // Machine-readable hook for headless verification.
    window.__forge_gate = { pass, got, expected: expected.ids };
  }
}
