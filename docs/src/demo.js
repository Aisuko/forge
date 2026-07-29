// The live demo: the char-level Shakespeare model, run through WebGPU on the
// visitor's own GPU. Everything is same-origin — no CDN, no HuggingFace fetch.
//
// Progressive enhancement throughout: every failure below replaces the panel
// with an explanation, and none of them blanks it or breaks the rest of the
// page. The three moving parts fail independently on purpose — WebGPU (the
// model), WebGL (the 3D stage), and the plain HTML readouts — and any one of
// them missing still leaves a complete page.

const $ = (id) => document.getElementById(id);
const panel = $("demo-panel");
const app = $("demo-app");

// How many blocks the explainer captures in full. It draws one head of one
// block in detail, so one is all it needs — and one layer of readback per step
// instead of three. The rest are the folded stack: they run the identical
// code, and their attention still feeds the slabs.
const DETAIL = 1;
// Probability bars to rank. 24 fits the panel; the char vocabulary is 65.
const TOP_N = 24;
// Above this the prefill detail readback is worth megabytes for one frame, so
// it is skipped and the attention triangle carries the section alone. Lowered
// from 96 when the trace grew the LayerNorm, raw-score and projection
// captures: at 96 positions the prefill readback is 3.75 MB, at 64 it is 2.16.
// The stage draws the last 8 positions either way, so nothing visible is lost.
const MAX_DETAIL_PROMPT = 64;

// ── The stage cover ───────────────────────────────────────────────────────
// The stage draws an empty scaffold until a trace arrives, and the control
// that produces one is a button in the card underneath it. Anyone who reads
// "Watch it think" without scrolling past the whole stage sees grey cells and
// no explanation, so the cover carries the section's state — idle, loading,
// failed — and starts the run itself. It is looked up on each call because
// startStage() removes the whole stage on a WebGL failure.

/** True once the stage has numbers of its own; the cover never returns after. */
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

/** The model's next-token distribution, as bars. */
function renderProbs(el, top) {
  if (!el) return;
  el.textContent = "";
  for (const t of top) {
    const row = document.createElement("div");
    row.className = "prob-row";
    const fill = document.createElement("span");
    fill.className = "prob-fill";
    fill.style.width = `${(t.p * 100).toFixed(1)}%`;
    const text = document.createElement("span");
    text.className = "prob-text";
    const name = document.createElement("span");
    name.textContent = glyph(t.token);
    const p = document.createElement("span");
    p.className = "text-ink-500 dark:text-ink-300";
    p.textContent = t.p.toFixed(3);
    text.append(name, p);
    row.append(fill, text);
    el.append(row);
  }
}

/** The token columns, for the no-WebGL panel. */
function renderTokens(el, tokens) {
  if (!el) return;
  el.textContent = tokens.length
    ? `positions: ${tokens.map(glyph).join(" ")}`
    : "";
}

// ── The 3D stage ──────────────────────────────────────────────────────────
// Separate from the WebGPU check on purpose: WebGL and WebGPU fail
// independently, and either one missing must still leave a complete page.

let stagePromise = null;

/** Start the pipeline at most once; resolves to the controller or null. */
function ensureStage() {
  stagePromise = stagePromise || startStage();
  return stagePromise;
}

async function startStage() {
  const canvas = $("stage");
  if (!canvas) return null;
  try {
    // three.js is 751 KB and lives behind this call, so it is fetched when
    // the section is reached rather than on first paint.
    const { createPipeline } = await import("./pipeline.js");
    const pipeline = createPipeline({
      canvas,
      overlay: $("stage-overlay"),
      readout: $("stage-readout"),
      onStage: showStage,
    });
    wireStepper(pipeline);
    return pipeline;
  } catch (e) {
    // No WebGL, or the module itself failed to load. Drop the canvas
    // entirely — an empty rectangle is worse than no rectangle — and show
    // the panel that says the same thing in words. The caption strip, the
    // probability bars and the token list beside it are plain HTML and keep
    // working.
    console.warn("3D pipeline unavailable:", e);
    $("stage-wrap")?.remove();
    const fallback = $("stage-fallback");
    if (fallback) fallback.hidden = false;
    return null;
  }
}

// ── the walk ──────────────────────────────────────────────────────────────
// The stepper is DOM; the camera is three.js. They meet here, and only here,
// so the caption strip is readable prose in index.html rather than strings in
// the JS bundle.

/** Stage id → its <p>, read once out of the template in index.html. */
const PROSE = (() => {
  const t = document.getElementById("stage-prose");
  const out = {};
  for (const p of t ? t.content.querySelectorAll("p[data-stage]") : []) {
    out[p.dataset.stage] = p.innerHTML;
  }
  return out;
})();

let pipelineRef = null;

/** Reflect the walk's position into the caption strip and the counter. */
function showStage(index, stage) {
  const count = pipelineRef?.stageCount ?? 15;
  const counter = $("stage-counter");
  const title = $("stage-caption-title");
  const body = $("stage-caption");
  const slice = $("stage-caption-slice");
  if (!counter) return;
  if (index < 0 || !stage) {
    counter.textContent = `Overview · ${count} stages`;
    title.textContent = "The whole forward pass";
    body.innerHTML =
      "Fifteen stages, from the embedding to the next character. Step " +
      "through them, or hover any cell at any time to read the number under " +
      "it.";
    slice.textContent = "";
    return;
  }
  counter.textContent = `${index + 1} / ${count} · ${stage.name}`;
  title.textContent = stage.name;
  body.innerHTML = PROSE[stage.id] || "";
  slice.textContent = `showing: ${stage.slice}`;
}

function wireStepper(pipeline) {
  pipelineRef = pipeline;
  const buttons = ["stage-prev", "stage-next", "stage-overview"].map($);
  const [prev, next, over] = buttons;
  if (!prev) return;
  for (const b of buttons) b.disabled = false;
  prev.addEventListener("click", () => pipeline.prevStage());
  next.addEventListener("click", () => pipeline.nextStage());
  over.addEventListener("click", () => pipeline.overview());
  // Arrow keys once a stepper button has focus. Bound here rather than to the
  // document: a visitor typing a prompt is not trying to walk the pipeline.
  // Not to the canvas either — it is aria-hidden, and making it focusable
  // would put a stop on the tab order that a screen reader cannot describe.
  prev.parentElement?.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") pipeline.nextStage();
    else if (e.key === "ArrowLeft") pipeline.prevStage();
    else if (e.key === "Escape") pipeline.overview();
    else return;
    e.preventDefault();
  });
  showStage(pipeline.stage(), null);
}

const section = $("demo");
if (section && "IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (entries, obs) => {
      if (entries.some((e) => e.isIntersecting)) {
        obs.disconnect();
        ensureStage();
      }
    },
    { rootMargin: "200px" },
  );
  io.observe(section);
} else {
  ensureStage();
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
  const el = $("demo-quality");
  if (!el) return;
  try {
    const m = await fetch("./model/metrics.json").then((r) =>
      r.ok ? r.json() : null,
    );
    if (!m || typeof m.val_loss !== "number") return;
    const when = m.measured ? ` on ${m.measured}` : "";
    el.textContent =
      `Held-out validation loss ${m.val_loss.toFixed(3)} on Tiny Shakespeare, ` +
      `measured${when} over ${m.val_windows ?? "?"} windows of the 10% this ` +
      `model never trained on. nanoGPT's published reference for this ` +
      `configuration is 1.4697.`;
    el.hidden = false;
  } catch {
    // No metrics file, or it is not JSON. Say nothing.
  }
}

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

  /** Fetch with a progress callback — 43 MB deserves a bar. */
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

  // Lazy by design: a visitor reading the explainer must not pay 43 MB. The
  // fetch starts on the first Run, not on page load.
  async function load() {
    const { default: init, WasmGpt2 } = await import("./forge/forge.js");
    await init();

    status("fetching weights (43 MB, cached by your browser after this)…");
    progress(0);
    const base = "./model";
    const [bytes, config, vocab] = await Promise.all([
      fetchBytes(`${base}/model.safetensors`, progress),
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

  showQuality();

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

      const detailLayers = Math.min(DETAIL, model.n_layer());

      // The visualisation must describe the model that is running, not the
      // defaults it was built with.
      const stage = await ensureStage();
      stage?.setConfig({
        nLayer: model.n_layer(),
        nHead: model.n_head(),
        nEmbd: model.n_embd(),
        nCtx: model.n_ctx(),
      });
      stage?.reset();

      const prompt = $("demo-prompt").value;
      const tokens = Array.from(model.tokenize_display(prompt));
      stage?.setTokens(tokens);
      renderTokens($("stage-fallback-tokens"), tokens);

      // The §3.6 guard: a long prompt would make the prefill readback the most
      // expensive thing on the page, for one frame nobody asked for.
      const withDetail = tokens.length <= MAX_DETAIL_PROMPT;
      stage?.setDetailEnabled(withDetail);

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
          return;
        }
        count += 1;
        stage?.pushToken(s);
        const dt = (performance.now() - decodeStart) / 1000;
        if (dt > 0) status(`generating — ${(count / dt).toFixed(1)} tok/s`);
      };

      const onTrace = (trace) => {
        // A failing visualization must never take generation down with it.
        try {
          // The first trace is the moment the stage stops being a scaffold.
          if (stage) stageGoLive();
          stage?.pushTrace(trace);
          renderProbs($("stage-probs"), trace.top);
        } catch (e) {
          console.warn("visualization step failed:", e);
        }
      };

      // Text generation never depends on the views. onTrace swallows its own
      // failures, and the trace path is taken even with no WebGL at all —
      // the probability bars are HTML, and they are the readout that must
      // survive everything else going missing.
      await model.generate_with_trace(
        prompt,
        n,
        topk,
        temp,
        BigInt(Date.now() % 100000),
        onText,
        onTrace,
        stage && withDetail ? detailLayers : 0,
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
