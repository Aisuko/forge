// The live demo: the char-level Shakespeare model, run through WebGPU on the
// visitor's own GPU. Everything is same-origin — no CDN, no HuggingFace fetch.
//
// Progressive enhancement throughout: every failure below replaces the panel
// with an explanation, and none of them blanks it or breaks the rest of the
// page.

const $ = (id) => document.getElementById(id);
const panel = $("demo-panel");
const app = $("demo-app");

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

// ── The 3D stack ──────────────────────────────────────────────────────────
// Separate from the WebGPU check on purpose: WebGL and WebGPU fail
// independently, and either one missing must still leave a complete page.

let scenePromise = null;

/** Start the scene at most once; resolves to the controller or null. */
function ensureScene() {
  scenePromise = scenePromise || startScene();
  return scenePromise;
}

async function startScene() {
  const canvas = $("scene");
  if (!canvas) return null;
  try {
    // three.js is 751 KB and lives behind this call, so it is fetched when
    // the section is reached rather than on first paint.
    const { createStack } = await import("./scene.js");
    return createStack({ canvas, label: $("scene-label") });
  } catch (e) {
    // No WebGL, or the module itself failed to load. Drop the canvas
    // entirely — an empty rectangle is worse than no rectangle — and open
    // the text architecture, which says the same thing in words.
    console.warn("3D stack unavailable:", e);
    $("scene-card")?.remove();
    $("demo-grid")?.classList.remove("md:grid-cols-2");
    const text = $("stack-text");
    if (text) text.open = true;
    return null;
  }
}

const section = $("demo");
if (section && "IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (entries, obs) => {
      if (entries.some((e) => e.isIntersecting)) {
        obs.disconnect();
        ensureScene();
      }
    },
    { rootMargin: "200px" },
  );
  io.observe(section);
} else {
  ensureScene();
}

// ── The demo itself ───────────────────────────────────────────────────────

if (!("gpu" in navigator)) {
  explain(
    "WebGPU is not available in this browser",
    "The demo needs WebGPU: Chrome or Edge 113+, or Safari 26+. Everything " +
      "else on this page works without it, and the model runs natively with " +
      "cargo run --release --example generate.",
  );
} else {
  app.hidden = false;
  wire();
}

function wire() {
  const status = (t) => {
    $("demo-status").textContent = t;
  };
  const progress = (frac) => {
    const wrap = $("demo-progress-wrap");
    wrap.hidden = frac === null;
    if (frac !== null) $("demo-progress").style.width = `${(frac * 100).toFixed(1)}%`;
  };

  let model = null;
  let loading = null;
  let stop = false;

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

    status("requesting a WebGPU adapter and uploading weights…");
    const m = await WasmGpt2.load_char(bytes, config, vocab);
    status(`ready — ${m.device_info()} · ${m.vocab_size()}-token ${m.tokenizer_kind()} vocab`);
    return m;
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

  $("demo-stop").addEventListener("click", () => {
    stop = true;
    status("stopping…");
  });

  $("demo-run").addEventListener("click", async () => {
    const run = $("demo-run");
    run.disabled = true;
    try {
      if (!model) {
        // One in-flight load, however many times Run is pressed.
        loading = loading || load();
        model = await loading;
      }
      if (!checkCharset()) return;

      // The visualization must describe the model that is running, not the
      // defaults it was built with.
      const scene = await ensureScene();
      scene?.setConfig({
        nLayer: model.n_layer(),
        nHead: model.n_head(),
        nEmbd: model.n_embd(),
        nCtx: model.n_ctx(),
      });
      scene?.reset();

      $("demo-output").textContent = "";
      $("demo-stop").hidden = false;
      stop = false;

      // n_ctx is 256; the prompt takes the rest, so cap well below it.
      const n = Math.max(1, Math.min(240, Number($("demo-tokens").value) || 200));
      const topk = Number($("demo-sampling").value);
      const t0 = performance.now();
      let count = 0;
      let first = null;

      const onText = (s) => {
        // Returning false stops generation after the current token.
        if (stop) return false;
        if (first === null) first = performance.now();
        count += 1;
        $("demo-output").textContent += s;
        const dt = (performance.now() - first) / 1000;
        if (dt > 0) {
          status(`generating — ${(count / dt).toFixed(1)} tok/s`);
        }
      };
      const args = [
        $("demo-prompt").value,
        n,
        topk,
        0.8,
        BigInt(Date.now() % 100000),
        onText,
      ];

      // Text generation never depends on the 3D view: without it the plain
      // path runs, and it does no attention readback at all.
      await (scene
        ? model.generate_with_attention(...args, (layer, nHead, weights) =>
            scene.pushAttention(layer, nHead, weights),
          )
        : model.generate(...args));

      const dt = (performance.now() - t0) / 1000;
      const decode = first === null ? dt : (performance.now() - first) / 1000;
      status(
        `${count} tokens in ${dt.toFixed(1)}s · ${(count / Math.max(decode, 1e-3)).toFixed(1)} tok/s · ` +
          `${model.device_info()}`,
      );
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
      } else {
        status(`error: ${msg}`);
      }
    } finally {
      $("demo-run").disabled = false;
      $("demo-stop").hidden = true;
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
