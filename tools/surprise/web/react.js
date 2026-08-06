/**
 * The reactive page: select text, and the model tints it by how surprised it
 * was to find those characters there.
 *
 * Everything drawn comes from one call — `WasmGpt2.surprisal(text)`, which runs
 * a single forward pass and returns per-position bits, plus the character the
 * model would have chosen instead. Nothing here re-derives a probability in
 * JavaScript; the only thing this file decides is colour.
 *
 * Three constraints shape the code, and all three are about the model being
 * genuinely local rather than behind a network call:
 *
 *  - load once, at page open, so every interaction after it is instant;
 *  - cancel in flight, because a reader dragging a selection outruns the GPU;
 *  - never reload the model, whatever happens.
 */

import init, { WasmGpt2 } from "./forge/forge.js";

const $ = (id) => document.getElementById(id);

/** Bits at which the scale saturates. log2(65) ≈ 6.02 is the char model's
 *  uniform-guess entropy, so 6 is "it had no idea" and makes a natural top. */
const MAX_BITS = 6;

/** How long the reader has to stop moving before we ask the GPU anything. */
const DEBOUNCE_MS = 150;

const PASSAGES = [
  {
    label: "Shakespeare — what it was trained on",
    text: `ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief.`,
  },
  {
    label: "Shakespeare — a different play",
    text: `MENENIUS:
There was a time when all the body's members
Rebell'd against the belly, thus accused it:
That only like a gulf it did remain
I' the midst o' the body, idle and unactive.`,
  },
  {
    // Deliberately inside the 65-character vocabulary — the model's alphabet is
    // essentially tinyshakespeare's, which has letters, a handful of
    // punctuation and, oddly, the digit 3. A passage full of characters it has
    // never seen would demonstrate a missing alphabet, not surprise, and this
    // passage is here to demonstrate surprise.
    label: "Modern prose — nothing like what it was trained on",
    text: `The deployment pipeline failed again this morning because the
container registry rate-limited our pull, and nobody had configured a retry.`,
  },
];

const state = {
  model: null,
  /** Monotonic id; a result whose id is stale is dropped rather than drawn. */
  generation: 0,
  timer: null,
  /** The element whose text is currently scored, so hover can find its data. */
  scored: new WeakMap(),
};

const el = {
  status: (t) => ($("react-status").textContent = t),
  progress: (frac) => {
    const wrap = $("react-progress-wrap");
    wrap.hidden = frac === null;
    if (frac !== null) $("react-progress").style.width = `${(frac * 100).toFixed(0)}%`;
  },
  stat: (t) => ($("react-stat").textContent = t),
  readout: (t) => ($("react-readout").textContent = t),
  warn: (t) => {
    const w = $("react-warn");
    w.hidden = !t;
    w.textContent = t || "";
  },
};

/* ── colour ─────────────────────────────────────────────────────────────
 *
 * Sequential, not diverging: surprise runs one way, and a diverging scale
 * would invent a midpoint that means nothing. Low surprise stays near the page
 * background so the text still reads as text; high surprise saturates toward
 * Forge orange. Lightness carries the signal as well as hue, so the scale
 * survives being colour-blind or printed in grey.
 */

function tint(bits) {
  const f = Math.min(Math.max(bits / MAX_BITS, 0), 1);
  // Hue 40° (amber) → 18° (Forge orange); saturation and alpha ramp together.
  const hue = 40 - 22 * f;
  const sat = 25 + 65 * f;
  const alpha = 0.10 + 0.62 * f;
  return `hsl(${hue} ${sat}% 55% / ${alpha})`;
}

function paintScale() {
  const stops = [];
  for (let i = 0; i <= 10; i++) stops.push(tint((i / 10) * MAX_BITS));
  $("react-scale").style.background = `linear-gradient(to right, ${stops.join(",")})`;
}

/* ── loading ─────────────────────────────────────────────────────────── */

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress(got / total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/**
 * Eager, unlike the demo page's lazy load — and that is the whole design. A
 * page that reacts to reading cannot ask the reader to press something first,
 * so it pays the 6.7 MB up front and everything after it is a few milliseconds.
 */
async function load() {
  el.status("starting wasm…");
  await init();

  el.status("fetching weights (6.7 MB, cached by your browser after this)…");
  el.progress(0);
  const base = "./model";
  const [bytes, config, vocab] = await Promise.all([
    fetchBytes(`${base}/model.fzm`, el.progress),
    fetch(`${base}/config.json`).then((r) => r.text()),
    fetch(`${base}/vocab.json`).then((r) => r.text()),
  ]);
  el.progress(null);

  el.status("requesting a WebGPU adapter and uploading weights…");
  const m = await WasmGpt2.load_char(bytes, config, vocab);
  $("react-device").textContent = m.device_info();
  el.status(`ready — ${m.vocab_size()}-character vocabulary, ${m.n_layer()} layers`);
  return m;
}

/* ── scoring ─────────────────────────────────────────────────────────── */

/**
 * Score `text` and paint it into `target`.
 *
 * Takes a generation id and drops its own result if a newer request started
 * while the GPU was busy. Without this the page paints answers to selections
 * the reader has already left — the single most visible way an interactive
 * model demo looks broken.
 */
async function score(text, target, gen, { warns = false, stat = el.stat, onDone } = {}) {
  if (!state.model || !text.trim()) return;

  const unsupported = state.model.unsupported_chars(text);
  if (unsupported) {
    text = [...text].filter((c) => !unsupported.includes(c)).join("");
    if (!text.trim()) {
      if (warns) el.warn("none of those characters are in the model's alphabet");
      return;
    }
  }
  // Only the reader's own text can raise this: the passages above are curated
  // to stay inside the alphabet. Scoping it means scoring one region never
  // clears or invents a warning about another.
  if (warns) {
    el.warn(
      unsupported
        ? `not in the model's 65-character alphabet, so skipped: ` +
            `${[...unsupported].join(" ")}`
        : "",
    );
  }

  const t0 = performance.now();
  let out;
  try {
    out = await state.model.surprisal(text);
  } catch (e) {
    el.status(`scoring failed: ${e}`);
    return;
  }
  if (gen !== state.generation) return; // a newer selection won

  const ms = performance.now() - t0;
  paint(target, out);

  // Mean over positions 1.. — position 0 has no context and is always 0.
  const bits = out.bits;
  let sum = 0;
  for (let i = 1; i < bits.length; i++) sum += bits[i];
  const mean = bits.length > 1 ? sum / (bits.length - 1) : 0;
  stat(
    `${bits.length} characters · ${mean.toFixed(2)} bits/char average · ` +
      `${ms.toFixed(1)} ms, one forward pass`,
  );
  if (onDone) onDone(mean, bits.length);
}

function paint(target, out) {
  const frag = document.createDocumentFragment();
  const { tokens, bits, top, topP } = out;
  for (let i = 0; i < tokens.length; i++) {
    const span = document.createElement("span");
    span.textContent = tokens[i];
    span.dataset.i = String(i);
    // Newlines carry no useful tint and a coloured block at the end of a line
    // reads as a rendering bug rather than as information.
    if (!/^\s+$/.test(tokens[i])) {
      span.style.background = tint(bits[i]);
      span.style.borderRadius = "2px";
    }
    frag.appendChild(span);
  }
  target.replaceChildren(frag);
  state.scored.set(target, { tokens, bits, top, topP });
}

function describe(target, i) {
  const d = state.scored.get(target);
  if (!d || i == null || i < 0 || i >= d.bits.length) return;
  const show = (s) => (s === "\n" ? "\\n" : s === " " ? "␣" : s);
  if (i === 0) {
    el.readout(`"${show(d.tokens[0])}" — the first character; nothing precedes it.`);
    return;
  }
  const pct = (d.topP[i] * 100).toFixed(0);
  el.readout(
    `"${show(d.tokens[i])}" — ${d.bits[i].toFixed(2)} bits. ` +
      `The model expected "${show(d.top[i])}" (${pct}% sure).`,
  );
}

/* ── reacting ────────────────────────────────────────────────────────── */

/** One debounce and one generation counter for every trigger on the page. */
function request(text, target, opts) {
  const gen = ++state.generation;
  clearTimeout(state.timer);
  state.timer = setTimeout(() => score(text, target, gen, opts), DEBOUNCE_MS);
}

function buildPassages(root) {
  root.replaceChildren();
  for (const p of PASSAGES) {
    const wrap = document.createElement("div");
    wrap.className = "mb-6 last:mb-0";

    const label = document.createElement("p");
    label.className = "mb-2 text-[11px] uppercase tracking-wide text-ink-500 dark:text-ink-300";
    label.textContent = p.label;
    wrap.appendChild(label);

    const body = document.createElement("div");
    body.dataset.scored = "1";
    body.textContent = p.text;
    body.addEventListener("mousemove", (e) => {
      const t = e.target;
      if (t instanceof HTMLElement && t.dataset.i) describe(body, Number(t.dataset.i));
    });
    wrap.appendChild(body);
    root.appendChild(wrap);

    // Scored once at load, so the page is already coloured when the reader
    // arrives — a blank page with an instruction is a worse first frame than a
    // page that has already done the thing.
    p.el = body;
  }
}

async function main() {
  paintScale();
  const root = $("react-text");
  buildPassages(root);

  try {
    state.model = await load();
  } catch (e) {
    el.status(`could not start: ${e}`);
    return;
  }

  // Paint every passage once, sequentially — three forward passes, and they
  // must not race each other for the same generation id.
  for (const p of PASSAGES) {
    await score(p.text, p.el, ++state.generation, {
      onDone: (mean) => {
        p.mean = mean;
      },
    });
  }
  el.readout("Hover a character to see what the model expected instead.");

  // Selecting rescores exactly what was selected — and that is the page's real
  // point, because a selection is a *different question*: the model sees only
  // those characters, with none of the context that preceded them. The passage
  // itself is left alone so the two colourings can be compared side by side.
  const selView = $("react-selection");
  selView.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t instanceof HTMLElement && t.dataset.i) describe(selView, Number(t.dataset.i));
  });

  document.addEventListener("selectionchange", () => {
    const sel = document.getSelection();
    if (!sel || sel.isCollapsed) return;
    for (const p of PASSAGES) {
      if (!p.el.contains(sel.anchorNode)) continue;
      const text = sel.toString();
      if (text.trim().length < 2) return;
      request(text, selView, {
        stat: (t) => ($("react-selection-stat").textContent = t),
        onDone: (mean) => {
          $("react-selection-card").hidden = false;
          $("react-selection-note").textContent =
            `${mean.toFixed(2)} bits per character with only the selection for ` +
            `context, against ${p.mean.toFixed(2)} for the whole passage. ` +
            `Same characters, less to go on.`;
        },
      });
      return;
    }
  });

  const input = $("react-input");
  const own = document.createElement("div");
  own.className = "mt-3 whitespace-pre-wrap font-mono text-[13px] leading-7";
  own.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t instanceof HTMLElement && t.dataset.i) describe(own, Number(t.dataset.i));
  });
  input.parentElement.parentElement.appendChild(own);

  input.addEventListener("input", () => request(input.value, own, { warns: true }));
  await score(input.value, own, ++state.generation, { warns: true });
}

main();
