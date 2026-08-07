/**
 * The reactive page: the model reads text, and every character resolves in the
 * time the model needed to be sure of it.
 *
 * Everything drawn comes from one call — `WasmSurprise.surprisal(text, k)`,
 * which runs a single forward pass and returns, per position, the surprisal in
 * bits and the k characters the model actually weighed there. Nothing here
 * re-derives a probability in JavaScript; this file decides colour and timing,
 * and both are read straight off those numbers.
 *
 * **The reveal is a replay, and the page says so.** The pass has already
 * finished — in ~15 ms — before the first frame is drawn; the flicker is that
 * one result played back, with each position's lock time set by its own
 * surprisal and each flickered character drawn from its own top-k. No
 * inference happens while a character is spinning, and nothing about the
 * animation staggers left to right, because nothing was computed left to
 * right: all positions are scored in the same pass, so all of them start
 * together and only their certainty separates them.
 *
 * Three constraints shape the code, and all three are about the model being
 * genuinely local rather than behind a network call:
 *
 *  - load once, at page open, so every interaction after it is instant;
 *  - cancel in flight, because a reader dragging a selection outruns the GPU;
 *  - never reload the model, whatever happens.
 */

import init, { WasmSurprise } from "./forge-surprise/forge_surprise.js";

const $ = (id) => document.getElementById(id);

/** Bits at which the scale saturates. log2(65) ≈ 6.02 is the char model's
 *  uniform-guess entropy, so 6 is "it had no idea" and makes a natural top. */
const MAX_BITS = 6;

/** Alternatives kept per position. Eight is enough for a spin to look
 *  considered and short enough to read as a panel of bars. */
const K = 8;

/** Below this the model was as good as certain — 0.3 bits is p ≈ 0.81 — and a
 *  frame of manufactured doubt would misrepresent it. These lock immediately,
 *  which is why the mass of a passage goes solid at once. */
const SNAP_BITS = 0.3;

/** Lock times in ms, spanning SNAP_BITS..MAX_BITS. */
const REVEAL = { min: 120, max: 1100 };

/** Shorter for the two surfaces a reader retriggers continuously — the
 *  selection panel under a drag, the text box under typing. A full-length
 *  reveal per keystroke would be punishing. */
const RETRIGGER_REVEAL = { min: 80, max: 600 };

/** Swap interval, eased from fast to slow, so a position decelerates into
 *  place rather than stopping dead. */
const SWAP = { from: 50, to: 180 };

/** The blanket rule in input.css cannot reach a requestAnimationFrame loop,
 *  so the reveal has to honour the setting itself: it jumps to the last frame.
 *  Read live rather than once, so changing the OS setting takes effect on the
 *  next replay. */
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

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
  /** Every element that currently holds a scored result, so "Read it again"
   *  can replay all of them without a second forward pass. */
  targets: new Set(),
};

const el = {
  status: (t) => ($("react-status").textContent = t),
  progress: (frac) => {
    const wrap = $("react-progress-wrap");
    wrap.hidden = frac === null;
    if (frac !== null) $("react-progress").style.width = `${(frac * 100).toFixed(0)}%`;
  },
  stat: (t) => ($("react-stat").textContent = t),
  at: (t) => ($("react-at").textContent = t),
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
  const m = await WasmSurprise.load_char(bytes, config, vocab);
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
async function score(
  text,
  target,
  gen,
  { warns = false, stat = el.stat, onDone, pace = REVEAL } = {},
) {
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
    out = await state.model.surprisal(text, K);
  } catch (e) {
    el.status(`scoring failed: ${e}`);
    return;
  }
  if (gen !== state.generation) return; // a newer selection won

  const ms = performance.now() - t0;
  reveal(layout(target, out, pace), pace);

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

/* ── the reveal ──────────────────────────────────────────────────────────
 *
 * The old paint() built the finished picture in one shot. It is now two steps:
 * layout() puts the characters on the page untinted, and reveal() resolves
 * them. What settles is exactly what paint() used to draw.
 */

const isSpace = (s) => /^\s+$/.test(s);

/**
 * Build the spans and stash everything the reveal and the hover panel need.
 * No tint yet — a finished heat map cannot distinguish a character the model
 * was certain of from one it was torn about, and that difference is the thing
 * worth showing.
 */
function layout(target, out, pace = REVEAL) {
  const frag = document.createDocumentFragment();
  const spans = [];
  for (let i = 0; i < out.tokens.length; i++) {
    const span = document.createElement("span");
    span.textContent = out.tokens[i];
    span.dataset.i = String(i);
    frag.appendChild(span);
    spans.push(span);
  }
  target.replaceChildren(frag);
  // A fresh object identity per score: the frame loop compares against it to
  // notice that a newer result has replaced the one it was animating.
  const data = { ...out, spans, target, pace };
  state.scored.set(target, data);
  state.targets.add(target);
  return data;
}

/**
 * The characters position `i` may flicker through: the model's own top-k with
 * whitespace dropped — a blinking blank reads as a rendering fault — and the
 * survivors renormalised, so a draw is still in the proportions the model gave
 * them.
 */
function candidates(d, i) {
  const out = [];
  let mass = 0;
  for (let j = 0; j < d.k; j++) {
    const ch = d.alt[i * d.k + j];
    const p = d.altP[i * d.k + j];
    if (!ch || isSpace(ch) || !(p > 0)) continue;
    out.push({ ch, p });
    mass += p;
  }
  if (mass > 0) for (const c of out) c.p /= mass;
  return out;
}

function drawCandidate(cands) {
  let r = Math.random();
  for (const c of cands) {
    r -= c.p;
    if (r <= 0) return c.ch;
  }
  return cands[cands.length - 1].ch;
}

/** Settle one position: the real character, its tint, and the landing pulse. */
function lock(span, text, bits) {
  span.textContent = text;
  // Newlines carry no useful tint and a coloured block at the end of a line
  // reads as a rendering bug rather than as information.
  if (isSpace(text)) {
    span.className = "";
    return;
  }
  span.style.background = tint(bits);
  span.className = "tok-locked";
}

/**
 * Resolve every position at once, each in its own time.
 *
 * The lock time *is* the surprisal: `bits[i]` mapped through `pace`. Under
 * `SNAP_BITS` the model was certain and the character never spins, which is
 * why most of a Shakespeare passage is solid on the first frame and what is
 * left blinking is precisely the set it was unsure about.
 */
function reveal(data, pace = REVEAL) {
  const { tokens, bits, spans } = data;
  const t0 = performance.now();
  const jump = reducedMotion.matches;
  // Claims the spans. A second reveal over the same data — the reader pressing
  // the button twice — supersedes this one on its next frame.
  const token = (data.reveal = {});
  let live = [];

  for (let i = 0; i < tokens.length; i++) {
    // Position 0 was never predicted — nothing precedes it — so it has no
    // alternatives to show and no surprisal to spend.
    const cands = jump || i === 0 || isSpace(tokens[i]) ? [] : candidates(data, i);
    const f = (bits[i] - SNAP_BITS) / (MAX_BITS - SNAP_BITS);
    if (cands.length < 2 || f <= 0) {
      lock(spans[i], tokens[i], bits[i]);
      continue;
    }
    // On a replay this span is still wearing the tint the last reveal left. It
    // has not been earned again yet.
    spans[i].style.background = "";
    spans[i].className = "tok-spin";
    live.push({
      i,
      span: spans[i],
      cands,
      lockAt: t0 + pace.min + (pace.max - pace.min) * Math.min(f, 1),
      nextSwap: t0,
    });
  }

  function frame(now) {
    // Guarded per element, not against state.generation directly: the three
    // passages are scored one after another and their reveals overlap on
    // purpose, but a newer result — or a newer replay — for *this* element
    // must abort the loop running over it rather than fight it for the spans.
    if (state.scored.get(data.target) !== data || data.reveal !== token) return;
    live = live.filter((p) => {
      if (now >= p.lockAt) {
        lock(p.span, tokens[p.i], bits[p.i]);
        return false;
      }
      if (now >= p.nextSwap) {
        p.span.textContent = drawCandidate(p.cands);
        const done = (now - t0) / (p.lockAt - t0);
        p.nextSwap = now + SWAP.from + (SWAP.to - SWAP.from) * done;
      }
      return true;
    });
    if (live.length) requestAnimationFrame(frame);
  }
  if (live.length) requestAnimationFrame(frame);
}

/** Replay what is already on the page. No forward pass: the numbers driving
 *  the second reveal are the numbers that drove the first. */
function replay() {
  for (const target of state.targets) {
    const d = state.scored.get(target);
    if (d) reveal(d, d.pace);
  }
}

/* ── the readout ─────────────────────────────────────────────────────────
 *
 * The same top-k the flicker cycles through, as bars. That is the point of
 * showing them: a reader who watched a character hesitate between `e` and `a`
 * can hover it and find `e` and `a`, with the weights that decided how often
 * each appeared.
 */

const bars = { rows: [], root: null };

function barRow() {
  const el = document.createElement("div");
  el.className = "bar-row";
  const g = document.createElement("span");
  g.className = "bar-glyph";
  const track = document.createElement("span");
  track.className = "bar-track";
  const fill = document.createElement("span");
  fill.className = "bar-fill";
  track.append(fill);
  const p = document.createElement("span");
  p.className = "bar-p";
  const mark = document.createElement("span");
  mark.className = "bar-mark";
  el.append(g, track, p, mark);
  return { el, g, fill, p, mark };
}

function ensureRows(n) {
  bars.root ??= $("react-bars");
  while (bars.rows.length < n) {
    const r = barRow();
    bars.rows.push(r);
    bars.root.append(r.el);
  }
  for (let i = 0; i < bars.rows.length; i++) bars.rows[i].el.hidden = i >= n;
}

const show = (s) => (s === "\n" ? "\\n" : s === " " ? "␣" : s);

function describe(target, i) {
  const d = state.scored.get(target);
  if (!d || i == null || i < 0 || i >= d.bits.length) return;
  if (i === 0) {
    el.at(`"${show(d.tokens[0])}" — the first character; nothing precedes it.`);
    ensureRows(0);
    return;
  }

  const k = d.k;
  // Scaled to the top bar, not to 1: the model is often 90% sure, and against
  // an absolute axis every alternative it weighed would be an invisible sliver.
  const peak = d.altP[i * k] || 1;
  let found = false;
  ensureRows(k);
  for (let j = 0; j < k; j++) {
    const ch = d.alt[i * k + j];
    const p = d.altP[i * k + j];
    const r = bars.rows[j];
    r.g.textContent = show(ch);
    r.fill.style.width = `${Math.max(1, (p / peak) * 100).toFixed(1)}%`;
    r.p.textContent = p < 0.005 ? "<1%" : `${(p * 100).toFixed(0)}%`;
    const actual = ch === d.tokens[i];
    found ||= actual;
    r.mark.textContent = actual ? "actual" : "";
    r.el.classList.toggle("is-chosen", actual);
  }
  el.at(
    `"${show(d.tokens[i])}" — ${d.bits[i].toFixed(2)} bits` +
      (found ? "" : `, not in its top ${k}`),
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

  // Score every passage once, sequentially — three forward passes, and they
  // must not race each other for the same generation id. Their *reveals*
  // overlap: each starts as its own pass lands, and the loops are guarded per
  // element rather than by the shared counter.
  for (const p of PASSAGES) {
    await score(p.text, p.el, ++state.generation, {
      onDone: (mean) => {
        p.mean = mean;
      },
    });
  }

  const replayBtn = $("react-replay");
  replayBtn.disabled = false;
  replayBtn.addEventListener("click", replay);

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
        pace: RETRIGGER_REVEAL,
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

  const ownOpts = { warns: true, pace: RETRIGGER_REVEAL };
  input.addEventListener("input", () => request(input.value, own, ownOpts));
  await score(input.value, own, ++state.generation, ownOpts);
}

main();
