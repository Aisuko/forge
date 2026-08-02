/**
 * The council page: four small GPT-2s run in parallel on your GPU, exchange
 * hidden states rather than text, and merge into one character at a time.
 *
 * Everything drawn here comes out of `WasmCouncil.step()` — the expert cards,
 * the edge weights and the dots in the latent plane are the vectors and weights
 * the model actually used, never a re-derivation in JavaScript. The only thing
 * this file invents is the 2-D projection, and it says so on the panel.
 */

import init, { WasmCouncil } from "./forge-council/forge_council.js";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";

/** One hue per expert; the merge gets Forge orange, because it is the answer. */
const EXPERT_COLORS = ["#4c8dff", "#35c48a", "#b478ff", "#ffb020"];
const MERGE_COLOR = "#ea6a24";

const VIEW_H = 470;

const LAYOUT = {
  card: { x: 24, w: 214, h: 76, top: 22, gap: 112 },
  merge: { x: 502, y: 230, r: 50 },
};

const state = {
  council: null,
  running: false,
  stop: false,
  names: [],
  nEmbd: 0,
  proj: null, // [2][nEmbd] fixed random projection
  radius: null, // eased half-width of the view, in projection units
  prev: null, // last frame's merged position, for the tail's bearing
  nodes: [], // per-expert SVG handles
  paths: [], // per-expert edge <path>
};

/* ── loading ────────────────────────────────────────────────────────── */

const el = {
  idle: () => $("council-idle"),
  status: (t) => ($("council-status").textContent = t),
  progress: (frac) => {
    const wrap = $("council-progress-wrap");
    wrap.hidden = frac === null;
    if (frac !== null) $("council-progress").style.width = `${(frac * 100).toFixed(0)}%`;
  },
};

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function load() {
  el.status("starting wasm…");
  await init();

  el.status("reading the manifest…");
  const manifest = await fetchJson("./council/manifest.json");
  const [configJson, vocabJson] = await Promise.all([
    fetch("./council/config.json").then((r) => r.text()),
    fetch("./council/vocab.json").then((r) => r.text()),
  ]);

  const blobs = [];
  for (const [i, e] of manifest.experts.entries()) {
    el.status(`loading ${e.label}…`);
    el.progress(i / manifest.experts.length);
    const buf = await fetch(`./council/${e.file}`).then((r) => r.arrayBuffer());
    blobs.push(new Uint8Array(buf));
  }
  el.progress(1);

  el.status("compiling kernels…");
  const names = manifest.experts.map((e) => e.label);
  const council = await WasmCouncil.load(blobs, names, configJson, vocabJson, 1337n);

  state.council = council;
  state.names = names;
  state.nEmbd = council.n_embd();
  state.proj = randomProjection(state.nEmbd, 2, 0x5eed);
  $("council-device").textContent = council.device_info();
  el.progress(null);
  el.status(`${names.length} experts · ${state.nEmbd}-D hidden · ready`);
  el.idle().hidden = true;

  buildFlow();
  buildLegend();
}

/* ── the flow diagram ───────────────────────────────────────────────── */

function svg(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function buildFlow() {
  const edges = $("flow-edges");
  const nodes = $("flow-nodes");
  const packets = $("flow-packets");
  edges.replaceChildren();
  nodes.replaceChildren();
  packets.replaceChildren();
  state.nodes = [];
  state.paths = [];

  const { card, merge } = LAYOUT;
  const n = state.names.length;
  // Fit the stack to the four cards, whatever four turns out to be.
  const gap = Math.min(card.gap, (VIEW_H - 2 * card.top) / n);

  state.names.forEach((name, k) => {
    const y = card.top + k * gap;
    const color = EXPERT_COLORS[k % EXPERT_COLORS.length];
    const mid = y + card.h / 2;

    // The edge is drawn first so the cards paint over its stub.
    const path = svg("path", {
      d: `M ${card.x + card.w} ${mid} C ${card.x + card.w + 90} ${mid}, ${merge.x - 110} ${merge.y}, ${merge.x - merge.r} ${merge.y}`,
      fill: "none",
      stroke: color,
      "stroke-width": 2,
      "stroke-linecap": "round",
      opacity: 0.35,
    });
    edges.append(path);
    state.paths.push(path);

    const g = svg("g", {});
    g.append(
      svg("rect", {
        x: card.x,
        y,
        width: card.w,
        height: card.h,
        rx: 12,
        fill: color,
        "fill-opacity": 0.08,
        stroke: color,
        "stroke-opacity": 0.5,
      }),
    );
    const label = svg("text", {
      x: card.x + 12,
      y: y + 20,
      "font-size": 11,
      fill: color,
    });
    label.textContent = name;
    const glyph = svg("text", {
      x: card.x + 12,
      y: y + 54,
      "font-size": 28,
      "font-family": "ui-monospace, monospace",
      fill: "currentColor",
    });
    glyph.textContent = "·";
    const prob = svg("text", {
      x: card.x + card.w - 12,
      y: y + 54,
      "font-size": 12,
      "font-family": "ui-monospace, monospace",
      "text-anchor": "end",
      fill: "currentColor",
      opacity: 0.55,
    });
    // The weight bar: how much of the merged vector came from this expert.
    const track = svg("rect", {
      x: card.x + 12,
      y: y + card.h - 12,
      width: card.w - 24,
      height: 5,
      rx: 2.5,
      fill: "currentColor",
      "fill-opacity": 0.12,
    });
    const fill = svg("rect", {
      x: card.x + 12,
      y: y + card.h - 12,
      width: 0,
      height: 5,
      rx: 2.5,
      fill: color,
    });
    g.append(label, glyph, prob, track, fill);
    nodes.append(g);

    const packet = svg("circle", { r: 4.5, fill: color, opacity: 0 });
    packets.append(packet);

    state.nodes.push({ glyph, prob, fill, packet, path, barW: card.w - 24 });
  });

  const g = svg("g", {});
  g.append(
    svg("circle", {
      cx: merge.x,
      cy: merge.y,
      r: merge.r,
      fill: MERGE_COLOR,
      "fill-opacity": 0.1,
      stroke: MERGE_COLOR,
      "stroke-width": 2,
    }),
  );
  const chosen = svg("text", {
    x: merge.x,
    y: merge.y + 12,
    "font-size": 34,
    "font-family": "ui-monospace, monospace",
    "text-anchor": "middle",
    fill: "currentColor",
  });
  chosen.textContent = "·";
  const sigma = svg("text", {
    x: merge.x,
    y: merge.y + merge.r + 20,
    "font-size": 12,
    "font-family": "ui-monospace, monospace",
    "text-anchor": "middle",
    fill: "currentColor",
    opacity: 0.55,
  });
  sigma.textContent = "Σ wᵢ·hᵢ";
  g.append(chosen, sigma);
  $("flow-nodes").append(g);
  state.chosenGlyph = chosen;
}

function buildLegend() {
  const box = $("council-legend");
  box.replaceChildren();
  const row = (color, text) => {
    const s = document.createElement("span");
    s.className = "flex items-center gap-1.5 font-mono text-[11px]";
    const dot = document.createElement("span");
    dot.className = "inline-block h-2.5 w-2.5 rounded-full";
    dot.style.background = color;
    s.append(dot, document.createTextNode(text));
    return s;
  };
  state.names.forEach((n, k) =>
    box.append(row(EXPERT_COLORS[k % EXPERT_COLORS.length], n)),
  );
  box.append(row(MERGE_COLOR, "merged"));
}

/** Printable stand-in for a character that has no width of its own. */
function glyphOf(token) {
  if (token === "\n") return "⏎";
  if (token === " ") return "␣";
  if (token === "\t") return "⇥";
  return token;
}

/* ── the latent plane ───────────────────────────────────────────────── */

/**
 * A fixed 128→2 projection from a seeded PRNG. Fixed so the picture is stable
 * across characters and across runs — a projection recomputed per frame would
 * make the dots dance for reasons that have nothing to do with the model.
 */
function randomProjection(dim, out, seed) {
  let s = seed >>> 0;
  const rand = () => {
    // xorshift32, then a Box-Muller pair would be overkill: a uniform basis is
    // enough to separate points that differ.
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 0xffffffff - 0.5;
  };
  const rows = [];
  for (let i = 0; i < out; i++) {
    const row = new Float32Array(dim);
    for (let j = 0; j < dim; j++) row[j] = rand();
    rows.push(row);
  }
  return rows;
}

function project(h) {
  const [rx, ry] = state.proj;
  let x = 0;
  let y = 0;
  for (let i = 0; i < h.length; i++) {
    x += h[i] * rx[i];
    y += h[i] * ry[i];
  }
  return [x, y];
}

/**
 * Half-width of the view: the furthest expert's distance from the merge, plus
 * headroom. Every expert is inside it — the merge is their convex combination.
 *
 * Fitting a box over recent frames instead scales the view to the merge's
 * journey, which is ~2 units per character against 2.45 of disagreement across
 * the whole council: the five dots came out at 17% of the panel, measured.
 */
const MARGIN = 1.25;

function fit(experts, merged) {
  let r = 0;
  for (const [x, y] of experts) {
    r = Math.max(r, Math.hypot(x - merged[0], y - merged[1]));
  }
  const target = Math.max(r, 1e-3) * MARGIN;
  // Grow on the frame it happens — easing a growing radius clips the dot that
  // grew it — and ease the shrink, or the plane breathes every character. Fast,
  // because the spread swings ~11× per character and this tracks its maximum.
  state.radius =
    state.radius === null
      ? target
      : Math.max(target, state.radius + (target - state.radius) * 0.5);
  return state.radius;
}

function drawPlane(step) {
  const canvas = $("council-plane");
  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth;
  if (!size) return;
  if (canvas.width !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  const pts = step.experts.map((e) => project(e.hidden));
  const merged = project(step.hidden);
  const radius = fit(pts, merged);

  // The merge is the origin: it sits at the centre every frame, and the scale
  // is the disagreement around it. Nothing else competes for the scale.
  const pad = 22;
  const half = (size - 2 * pad) / 2;
  const to = ([x, y]) => [
    size / 2 + ((x - merged[0]) / radius) * half,
    size / 2 + ((y - merged[1]) / radius) * half,
  ];

  // Where it just came from. The whole path cannot be drawn in these units — a
  // character of travel is more than a radius, and a segment joining two
  // off-panel vertices still cuts across the view, which was the old scribble.
  // Only the bearing survives the rescale, so only the bearing is drawn: the
  // length is capped and means nothing.
  const [mx, my] = [size / 2, size / 2];
  const prev = state.prev;
  state.prev = merged;
  if (prev) {
    const [dx, dy] = [prev[0] - merged[0], prev[1] - merged[1]];
    const len = Math.hypot(dx, dy);
    if (len > 1e-6) {
      const [tx, ty] = [mx + (dx / len) * half, my + (dy / len) * half];
      const grad = ctx.createLinearGradient(mx, my, tx, ty);
      grad.addColorStop(0, MERGE_COLOR);
      grad.addColorStop(1, "transparent");
      ctx.strokeStyle = grad;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(mx, my);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  pts.forEach((p, k) => {
    const [x, y] = to(p);
    const color = EXPERT_COLORS[k % EXPERT_COLORS.length];
    // The tie back to the merge, as thick as the weight it earned.
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 0.5 + 4 * step.experts[k].weight;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(mx, my);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = MERGE_COLOR;
  ctx.beginPath();
  ctx.arc(mx, my, 8, 0, Math.PI * 2);
  ctx.fill();
}

/* ── the bars ───────────────────────────────────────────────────────── */

function drawBars(step) {
  const box = $("council-bars");
  box.replaceChildren();
  for (const t of step.top) {
    const row = document.createElement("div");
    row.className = "bar-row" + (t.id === step.id ? " is-chosen" : "");
    const glyph = document.createElement("span");
    glyph.className = "bar-glyph";
    glyph.textContent = glyphOf(t.token);
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = `${(t.p * 100).toFixed(1)}%`;
    track.append(fill);
    const p = document.createElement("span");
    p.className = "bar-p";
    p.textContent = `${(t.p * 100).toFixed(1)}%`;
    const mark = document.createElement("span");
    mark.className = "bar-mark";
    mark.textContent = t.id === step.id ? "chosen" : "";
    row.append(glyph, track, p, mark);
    box.append(row);
  }
}

/* ── one character ──────────────────────────────────────────────────── */

function render(step) {
  step.experts.forEach((e, k) => {
    const n = state.nodes[k];
    if (!n) return;
    const top = e.top[0];
    n.glyph.textContent = glyphOf(top.token);
    n.prob.textContent = `${(top.p * 100).toFixed(0)}%  w ${e.weight.toFixed(2)}`;
    n.fill.setAttribute("width", (n.barW * e.weight).toFixed(1));
    // The edge carries the weight too, so the picture reads without the bars.
    n.path.setAttribute("stroke-width", (1 + 7 * e.weight).toFixed(2));
    n.path.setAttribute("opacity", (0.2 + 0.7 * e.weight).toFixed(2));
  });
  state.chosenGlyph.textContent = glyphOf(step.token);
  drawPlane(step);
  drawBars(step);
  flyPackets();
}

/** Send one dot down every edge, sized by the weight already on the edge. */
function flyPackets() {
  const start = performance.now();
  const dur = 260;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur);
    for (const n of state.nodes) {
      const len = n.path.getTotalLength();
      const p = n.path.getPointAtLength(t * len);
      n.packet.setAttribute("cx", p.x);
      n.packet.setAttribute("cy", p.y);
      n.packet.setAttribute("opacity", (1 - t) * 0.9);
    }
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ── the run loop ───────────────────────────────────────────────────── */

async function run() {
  const council = state.council;
  if (!council || state.running) return;
  const prompt = $("council-prompt").value || "ROMEO:";
  const bad = council.unsupported_chars(prompt);
  if (bad) {
    el.status(`this vocabulary has no ${JSON.stringify(bad)}`);
    return;
  }

  state.running = true;
  state.stop = false;
  state.prev = null;
  state.radius = null;
  $("council-run").disabled = true;
  $("council-stop").disabled = false;
  $("council-out").textContent = prompt;

  const seed = BigInt(Date.now() % 1_000_000);
  council.reset(seed);
  council.set_beta(Number($("council-beta").value));

  let ids = council.encode(prompt);
  const limit = council.n_ctx() - ids.length - 1;
  const started = performance.now();
  let produced = 0;
  let agreed = 0;

  for (let i = 0; i < Math.min(240, limit); i++) {
    if (state.stop) break;
    // Read the slider every character: dragging it mid-run is the point.
    council.set_beta(Number($("council-beta").value));
    const step = await council.step(ids, 6, 0.85, seed + BigInt(i));
    render(step);
    $("council-out").textContent += step.token;
    produced++;
    if (step.consensus === 1) agreed++;
    ids = new Uint32Array([step.id]);
    const rate = produced / ((performance.now() - started) / 1000);
    el.status(
      `${produced} chars · ${rate.toFixed(1)}/s · experts split on ` +
        `${(100 * (1 - agreed / produced)).toFixed(0)}%`,
    );
    await new Promise((r) => requestAnimationFrame(r));
  }

  state.running = false;
  $("council-run").disabled = false;
  $("council-stop").disabled = true;
}

/* ── wiring ─────────────────────────────────────────────────────────── */

$("council-beta").addEventListener("input", (e) => {
  $("council-beta-val").textContent = Number(e.target.value).toFixed(1);
  state.council?.set_beta(Number(e.target.value));
});

$("council-start").addEventListener("click", async () => {
  $("council-start").disabled = true;
  try {
    await load();
    await run();
  } catch (err) {
    el.status(String(err));
    $("council-start").disabled = false;
  }
});

$("council-run").addEventListener("click", () => {
  if (state.council) run();
  else $("council-start").click();
});

$("council-stop").addEventListener("click", () => {
  state.stop = true;
});
