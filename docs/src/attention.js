// The attention matrix, in two dimensions.
//
// This replaces the 3D pipeline walk that stood here through v5. That view
// could show the *shape* of the calculation and never its values: perspective
// plus a 384-number tile folded 24 x 16 is a picture you can orbit and cannot
// read. Attention is natively a matrix — rows are queries, columns are keys —
// and the moment it is drawn flat, every cell is a number you can point at.
//
// Nothing here needs WebGL, three.js, or a single line of new Rust:
// generate_with_trace already emits `attn` for every block and every head, and
// the old renderer threw away 35 of the 36 pairs. This one keeps them all.

/** How a character is spelled on an axis. Whitespace has to be visible. */
const glyph = (s) => (s ?? "").replace(/\n/g, "↵").replace(/ /g, "␣") || "·";

/** Below this many positions the axes carry characters; above it they cannot
    fit and are dropped rather than drawn illegibly. */
const LABEL_MAX = 40;

/** Attention is heavily skewed — a handful of cells near 1, the rest near 0 —
    so a linear ramp renders as one bright diagonal on black. The colour is
    p ** GAMMA, and the readout always quotes the raw probability, which is the
    only number this section promises. */
const GAMMA = 0.45;

const dark = () =>
  window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;

/**
 * A 2D attention heatmap over one (block, head) pair, with every other pair
 * kept in memory so switching is instant and needs no re-run.
 *
 * `canvas` is drawn into, `readout` receives the hover text, and `onSelect`
 * fires whenever the selection changes so the caller can restyle its buttons.
 */
export function createAttention({ canvas, readout, onSelect } = {}) {
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) throw new Error("no 2D context");

  // Off-screen, one pixel per cell, scaled up with smoothing off. 256 x 256
  // fillRect calls per frame is the version of this that drops frames.
  const off = document.createElement("canvas");
  const offCtx = off.getContext("2d");

  let cfg = { nLayer: 6, nHead: 6, nCtx: 256 };
  let probs = null; // Float32Array, [layer][head][q][k]
  let tokens = [];
  let live = 0; // positions the model has actually computed
  let block = 0;
  let head = 0;
  let hover = null; // { q, k }
  // Written by draw(), read by the pointer handler: the two must agree about
  // where the grid landed, and only one of them can decide.
  let layout = null;

  function allocate() {
    const { nLayer, nHead, nCtx } = cfg;
    // 6 * 6 * 256 * 256 * 4 B = 9.4 MB, beside 43 MB of weights. Cheap enough
    // to keep every head, which is the whole point of the rewrite.
    probs = new Float32Array(nLayer * nHead * nCtx * nCtx);
    off.width = nCtx;
    off.height = nCtx;
  }

  const at = (l, h, q, k) =>
    probs[((l * cfg.nHead + h) * cfg.nCtx + q) * cfg.nCtx + k];

  // ── input ───────────────────────────────────────────────────────────────

  function setConfig(next) {
    cfg = { ...cfg, ...next };
    allocate();
    if (block >= cfg.nLayer) block = 0;
    if (head >= cfg.nHead) head = 0;
    onSelect?.(block, head, cfg);
  }

  function reset() {
    if (!probs) allocate();
    probs.fill(0);
    tokens = [];
    live = 0;
    hover = null;
    draw();
  }

  function setTokens(next) {
    tokens = Array.from(next, glyph);
    draw();
  }

  /** A generated character, appended so the axes keep naming real positions. */
  function pushToken(s) {
    tokens.push(glyph(s));
  }

  /**
   * One step's trace. The prefill lands `qLen` rows at once; every decode step
   * after it lands exactly one. `probs` is head-major:
   * `probs[(h * qLen + q) * kvLen + k]`.
   */
  function pushTrace(trace) {
    if (!probs) allocate();
    const { nCtx, nHead } = cfg;
    const base = trace.kvLen - trace.qLen; // first new position
    for (const a of trace.attn) {
      if (a.layer >= cfg.nLayer) continue;
      const heads = Math.min(a.nHead, nHead);
      for (let h = 0; h < heads; h++) {
        for (let q = 0; q < a.qLen; q++) {
          const pos = base + q;
          if (pos >= nCtx) break;
          const from = (h * a.qLen + q) * a.kvLen;
          const to = ((a.layer * nHead + h) * nCtx + pos) * nCtx;
          const n = Math.min(a.kvLen, nCtx);
          for (let k = 0; k < n; k++) probs[to + k] = a.probs[from + k];
        }
      }
    }
    live = Math.min(Math.max(live, trace.kvLen), nCtx);
    draw();
    // Without this the readout stays empty from the end of a run until the
    // first hover, and the one line telling the visitor the grid is hoverable
    // is the line that never appears.
    say();
  }

  function select(nextBlock, nextHead) {
    block = Math.max(0, Math.min(cfg.nLayer - 1, nextBlock));
    head = Math.max(0, Math.min(cfg.nHead - 1, nextHead));
    onSelect?.(block, head, cfg);
    draw();
    say();
  }

  // ── drawing ─────────────────────────────────────────────────────────────

  /** Background → forge-600, through the gamma above. */
  function ramp(p, isDark, out, i) {
    const t = p <= 0 ? 0 : Math.pow(Math.min(1, p), GAMMA);
    const bg = isDark ? 13 : 247;
    out[i] = bg + (234 - bg) * t;
    out[i + 1] = bg + (106 - bg) * t;
    out[i + 2] = bg + (36 - bg) * t;
    out[i + 3] = 255;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const n = live;
    if (!n || !probs) {
      layout = null;
      return;
    }

    const isDark = dark();
    const labels = n <= LABEL_MAX;
    const padL = labels ? 30 : 10;
    const padT = labels ? 20 : 10;
    const side = Math.max(0, Math.min(w - padL - 10, h - padT - 10));
    const cell = side / n;
    layout = { padL, padT, side, cell, n };

    const img = offCtx.createImageData(n, n);
    const d = img.data;
    for (let q = 0; q < n; q++) {
      for (let k = 0; k < n; k++) {
        ramp(at(block, head, q, k), isDark, d, (q * n + k) * 4);
      }
    }
    offCtx.putImageData(img, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, n, n, padL, padT, side, side);
    ctx.imageSmoothingEnabled = true;

    ctx.strokeStyle = isDark ? "#35353f" : "#ebebee";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL + 0.5, padT + 0.5, side, side);

    if (labels) {
      ctx.fillStyle = isDark ? "#b4b4bd" : "#6b6b78";
      ctx.font = `${Math.min(11, Math.max(7, cell * 0.7))}px ui-monospace, monospace`;
      ctx.textBaseline = "middle";
      for (let i = 0; i < n; i++) {
        const c = tokens[i] ?? "·";
        ctx.textAlign = "center";
        ctx.fillText(c, padL + (i + 0.5) * cell, padT - 9);
        ctx.textAlign = "right";
        ctx.fillText(c, padL - 6, padT + (i + 0.5) * cell);
      }
    }

    if (hover) {
      ctx.strokeStyle = isDark ? "#ffb185" : "#c14f18";
      ctx.lineWidth = 1;
      ctx.strokeRect(
        padL + hover.k * cell + 0.5,
        padT + hover.q * cell + 0.5,
        Math.max(cell, 2),
        Math.max(cell, 2),
      );
    }
  }

  /** The hover line — or, with nothing under the cursor, what is on screen. */
  function say() {
    if (!readout) return;
    const where = `block ${block + 1}/${cfg.nLayer} · head ${head + 1}/${cfg.nHead}`;
    if (!hover || !live) {
      readout.textContent = live
        ? `${where} · ${live} positions · hover any cell for its probability`
        : "";
      return;
    }
    const { q, k } = hover;
    const p = at(block, head, q, k);
    const masked = k > q;
    readout.textContent =
      `${where} · p(q=${q} '${tokens[q] ?? "·"}' ← k=${k} '${tokens[k] ?? "·"}') = ` +
      (masked ? "0 — masked, a position may not see the future" : p.toFixed(4));
  }

  // ── pointer ─────────────────────────────────────────────────────────────

  function locate(ev) {
    if (!layout) return null;
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left - layout.padL;
    const y = ev.clientY - r.top - layout.padT;
    if (x < 0 || y < 0 || x >= layout.side || y >= layout.side) return null;
    return {
      k: Math.min(layout.n - 1, Math.floor(x / layout.cell)),
      q: Math.min(layout.n - 1, Math.floor(y / layout.cell)),
    };
  }

  const onMove = (ev) => {
    const next = locate(ev);
    if (next?.q === hover?.q && next?.k === hover?.k) return;
    hover = next;
    draw();
    say();
  };
  const onLeave = () => {
    if (!hover) return;
    hover = null;
    draw();
    say();
  };

  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);

  const ro = new ResizeObserver(() => draw());
  ro.observe(canvas);
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", draw);

  draw();

  return {
    setConfig,
    reset,
    setTokens,
    pushToken,
    pushTrace,
    select,
    selection: () => ({ block, head }),
  };
}
