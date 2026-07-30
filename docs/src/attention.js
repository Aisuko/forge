// Where the model looked: one query's attention as a labelled strip, the whole
// matrix as a zoom-out, and a description of the selected head computed from
// its own numbers.
//
// The matrix renderer here is the one v6 built, and it is correct. What v6 got
// wrong was making it the *lead*: at the default 240 tokens the grid is 246
// positions wide, which puts every cell at about two pixels and drops the
// character axes entirely (see LABEL_MAX) — so the picture a first-time visitor
// meets is an orange triangle with no characters anywhere on it, and no reason
// to care. The strip below fixes that by showing one row over a fixed 32-column
// window, where the characters always fit, and the matrix moves behind a
// disclosure for people who want the overview.
//
// Nothing here needs WebGL, three.js, or a single line of new Rust:
// generate_with_trace already emits `attn` for every block and every head, and
// the pre-v6 renderer threw away 35 of the 36 pairs. This one keeps them all.

/** How a character is spelled on an axis. Whitespace has to be visible. */
const glyph = (s) => (s ?? "").replace(/\n/g, "↵").replace(/ /g, "␣") || "·";

/** Below this many positions the axes carry characters; above it they cannot
    fit and are dropped rather than drawn illegibly. */
const LABEL_MAX = 40;

/** Keys in the strip. Fixed, not derived from the context length: a window that
    grows is a window whose labels eventually vanish, which is the failure the
    strip exists to end. */
const STRIP_W = 32;

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
 * `canvas` is drawn into, `readout` receives the hover text, `strip` the
 * current query's labelled bars, `caption` the description of the selected
 * head, and `onSelect` fires whenever the selection changes so the caller can
 * restyle its buttons.
 */
export function createAttention({
  canvas,
  readout,
  strip,
  caption,
  onSelect,
} = {}) {
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
    paint();
  }

  function setTokens(next) {
    tokens = Array.from(next, glyph);
    paint();
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
    paint();
  }

  function select(nextBlock, nextHead) {
    block = Math.max(0, Math.min(cfg.nLayer - 1, nextBlock));
    head = Math.max(0, Math.min(cfg.nHead - 1, nextHead));
    onSelect?.(block, head, cfg);
    paint();
  }

  /** Everything that depends on the data or the selection, in one call. The
      hover handlers deliberately do not use this: a cursor moving across the
      grid must not recompute the head statistics on every pixel. */
  function paint() {
    draw();
    drawStrip();
    drawCaption();
    // Without the readout here it stays empty from the end of a run until the
    // first hover, and the one line telling the visitor the grid is hoverable
    // is the line that never appears.
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

  // ── one row, labelled ───────────────────────────────────────────────────
  // The matrix answers "where did every position look?" in one picture and
  // answers it too small to read. This answers "where did the position it just
  // computed look?" over a fixed window, at a size where the characters fit —
  // which is the question a first-time reader actually has.

  let cols = []; // reused DOM, one per key column

  function ensureCols(n) {
    while (cols.length < n) {
      const el = document.createElement("div");
      el.className = "strip-col";
      const track = document.createElement("span");
      track.className = "strip-track";
      const fill = document.createElement("span");
      fill.className = "strip-fill";
      track.append(fill);
      const label = document.createElement("span");
      label.className = "strip-label";
      el.append(track, label);
      cols.push({ el, fill, label });
      strip?.append(el);
    }
    for (let i = 0; i < cols.length; i++) cols[i].el.hidden = i >= n;
  }

  /** The strongest key in row `q`, over every position it can see. */
  function peakOf(q) {
    let best = 0;
    let where = 0;
    for (let k = 0; k <= q; k++) {
      const p = at(block, head, q, k);
      if (p > best) {
        best = p;
        where = k;
      }
    }
    return { p: best, k: where };
  }

  function drawStrip() {
    if (!strip) return;
    if (!live || !probs) {
      ensureCols(0);
      return;
    }
    const q = live - 1;
    const lo = Math.max(0, q - STRIP_W + 1);
    const n = q - lo + 1;
    ensureCols(n);
    // Normalised to the row's own peak wherever that peak is — including
    // outside this window, which leaves the bars honestly short and is what the
    // caption's "strongest position" clause is for. Normalising to the window
    // instead would fill the panel by misreporting the scale.
    const peak = peakOf(q).p || 1;
    for (let i = 0; i < n; i++) {
      const k = lo + i;
      const p = at(block, head, q, k);
      const c = tokens[k] ?? "·";
      cols[i].fill.style.height = `${Math.max(1, (p / peak) * 100).toFixed(1)}%`;
      cols[i].label.textContent = c;
      cols[i].el.title = `p(q=${q} ← k=${k} '${c}') = ${p.toFixed(4)}`;
      cols[i].el.classList.toggle("is-self", k === q);
    }
  }

  // ── what this head does ─────────────────────────────────────────────────
  // Four numbers over every row the model has computed, and a sentence built
  // from them. Computed rather than written down: hand-written head labels
  // cannot survive a retrain, and this project retrains. Every clause below
  // quotes the number that produced it.

  /** Row 0 is skipped throughout: it can only attend to itself, so it scores a
      perfect focus of 1.0 at distance 0 and would drag every average. */
  function headStats() {
    if (!probs || live < 2) return null;
    let dist = 0;
    let focus = 0;
    let prev = 0;
    let sink = 0;
    for (let q = 1; q < live; q++) {
      let best = 0;
      for (let k = 0; k <= q; k++) {
        const p = at(block, head, q, k);
        dist += p * (q - k);
        if (p > best) best = p;
      }
      focus += best;
      prev += at(block, head, q, q - 1);
      sink += at(block, head, q, 0);
    }
    const rows = live - 1;
    return {
      rows,
      dist: dist / rows,
      focus: focus / rows,
      prev: prev / rows,
      sink: sink / rows,
    };
  }

  function drawCaption() {
    if (!caption) return;
    const s = headStats();
    if (!s) {
      caption.textContent = "";
      return;
    }
    const reach =
      s.dist < 1.5
        ? `reads the character just before it`
        : s.dist < 6
          ? `stays inside the last few characters`
          : `looks a long way back`;
    const sharp =
      s.focus > 0.6
        ? `sharply focused`
        : s.focus > 0.25
          ? `moderately focused`
          : `spread thin`;
    let line =
      `${reach} — ${s.dist.toFixed(1)} positions back on average, ` +
      `${sharp}: ${(s.focus * 100).toFixed(0)}% of a row's weight on its ` +
      `strongest position, ${(s.prev * 100).toFixed(0)}% on the previous ` +
      `character`;
    // The attention sink is real, well known, and the most surprising thing a
    // reader can find in here on their own. Worth naming when it shows up.
    if (s.sink > 0.3) {
      line += `. It also parks ${(s.sink * 100).toFixed(0)}% of every row on the very first position`;
    }
    // Where the current row's peak actually is, since the strip is normalised
    // to it and may not contain it.
    const q = live - 1;
    const pk = peakOf(q);
    const lo = Math.max(0, q - STRIP_W + 1);
    if (pk.k < lo) {
      line +=
        `. Strongest for the position just computed is k=${pk.k} ` +
        `'${tokens[pk.k] ?? "·"}' at ${pk.p.toFixed(3)}, off the left of the strip`;
    }
    caption.textContent = `${line}.`;
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

  paint();

  return {
    setConfig,
    reset,
    setTokens,
    pushToken,
    pushTrace,
    select,
    // For the disclosure holding the matrix: a canvas inside a closed
    // <details> has clientWidth 0, and draw() returns early on that, so it
    // would open blank without a redraw on toggle.
    redraw: paint,
    selection: () => ({ block, head }),
  };
}
