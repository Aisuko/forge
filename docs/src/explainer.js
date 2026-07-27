// The attention half of one block, in 3D — one head, one query, six stations.
//
// Every cell below is a number the model computed during the run beside it:
// the Query for the newest position, the Keys and Values it is scored against,
// the scores themselves, the softmax over them, and the weighted sum that
// leaves the head. Nothing is bucketed, resampled or averaged, and nothing is
// Math.random().
//
// The previous version of this file drew seven stations over 384 embedding
// dimensions and 32 positions. It was all real, and none of it was legible:
// 192 texture columns landed on under a pixel each, so every stage read as
// noise. This one shows less on purpose — one head is 64 dimensions, and 64
// cells fit on a screen honestly.
//
// Two things are computed here rather than captured, from tensors that were:
// the scores, as dot(q, k) / sqrt(head_dim), and the output, as the attention-
// weighted sum of V. Drawing the dot product next to its two operands is the
// whole point of the section — the arithmetic is shown, not asserted.
//
// One shape worth understanding: with a KV cache, only the prompt prefill has
// a full q_len. The prefill fills the triangle in one go; every decode step
// after it adds exactly one query column — the one the model really attended
// with.
//
// three.js is imported here and in scene.js, and both are dynamically imported
// by demo.js, so a visitor who never scrolls pays for neither.
//
// Nothing here is required for the page to work: if WebGL is missing,
// createExplainer throws and demo.js shows the static panel instead.

import {
  AdditiveBlending,
  AmbientLight,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CatmullRomCurve3,
  ClampToEdgeWrapping,
  Color,
  DataTexture,
  DirectionalLight,
  DoubleSide,
  Group,
  InstancedMesh,
  LinearFilter,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  RGBAFormat,
  Raycaster,
  RepeatWrapping,
  SRGBColorSpace,
  Scene,
  UnsignedByteType,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";

import { RAMP, ramp } from "./scene.js";

// The one head and the one block this stage draws. Deliberately constants and
// not controls: a stepper over 6 heads × 6 blocks is 36 pictures, and the
// section's job is to make one of them readable.
const HEAD = 0;
const BLOCK = 0;

// Positions on screen. 8 keys and 8 queries is what leaves a cell big enough
// to read a number off; older positions scroll out of the window as the run
// goes on, and the model keeps attending to them either way.
const WIN = 8;

// Stage hues. Query/key/value keep the blue/red/green language the field uses,
// so the picture is readable by anyone who has seen an attention diagram.
const HUE = {
  q: [0x4c, 0x8d, 0xff],
  k: [0xff, 0x5c, 0x5c],
  v: [0x3e, 0xcf, 0x8e],
  score: [0xff, 0xb1, 0x85],
  out: [0xea, 0x6a, 0x24],
};

// Cell geometry, in world units. A vector runs *down* (64 dimensions), a
// position runs *across* (columns), and magnitude is depth toward the viewer —
// so a Q vector is a tall narrow card of 64 bars rather than a smear.
const ROW_PITCH = 0.11;
const ROW_H = 0.09;
const COL_PITCH = 0.34;
const COL_W = 0.28;
// The 8-wide stations (scores, attention) get bigger cells: they carry numbers.
const GRID_PITCH = 0.55;
const GRID_W = 0.46;

// Station centres along +X, in the order the calculation happens:
// q · k → scores → softmax → weighted by V → out.
const X = {
  // Q sits right against the Keys card: they are the two operands of one dot
  // product, and one ribbon leaves the pair together.
  q: -9.3,
  k: -7.4,
  scores: -1.9,
  attn: 3.8,
  // Clear of the matrix by more than the probabilities printed beside it.
  v: 10.4,
  out: 13.4,
};
// The camera looks at the middle of that run, nudged right so the pipeline
// sits clear of the probability panel that overlays the right edge on md up.
const CENTRE = (X.q + X.out) / 2 + 0.8;

// Height of a 64-dimension card, and of the whole stage.
const SPAN = 64 * ROW_PITCH;

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

const glyph = (s) =>
  s === undefined ? "·" : s.replace(/\n/g, "↵").replace(/ /g, "␣") || "·";

/** A 1 x 64 gradient used to make ribbon flow visible when it scrolls. */
function flowTexture() {
  const data = new Uint8Array(64 * 4);
  for (let i = 0; i < 64; i++) {
    const t = (Math.sin((i / 64) * Math.PI * 2) + 1) / 2;
    data[i * 4] = 0xea * t;
    data[i * 4 + 1] = 0x8a * t;
    data[i * 4 + 2] = 0x4c * t;
    data[i * 4 + 3] = 255 * (0.25 + 0.75 * t);
  }
  const tex = new DataTexture(data, 64, 1, RGBAFormat, UnsignedByteType);
  tex.colorSpace = SRGBColorSpace;
  tex.wrapS = RepeatWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.minFilter = LinearFilter;
  tex.magFilter = LinearFilter;
  return tex;
}

/**
 * Build the explainer stage in `canvas`, writing anchored labels into
 * `overlay` and the hovered-cell readout into `readout`.
 *
 * Throws if a WebGL context cannot be created — the caller's cue to show the
 * static panel rather than leave an empty rectangle.
 */
export function createExplainer({ canvas, overlay, readout, onSelect }) {
  const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new Scene();
  // A long lens: at 40° the far end of a 23-unit pipeline is half the size of
  // the near end, and cells that are the same number have to look the same.
  const camera = new PerspectiveCamera(32, 16 / 9, 0.5, 400);
  // One rig for the whole pipeline: resetting the view is one assignment.
  const rig = new Group();
  scene.add(rig);

  // Dimmer than a product render on purpose: these cells carry meaning in
  // their colour, and a bright key light washes a saturated hue out to pastel.
  scene.add(new AmbientLight(0xffffff, 1.05));
  const key = new DirectionalLight(0xffffff, 1.35);
  key.position.set(6, 12, 10);
  scene.add(key);
  const rim = new DirectionalLight(0xff8a4c, 0.7);
  rim.position.set(-10, -4, -8);
  scene.add(rim);

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  let still = reduced.matches;

  let cfg = { nLayer: 6, nHead: 6, nEmbd: 384, nCtx: 256 };
  let detailOff = false; // long prompt: attention only (see the §3.6 guard)

  // ── store: head HEAD of block BLOCK, rebuilt on the CPU from the captures ─
  // Each step hands back only its *new* positions, which is exactly what the
  // model computed. Accumulating them here is what lets a hover re-read a
  // value from ten tokens ago without re-running anything.
  let store = null;
  let live = 0; // positions filled
  let tokens = []; // display string per position
  let lastTop = [];

  function allocate() {
    const { nCtx, nEmbd, nHead } = cfg;
    const hd = Math.round(nEmbd / nHead);
    store = {
      hd,
      q: new Float32Array(nCtx * hd),
      k: new Float32Array(nCtx * hd),
      v: new Float32Array(nCtx * hd),
      // [query][key] for this head alone — the whole triangle, so a decode
      // step only ever writes the one column it produced.
      attn: new Float32Array(nCtx * nCtx),
    };
    if (outVec.length !== hd) outVec = new Float32Array(hd);
    live = 0;
  }

  // The window on screen: `origin` is the oldest position shown, `shown` how
  // many of the WIN slots have data.
  let origin = 0;
  let shown = 0;
  const query = () => live - 1; // the position being computed

  // ── stations ────────────────────────────────────────────────────────────
  // One InstancedMesh each: one draw call per station, whatever the cell
  // count, and one shared unit box scaled per instance.

  const stations = {};
  const disposables = [];
  const cellGeometry = new BoxGeometry(1, 1, 1);
  const cellMaterial = new MeshStandardMaterial({
    roughness: 0.5,
    metalness: 0.04,
    emissive: new Color(0xffffff),
    emissiveIntensity: 0.14,
  });
  disposables.push(cellGeometry, cellMaterial);

  const tmpMatrix = new Matrix4();
  const tmpColor = new Color();

  /**
   * A grid of `cols × rows` cells centred on `x`. Instance `c * rows + r` is
   * column `c`, row `r`, counting from the top — the same order every paint
   * function and the hover readout use.
   */
  function station(name, { x, cols, rows, colPitch, rowPitch, cellW, cellH }) {
    const mesh = new InstancedMesh(cellGeometry, cellMaterial, cols * rows);
    mesh.position.set(x, 0, 0);
    mesh.frustumCulled = false;
    mesh.userData.name = name;
    rig.add(mesh);
    stations[name] = {
      mesh,
      cols,
      rows,
      colPitch,
      rowPitch,
      cellW,
      cellH,
      x,
    };
    return stations[name];
  }

  function buildStations() {
    const card = {
      rows: 64,
      colPitch: COL_PITCH,
      rowPitch: ROW_PITCH,
      cellW: COL_W,
      cellH: ROW_H,
    };
    const hd = store ? store.hd : 64;
    station("q", { ...card, rows: hd, x: X.q, cols: 1 });
    station("k", { ...card, rows: hd, x: X.k, cols: WIN });
    station("v", { ...card, rows: hd, x: X.v, cols: WIN });
    station("out", { ...card, rows: hd, x: X.out, cols: 1 });
    // The two stations that carry numbers get cells four times the size.
    const grid = {
      colPitch: GRID_PITCH,
      rowPitch: GRID_PITCH,
      cellW: GRID_W,
      cellH: GRID_W,
    };
    station("scores", { ...grid, x: X.scores, cols: WIN, rows: 1 });
    station("attn", { ...grid, x: X.attn, cols: WIN, rows: WIN });
  }

  /** Column `c` of `st`, in world X relative to the station. */
  const colX = (st, c) => (c - (st.cols - 1) / 2) * st.colPitch;
  /** Row `r` of `st`, in world Y. Row 0 is the top. */
  const rowY = (st, r) => (-(r - (st.rows - 1) / 2)) * st.rowPitch;

  const ZERO = new Matrix4().makeScale(0, 0, 0);
  const BASE = [0x14, 0x16, 0x1e];

  /**
   * Paint one station. `read(c, r)` returns the value for that cell, or `NaN`
   * for a cell that does not exist — masked attention, or a position the run
   * has not reached. Those are not drawn at all, so the causal triangle is a
   * shape rather than a convention to be explained.
   *
   * Magnitude becomes depth toward the viewer and brightness against the
   * station's hue; sign becomes a dimmer, cooler cell rather than a second
   * hue, because these are activations and what matters is where the energy
   * is. `scale` is the value that saturates: for Q, K, V it is the largest
   * magnitude on screen, so a card is always readable whatever its range.
   */
  const cellRgba = new Uint8Array(4);

  function paint(
    name,
    hue,
    scale,
    read,
    { maxDepth = 1.4, ramped = false, ghost = false, lit: litCol = () => false } = {},
  ) {
    const st = stations[name];
    if (!st) return;
    const inv = scale > 0 ? 1 / scale : 0;
    for (let c = 0; c < st.cols; c++) {
      for (let r = 0; r < st.rows; r++) {
        const i = c * st.rows + r;
        const value = read(c, r);
        if (Number.isNaN(value)) {
          st.mesh.setMatrixAt(i, ZERO);
          continue;
        }
        // An empty station is tinted just enough to carry its colour — the
        // legend for the whole picture is which card is which hue, and before
        // the first run there is no data to say it with.
        const t = ghost ? 0.32 : clamp01(Math.abs(value) * inv) ** 0.65;
        const depth = ghost ? 0.05 : 0.05 + t * maxDepth;
        tmpMatrix.makeScale(st.cellW, st.cellH, depth);
        tmpMatrix.setPosition(colX(st, c), rowY(st, r), depth / 2);
        st.mesh.setMatrixAt(i, tmpMatrix);
        // Highlighting a key lifts every cell that key touches.
        const lift = litCol(c, r) ? 1.45 : 1;
        if (ramped) {
          // Probabilities share scene.js's ramp: two ramps for the same
          // quantity would read as two different quantities.
          ramp(t, cellRgba, 0);
          tmpColor.setRGB(
            clamp01((cellRgba[0] * lift) / 255),
            clamp01((cellRgba[1] * lift) / 255),
            clamp01((cellRgba[2] * lift) / 255),
            SRGBColorSpace,
          );
        } else {
          const neg = value < 0 ? 0.45 : 1;
          tmpColor.setRGB(
            clamp01(((BASE[0] + (hue[0] * neg - BASE[0]) * t) * lift) / 255),
            clamp01(((BASE[1] + (hue[1] * neg - BASE[1]) * t) * lift) / 255),
            clamp01(((BASE[2] + (hue[2] * neg - BASE[2]) * t) * lift) / 255),
            SRGBColorSpace,
          );
        }
        st.mesh.setColorAt(i, tmpColor);
      }
    }
    st.mesh.instanceMatrix.needsUpdate = true;
    if (st.mesh.instanceColor) st.mesh.instanceColor.needsUpdate = true;
  }

  // ── ribbons: the flow between stations ──────────────────────────────────

  const ribbons = [];
  const flowTex = flowTexture();
  disposables.push(flowTex);

  function ribbon(from, to, width, lift = 0.9) {
    const curve = new CatmullRomCurve3([
      new Vector3(from, 0, 0),
      new Vector3((from + to) / 2, lift, 0),
      new Vector3(to, 0, 0),
    ]);
    const n = 24;
    const pos = new Float32Array((n + 1) * 2 * 3);
    const uv = new Float32Array((n + 1) * 2 * 2);
    const idx = [];
    for (let i = 0; i <= n; i++) {
      const p = curve.getPoint(i / n);
      for (const s of [-1, 1]) {
        const o = (i * 2 + (s > 0 ? 1 : 0)) * 3;
        pos[o] = p.x;
        pos[o + 1] = p.y;
        pos[o + 2] = p.z + (s * width) / 2;
        const u = (i * 2 + (s > 0 ? 1 : 0)) * 2;
        uv[u] = i / n;
        uv[u + 1] = s > 0 ? 1 : 0;
      }
      if (i < n) {
        const a = i * 2;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(pos, 3));
    geometry.setAttribute("uv", new BufferAttribute(uv, 2));
    geometry.setIndex(idx);
    const map = flowTex.clone();
    map.needsUpdate = true;
    map.repeat.set(3, 1);
    const material = new MeshBasicMaterial({
      map,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
      side: DoubleSide,
      // Faint on purpose: these are a hint that data moved, and they sit in
      // the same frame as six stations of cells that carry actual numbers.
      opacity: 0.3,
    });
    const mesh = new Mesh(geometry, material);
    rig.add(mesh);
    disposables.push(geometry, material, map);
    ribbons.push({ map, mesh });
  }

  function buildRibbons() {
    const half = (st) => (st.cols * st.colPitch) / 2;
    const scoresL = X.scores - half(stations.scores) - 0.2;
    // One ribbon for the Query and the Keys together. Giving the Query its own
    // would mean arcing it over or through 64 × 8 cells of solid Keys card,
    // and a loop that big reads as decoration rather than as flow.
    ribbon(X.k + half(stations.k) + 0.2, scoresL, 1.6, 0.9);
    ribbon(X.scores + half(stations.scores) + 0.2, X.attn - half(stations.attn) - 0.2, 2.2);
    ribbon(X.attn + half(stations.attn) + 0.2, X.v - half(stations.v) - 0.2, 2.2);
    ribbon(X.v + half(stations.v) + 0.2, X.out - half(stations.out) - 0.2, 1.4);
    // The last two feed stations that fold away on a phone; a ribbon into
    // nothing is worse than no ribbon.
    ribbons[2].foldable = true;
    ribbons[3].foldable = true;
  }

  // Ribbons scroll for ~400 ms as a step's data lands, then settle.
  let flowUntil = 0;

  // ── overlay ─────────────────────────────────────────────────────────────
  // Projected HTML rather than 3D text: crisp at any DPR, selectable, and
  // reachable by a screen reader — for no SDF font atlas.

  const anchors = [];

  function anchor(text, pos, cls) {
    const el = document.createElement("span");
    el.className = `stage-label ${cls || ""}`;
    el.textContent = text;
    overlay.appendChild(el);
    const a = { el, pos };
    anchors.push(a);
    return a;
  }

  const caps = {};
  let colLabels = []; // the character over each key column
  let queryLabels = []; // the character over each query column of the matrix
  let dimTicks = []; // 0, 8, 16 … down the side of Q
  let scoreNums = []; // one number under each score bar
  let probNums = []; // the softmax of those scores, beside the matrix

  function buildLabels() {
    const top = SPAN / 2 + 0.55;
    const gtop = (WIN * GRID_PITCH) / 2 + 0.55;
    caps.q = anchor("Query", new Vector3(X.q, top, 0), "stage-cap cap-q");
    caps.k = anchor("Keys", new Vector3(X.k, top + 0.75, 0), "stage-cap cap-k");
    caps.scores = anchor(
      "scores  q · kᵢ / √64",
      new Vector3(X.scores, gtop + 0.9, 0),
      "stage-cap",
    );
    caps.attn = anchor(
      "softmax → attention",
      new Vector3(X.attn, gtop, 0),
      "stage-cap",
    );
    caps.v = anchor("Values", new Vector3(X.v, top, 0), "stage-cap cap-v");
    caps.out = anchor(
      "out  Σ pᵢ vᵢ",
      new Vector3(X.out, top + 0.75, 0),
      "stage-cap",
    );
    // Says what the stage is fixed to, so a visitor never wonders which of the
    // 36 head/block pairs they are looking at.
    caps.fixed = anchor(
      `block ${BLOCK + 1} · head ${HEAD + 1}`,
      new Vector3(X.scores, -SPAN / 2 - 0.55, 0),
      "stage-cap",
    );
    // Which position the Query belongs to — rewritten every step.
    caps.pos = anchor("", new Vector3(X.q + 1.1, -SPAN / 2 - 0.55, 0), "stage-cap");

    const k = stations.k;
    const attn = stations.attn;
    for (let c = 0; c < WIN; c++) {
      const a = anchor("", new Vector3(X.k + colX(k, c), SPAN / 2 + 0.22, 0), "stage-token");
      a.col = c;
      colLabels.push(a);
      const qa = anchor(
        "",
        new Vector3(X.attn + colX(attn, c), (WIN * GRID_PITCH) / 2 + 0.22, 0),
        "stage-token",
      );
      qa.col = c;
      queryLabels.push(qa);
      // Two rows, alternating: "-0.42" is wider than the gap between bars.
      const sn = anchor(
        "",
        new Vector3(
          X.scores + colX(stations.scores, c),
          -0.5 - (c % 2) * 0.55,
          0,
        ),
        "stage-num",
      );
      sn.col = c;
      scoreNums.push(sn);
      const pn = anchor(
        "",
        new Vector3(
          X.attn + colX(attn, WIN - 1) + 0.55,
          rowY(attn, c),
          0,
        ),
        "stage-num",
      );
      pn.row = c;
      probNums.push(pn);
    }
    const hd = store ? store.hd : 64;
    for (let d = 0; d < hd; d += 8) {
      const a = anchor(
        `${d}`,
        new Vector3(X.q - 0.85, rowY(stations.q, d), 0),
        "stage-dim",
      );
      dimTicks.push(a);
    }
  }

  function clearLabels() {
    for (const a of anchors.splice(0)) a.el.remove();
    colLabels = [];
    queryLabels = [];
    dimTicks = [];
    scoreNums = [];
    probNums = [];
  }

  const projected = new Vector3();

  function projectAnchors() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    for (const a of anchors) {
      if (a.el.hidden) continue;
      projected.copy(a.pos).applyMatrix4(rig.matrixWorld).project(camera);
      const x = (projected.x * 0.5 + 0.5) * w;
      const y = (-projected.y * 0.5 + 0.5) * h;
      const behind = projected.z > 1;
      a.el.style.opacity = behind ? "0" : "1";
      a.el.style.transform = `translate(-50%,-50%) translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
    }
  }

  // ── the calculation ─────────────────────────────────────────────────────

  const scores = new Float32Array(WIN);
  let outVec = new Float32Array(64);

  /** dot(q_t, k_i) / sqrt(head_dim) for each key on screen. */
  function computeScores() {
    scores.fill(NaN);
    if (!store || live < 1) return;
    const { hd, q: Q, k: K } = store;
    const t = query();
    const inv = 1 / Math.sqrt(hd);
    for (let c = 0; c < shown; c++) {
      const p = origin + c;
      let dot = 0;
      for (let d = 0; d < hd; d++) dot += Q[t * hd + d] * K[p * hd + d];
      scores[c] = dot * inv;
    }
  }

  /**
   * Σ p_i · v_i over *every* key the model attended to, not only the eight on
   * screen — that is what the head actually output, and windowing its inputs
   * must not quietly window its result.
   */
  function computeOut() {
    outVec.fill(0);
    if (!store || live < 1) return;
    const { hd, v: V, attn } = store;
    const { nCtx } = cfg;
    const t = query();
    for (let p = 0; p <= t && p < nCtx; p++) {
      const w = attn[t * nCtx + p];
      if (w === 0) continue;
      for (let d = 0; d < hd; d++) outVec[d] += w * V[p * hd + d];
    }
  }

  /** Largest magnitude over the visible columns of a per-position tensor. */
  function windowScale(src, hd) {
    let max = 0;
    for (let c = 0; c < shown; c++) {
      const base = (origin + c) * hd;
      for (let d = 0; d < hd; d++) {
        const a = Math.abs(src[base + d]);
        if (a > max) max = a;
      }
    }
    return max;
  }

  let highlight = -1; // a key column, lit across every station that uses it

  /** Repaint every station from the retained data — no model work. */
  function repaint() {
    if (!store) return;
    const { hd } = store;
    const { nCtx } = cfg;
    shown = Math.min(live, WIN);
    origin = Math.max(0, live - WIN);
    const t = query();

    computeScores();
    computeOut();

    // Before the first run there is nothing to draw and everything to
    // explain, so every station shows its grid as empty cells: the shape of
    // the calculation is on screen, and the first token fills it in rather
    // than conjuring it out of a blank rectangle.
    const empty = live === 0;
    const cols = empty ? WIN : shown;
    // A key column lights across every station indexed by keys.
    const litKey = (c) => highlight >= 0 && c === highlight;

    // Q, K and V share one scale so their cards are comparable to each other,
    // which is the only way "this key matched that query" is visible.
    const qkv = Math.max(
      windowScale(store.q, hd),
      windowScale(store.k, hd),
      windowScale(store.v, hd),
      1e-6,
    );
    paint("q", HUE.q, qkv, (c, r) => (empty ? 0 : store.q[t * hd + r]), { ghost: empty });
    paint("k", HUE.k, qkv, (c, r) => (c >= cols ? NaN : empty ? 0 : store.k[(origin + c) * hd + r]), {
      ghost: empty,
      lit: litKey,
    });
    paint("v", HUE.v, qkv, (c, r) => (c >= cols ? NaN : empty ? 0 : store.v[(origin + c) * hd + r]), {
      ghost: empty,
      lit: litKey,
    });

    let smax = 1e-6;
    for (let c = 0; c < shown; c++) smax = Math.max(smax, Math.abs(scores[c]));
    paint("scores", HUE.score, smax, (c) => (c >= cols ? NaN : empty ? 0 : scores[c]), {
      maxDepth: 2.0,
      ghost: empty,
      lit: litKey,
    });

    // Queries across, keys down; k > q is masked and simply not drawn, so the
    // causal triangle is a shape rather than a convention to be explained.
    paint(
      "attn",
      RAMP[1],
      1,
      (c, r) => {
        if (r > c) return NaN;
        if (empty) return 0;
        const q = origin + c;
        return q >= live ? NaN : store.attn[q * nCtx + origin + r];
      },
      { maxDepth: 2.4, ramped: !empty, ghost: empty, lit: (c, r) => litKey(r) },
    );

    let omax = 1e-6;
    for (let d = 0; d < hd; d++) omax = Math.max(omax, Math.abs(outVec[d]));
    paint("out", HUE.out, omax, (c, r) => (empty ? 0 : outVec[r]), { ghost: empty });

    paintLabels();
    applyVisibility();
    if (still) render();
  }

  function paintLabels() {
    const t = query();
    const { nCtx } = cfg;
    caps.pos.el.textContent =
      live > 0 ? `position ${t}  ${JSON.stringify(glyph(tokens[t]))}` : "position —";
    for (const a of colLabels) {
      const p = origin + a.col;
      a.el.hidden = a.col >= shown;
      a.el.textContent = `${glyph(tokens[p])}`;
      a.el.title = `key at position ${p}`;
      a.el.classList.toggle("is-lit", a.col === highlight);
    }
    for (const a of queryLabels) {
      const p = origin + a.col;
      a.el.hidden = a.col >= shown;
      a.el.textContent = `${glyph(tokens[p])}`;
      a.el.title = `query at position ${p}`;
      a.el.classList.toggle("is-lit", p === t);
    }
    for (const a of scoreNums) {
      a.el.hidden = a.col >= shown;
      if (!a.el.hidden) a.el.textContent = scores[a.col].toFixed(2);
      a.el.classList.toggle("is-lit", a.col === highlight);
    }
    // The probabilities beside the matrix are the *current* query's column —
    // the softmax of the numbers one station to the left. They will not sum to
    // 1 once the run is past 8 positions, because the softmax normalises over
    // every key, not the 8 on screen. The caption says so; the numbers are the
    // model's own either way.
    const attn = stations.attn;
    // Clear of the last column's cells, which are extruded toward the viewer
    // and so reach further right on screen than their footprint does.
    const last = colX(attn, Math.max(0, shown - 1)) + 1.5;
    for (const a of probNums) {
      const k = origin + a.row;
      a.el.hidden = live < 1 || a.row >= shown || k > t;
      if (!a.el.hidden) a.el.textContent = store.attn[t * nCtx + k].toFixed(3);
      a.el.classList.toggle("is-lit", a.row === highlight);
      a.pos.x = X.attn + last;
    }
  }

  // ── ingest ──────────────────────────────────────────────────────────────

  /**
   * One step's trace. The prefill lands `qLen` positions at once; every decode
   * step after it lands exactly one. Only block BLOCK, head HEAD is kept — the
   * rest of the trace feeds the folded stack in scene.js.
   */
  function ingest(trace) {
    if (!store) allocate();
    const { nCtx, nHead } = cfg;
    const { hd } = store;
    const base = trace.kvLen - trace.qLen; // first new position
    const qLen = trace.qLen;

    for (const a of trace.attn) {
      if (a.layer !== BLOCK || HEAD >= a.nHead) continue;
      for (let q = 0; q < a.qLen; q++) {
        const pos = base + q;
        if (pos >= nCtx) break;
        const from = (HEAD * a.qLen + q) * a.kvLen;
        for (let k = 0; k < a.kvLen && k < nCtx; k++) {
          store.attn[pos * nCtx + k] = a.probs[from + k];
        }
      }
    }

    for (const d of trace.detail) {
      if (d.layer !== BLOCK) continue;
      for (const [name, src] of [
        ["q", d.q],
        ["k", d.k],
        ["v", d.v],
      ]) {
        if (!src.length) continue;
        // Head count from the tensor itself, not from cfg: if the two ever
        // disagree, reading the wrong stride would shear every column rather
        // than fail.
        const heads = Math.min(nHead, src.length / (qLen * hd));
        if (HEAD >= heads) continue;
        for (let q = 0; q < qLen; q++) {
          const pos = base + q;
          if (pos >= nCtx) break;
          const from = (HEAD * qLen + q) * hd;
          store[name].set(src.subarray(from, from + hd), pos * hd);
        }
      }
    }

    live = Math.min(trace.kvLen, nCtx);
    lastTop = trace.top;
  }

  // ── interaction ─────────────────────────────────────────────────────────

  const pointer = new Vector2();
  const raycaster = new Raycaster();
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  // Near side-on, tipped down far enough to read the cards as relief:
  // straight-on hides the extrusion, top-down loses the pipeline.
  const YAW0 = -0.2;
  const PITCH0 = 0.22;
  let yaw = YAW0;
  let pitch = PITCH0;
  let dist = 24;
  let hovered = -1;
  let hoveredName = "";
  // Once the visitor has zoomed, the stage stops second-guessing them.
  let zoomed = false;
  let needsFit = true;

  const narrow = () => canvas.clientWidth < 640;

  const probe = new Vector3();
  // What the camera is pointed at. Panned by the fit below rather than fixed
  // at the origin: which stations are on screen changes with width, and a
  // pipeline centred for a desktop sits half off a phone.
  let targetX = CENTRE;
  let targetY = 0;

  /** World-space corners of everything currently drawn, labels included. */
  function corners() {
    const out = [];
    for (const [name, st] of Object.entries(stations)) {
      if (!st.mesh.visible) continue;
      // Margins for the projected HTML: captions above, the position line
      // below, dimension ticks left of Q and probabilities right of the
      // matrix. Labels are not geometry, so nothing else would count them.
      const w = (st.cols * st.colPitch) / 2;
      const h = (st.rows * st.rowPitch) / 2;
      // A one-column station is narrower than its own caption, so the caption
      // is what has to fit; the matrix has the probabilities down its right.
      const wide = st.cols === 1 ? 1.5 : 0.2;
      const left = name === "q" ? Math.max(wide, 1.3) : wide;
      const right = name === "attn" ? 2.4 : wide;
      const below = name === "attn" || name === "scores" ? 1.1 : 0.9;
      for (const x of [st.x - w - left, st.x + w + right]) {
        for (const y of [-h - below, h + 1.5]) {
          for (const z of [0, 2.2]) out.push([x, y, z]);
        }
      }
    }
    return out;
  }

  /**
   * Frame everything visible: pan until it is centred, zoom until it just
   * fits. Measured, not derived — the camera is off-axis in both yaw and
   * pitch, so the screen extent of a world-space box is not its width over
   * the distance, and a closed form for it is longer than this loop and
   * easier to get wrong. Four passes is comfortably enough to converge.
   */
  function fitView() {
    const box = corners();
    if (!box.length) return;
    for (let pass = 0; pass < 4; pass++) {
      applyCamera();
      camera.updateMatrixWorld(true);
      let x0 = Infinity;
      let x1 = -Infinity;
      let y0 = Infinity;
      let y1 = -Infinity;
      for (const [x, y, z] of box) {
        probe.set(x, y, z).project(camera);
        x0 = Math.min(x0, probe.x);
        x1 = Math.max(x1, probe.x);
        y0 = Math.min(y0, probe.y);
        y1 = Math.max(y1, probe.y);
      }
      // NDC → world at the target plane, for the pan.
      const k = dist * Math.tan(((camera.fov * Math.PI) / 180) / 2);
      targetX += ((x0 + x1) / 2) * k * camera.aspect * Math.cos(yaw);
      targetY += ((y0 + y1) / 2) * k;
      dist = Math.max(
        4,
        Math.min(90, dist * Math.max((x1 - x0) / 1.84, (y1 - y0) / 1.84)),
      );
    }
    applyCamera();
  }

  function applyCamera() {
    camera.position.set(
      targetX + Math.sin(yaw) * Math.cos(pitch) * dist,
      targetY + Math.sin(pitch) * dist,
      Math.cos(yaw) * Math.cos(pitch) * dist,
    );
    camera.lookAt(targetX, targetY, 0);
  }
  applyCamera();

  const onDown = (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onUp = (e) => {
    dragging = false;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      // The pointer was already released (leave, or a cancelled gesture).
    }
  };
  const onMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    if (dragging) {
      yaw -= (e.clientX - lastX) * 0.006;
      pitch = Math.max(-0.15, Math.min(1.15, pitch + (e.clientY - lastY) * 0.004));
      lastX = e.clientX;
      lastY = e.clientY;
      applyCamera();
      if (still) render();
    }
    hover();
  };
  const onLeave = () => {
    dragging = false;
    if (hovered !== -1) {
      hovered = -1;
      hoveredName = "";
      if (readout) readout.textContent = "";
      setHighlight(-1);
    }
  };
  const onWheel = (e) => {
    e.preventDefault();
    zoomed = true;
    dist = Math.max(6, Math.min(90, dist + e.deltaY * 0.02));
    applyCamera();
    if (still) render();
  };

  function setHighlight(col) {
    if (col === highlight) return;
    highlight = col;
    repaint();
  }

  /** What the hovered cell is, in words and in the model's own numbers. */
  function describe(name, c, r) {
    const { hd } = store;
    const { nCtx } = cfg;
    const p = origin + c;
    const at = (i) => `${i} ${JSON.stringify(glyph(tokens[i]))}`;
    if (name === "q") {
      return `q[${at(query())}][dim ${r}] = ${store.q[query() * hd + r].toFixed(4)}`;
    }
    if (name === "k" || name === "v") {
      return `${name}[${at(p)}][dim ${r}] = ${store[name][p * hd + r].toFixed(4)}`;
    }
    if (name === "scores") {
      return `q · k[${at(p)}] / √${hd} = ${scores[c].toFixed(4)}`;
    }
    if (name === "attn") {
      const q = origin + c;
      const k = origin + r;
      return (
        `p(q=${at(q)} ← k=${at(k)}) = ${store.attn[q * nCtx + k].toFixed(4)}` +
        `  ·  softmax over all ${q + 1} keys`
      );
    }
    if (name === "out") {
      return `out[${at(query())}][dim ${r}] = ${outVec[r].toFixed(4)}  ·  Σ pᵢ vᵢ over ${query() + 1} keys`;
    }
    return "";
  }

  function hover() {
    if (!store || live < 1) return;
    raycaster.setFromCamera(pointer, camera);
    const meshes = Object.values(stations)
      .map((s) => s.mesh)
      .filter((m) => m.visible);
    const hit = raycaster.intersectObjects(meshes, false)[0];
    const id = hit && hit.instanceId !== undefined ? hit.instanceId : -1;
    const name = hit ? hit.object.userData.name : "";
    if (id === hovered && name === hoveredName) return;
    hovered = id;
    hoveredName = name;
    canvas.style.cursor = id >= 0 ? "crosshair" : "grab";
    if (id < 0) {
      if (readout) readout.textContent = "";
      setHighlight(-1);
      return;
    }
    const st = stations[name];
    const c = Math.floor(id / st.rows);
    const r = id % st.rows;
    if (readout) readout.textContent = describe(name, c, r);
    // Hovering anything that is indexed by a key lights that key everywhere.
    setHighlight(name === "attn" ? r : name === "q" || name === "out" ? -1 : c);
    onSelect?.({ station: name, col: origin + c, row: r });
  }

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // ── visibility ──────────────────────────────────────────────────────────

  /**
   * Below 640 px six stations cannot be read end to end at any distance that
   * keeps a cell legible, so Values and the output fold away — Q, the keys,
   * the scores and the triangle still tell the whole "which key wins" story.
   *
   * With a long prompt the detail readback never happened (the §3.6 guard), so
   * only the triangle has data. The empty stations hide rather than drawing a
   * grid of zeros, which would read as "the model computed nothing".
   */
  let shape = "";

  function applyVisibility() {
    const n = narrow();
    const folded = (name) => n && (name === "v" || name === "out");
    for (const [name, st] of Object.entries(stations)) {
      st.mesh.visible = name === "attn" ? true : !detailOff && !folded(name);
    }
    for (const [name, a] of Object.entries(caps)) {
      // The two captions that belong to a station rather than being one.
      const own = { pos: "q", fixed: "attn" }[name] || name;
      a.el.hidden = (own !== "attn" && detailOff) || folded(own);
    }
    // On a phone the labels are what collide first: the queries along the top
    // of the matrix repeat the keys already named over the Keys card, and the
    // probabilities beside it repeat what the triangle's own relief says.
    // The eight score numbers are 40 px of text in a 22 px column at this
    // width; printed on top of each other they are worse than not printed.
    // The bars still rank the keys, and the readout still names any cell.
    for (const a of [...queryLabels, ...probNums, ...scoreNums]) {
      a.el.hidden = a.el.hidden || n;
    }
    if (detailOff) {
      for (const a of [...colLabels, ...scoreNums]) a.el.hidden = true;
    }
    // The dimension ticks are the first thing to go when there is no room:
    // they annotate the Query card, and on a phone it is 40 px wide.
    for (const a of dimTicks) a.el.hidden = detailOff || n;
    for (const m of ribbons) m.mesh.visible = !detailOff && !(n && m.foldable);
    // What is on screen decides how the stage is framed, so a change here is
    // a reason to reframe it.
    const next = `${n}${detailOff}`;
    if (next !== shape) {
      shape = next;
      needsFit = true;
    }
  }

  // ── loop ────────────────────────────────────────────────────────────────

  let frame = 0;
  let prev = 0;
  let lastNarrow = null;
  let lastW = 0;
  let lastH = 0;

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    if (w === lastW && h === lastH) return;
    lastW = w;
    lastH = h;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    needsFit = true;
    // Set first, then repaint: repaint can render, render calls resize, and a
    // stale flag would make that recurse.
    const was = lastNarrow;
    lastNarrow = narrow();
    if (was !== null && was !== lastNarrow) repaint();
  }

  function render() {
    resize();
    // Reframing is a response to the stage changing shape or contents, not
    // something to redo 60 times a second — and never while a visitor is in
    // the middle of moving the camera themselves.
    if (needsFit && !dragging && !zoomed) {
      fitView();
      needsFit = false;
    }
    rig.updateMatrixWorld();
    renderer.render(scene, camera);
    projectAnchors();
  }

  function loop(now) {
    frame = requestAnimationFrame(loop);
    const dt = prev ? Math.min(0.05, (now - prev) / 1000) : 0;
    prev = now;
    if (now < flowUntil) {
      for (const m of ribbons) m.map.offset.x = (m.map.offset.x - dt * 1.6) % 1;
    }
    render();
  }

  function start() {
    if (still) {
      render();
      return;
    }
    if (!frame) {
      prev = 0;
      frame = requestAnimationFrame(loop);
    }
  }

  function stop() {
    if (frame) {
      cancelAnimationFrame(frame);
      frame = 0;
    }
  }

  const onReduced = () => {
    still = reduced.matches;
    stop();
    start();
  };
  const onResize = () => {
    if (still) render();
  };
  reduced.addEventListener("change", onReduced);
  window.addEventListener("resize", onResize);

  // Only render while on screen: a permanently spinning scene drains laptop
  // batteries for a section nobody is looking at.
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) start();
        else stop();
      }
    },
    { threshold: 0.02 },
  );
  io.observe(canvas);

  allocate();
  buildStations();
  buildRibbons();
  buildLabels();
  repaint();
  start();

  // ── public surface ──────────────────────────────────────────────────────

  return {
    /** Resize the store to the model that is actually loaded. */
    setConfig(next) {
      const merged = { ...cfg, ...next };
      const rebuild =
        merged.nCtx !== cfg.nCtx ||
        merged.nHead !== cfg.nHead ||
        merged.nEmbd !== cfg.nEmbd;
      cfg = merged;
      if (rebuild) {
        for (const st of Object.values(stations)) {
          rig.remove(st.mesh);
          st.mesh.dispose();
        }
        for (const name of Object.keys(stations)) delete stations[name];
        clearLabels();
        allocate();
        buildStations();
        buildLabels();
      }
      repaint();
    },

    /** Prompt token strings, before the run starts. */
    setTokens(list) {
      tokens = [...list];
      paintLabels();
      if (still) render();
    },

    /** One generated token's display text, appended as its position fills. */
    pushToken(text) {
      tokens.push(text);
    },

    /**
     * Attention only, for a prompt long enough that the detail readback would
     * be worth megabytes. The triangle still tells the story.
     */
    setDetailEnabled(on) {
      detailOff = !on;
      repaint();
    },

    /** One step's trace, straight from the model. */
    pushTrace(trace) {
      ingest(trace);
      flowUntil = performance.now() + 400;
      repaint();
    },

    resetView() {
      yaw = YAW0;
      pitch = PITCH0;
      zoomed = false;
      targetX = CENTRE;
      targetY = 0;
      needsFit = true;
      if (still) render();
    },

    /** What the last step said the next token would probably be. */
    top() {
      return lastTop;
    },

    /** Back to an empty stage — called at the start of every run. */
    reset() {
      tokens = [];
      lastTop = [];
      highlight = -1;
      allocate();
      repaint();
    },

    dispose() {
      stop();
      io.disconnect();
      reduced.removeEventListener("change", onReduced);
      window.removeEventListener("resize", onResize);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      clearLabels();
      for (const st of Object.values(stations)) st.mesh.dispose();
      for (const d of disposables.splice(0)) d.dispose();
      renderer.dispose();
    },
  };
}
