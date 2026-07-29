// The whole forward pass of one block, in 3D, as a walk the visitor steps
// through — embedding, LayerNorm, Q/K/V, scores, softmax, the weighted sum,
// the concat, the projection, both residual adds, the MLP, the remaining five
// blocks, and the distribution over the next character.
//
// Every cell below is a number this model computed on the visitor's GPU during
// the run beside it. Nothing is bucketed, resampled, averaged, or
// Math.random(). Exactly two quantities are computed here rather than
// captured — the score dot products and the attention-weighted sum of V — and
// both are drawn next to their operands *and* checked against the captured
// tensor that supersedes them, because drawing the arithmetic is the point and
// drawing it wrong would be worse than not drawing it.
//
// This file replaces explainer.js (one head of one block) and scene.js (the
// folded remainder) with one scene graph and one canvas. It is table-driven —
// STAGES below is the specification, and there is no per-stage geometry code —
// because fifteen hand-placed stations would not survive one layout change,
// and would not fit the site's 45 KB gzipped core budget.
//
// Our model does not fit on a screen the way a 3-layer/48-dim toy network
// does. So every station states what it is showing and what it is not, in the
// caption anchored above it. A silent crop would break the section's one
// promise.
//
// Nothing here is required for the page to work: if WebGL is missing,
// createPipeline throws and demo.js shows the static panel instead.

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

// The one head and the one block drawn in full. Deliberately constants and not
// controls: 6 heads × 6 blocks is 36 pictures, and the section's job is to make
// one path legible. Blocks 2-6 are stage 14.
const HEAD = 0;
const BLOCK = 0;

// Positions on screen. 8 is what leaves a cell big enough to read a number off;
// older positions scroll out of the window as the run goes on, and the model
// keeps attending to them either way.
const WIN = 8;

// Hues by role. Q/K/V keep the blue/red/green language the field uses, so the
// picture is readable by anyone who has seen an attention diagram. The residual
// stream has one colour throughout — it is one tensor being rewritten, and
// giving each rewrite its own hue would say otherwise.
const HUE = {
  x: [0x7c, 0x9a, 0xd8],
  q: [0x4c, 0x8d, 0xff],
  k: [0xff, 0x5c, 0x5c],
  v: [0x3e, 0xcf, 0x8e],
  score: [0xff, 0xb1, 0x85],
  out: [0xea, 0x6a, 0x24],
  mlp: [0xc9, 0x8d, 0xff],
  norm: [0x6f, 0xd2, 0xe0],
};

// The probability ramp, previously exported by scene.js. Two ramps for the
// same quantity would read as two different quantities.
const RAMP = [
  [0x1c, 0x1c, 0x24],
  [0xea, 0x6a, 0x24],
  [0xff, 0xe6, 0xd5],
];

/** Write RAMP(t) as RGBA bytes into `out` at offset `o`. `t` is clamped 0..1. */
function ramp(t, out, o) {
  const i = t < 0.5 ? 0 : 1;
  const f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  for (let c = 0; c < 3; c++) {
    out[o + c] = RAMP[i][c] + (RAMP[i + 1][c] - RAMP[i][c]) * f;
  }
  out[o + 3] = 255;
}

// Cell geometry, in world units. A vector runs *down*, a position runs
// *across*, and magnitude is depth toward the viewer — so a Q vector is a tall
// narrow card of 64 bars rather than a smear.
const ROW_PITCH = 0.11;
const ROW_H = 0.09;
const COL_PITCH = 0.34;
const COL_W = 0.28;
// The stations that carry printed numbers get cells four times the size.
const GRID_PITCH = 0.55;
const GRID_W = 0.46;
// A 384-channel tile is 24 × 16 of these; the cells are small because there
// are 384 of them per position and all 384 are drawn.
const TILE_PITCH = 0.13;
const TILE_W = 0.11;
// Gap between one position's tile and the next.
const GROUP_GAP = 0.5;
// Clear space between two stations.
const STATION_GAP = 2.1;

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

const glyph = (s) =>
  s === undefined ? "·" : s.replace(/\n/g, "↵").replace(/ /g, "␣") || "·";

/**
 * The pipeline, one row per stage.
 *
 * `slice` is drawn on screen under the station's name — it is the contract
 * with the visitor about what has been left out. `buf` names a store buffer
 * and `width` its stride per position; `kind` decides the geometry. `single`
 * marks a station that shows only the position being computed rather than the
 * window of eight.
 */
const STAGES = [
  {
    id: "tokens",
    name: "Tokens",
    slice: `last ${WIN} of up to 256 positions`,
    stations: [{ n: "tokens", kind: "labels", hue: HUE.x }],
  },
  {
    id: "embed",
    name: "Embedding — wte + wpe",
    slice: "the newest position, all 384 channels, folded 24 × 16",
    stations: [{ n: "embed", kind: "tile", buf: "embed", unit: "ch", hue: HUE.x, single: true }],
  },
  {
    id: "ln1",
    name: "LayerNorm 1",
    slice: "the newest position, all 384 — same tile as the last stage, so the flattening shows",
    stations: [{ n: "ln1", kind: "tile", buf: "ln1", unit: "ch", hue: HUE.norm, single: true }],
  },
  {
    id: "qkv",
    name: "Q, K and V",
    slice: "head 1 of 6 — 64 of the 384 dimensions",
    stations: [
      { n: "q", kind: "card", buf: "q", unit: "dim", hue: HUE.q, single: true, label: "Query" },
      { n: "k", kind: "card", buf: "k", unit: "dim", hue: HUE.k, label: "Keys" },
      { n: "v", kind: "card", buf: "v", unit: "dim", hue: HUE.v, label: "Values" },
    ],
  },
  {
    id: "scores",
    name: "Scores — q · kᵢ / √64",
    slice: "head 1, before the causal mask and the softmax",
    stations: [{ n: "scores", kind: "grid", hue: HUE.score }],
  },
  {
    id: "softmax",
    name: "Softmax",
    slice: "head 1 — the mask is why this is a triangle",
    stations: [{ n: "attn", kind: "grid", hue: RAMP[1], ramped: true }],
  },
  {
    id: "headout",
    name: "Σ pᵢ · vᵢ — the head's output",
    slice: "head 1 of 6, summed over every key, not only the eight drawn",
    stations: [{ n: "headout", kind: "card", unit: "dim", hue: HUE.v, single: true }],
  },
  {
    id: "concat",
    name: "6 heads concatenated",
    slice: "the newest position, all 384 — 6 × 64 becomes 384 here",
    stations: [{ n: "concat", kind: "tile", buf: "concat", unit: "ch", hue: HUE.out, single: true }],
  },
  {
    id: "proj",
    name: "Output projection",
    slice: "the newest position, all 384 channels",
    stations: [{ n: "proj", kind: "tile", buf: "proj", unit: "ch", hue: HUE.out, single: true }],
  },
  {
    id: "resid1",
    name: "Residual add",
    slice: "the newest position, all 384 — the embedding with attention added to it",
    stations: [{ n: "resid", kind: "tile", buf: "resid", unit: "ch", hue: HUE.x, single: true }],
  },
  {
    id: "ln2",
    name: "LayerNorm 2",
    slice: "the newest position, all 384 channels",
    stations: [{ n: "ln2", kind: "tile", buf: "ln2", unit: "ch", hue: HUE.norm, single: true }],
  },
  {
    id: "mlp",
    name: "MLP 384 → 1536, GELU",
    slice: "the newest position, all 1536 channels, folded 48 × 32",
    stations: [
      { n: "mlp", kind: "tile", buf: "mlp", unit: "ch", hue: HUE.mlp, cols: 48, rows: 32, single: true },
    ],
  },
  {
    id: "blockout",
    name: "MLP 1536 → 384, added back",
    slice: "the newest position, all 384 — block 1 is complete",
    stations: [{ n: "blockout", kind: "tile", buf: "blockout", unit: "ch", hue: HUE.x, single: true }],
  },
  {
    id: "folded",
    name: "Blocks 2 – 6",
    slice: "newest query row, all 6 heads, one slab per block",
    stations: [{ n: "folded", kind: "folded", hue: RAMP[1], ramped: true }],
  },
  {
    id: "out",
    name: "Final LayerNorm → next character",
    slice: "top 8 of 65 — the full ranking is in the panel below",
    stations: [
      { n: "lnf", kind: "tile", buf: "lnf", unit: "ch", hue: HUE.norm, single: true, label: "Final LayerNorm" },
      { n: "top", kind: "bars", hue: HUE.out, label: "Next character" },
    ],
  },
];

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
 * Build the pipeline in `canvas`, writing anchored labels into `overlay`, the
 * hovered-cell readout into `readout`, and driving `onStage(index, stage)` as
 * the walk moves so the page can swap its caption.
 *
 * Throws if a WebGL context cannot be created — the caller's cue to show the
 * static panel rather than leave an empty rectangle.
 */
export function createPipeline({ canvas, overlay, readout, onStage }) {
  const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new Scene();
  // A long lens: at 40° the far end of the pipeline is half the size of the
  // near end, and cells that are the same number have to look the same.
  const camera = new PerspectiveCamera(32, 16 / 9, 0.5, 800);
  const rig = new Group();
  scene.add(rig);

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
  let detailOff = false; // long prompt: attention only (the prefill guard)

  // ── store ───────────────────────────────────────────────────────────────
  // Each step hands back only its *new* positions, which is exactly what the
  // model computed. Accumulating them here is what lets a hover re-read a
  // value from ten tokens ago without re-running anything.

  let store = null;
  let live = 0; // positions filled
  let tokens = []; // display string per position
  let lastTop = [];

  function allocate() {
    const { nCtx, nEmbd, nHead, nLayer } = cfg;
    const hd = Math.round(nEmbd / nHead);
    const wide = () => new Float32Array(nCtx * nEmbd);
    store = {
      hd,
      nEmbd,
      nMlp: 4 * nEmbd,
      // head HEAD of block BLOCK
      q: new Float32Array(nCtx * hd),
      k: new Float32Array(nCtx * hd),
      v: new Float32Array(nCtx * hd),
      // [query][key] for this head alone — the whole triangle, so a decode
      // step only ever writes the one row it produced.
      attn: new Float32Array(nCtx * nCtx),
      rawScores: new Float32Array(nCtx * nCtx),
      // the residual stream through block BLOCK, full width
      embed: wide(),
      ln1: wide(),
      concat: wide(),
      proj: wide(),
      resid: wide(),
      ln2: wide(),
      blockout: wide(),
      lnf: wide(),
      mlp: new Float32Array(nCtx * 4 * nEmbd),
      // stage 14: the newest query row of every remaining block, per head
      folded: new Float32Array(nLayer * nHead * nCtx),
    };
    outVec = new Float32Array(hd);
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
  const order = []; // station names, in pipeline order — ribbons follow this
  const disposables = [];
  const cellGeometry = new BoxGeometry(1, 1, 1);
  const cellMaterial = new MeshStandardMaterial({
    roughness: 0.5,
    metalness: 0.04,
    emissive: new Color(0xffffff),
    emissiveIntensity: 0.14,
  });
  // Stations outside the current stage stay drawn, faintly, so the walk never
  // loses the shape of the whole pipeline. A second material because opacity
  // is a material property and every station shares one mesh.
  const dimMaterial = new MeshStandardMaterial({
    roughness: 0.5,
    metalness: 0.04,
    transparent: true,
    opacity: 0.07,
    depthWrite: false,
  });
  disposables.push(cellGeometry, cellMaterial, dimMaterial);

  const tmpMatrix = new Matrix4();
  const tmpColor = new Color();

  /** Geometry for one station spec, before it knows where it sits. */
  function geometryFor(spec) {
    const { nHead, nLayer, nEmbd } = cfg;
    const hd = store ? store.hd : 64;
    switch (spec.kind) {
      case "labels":
        return { cols: WIN, rows: 1, colPitch: GRID_PITCH, rowPitch: GRID_PITCH,
                 cellW: GRID_W, cellH: GRID_W };
      case "card":
        return { cols: spec.single ? 1 : WIN, rows: hd, colPitch: COL_PITCH,
                 rowPitch: ROW_PITCH, cellW: COL_W, cellH: ROW_H };
      case "grid":
        return { cols: WIN, rows: WIN, colPitch: GRID_PITCH, rowPitch: GRID_PITCH,
                 cellW: GRID_W, cellH: GRID_W };
      case "bars":
        return { cols: 8, rows: 1, colPitch: GRID_PITCH, rowPitch: GRID_PITCH,
                 cellW: GRID_W, cellH: GRID_W };
      case "folded":
        return { cols: Math.max(1, nLayer - 1) * WIN, rows: nHead,
                 colPitch: GRID_PITCH * 0.6, rowPitch: GRID_PITCH * 0.6,
                 cellW: GRID_W * 0.55, cellH: GRID_W * 0.55, group: WIN };
      default: {
        // A tile folds one position's whole vector into a rectangle. The
        // defaults are 24 × 16 = 384; the MLP overrides them for 1536.
        const tc = spec.cols || 24;
        const tr = spec.rows || 16;
        const width = spec.buf === "mlp" ? 4 * nEmbd : nEmbd;
        return { cols: (spec.single ? 1 : WIN) * tc, rows: tr,
                 colPitch: TILE_PITCH, rowPitch: TILE_PITCH,
                 cellW: TILE_W, cellH: TILE_W, group: tc, tc, tr, width };
      }
    }
  }

  /**
   * A grid of `cols × rows` cells centred on `x`. Instance `c * rows + r` is
   * column `c`, row `r`, counting from the top — the same order every paint
   * function and the hover readout use.
   */
  function station(spec, g, x) {
    const mesh = new InstancedMesh(cellGeometry, cellMaterial, g.cols * g.rows);
    mesh.position.set(x, 0, 0);
    mesh.frustumCulled = false;
    mesh.userData.name = spec.n;
    rig.add(mesh);
    stations[spec.n] = { ...g, spec, mesh, x, active: true };
    order.push(spec.n);
    return stations[spec.n];
  }

  /** Half-width of a station, including the gaps between its groups. */
  function halfWidth(g) {
    const groups = g.group ? g.cols / g.group : 0;
    return (g.cols * g.colPitch + Math.max(0, groups - 1) * GROUP_GAP) / 2;
  }

  /**
   * Place every station along +X by accumulating widths. Hand-tuned positions
   * were workable for six stations; for seventeen they would not survive one
   * change to a tile shape.
   */
  function buildStations() {
    let x = 0;
    for (const stage of STAGES) {
      stage.first = null;
      for (const spec of stage.stations) {
        const g = geometryFor(spec);
        const w = halfWidth(g);
        x += w;
        const st = station(spec, g, x);
        stage.first = stage.first ?? st;
        stage.last = st;
        x += w + STATION_GAP;
      }
    }
  }

  /** Column `c` of `st`, in world X relative to the station. */
  function colX(st, c) {
    const gap = st.group ? Math.floor(c / st.group) * GROUP_GAP : 0;
    const groups = st.group ? st.cols / st.group : 0;
    const total = st.cols * st.colPitch + Math.max(0, groups - 1) * GROUP_GAP;
    return c * st.colPitch + gap - total / 2 + st.colPitch / 2;
  }
  /** Row `r` of `st`, in world Y. Row 0 is the top. */
  const rowY = (st, r) => (-(r - (st.rows - 1) / 2)) * st.rowPitch;

  const ZERO = new Matrix4().makeScale(0, 0, 0);
  const BASE = [0x14, 0x16, 0x1e];
  const cellRgba = new Uint8Array(4);

  /**
   * Paint one station. `read(c, r)` returns the value for that cell, or `NaN`
   * for a cell that does not exist — masked attention, or a position the run
   * has not reached. Those are not drawn at all, so the causal triangle is a
   * shape rather than a convention to be explained.
   *
   * Magnitude becomes depth toward the viewer and brightness against the
   * station's hue; sign becomes a dimmer, cooler cell rather than a second
   * hue, because these are activations and what matters is where the energy
   * is.
   */
  function paint(name, hue, scale, read, opts = {}) {
    const st = stations[name];
    if (!st) return;
    const { maxDepth = 1.4, ramped = false, ghost = false, lit = () => false } = opts;
    const inv = scale > 0 ? 1 / scale : 0;
    for (let c = 0; c < st.cols; c++) {
      const x = colX(st, c);
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
        tmpMatrix.setPosition(x, rowY(st, r), depth / 2);
        st.mesh.setMatrixAt(i, tmpMatrix);
        const boost = lit(c, r) ? 1.45 : 1;
        if (ramped) {
          ramp(t, cellRgba, 0);
          tmpColor.setRGB(
            clamp01((cellRgba[0] * boost) / 255),
            clamp01((cellRgba[1] * boost) / 255),
            clamp01((cellRgba[2] * boost) / 255),
            SRGBColorSpace,
          );
        } else {
          const neg = value < 0 ? 0.45 : 1;
          tmpColor.setRGB(
            clamp01(((BASE[0] + (hue[0] * neg - BASE[0]) * t) * boost) / 255),
            clamp01(((BASE[1] + (hue[1] * neg - BASE[1]) * t) * boost) / 255),
            clamp01(((BASE[2] + (hue[2] * neg - BASE[2]) * t) * boost) / 255),
            SRGBColorSpace,
          );
        }
        st.mesh.setColorAt(i, tmpColor);
      }
    }
    st.mesh.instanceMatrix.needsUpdate = true;
    if (st.mesh.instanceColor) st.mesh.instanceColor.needsUpdate = true;
  }

  // ── ribbons ─────────────────────────────────────────────────────────────
  // A hint that data moved between two stations. Faint on purpose: they sit in
  // the same frame as stations of cells that carry actual numbers.

  const ribbons = [];
  const flowTex = flowTexture();
  disposables.push(flowTex);

  function ribbon(from, to, width, lift = 0.9) {
    const curve = new CatmullRomCurve3([
      new Vector3(from, 0, 0),
      new Vector3((from + to) / 2, lift, 0),
      new Vector3(to, 0, 0),
    ]);
    const n = 20;
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
      opacity: 0.3,
    });
    const mesh = new Mesh(geometry, material);
    rig.add(mesh);
    disposables.push(geometry, material, map);
    ribbons.push({ map, mesh, from: null, to: null });
    return ribbons[ribbons.length - 1];
  }

  /** One ribbon between each consecutive pair of stations. */
  function buildRibbons() {
    for (let i = 0; i + 1 < order.length; i++) {
      const a = stations[order[i]];
      const b = stations[order[i + 1]];
      const r = ribbon(a.x + halfWidth(a) + 0.2, b.x - halfWidth(b) - 0.2, 1.6);
      r.from = order[i];
      r.to = order[i + 1];
    }
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

  let caps = []; // one name + slice per station
  let colLabels = []; // the character over each key column
  let scoreNums = []; // one number under each score cell
  let probNums = []; // the current query's softmax row, beside the triangle
  let posLine = null;

  function buildLabels() {
    for (const stage of STAGES) {
      let lo = Infinity;
      let hi = -Infinity;
      let tallest = 0;
      for (const spec of stage.stations) {
        const st = stations[spec.n];
        if (!st) continue;
        lo = Math.min(lo, st.x);
        hi = Math.max(hi, st.x);
        tallest = Math.max(tallest, (st.rows * st.rowPitch) / 2);
      }
      for (const spec of stage.stations) {
        const st = stations[spec.n];
        if (!st) continue;
        // A station names itself when the stage holds more than one.
        const a = anchor(
          spec.label || stage.name,
          new Vector3(st.x, (st.rows * st.rowPitch) / 2 + 0.55, 0),
          "is-title",
        );
        caps.push({ name: spec.n, title: a, slice: null });
      }
      // One slice note per stage, centred over it and clear of every title:
      // anchored under the first station it collided with the second's name.
      if (caps.length) {
        caps[caps.length - 1].slice = anchor(
          stage.slice,
          new Vector3((lo + hi) / 2, tallest + 1.15, 0),
          "is-slice",
        );
      }
    }
    const kSt = stations.k;
    const kTop = (kSt.rows * kSt.rowPitch) / 2 + 0.28;
    for (let c = 0; c < WIN; c++) {
      colLabels.push({
        col: c,
        el: anchor("", new Vector3(kSt.x + colX(kSt, c), kTop, 0), "is-token").el,
      });
    }
    const sc = stations.scores;
    for (let c = 0; c < WIN; c++) {
      scoreNums.push({
        col: c,
        el: anchor("", new Vector3(sc.x + colX(sc, c), -(sc.rows * sc.rowPitch) / 2 - 0.35, 0), "is-num").el,
      });
    }
    const at = stations.attn;
    for (let r = 0; r < WIN; r++) {
      const a = anchor("", new Vector3(at.x, rowY(at, r), 0), "is-num");
      probNums.push({ row: r, el: a.el, pos: a.pos });
    }
    posLine = anchor("position —", new Vector3(stations.q.x, -(stations.q.rows * ROW_PITCH) / 2 - 0.6, 0), "is-pos");
  }

  function clearLabels() {
    for (const a of anchors.splice(0)) a.el.remove();
    caps = [];
    colLabels = [];
    scoreNums = [];
    probNums = [];
    posLine = null;
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
      a.el.style.opacity = projected.z > 1 ? "0" : "1";
      a.el.style.transform = `translate(-50%,-50%) translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
    }
  }

  // ── the two derived quantities ──────────────────────────────────────────

  const winScores = new Float32Array(WIN);
  let outVec = new Float32Array(64);

  /**
   * The scores for the eight keys on screen. Read from the captured raw scores
   * when the model gave us them, and recomputed as dot(q, kᵢ)/√hd otherwise —
   * drawing the dot product next to its two operands is the point of the
   * stage, and the captured tensor is what proves the recomputation right.
   */
  function computeScores() {
    winScores.fill(NaN);
    if (!store || live < 1) return;
    const { hd, q: Q, k: K, rawScores } = store;
    const { nCtx } = cfg;
    const t = query();
    const inv = 1 / Math.sqrt(hd);
    for (let c = 0; c < shown; c++) {
      const p = origin + c;
      const captured = rawScores[t * nCtx + p];
      if (captured !== 0) {
        winScores[c] = captured;
        continue;
      }
      let dot = 0;
      for (let d = 0; d < hd; d++) dot += Q[t * hd + d] * K[p * hd + d];
      winScores[c] = dot * inv;
    }
  }

  /**
   * Σ pᵢ · vᵢ over *every* key the model attended to, not only the eight on
   * screen — that is what the head actually output, and windowing its inputs
   * must not quietly window its result.
   *
   * The model also hands back the concatenated heads, so this sum can be
   * checked rather than trusted: a drawn number that is not the model's number
   * is the one failure this section cannot have.
   */
  let driftWarned = false;

  function computeOut() {
    outVec.fill(0);
    if (!store || live < 1) return;
    const { hd, v: V, attn, concat, nEmbd } = store;
    const { nCtx } = cfg;
    const t = query();
    for (let p = 0; p <= t && p < nCtx; p++) {
      const w = attn[t * nCtx + p];
      if (w === 0) continue;
      for (let d = 0; d < hd; d++) outVec[d] += w * V[p * hd + d];
    }
    if (driftWarned) return;
    // Head HEAD occupies dims [HEAD*hd, (HEAD+1)*hd) of the concatenation.
    let drift = 0;
    for (let d = 0; d < hd; d++) {
      drift = Math.max(drift, Math.abs(outVec[d] - concat[t * nEmbd + HEAD * hd + d]));
    }
    if (drift > 1e-3 && concat[t * nEmbd + HEAD * hd] !== 0) {
      driftWarned = true;
      console.warn(
        `pipeline: Σ pᵢvᵢ drifts from the captured head output by ${drift.toExponential(2)}`,
      );
    }
  }

  /** Largest magnitude over the visible columns of a per-position tensor. */
  function windowScale(src, width, single) {
    let max = 0;
    const from = single ? query() : origin;
    const n = single ? 1 : shown;
    for (let c = 0; c < n; c++) {
      const base = (from + c) * width;
      for (let d = 0; d < width; d++) {
        const a = Math.abs(src[base + d]);
        if (a > max) max = a;
      }
    }
    return max;
  }

  // ── repaint ─────────────────────────────────────────────────────────────

  let highlight = -1; // a key column, lit across every station that uses it
  let stage = 0;

  /** The (position, index) a tile or card cell reads. */
  function cellSource(st, c, r) {
    const g = st;
    if (g.spec.kind === "tile") {
      const slot = g.spec.single ? 0 : Math.floor(c / g.tc);
      return { pos: g.spec.single ? query() : origin + slot, idx: (c % g.tc) * g.tr + r };
    }
    return { pos: g.spec.single ? query() : origin + c, idx: r };
  }

  function repaint() {
    if (!store) return;
    const { hd, nEmbd } = store;
    const { nCtx, nHead, nLayer } = cfg;
    shown = Math.min(live, WIN);
    origin = Math.max(0, live - WIN);
    const t = query();

    computeScores();
    computeOut();

    // Before the first run there is nothing to draw and everything to explain,
    // so every station shows its grid as empty cells: the shape of the
    // calculation is on screen, and the first token fills it in rather than
    // conjuring it out of a blank rectangle.
    const empty = live === 0;
    const litKey = (c) => highlight >= 0 && c === highlight;

    // Q, K and V share one scale so their cards are comparable to each other,
    // which is the only way "this key matched that query" is visible.
    const qkv = Math.max(
      windowScale(store.q, hd, true),
      windowScale(store.k, hd, false),
      windowScale(store.v, hd, false),
      1e-6,
    );

    for (const name of order) {
      const st = stations[name];
      const spec = st.spec;
      const cols = empty ? st.cols : spec.kind === "tile" ? st.cols : shown;
      switch (spec.kind) {
        case "labels":
          paint(name, spec.hue, 1, (c) => (c >= (empty ? WIN : shown) ? NaN : 0.55), {
            ghost: true,
            lit: litKey,
          });
          break;
        case "card": {
          const src = name === "headout" ? null : store[spec.buf];
          const scale = name === "headout"
            ? Math.max(1e-6, outVec.reduce((m, x) => Math.max(m, Math.abs(x)), 0))
            : qkv;
          paint(
            name,
            spec.hue,
            scale,
            (c, r) => {
              if (c >= cols) return NaN;
              if (empty) return 0;
              if (!src) return outVec[r];
              const { pos } = cellSource(st, c, r);
              return src[pos * hd + r];
            },
            { ghost: empty, lit: spec.single ? () => false : litKey },
          );
          break;
        }
        case "grid": {
          if (name === "scores") {
            let smax = 1e-6;
            for (let c = 0; c < shown; c++) {
              if (!Number.isNaN(winScores[c])) smax = Math.max(smax, Math.abs(winScores[c]));
            }
            // Queries across, keys down — the same axes as the triangle beside
            // it, so the two stations can be read as one calculation.
            paint(
              name,
              spec.hue,
              smax,
              (c, r) => {
                if (empty) return r > c ? NaN : 0;
                const q = origin + c;
                if (q >= live || r >= shown) return NaN;
                const raw = store.rawScores[q * nCtx + origin + r];
                return q === t ? winScores[r] : raw === 0 ? NaN : raw;
              },
              { maxDepth: 2.0, ghost: empty, lit: (c, r) => litKey(r) },
            );
          } else {
            // k > q is masked and simply not drawn, so the causal triangle is
            // a shape rather than a convention to be explained.
            paint(
              name,
              spec.hue,
              1,
              (c, r) => {
                if (r > c) return NaN;
                if (empty) return 0;
                const q = origin + c;
                return q >= live ? NaN : store.attn[q * nCtx + origin + r];
              },
              { maxDepth: 2.4, ramped: !empty, ghost: empty, lit: (c, r) => litKey(r) },
            );
          }
          break;
        }
        case "tile": {
          const src = store[spec.buf];
          const width = st.width;
          const scale = Math.max(1e-6, windowScale(src, width, spec.single));
          paint(
            name,
            spec.hue,
            scale,
            (c, r) => {
              const { pos, idx } = cellSource(st, c, r);
              if (empty) return 0;
              if (pos >= live || pos < 0 || idx >= width) return NaN;
              return src[pos * width + idx];
            },
            { maxDepth: 1.1, ghost: empty, lit: () => false },
          );
          break;
        }
        case "folded": {
          // One slab per remaining block: the newest query row, every head.
          paint(
            name,
            spec.hue,
            1,
            (c, r) => {
              if (empty) return 0;
              const b = Math.floor(c / WIN) + 1; // block 2..nLayer, 0-indexed
              const kk = origin + (c % WIN);
              if (b >= nLayer || kk >= live) return NaN;
              return store.folded[(b * nHead + r) * nCtx + kk];
            },
            { maxDepth: 1.6, ramped: !empty, ghost: empty, lit: (c) => litKey(c % WIN) },
          );
          break;
        }
        case "bars": {
          const top = lastTop;
          paint(
            name,
            spec.hue,
            1,
            (c) => (empty || c >= top.length ? (empty ? 0 : NaN) : top[c].p),
            { maxDepth: 3.0, ghost: empty },
          );
          break;
        }
      }
    }

    paintLabels();
    applyVisibility();
    if (still) render();
  }

  function paintLabels() {
    const t = query();
    const { nCtx } = cfg;
    posLine.el.textContent =
      live > 0 ? `position ${t}  ${JSON.stringify(glyph(tokens[t]))}` : "position —";
    for (const a of colLabels) {
      const p = origin + a.col;
      a.el.hidden = a.col >= shown;
      a.el.textContent = glyph(tokens[p]);
      a.el.title = `key at position ${p}`;
      a.el.classList.toggle("is-lit", a.col === highlight);
    }
    for (const a of scoreNums) {
      a.el.hidden = a.col >= shown || Number.isNaN(winScores[a.col]);
      if (!a.el.hidden) a.el.textContent = winScores[a.col].toFixed(2);
      a.el.classList.toggle("is-lit", a.col === highlight);
    }
    // The probabilities beside the matrix are the *current* query's row. They
    // will not sum to 1 once the run is past 8 positions, because the softmax
    // normalises over every key, not the 8 on screen.
    const at = stations.attn;
    const right = colX(at, Math.max(0, shown - 1)) + 1.5;
    for (const a of probNums) {
      const k = origin + a.row;
      a.el.hidden = live < 1 || a.row >= shown || k > t;
      if (!a.el.hidden) a.el.textContent = store.attn[t * nCtx + k].toFixed(3);
      a.el.classList.toggle("is-lit", a.row === highlight);
      a.pos.x = at.x + right;
    }
  }

  // ── ingest ──────────────────────────────────────────────────────────────

  /** Copy `qLen` rows of a `[qLen, width]` capture into a per-position store. */
  function spread(dst, src, width, base, qLen) {
    if (!src || !src.length) return;
    const { nCtx } = cfg;
    for (let q = 0; q < qLen; q++) {
      const pos = base + q;
      if (pos >= nCtx) break;
      dst.set(src.subarray(q * width, (q + 1) * width), pos * width);
    }
  }

  /**
   * One step's trace. The prefill lands `qLen` positions at once; every decode
   * step after it lands exactly one.
   */
  function ingest(trace) {
    if (!store) allocate();
    const { nCtx, nHead, nLayer } = cfg;
    const { hd, nEmbd, nMlp } = store;
    const base = trace.kvLen - trace.qLen; // first new position
    const qLen = trace.qLen;

    for (const a of trace.attn) {
      if (a.layer === BLOCK && HEAD < a.nHead) {
        for (let q = 0; q < a.qLen; q++) {
          const pos = base + q;
          if (pos >= nCtx) break;
          const from = (HEAD * a.qLen + q) * a.kvLen;
          for (let k = 0; k < a.kvLen && k < nCtx; k++) {
            store.attn[pos * nCtx + k] = a.probs[from + k];
          }
        }
      }
      // Stage 14 keeps only the newest query row of each block, per head —
      // one slab each, which is what the folded view draws.
      if (a.layer > BLOCK && a.layer < nLayer) {
        const last = (a.qLen - 1) * a.kvLen;
        for (let h = 0; h < a.nHead && h < nHead; h++) {
          for (let k = 0; k < a.kvLen && k < nCtx; k++) {
            store.folded[(a.layer * nHead + h) * nCtx + k] = a.probs[h * a.qLen * a.kvLen + last + k];
          }
        }
      }
    }

    spread(store.embed, trace.embedding, nEmbd, base, qLen);
    spread(store.lnf, trace.lnFOut, nEmbd, base, qLen);

    for (const d of trace.detail) {
      if (d.layer !== BLOCK) continue;
      for (const [name, src] of [["q", d.q], ["k", d.k], ["v", d.v]]) {
        if (!src || !src.length) continue;
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
      if (d.scores && d.scores.length) {
        const kvLen = d.scores.length / (Math.max(1, nHead) * qLen);
        for (let q = 0; q < qLen; q++) {
          const pos = base + q;
          if (pos >= nCtx) break;
          const from = (HEAD * qLen + q) * kvLen;
          for (let k = 0; k < kvLen && k < nCtx; k++) {
            store.rawScores[pos * nCtx + k] = d.scores[from + k];
          }
        }
      }
      spread(store.ln1, d.ln1Out, nEmbd, base, qLen);
      spread(store.concat, d.attnHeadOut, nEmbd, base, qLen);
      spread(store.proj, d.attnProjOut, nEmbd, base, qLen);
      spread(store.resid, d.residAttn, nEmbd, base, qLen);
      spread(store.ln2, d.ln2Out, nEmbd, base, qLen);
      spread(store.blockout, d.blockOut, nEmbd, base, qLen);
      spread(store.mlp, d.mlpHidden, nMlp, base, qLen);
    }

    live = Math.min(trace.kvLen, nCtx);
    lastTop = trace.top || [];
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
  let zoomed = false;
  let needsFit = true;

  const narrow = () => canvas.clientWidth < 640;

  const probe = new Vector3();
  let targetX = 0;
  let targetY = 0;
  // Where the walk is heading. The camera eases toward these rather than
  // jumping, so a step reads as travel through one object rather than as a cut
  // between fifteen pictures.
  let wantX = 0;
  let wantY = 0;
  let wantDist = 24;

  /** World-space corners of the stations the walk is currently looking at. */
  function corners(only) {
    const out = [];
    for (const [name, st] of Object.entries(stations)) {
      if (!st.mesh.visible) continue;
      if (only && !only.includes(name)) continue;
      const w = halfWidth(st);
      const h = (st.rows * st.rowPitch) / 2;
      // Margins for the projected HTML: the caption and slice note above, the
      // position line below, probabilities right of the triangle. Labels are
      // not geometry, so nothing else would count them.
      const side = st.cols === 1 || st.spec.single ? 1.5 : 0.3;
      const right = name === "attn" ? 2.6 : side;
      // Above a station sit two lines of projected HTML — its name and the
      // stage's slice rule — and neither is geometry, so nothing else would
      // count them and both would be framed off the top edge.
      for (const x of [st.x - w - side, st.x + w + right]) {
        for (const y of [-h - 1.0, h + 2.6]) {
          for (const z of [0, 2.2]) out.push([x, y, z]);
        }
      }
    }
    return out;
  }

  /**
   * Frame a set of stations: pan until they are centred, zoom until they just
   * fit. Measured, not derived — the camera is off-axis in both yaw and pitch,
   * so the screen extent of a world-space box is not its width over the
   * distance, and a closed form is longer than this loop and easier to get
   * wrong. Four passes converges comfortably.
   */
  function fitView(only, immediate) {
    const box = corners(only);
    if (!box.length) return;
    const keepX = targetX;
    const keepY = targetY;
    const keepD = dist;
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
      const k = dist * Math.tan(((camera.fov * Math.PI) / 180) / 2);
      targetX += ((x0 + x1) / 2) * k * camera.aspect * Math.cos(yaw);
      targetY += ((y0 + y1) / 2) * k;
      dist = Math.max(4, Math.min(240, dist * Math.max((x1 - x0) / 1.84, (y1 - y0) / 1.84)));
    }
    wantX = targetX;
    wantY = targetY;
    wantDist = dist;
    if (!immediate) {
      // Solved for, then handed to the easing: restore the camera so the move
      // is seen rather than jumped.
      targetX = keepX;
      targetY = keepY;
      dist = keepD;
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
    dist = Math.max(4, Math.min(240, dist + e.deltaY * 0.02));
    wantDist = dist;
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
    const st = stations[name];
    const { hd, nEmbd } = store;
    const { nCtx } = cfg;
    const at = (i) => `${i} ${JSON.stringify(glyph(tokens[i]))}`;
    if (name === "scores") {
      const q = origin + c;
      const k = origin + r;
      const val = q === query() ? winScores[r] : store.rawScores[q * nCtx + k];
      return `q[${at(q)}] · k[${at(k)}] / √${hd} = ${val.toFixed(4)}`;
    }
    if (name === "attn") {
      const q = origin + c;
      const k = origin + r;
      return (
        `p(q=${at(q)} ← k=${at(k)}) = ${store.attn[q * nCtx + k].toFixed(4)}` +
        `  ·  softmax over all ${q + 1} keys`
      );
    }
    if (name === "folded") {
      const b = Math.floor(c / WIN) + 1;
      const k = origin + (c % WIN);
      return `block ${b + 1}, head ${r + 1}: p(newest ← k=${at(k)}) = ${store.folded[(b * cfg.nHead + r) * nCtx + k].toFixed(4)}`;
    }
    if (name === "tokens") {
      const p = origin + c;
      return `position ${p} = ${JSON.stringify(glyph(tokens[p]))}`;
    }
    if (name === "top") {
      const t = lastTop[c];
      return t ? `p(next = ${JSON.stringify(glyph(t.token))}) = ${t.p.toFixed(4)}` : "";
    }
    if (name === "headout") {
      return `Σ pᵢ vᵢ [${at(query())}][dim ${r}] = ${outVec[r].toFixed(4)}  ·  over ${query() + 1} keys`;
    }
    const { pos, idx } = cellSource(st, c, r);
    const width = st.spec.kind === "tile" ? st.width : hd;
    const src = store[st.spec.buf];
    if (!src || pos < 0 || pos >= live) return "";
    return `${st.spec.n}[${at(pos)}][${st.spec.unit} ${idx}] = ${src[pos * width + idx].toFixed(4)}`;
  }

  function hover() {
    if (!store || live < 1) return;
    raycaster.setFromCamera(pointer, camera);
    // Only the stations the walk is actually on: raycasting every instance of
    // every station is tens of thousands of boxes per pointer move.
    const meshes = order
      .map((n) => stations[n])
      .filter((s) => s.mesh.visible && s.active)
      .map((s) => s.mesh);
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
    // Hovering anything indexed by a key lights that key everywhere.
    const keyed = { k: c, v: c, tokens: c, attn: r, scores: r, folded: c % WIN };
    setHighlight(keyed[name] ?? -1);
  }

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // ── visibility and the walk ─────────────────────────────────────────────

  let shape = "";

  /** Station names belonging to stage `i`; `-1` is the overview: all of them. */
  function stageStations(i) {
    if (i < 0) return order.slice();
    return STAGES[i].stations.map((s) => s.n);
  }

  /**
   * With a long prompt the detail readback never happened (the prefill guard),
   * so only the attention stages have data. Their stations hide rather than
   * drawing a grid of zeros, which would read as "the model computed nothing".
   */
  const ATTN_ONLY = new Set(["tokens", "attn", "folded", "top"]);

  function applyVisibility() {
    const names = stageStations(stage);
    const near = new Set();
    for (let i = Math.max(0, stage - 2); i <= Math.min(STAGES.length - 1, stage + 2); i++) {
      for (const n of stageStations(i)) near.add(n);
    }
    for (const [name, st] of Object.entries(stations)) {
      const has = !detailOff || ATTN_ONLY.has(name);
      st.active = has && names.includes(name);
      // Two stages either side stay drawn so the walk keeps its bearings;
      // seventeen stations at once is noise in the frame.
      st.mesh.visible = has && (stage < 0 || near.has(name));
      st.mesh.material = st.active ? cellMaterial : dimMaterial;
    }
    for (const cap of caps) {
      const st = stations[cap.name];
      if (stage < 0) {
        // Fifteen captions at overview distance overlap into a single band.
        // The shape of the pipeline is the point of this shot; the names
        // arrive as the walk reaches them.
        cap.title.el.hidden = true;
        if (cap.slice) cap.slice.el.hidden = true;
        continue;
      }
      cap.title.el.hidden = !st.active;
      if (cap.slice) cap.slice.el.hidden = !st.active || narrow();
    }
    // Every projected label lands on top of every other one at overview
    // distance, so the overview carries the shape and none of the text.
    const on = (n) => stage >= 0 && stations[n].active;
    for (const a of colLabels) a.el.hidden = a.el.hidden || !(on("k") || on("tokens"));
    for (const a of scoreNums) a.el.hidden = a.el.hidden || !on("scores") || narrow();
    for (const a of probNums) a.el.hidden = a.el.hidden || !on("attn") || narrow();
    posLine.el.hidden = !(on("q") || on("headout"));
    for (const m of ribbons) {
      m.mesh.visible =
        !detailOff && (stage < 0 || (stations[m.from].active && stations[m.to].active));
    }
    const next = `${narrow()}${detailOff}${stage}`;
    if (next !== shape) {
      shape = next;
      needsFit = true;
    }
  }

  /** Move the walk to stage `i` (`-1` = the overview) and reframe. */
  function goTo(i, immediate) {
    stage = Math.max(-1, Math.min(STAGES.length - 1, i));
    zoomed = false;
    applyVisibility();
    fitView(stage < 0 ? null : stageStations(stage), immediate);
    onStage?.(stage, stage < 0 ? null : STAGES[stage]);
    if (still) render();
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

  /** Ease the camera toward the stage the walk is on. */
  function advance(dt) {
    if (dragging || zoomed) return;
    const k = still ? 1 : 1 - Math.exp(-dt * 6);
    const dx = wantX - targetX;
    const dy = wantY - targetY;
    const dd = wantDist - dist;
    if (Math.abs(dx) < 1e-3 && Math.abs(dy) < 1e-3 && Math.abs(dd) < 1e-3) return;
    targetX += dx * k;
    targetY += dy * k;
    dist += dd * k;
    applyCamera();
  }

  function render() {
    resize();
    // Reframing is a response to the stage changing shape or contents, not
    // something to redo 60 times a second — and never while a visitor is in
    // the middle of moving the camera themselves.
    if (needsFit && !dragging && !zoomed) {
      fitView(stage < 0 ? null : stageStations(stage), still);
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
    advance(dt);
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

  function rebuild() {
    for (const st of Object.values(stations)) {
      rig.remove(st.mesh);
      st.mesh.dispose();
    }
    for (const name of Object.keys(stations)) delete stations[name];
    for (const r of ribbons.splice(0)) rig.remove(r.mesh);
    order.length = 0;
    clearLabels();
    allocate();
    buildStations();
    buildRibbons();
    buildLabels();
  }

  allocate();
  buildStations();
  buildRibbons();
  buildLabels();
  repaint();
  goTo(-1, true);
  start();

  // ── public surface ──────────────────────────────────────────────────────

  return {
    /** Every stage's name and slicing rule, for the page's caption strip. */
    stages: STAGES.map((s) => ({ id: s.id, name: s.name, slice: s.slice })),

    /** Resize the store to the model that is actually loaded. */
    setConfig(next) {
      const merged = { ...cfg, ...next };
      const changed =
        merged.nCtx !== cfg.nCtx ||
        merged.nHead !== cfg.nHead ||
        merged.nEmbd !== cfg.nEmbd ||
        merged.nLayer !== cfg.nLayer;
      cfg = merged;
      if (changed) rebuild();
      repaint();
      if (changed) goTo(stage, true);
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

    stageCount: STAGES.length,
    stage: () => stage,
    goToStage: (i) => goTo(i, false),
    nextStage: () => goTo(stage + 1, false),
    prevStage: () => goTo(stage < 0 ? 0 : stage - 1, false),
    overview: () => goTo(-1, false),

    resetView() {
      yaw = YAW0;
      pitch = PITCH0;
      zoomed = false;
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
      driftWarned = false;
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
