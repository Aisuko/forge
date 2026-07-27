// A live view of the model the page is actually running: one translucent slab
// per transformer block, each carrying a heads × positions attention strip
// that updates once per generated token from probabilities read back off the
// GPU. Idle, it is just the stack; generating, it shows which earlier
// characters every head is looking at.
//
// three.js is imported here and nowhere else, so demo.js can `await import()`
// this module only when the section scrolls into view — 751 KB a visitor who
// never scrolls never pays for.
//
// Nothing here is required for the page to work: if WebGL is missing,
// createStack throws and the caller shows the text architecture instead.

import {
  AmbientLight,
  BoxGeometry,
  ClampToEdgeWrapping,
  Color,
  DataTexture,
  DirectionalLight,
  Group,
  LinearFilter,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  RGBAFormat,
  Raycaster,
  SRGBColorSpace,
  Scene,
  UnsignedByteType,
  Vector2,
  WebGLRenderer,
} from "three";

// The config of the shipped char model, used until the wasm module reports
// the real one. Never GPT-2 124M's — the page must describe what it runs.
export const DEFAULT_CONFIG = {
  name: "Shakespeare char GPT",
  nLayer: 6,
  nHead: 6,
  nEmbd: 384,
  nCtx: 256,
  // What this stack's first slab is called, counting blocks from 1 the way
  // the page does. The explainer draws block 1 in full, so the folded
  // remainder starts at 2 — labelling it "block 1" again would claim the
  // model has two of them.
  firstBlock: 2,
};

const BASE = new Color(0x9aa0ad);
const HOVER = new Color(0xff8a4c);
const OPEN = new Color(0xea6a24);

const SLAB_H = 0.26;
const GAP = 0.14;

// Attention ramp on the forge palette: cold ink → forge orange → hot ember.
// Exported so explainer.js colours its matrix identically — two ramps for the
// same quantity would read as two different quantities.
export const RAMP = [
  [0x1c, 0x1c, 0x24],
  [0xea, 0x6a, 0x24],
  [0xff, 0xe6, 0xd5],
];

/** Write RAMP(t) as RGBA bytes into `out` at offset `o`. `t` is clamped 0..1. */
export function ramp(t, out, o) {
  const i = t < 0.5 ? 0 : 1;
  const f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  for (let c = 0; c < 3; c++) {
    out[o + c] = RAMP[i][c] + (RAMP[i + 1][c] - RAMP[i][c]) * f;
  }
  out[o + 3] = 255;
}

function subLayers({ nHead, nEmbd }) {
  const hd = Math.round(nEmbd / nHead);
  return [
    `LayerNorm [${nEmbd}]`,
    `causal self-attention ${nHead} heads × ${hd}`,
    `LayerNorm [${nEmbd}]`,
    `MLP ${nEmbd} → ${4 * nEmbd} → ${nEmbd}, GELU`,
  ];
}

/**
 * Build the stack in `canvas`, labelling it into `label`.
 *
 * Throws if a WebGL context cannot be created — the caller's cue to show the
 * text architecture instead of leaving an empty rectangle.
 *
 * @returns {{setConfig: Function, pushAttention: Function, reset: Function,
 *            dispose: Function}}
 */
export function createStack({ canvas, label, config }) {
  const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
  // Uncapped DPR on a 4K screen tanks the framerate for no visible gain.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new Scene();
  const camera = new PerspectiveCamera(38, 16 / 9, 0.1, 100);
  camera.position.set(5.2, 3.4, 7.4);
  camera.lookAt(0, 0, 0);

  scene.add(new AmbientLight(0xffffff, 1.5));
  const key = new DirectionalLight(0xffffff, 2.2);
  key.position.set(4, 8, 6);
  scene.add(key);
  const rim = new DirectionalLight(0xff8a4c, 1.1);
  rim.position.set(-6, -2, -4);
  scene.add(rim);

  const stack = new Group();
  scene.add(stack);

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  let still = reduced.matches;
  let cfg = { ...DEFAULT_CONFIG, ...config };
  let blocks = [];
  let strips = [];
  let subs = [];
  let geometry;
  let subGeometry;
  let hovered = -1;
  let opened = -1;
  // Positions in context on the most recent token; 0 while idle.
  let live = 0;
  let spin = 0;
  let manual = 0;
  let frame = 0;

  // ── build / teardown ────────────────────────────────────────────────────

  function build() {
    geometry = new BoxGeometry(3.4, SLAB_H, 2.2);
    subGeometry = new BoxGeometry(3.0, 0.05, 1.9);

    for (let i = 0; i < cfg.nLayer; i++) {
      // One texture per block, allocated at full context width so a token
      // never rebuilds geometry or reallocates: the strip is written in
      // place and revealed by narrowing the UV repeat.
      const data = new Uint8Array(cfg.nCtx * cfg.nHead * 4);
      const texture = new DataTexture(
        data,
        cfg.nCtx,
        cfg.nHead,
        RGBAFormat,
        UnsignedByteType,
      );
      texture.colorSpace = SRGBColorSpace;
      texture.minFilter = LinearFilter;
      texture.magFilter = LinearFilter;
      texture.wrapS = ClampToEdgeWrapping;
      texture.wrapT = ClampToEdgeWrapping;

      const material = new MeshStandardMaterial({
        color: BASE,
        transparent: true,
        opacity: 0.62,
        roughness: 0.35,
        metalness: 0.1,
        emissive: new Color(0xffffff),
        emissiveIntensity: 0,
      });
      const mesh = new Mesh(geometry, material);
      mesh.position.y = (i - (cfg.nLayer - 1) / 2) * (SLAB_H + GAP);
      mesh.userData.index = i;
      stack.add(mesh);
      blocks.push(mesh);
      strips.push({ data, texture });
    }

    // Sub-layer slabs, hidden until a block is opened.
    subs = subLayers(cfg).map(() => {
      const mesh = new Mesh(
        subGeometry,
        new MeshStandardMaterial({
          color: OPEN,
          transparent: true,
          opacity: 0.9,
          roughness: 0.3,
        }),
      );
      mesh.visible = false;
      stack.add(mesh);
      return mesh;
    });
    paint();
  }

  function teardown() {
    for (const m of [...blocks, ...subs]) {
      stack.remove(m);
      m.material.dispose();
    }
    for (const s of strips) s.texture.dispose();
    geometry?.dispose();
    subGeometry?.dispose();
    blocks = [];
    strips = [];
    subs = [];
  }

  // ── painting ────────────────────────────────────────────────────────────

  function paint() {
    for (const b of blocks) {
      const i = b.userData.index;
      const on = i === hovered || i === opened;
      const dimmed = opened >= 0 && i !== opened;
      if (live) {
        // The strip carries the colour, so the slab tint must not fight it.
        b.material.color.setHex(on ? 0xffffff : 0xd8d8dd);
        b.material.opacity = dimmed ? 0.3 : 0.95;
      } else {
        b.material.color.copy(i === opened ? OPEN : on ? HOVER : BASE);
        b.material.opacity = dimmed ? 0.22 : on ? 0.95 : 0.62;
      }
    }
    subs.forEach((s, k) => {
      s.visible = opened >= 0;
      if (opened >= 0) {
        s.position.set(0, blocks[opened].position.y + (k - 1.5) * 0.085, 0);
      }
    });
    if (!label) return;
    const named = (i) => cfg.firstBlock + i;
    const range = `blocks ${named(0)}–${named(cfg.nLayer - 1)}`;
    if (opened >= 0) {
      label.textContent = `block ${named(opened)} — ${subLayers(cfg).join("  ·  ")}`;
    } else if (hovered >= 0) {
      label.textContent = live
        ? `block ${named(hovered)} — ${cfg.nHead} heads × ${live} positions`
        : `block ${named(hovered)}  ·  click to expand`;
    } else if (live) {
      label.textContent =
        `live attention — ${range} × ${cfg.nHead} heads over ` +
        `${live} of ${cfg.nCtx} positions`;
    } else {
      label.textContent =
        `${cfg.name} — ${range}, ${cfg.nHead} heads, n_embd ${cfg.nEmbd}`;
    }
  }

  // ── pointer: hand-rolled drag rotation, no OrbitControls addon ──────────

  const pointer = new Vector2();
  const raycaster = new Raycaster();
  let dragging = false;
  let last = 0;

  const onDown = (e) => {
    dragging = true;
    last = e.clientX;
    canvas.setPointerCapture(e.pointerId);
  };
  const onUp = (e) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  };
  const onMove = (e) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    if (dragging) {
      manual += (e.clientX - last) * 0.006;
      last = e.clientX;
    }
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(blocks, false)[0];
    const next = hit ? hit.object.userData.index : -1;
    if (next !== hovered) {
      hovered = next;
      canvas.style.cursor = next >= 0 ? "pointer" : "default";
      paint();
      if (still) render();
    }
  };
  const onLeave = () => {
    dragging = false;
    if (hovered !== -1) {
      hovered = -1;
      paint();
      if (still) render();
    }
  };
  const onClick = () => {
    opened = hovered >= 0 && hovered !== opened ? hovered : -1;
    paint();
    if (still) render();
  };

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("click", onClick);

  // ── loop ────────────────────────────────────────────────────────────────

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function render() {
    resize();
    renderer.render(scene, camera);
  }

  function loop() {
    frame = requestAnimationFrame(loop);
    spin += 0.0022;
    stack.rotation.y = spin + manual;
    render();
  }

  function start() {
    if (still) {
      stack.rotation.y = 0.5 + manual;
      render();
      return;
    }
    if (!frame) loop();
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

  // Only run while visible: a permanently spinning scene drains laptop
  // batteries, and starting on first scroll keeps it off the critical path.
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) start();
        else stop();
      }
    },
    { threshold: 0.05 },
  );
  io.observe(canvas);

  build();
  start();

  // ── public surface ──────────────────────────────────────────────────────

  return {
    /** Resize the stack to the model that is actually loaded. */
    setConfig(next) {
      const merged = { ...cfg, ...next };
      const rebuild =
        merged.nLayer !== cfg.nLayer ||
        merged.nHead !== cfg.nHead ||
        merged.nCtx !== cfg.nCtx;
      cfg = merged;
      if (rebuild) {
        teardown();
        hovered = -1;
        opened = -1;
        // New strips are empty, so the stack is idle again whatever it was.
        live = 0;
        build();
      } else {
        paint();
      }
      if (still) render();
    },

    /**
     * One block's attention for the token just produced: `weights` is
     * head-major `nHead × kvLen`, exactly what the model attended with.
     *
     * Each head's row is scaled by its own maximum before colouring —
     * otherwise a head spread over 200 positions is uniformly black next to
     * one focused on three. The gamma is display only; the numbers are the
     * model's.
     */
    pushAttention(layer, nHead, weights) {
      const strip = strips[layer];
      if (!strip || !nHead) return;
      // `stride` is the model's kv_len; `kv` is what fits the texture. They
      // differ only if the config went stale, and conflating them would shear
      // every row.
      const stride = Math.floor(weights.length / nHead);
      const kv = Math.min(stride, cfg.nCtx);
      const rows = Math.min(nHead, cfg.nHead);
      if (kv < 1) return;
      for (let h = 0; h < rows; h++) {
        let max = 0;
        for (let p = 0; p < kv; p++) {
          const w = weights[h * stride + p];
          if (w > max) max = w;
        }
        const inv = max > 0 ? 1 / max : 0;
        for (let p = 0; p < kv; p++) {
          ramp(
            Math.pow(weights[h * stride + p] * inv, 0.6),
            strip.data,
            (h * cfg.nCtx + p) * 4,
          );
        }
      }
      strip.texture.needsUpdate = true;
      // Show only the filled part of the texture, stretched across the slab.
      strip.texture.repeat.set(kv / cfg.nCtx, 1);

      const mat = blocks[layer].material;
      if (mat.map !== strip.texture) {
        mat.map = strip.texture;
        mat.emissiveMap = strip.texture;
        mat.emissiveIntensity = 0.55;
        mat.needsUpdate = true;
      }
      if (live !== kv) {
        live = kv;
        paint();
      }
      // Reduced motion means no render loop, but a strip update is data, not
      // decoration, so it still gets drawn.
      if (still) render();
    },

    /** Back to the idle stack — called at the start of every run. */
    reset() {
      live = 0;
      for (let i = 0; i < blocks.length; i++) {
        strips[i].data.fill(0);
        strips[i].texture.needsUpdate = true;
        const mat = blocks[i].material;
        mat.map = null;
        mat.emissiveMap = null;
        mat.emissiveIntensity = 0;
        mat.needsUpdate = true;
      }
      paint();
      if (still) render();
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
      canvas.removeEventListener("click", onClick);
      teardown();
      renderer.dispose();
    },
  };
}
