// Twelve translucent slabs — GPT-2 124M's twelve transformer blocks.
// Hover highlights a block; click expands it into its sub-layers with the
// real shapes from Gpt2Config::gpt2().
//
// Only what is used is imported, three.js is vendored (no CDN), the loop stops
// when the canvas scrolls off-screen, and reduced-motion renders one static
// frame. If WebGL is unavailable the static architecture diagram takes the
// canvas's place — the page must be fully informative without it.

import {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  Group,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Raycaster,
  Scene,
  Vector2,
  WebGLRenderer,
} from "three";

const N_LAYER = 12;
const N_HEAD = 12;
const N_EMBD = 768;
const N_CTX = 1024;
const HEAD_DIM = N_EMBD / N_HEAD;

const SUB_LAYERS = [
  ["ln_1", `LayerNorm  [${N_EMBD}]`],
  ["attn", `causal self-attention  ${N_HEAD} heads × ${HEAD_DIM}`],
  ["ln_2", `LayerNorm  [${N_EMBD}]`],
  ["mlp", `MLP  ${N_EMBD} → ${4 * N_EMBD} → ${N_EMBD}, GELU`],
];

const canvas = document.getElementById("scene");
const holder = document.getElementById("scene-holder");
const label = document.getElementById("scene-label");
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

function fallback() {
  // Move the static diagram into the canvas's place rather than leaving a
  // blank rectangle.
  const stat = document.getElementById("stack-static");
  if (holder && stat) {
    holder.replaceWith(stat);
  }
}

let renderer;
try {
  if (!canvas) throw new Error("no canvas");
  renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
} catch {
  fallback();
}

if (renderer) {
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

  const BASE = new Color(0x9aa0ad);
  const HOVER = new Color(0xff8a4c);
  const OPEN = new Color(0xea6a24);

  const SLAB_H = 0.26;
  const GAP = 0.14;
  const geometry = new BoxGeometry(3.4, SLAB_H, 2.2);
  const blocks = [];

  for (let i = 0; i < N_LAYER; i++) {
    const material = new MeshStandardMaterial({
      color: BASE,
      transparent: true,
      opacity: 0.62,
      roughness: 0.35,
      metalness: 0.1,
    });
    const mesh = new Mesh(geometry, material);
    mesh.position.y = (i - (N_LAYER - 1) / 2) * (SLAB_H + GAP);
    mesh.userData.index = i;
    stack.add(mesh);
    blocks.push(mesh);
  }

  // Sub-layer slabs, hidden until a block is opened.
  const subGeometry = new BoxGeometry(3.0, 0.05, 1.9);
  const subs = SUB_LAYERS.map(() => {
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

  let hovered = -1;
  let opened = -1;

  function paint() {
    for (const b of blocks) {
      const i = b.userData.index;
      const on = i === hovered || i === opened;
      b.material.color.copy(i === opened ? OPEN : on ? HOVER : BASE);
      b.material.opacity = opened >= 0 && i !== opened ? 0.22 : on ? 0.95 : 0.62;
    }
    subs.forEach((s, k) => {
      s.visible = opened >= 0;
      if (opened >= 0) {
        const base = blocks[opened].position.y;
        s.position.set(0, base + (k - 1.5) * 0.085, 0);
      }
    });
    if (label) {
      if (opened >= 0) {
        label.textContent = `block ${opened} — ${SUB_LAYERS.map((s) => s[1]).join("  ·  ")}`;
      } else if (hovered >= 0) {
        label.textContent = `block ${hovered} of ${N_LAYER}  ·  click to expand  ·  n_ctx ${N_CTX}`;
      } else {
        label.textContent = `GPT-2 124M — ${N_LAYER} blocks, ${N_HEAD} heads, n_embd ${N_EMBD}`;
      }
    }
  }
  paint();

  // ── pointer: hand-rolled drag rotation, ~15 lines, no OrbitControls addon
  const pointer = new Vector2();
  const raycaster = new Raycaster();
  let dragging = false;
  let last = 0;
  let spin = 0;
  let manual = 0;

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    last = e.clientX;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
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
  });
  canvas.addEventListener("pointerleave", () => {
    dragging = false;
    if (hovered !== -1) {
      hovered = -1;
      paint();
      if (still) render();
    }
  });
  canvas.addEventListener("click", () => {
    opened = hovered >= 0 && hovered !== opened ? hovered : -1;
    paint();
    if (still) render();
  });

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

  let still = reduced.matches;
  let frame = 0;

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

  reduced.addEventListener("change", () => {
    still = reduced.matches;
    stop();
    start();
  });

  window.addEventListener("resize", () => {
    if (still) render();
  });

  // Only run while visible: a permanently spinning scene drains laptop
  // batteries, and initializing on first scroll keeps it off the critical path.
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

  // The static diagram is a duplicate of what the scene shows; keep it for
  // no-WebGL visitors, but collapse it to a summary when the scene works.
  const stat = document.getElementById("stack-static");
  if (stat) stat.dataset.sceneOk = "1";
}
