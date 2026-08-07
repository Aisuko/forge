const WINDOW = 120;

const VIEW_W = 600;
const VIEW_H = 160;
const TOP_PAD = 12;

const $ = (id) => document.getElementById(id);

function niceMax(v) {
  if (!(v > 0)) return 1;
  const mag = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 1.5, 2, 3, 4, 5, 7.5]) {
    if (v <= m * mag) return m * mag;
  }
  return 10 * mag;
}

const round = (v, d = 1) => v.toFixed(v >= 100 ? 0 : d);

export function createCost() {
  const line = $("eff-line");
  const area = $("eff-area");
  if (!line || !area) return null;

  const meanEl = $("eff-mean");
  const liveEl = $("eff-live");
  const yMaxEl = $("eff-ymax");
  const statsEl = $("eff-tps-stats");
  const tpsEl = $("eff-tps");

  let series = [];
  let yMax = 0;

  const scaleX = (i) => (i / (WINDOW - 1)) * VIEW_W;
  const scaleY = (v) => VIEW_H - Math.min(v / yMax, 1) * (VIEW_H - TOP_PAD);

  function clear() {
    line.setAttribute("d", "");
    area.setAttribute("d", "");
    if (meanEl) meanEl.style.display = "none";
    if (yMaxEl) yMaxEl.textContent = "";
    if (statsEl) statsEl.textContent = "—";
  }

  function draw(avg) {
    if (series.length < 2) return;
    let d = "";
    for (let i = 0; i < series.length; i++) {
      d += `${i ? "L" : "M"}${scaleX(i).toFixed(1)} ${scaleY(series[i]).toFixed(1)}`;
    }
    line.setAttribute("d", d);
    const last = scaleX(series.length - 1).toFixed(1);
    area.setAttribute("d", `${d}L${last} ${VIEW_H}L0 ${VIEW_H}Z`);

    if (meanEl && avg > 0) {
      const y = scaleY(avg).toFixed(1);
      meanEl.setAttribute("y1", y);
      meanEl.setAttribute("y2", y);
      meanEl.style.display = "";
    }
  }

  return {
    reset() {
      series = [];
      yMax = 0;
      clear();
      if (tpsEl) tpsEl.textContent = "—";
      if (liveEl) liveEl.hidden = false;
    },

    push(instant, average) {
      if (!Number.isFinite(instant) || instant <= 0) return;
      series.push(instant);
      if (series.length > WINDOW) series.shift();

      let lo = Infinity;
      let hi = 0;
      for (const v of series) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      yMax = niceMax(Math.max(hi, average));

      if (tpsEl) tpsEl.textContent = round(average);
      if (yMaxEl) yMaxEl.textContent = `${round(yMax, 0)} tok/s`;
      if (statsEl) {
        statsEl.textContent =
          `min ${round(lo)} · max ${round(hi)} · ` +
          `${series.length} token${series.length === 1 ? "" : "s"} shown`;
      }
      draw(average);
    },

    prefill(n, ms) {
      const el = $("eff-prefill");
      if (!el || !(n > 0) || !(ms > 0)) return;
      el.textContent = round(n / (ms / 1000), 0);
      const sub = $("eff-prefill-sub");
      if (sub) {
        sub.textContent = `${n} prompt token${n === 1 ? "" : "s"} in one pass, ${ms.toFixed(0)} ms`;
      }
    },

    done(average) {
      if (liveEl) liveEl.hidden = true;
      if (tpsEl && Number.isFinite(average)) tpsEl.textContent = round(average);
    },

    ready({ gpu, weightBytes, format, params }) {
      const gpuEl = $("eff-gpu");
      if (gpuEl && gpu) gpuEl.textContent = gpu;

      const fp32 = params.total * 4;
      const sizeEl = $("eff-size");
      if (sizeEl) sizeEl.textContent = (weightBytes / 1e6).toFixed(1);
      const diskBar = $("eff-size-bar");
      if (diskBar) {
        diskBar.style.width = `${Math.min(100, (weightBytes / fp32) * 100).toFixed(1)}%`;
      }
      const fp32Bar = $("eff-size-fp32-bar");
      if (fp32Bar) fp32Bar.style.width = "100%";
      const fp32El = $("eff-size-fp32");
      if (fp32El) fp32El.textContent = `${(fp32 / 1e6).toFixed(1)} MB`;
      const diskEl = $("eff-size-disk");
      if (diskEl) diskEl.textContent = `${(weightBytes / 1e6).toFixed(1)} MB`;
      const diskKey = $("eff-size-key");
      if (diskKey && format) diskKey.textContent = format;
      const sizeNote = $("eff-size-note");
      if (sizeNote) {
        sizeNote.textContent =
          `${(fp32 / weightBytes).toFixed(1)}× smaller than the same weights ` +
          `in fp32 — ${((weightBytes * 8) / params.total).toFixed(1)} bits ` +
          `per parameter on the wire, dequantised to f32 on the way in.`;
      }

      const paramsEl = $("eff-params");
      if (paramsEl) paramsEl.textContent = (params.total / 1e6).toFixed(2);
      for (const k of ["embed", "attn", "mlp"]) {
        const pct = (params[k] / params.total) * 100;
        const seg = $(`eff-p-${k}`);
        if (seg) seg.style.width = `${pct.toFixed(2)}%`;
        const val = $(`eff-p-${k}-v`);
        if (val) {
          val.textContent =
            `${(params[k] / 1e6).toFixed(2)} M · ${pct < 10 ? pct.toFixed(1) : pct.toFixed(0)}%`;
        }
      }
    },
  };
}
