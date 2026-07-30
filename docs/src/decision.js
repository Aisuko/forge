// The decision: what character the model is about to produce, and how sure it
// was about it.
//
// This is the panel the section was missing. Everything else here shows *where
// the model looked*; this shows *what it was deciding*, which is the question
// the looking is an answer to. It is also the one number a visitor needs no
// explanation for.
//
// None of it is new work. `generate_with_trace` has been sending a full ranked
// shortlist — `trace.top`, 24 entries of { id, token, p }, decoded on the Rust
// side because only the tokenizer can turn an id back into a character — since
// the 3D pipeline drew these bars, and nothing has rendered it since that
// renderer was deleted. The plumbing was left running with the taps closed.
//
// Plain DOM on purpose: no canvas, no measurement, nothing to fail. When the
// GPU is fine but the attention grid is not, this panel still works.

/** How a character is spelled in a label. Whitespace has to be visible. */
const glyph = (s) => (s ?? "").replace(/\n/g, "↵").replace(/ /g, "␣") || "·";

/** Bars to draw. The trace carries 24; that is a table, and eight is a shape
    you read at a glance. The rest are accounted for in one line underneath. */
const BARS = 8;

/** Characters of preceding text shown above the bars. Enough to see what is
    being continued, short enough to stay on one line at the panel's width. */
const TAIL = 28;

/**
 * The next-character panel.
 *
 * `bars` receives the ranked rows, `context` the tail of the text so far, and
 * `note` the line accounting for everything below the last bar.
 *
 * The callbacks arrive in this order, which is what makes the panel possible:
 * `generate_async_trace` computes a step, fires the trace for it, and only
 * *then* samples and emits the character. So `pushTop` always runs one callback
 * before the `chose` that settles it.
 */
export function createDecision({ bars, context, note } = {}) {
  let rows = []; // reused DOM, one per bar — rebuilt on nothing
  let shown = []; // the entries currently drawn, so chose() can find the winner
  let text = "";
  let vocab = 0;
  let tally = ""; // the remainder line, kept so end() can extend it

  function row() {
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

  function ensure(n) {
    while (rows.length < n) {
      const r = row();
      rows.push(r);
      bars?.append(r.el);
    }
    for (let i = 0; i < rows.length; i++) rows[i].el.hidden = i >= n;
  }

  /** The characters this model knows, so the remainder line can count them. */
  function setVocab(n) {
    vocab = Number(n) || 0;
  }

  function reset() {
    text = "";
    shown = [];
    tally = "";
    ensure(0);
    if (context) context.textContent = "";
    if (note) note.textContent = "";
  }

  /** The prompt, echoed once the prefill lands. Not a generated character. */
  function setContext(s) {
    text = s ?? "";
    drawContext();
  }

  function drawContext() {
    if (!context) return;
    const tail = text.slice(-TAIL);
    // The elision is only honest when something was actually elided.
    context.textContent =
      (text.length > TAIL ? "…" : "") + tail.replace(/\n/g, "↵");
  }

  /**
   * One step's shortlist. `top` is ranked, most likely first, and its
   * probabilities are a real softmax over the whole vocabulary — the model's
   * own belief, *before* temperature and top-k reshape the distribution it
   * actually samples from. The panel says so; it must not quietly imply these
   * are the sampling odds.
   */
  function pushTop(top) {
    if (!bars || !top) return;
    shown = Array.from(top).slice(0, BARS);
    ensure(shown.length);
    const peak = shown[0]?.p || 1;
    let sum = 0;
    shown.forEach((t, i) => {
      const r = rows[i];
      sum += t.p;
      r.g.textContent = glyph(t.token);
      // Relative to the leader, not to 1: a 0.03 second place is a visible bar
      // this way and a hairline the other, and the printed number is right
      // there either way.
      r.fill.style.width = `${Math.max(1, (t.p / peak) * 100).toFixed(1)}%`;
      // When the model is certain — and on this corpus it often is — the other
      // seven round to 0.000, and eight rows of "0.000" read as a panel that
      // failed rather than as a model that was sure.
      r.p.textContent = t.p < 0.0005 && t.p > 0 ? "<0.001" : t.p.toFixed(3);
      r.mark.textContent = "";
      r.el.classList.remove("is-chosen");
    });
    if (note) {
      const rest = Math.max(0, 1 - sum);
      const others = vocab ? vocab - shown.length : 0;
      tally = others
        ? `the other ${others} characters share ${(rest * 100).toFixed(1)}%`
        : `everything else shares ${(rest * 100).toFixed(1)}%`;
      note.textContent = tally;
    }
  }

  /**
   * The character that was actually sampled, which also extends the context.
   *
   * With top-k at or below the 24 entries the trace carries, the winner is
   * always one of them. Above that it may not be, and then nothing is marked —
   * marking the wrong bar would be worse than marking none.
   */
  function chose(s) {
    text += s ?? "";
    drawContext();
    const i = shown.findIndex((t) => t.token === s);
    if (i < 0 || !rows[i]) return;
    rows[i].el.classList.add("is-chosen");
    rows[i].mark.textContent = "chosen";
  }

  /**
   * Generation is over. The last shortlist to arrive is the one for a character
   * that was never sampled, so it carries no `chosen` marker — and that resting
   * state is what most visitors look at longest. Left unexplained it reads as
   * the marker being broken.
   */
  function end() {
    if (!note || !shown.length) return;
    note.textContent =
      `${tally} — the run ended here, so nothing was sampled from this ` +
      `last shortlist.`;
  }

  return { setVocab, reset, setContext, pushTop, chose, end };
}
