# surprise

Every character resolves in the time the model needed to be sure of it. The
confident ones snap solid on the first frame; the ones it was torn about keep
flickering through the alternatives it actually weighed, until the last hold-out
lands. What settles is a heat map of surprisal. Selecting a phrase rescores it
with only the selection for context, which is a visibly different answer from
the same characters read in full context.

```bash
./tools/surprise/build.sh
python3 -m http.server -d tools/surprise/dist 8081
```

## The reveal is a replay, and the page says so

The scoring pass finishes in ~15 ms, before the first frame is drawn. Every
number driving the animation is real — a position's lock time **is** its
surprisal, and the characters it flickers through are its own top-8, sampled in
proportion to probability — but no inference happens during the flicker, and a
reader must not be left to infer that each blink cost a forward pass. The page
states this under the text, and [`web/react.js`](web/react.js) states it in the
module comment.

Nothing staggers left to right. A wave sweeping across the line would say "one
character at a time", which is what generation looks like; all positions start
together because all positions *are* scored together, in one pass, and the only
thing separating them is how sure the model was.

## The Rust

One struct, one function, one binding — [`src/`](src), a crate downstream of
the runtime like [`../council`](../council):

- `Surprisal` — per position, the surprisal in bits (`−log₂ p`) and `k`
  candidates with their probabilities, flat and descending. Column 0 is the
  character the model expected most, so `top()` and `top_p()` read it off
  rather than storing it twice.
- `surprisal(model, ids, k)` — one teacher-forced forward pass over the whole
  sequence rather than one decode step per character, then host arithmetic over
  the `[t, vocab]` logits `Gpt2::forward` hands back: a log-sum-exp, a gather,
  and `forge::top_probs` for the ranking.
- `WasmSurprise` — the six methods the page calls, and no more.

Through 0.4.0 this was `Gpt2::surprisal_async` and `WasmGpt2.surprisal`, in the
runtime. Scoring is still a primitive; what moved is the *presentation* of
scoring — bits, the expected character, and a ranked list of alternatives sized
for an animation — which is a page's vocabulary, not a runtime's.

Hovering does no model work at all: the arrays are stashed once and hover is an
index lookup, which is why the panel beside the text can show the very
candidates the flicker was cycling through.

The model is the 10.6 M-parameter char-level Shakespeare checkpoint in
`assets/shakespeare_char/` — 6.7 MB of `.fzm` q4, dequantized to f32 on load —
which the build copies into `dist/model/`.

Deployed as part of the site — `make site` builds all three pages together, and
they share one copy of that checkpoint. See [`../README.md`](../README.md).
