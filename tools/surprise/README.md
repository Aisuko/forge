# surprise

Select any stretch of text and the model tints it by how surprised it was to
find it there. No button: the page reacts to reading. Selecting a phrase
rescores it with only the selection for context, which is a visibly different
answer from the same characters read in full context.

```bash
./tools/surprise/build.sh
python3 -m http.server -d tools/surprise/dist 8081
```

## No Rust here

This tool is a page and a build script. The call it makes —
`WasmGpt2.surprisal(text)` — is part of the runtime, because scoring text that
already exists is a forward pass like any other: one pass over the whole
selection, teacher-forced, rather than one decode step per character. What is
*not* runtime is the reading of that number, which is everything in
[`web/`](web).

Per position, one call returns four parallel arrays: the character actually
there, its surprisal in bits (`−log₂ p`), the single character the model
expected most, and that character's probability. Not the full 65-way
distribution — the rest is discarded inside the forward pass. Hovering does no
model work at all: the arrays are stashed once and hover is an index lookup.

The model is the 10.6 M-parameter char-level Shakespeare checkpoint in
`assets/shakespeare_char/` — 6.7 MB of `.fzm` q4, dequantized to f32 on load —
which the build copies into `dist/model/`.

Deployed as part of the site — `make site` builds all three pages together, and
they share one copy of that checkpoint. See [`../README.md`](../README.md).
