#!/usr/bin/env python3
"""Score generated text against the corpus it was trained on.

    score_text.py CORPUS SAMPLES  ->  word_validity<TAB>copy32<TAB>max_copy<TAB>speaker_rate

Three numbers that say different things about a character-level model, and one
that only makes sense alongside the others:

* **word_validity** — the fraction of alphabetic words that appear anywhere in
  the training split. A char model has to spell from scratch, so this is the
  most direct measure of whether its output reads as English. It separates the
  models that held-out loss alone does not: the 5000-step run scores worse on
  validation loss than the 2000-step one and better here.

* **copy32 / max_copy** — how much of the output appears verbatim in the
  training text. These exist to stop word_validity being gamed: a model that
  memorised Shakespeare would score 100% on spelling while having learned
  nothing worth shipping. Ordinary English collocations run to about 25
  characters, so a longest-shared-span far above that is the signal.

* **speaker_rate** — lines that are a bare `NAME:` speaker tag. Reported for
  interest, not scored; it measures format rather than quality.

Only the *training* split counts as "seen" — the last 10% is held out, matching
the trainer's own split, so a model is never credited for reproducing text it
was legitimately evaluated on.
"""
import re
import sys

# The trainer holds out the last 10%; scoring against the whole corpus would
# count held-out text as memorised.
VAL_FRACTION = 0.1
# k-gram lengths probed for verbatim overlap. The largest with a hit is
# reported, which bounds the exact longest-common-substring from below at a
# fraction of its cost.
COPY_K = (8, 16, 24, 32, 48, 64, 96, 128)
WORD = re.compile(r"[A-Za-z']+")
SPEAKER = re.compile(r"[A-Z][A-Za-z' ]*:")


def kgrams(text, k):
    return {hash(text[i:i + k]) for i in range(len(text) - k + 1)}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    corpus = open(sys.argv[1], encoding="utf-8").read()
    sample = open(sys.argv[2], encoding="utf-8").read()
    train = corpus[: int(len(corpus) * (1 - VAL_FRACTION))]

    vocab = set(w.lower() for w in WORD.findall(train))
    words = [w.lower() for w in WORD.findall(sample)]
    validity = sum(w in vocab for w in words) / max(len(words), 1)

    # Rolling sets rather than a substring scan: the corpus is ~1 MB and the
    # sample ~12 KB, so the naive O(n*m) search is minutes and this is instant.
    max_copy = 0
    copy32 = 0.0
    for k in COPY_K:
        if len(sample) < k:
            break
        seen = kgrams(train, k)
        hits = sum(1 for i in range(0, len(sample) - k + 1, 4)
                   if hash(sample[i:i + k]) in seen)
        total = len(range(0, len(sample) - k + 1, 4))
        if k == 32:
            copy32 = hits / max(total, 1)
        if not hits:
            # k-gram containment is monotone, so no hit at k means no hit above
            # it. Stopping here also leaves copy32 at 0.0 when the ladder never
            # reaches 32 — which is the right answer, not a missed measurement.
            break
        max_copy = k

    lines = sample.split("\n")
    speaker = sum(1 for ln in lines if SPEAKER.fullmatch(ln.strip())) / max(len(lines), 1)

    print(f"{validity:.4f}\t{copy32:.4f}\t{max_copy}\t{speaker:.4f}")


if __name__ == "__main__":
    main()
