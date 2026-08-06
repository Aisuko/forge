#!/usr/bin/env python3
"""Split Tiny Shakespeare into the council's expert slices.

Tiny Shakespeare is a concatenation of whole plays in order, so *position* is
the play boundary. Four contiguous quarters therefore land on four distinct
casts without a single hand-written label, and each expert still gets ~279 KB
of text. Each slice's name is read back out of the data: its two most talkative
speakers.

    ./tools/council/scripts/split_corpus.py   # -> data/council/{0..3}.txt, manifest.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Three levels up: tools/council/scripts/ -> the repository root, where the
# corpus and the data/ output directory live.
ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "data" / "tinyshakespeare.txt"
OUT = ROOT / "data" / "council"
N_EXPERTS = 4

# A speaker line in this corpus is a bare capitalised name followed by a colon,
# alone on its line ("FIRST CITIZEN:", "Nurse:").
SPEAKER = re.compile(r"^([A-Z][A-Za-z ]*):$", re.M)


def main() -> int:
    if not CORPUS.exists():
        print(f"missing {CORPUS} — run scripts/local/download_shakespeare.sh", file=sys.stderr)
        return 1
    text = CORPUS.read_text()
    n = len(text)
    OUT.mkdir(parents=True, exist_ok=True)

    experts = []
    for k in range(N_EXPERTS):
        # Cut on a line boundary so no speech is split across two experts.
        start = text.index("\n", k * n // N_EXPERTS) + 1 if k else 0
        end = text.index("\n", (k + 1) * n // N_EXPERTS) + 1 if k + 1 < N_EXPERTS else n
        slice_ = text[start:end]
        path = OUT / f"{k}.txt"
        path.write_text(slice_)

        top = Counter(SPEAKER.findall(slice_)).most_common(2)
        # Title case: "DUKE VINCENTIO" shouted on every card reads as noise.
        label = " & ".join(name.title() for name, _ in top)
        experts.append({"id": k, "file": path.name, "label": label, "bytes": len(slice_)})
        print(f"{path.relative_to(ROOT)}  {len(slice_):>7} bytes  {label}")

    manifest = {"corpus": CORPUS.name, "n_experts": N_EXPERTS, "experts": experts}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
