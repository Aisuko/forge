#!/usr/bin/env python3
"""Rewrite the site's kernel inventory from the actual contents of shaders/.

The page states a kernel count and lists every kernel by name. Hand-maintaining
that list guarantees it eventually lies, so it is generated at build time and
the count is asserted against the directory.

Usage: gen_kernels.py docs/dist/index.html
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "shaders"

# Backward and optimizer kernels are named; everything else is forward.
BACKWARD = {"ce_bwd", "gelu_bwd", "layernorm_bwd_dp", "layernorm_bwd_dx",
            "scatter_add", "softmax_bwd"}
OPTIMIZER = {"adamw", "sumsq"}


def group(name):
    if name in BACKWARD:
        return "Backward"
    if name in OPTIMIZER:
        return "Optimizer"
    return "Forward"


def main(target):
    names = sorted(p.stem for p in SHADERS.glob("*.wgsl"))
    if not names:
        sys.exit(f"no *.wgsl under {SHADERS}")

    groups = {"Forward": [], "Backward": [], "Optimizer": []}
    for n in names:
        groups[group(n)].append(n)

    cards = []
    for title, members in groups.items():
        chips = "\n".join(
            f'                <li class="chip">{m}</li>' for m in members
        )
        cards.append(
            '            <div class="card">\n'
            f'              <h3 class="font-semibold">{title} — {len(members)}</h3>\n'
            '              <ul class="mt-3 flex flex-wrap gap-1.5 font-mono text-xs">\n'
            f"{chips}\n"
            "              </ul>\n"
            "            </div>"
        )

    html = pathlib.Path(target).read_text()
    block = (
        '<div id="kernel-list" class="mt-8 grid gap-5 md:grid-cols-3">\n'
        + "\n".join(cards)
        + "\n          </div>"
    )
    html, n = re.subn(
        r'<div id="kernel-list".*?</div>\s*</div>\s*</div>',
        block + "\n        </div>\n      </div>",
        html,
        count=1,
        flags=re.S,
    )
    if n != 1:
        sys.exit("could not find the #kernel-list block in " + target)

    # The headline count appears in the section heading and the hero tile.
    total = len(names)
    html = html.replace(">23 WGSL kernels<", f">{total} WGSL kernels<")
    html = html.replace(
        '<dd class="mt-1 text-2xl font-semibold">23</dd>',
        f'<dd class="mt-1 text-2xl font-semibold">{total}</dd>',
    )
    pathlib.Path(target).write_text(html)
    print(f"   kernels: {total} "
          f"({', '.join(f'{k.lower()} {len(v)}' for k, v in groups.items())})")


if __name__ == "__main__":
    main(sys.argv[1])
