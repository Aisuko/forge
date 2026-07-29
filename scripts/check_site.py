#!/usr/bin/env python3
"""Static checks on docs/dist/ that catch the ways this site silently breaks.

1. Every referenced asset exists — a 404 in production that works in local
   preview is the classic Pages failure.
2. Every module specifier inside every shipped .js resolves. three.js r185 is
   *two* files (three.module.min.js imports ./three.core.min.js), and a missing
   sibling breaks the module graph before a single line of pipeline.js runs — with
   no console error a build step would notice.
3. No root-absolute paths: the site is served from /forge/, not /.
4. No cross-origin runtime references: no CDN, no HuggingFace fetch. (Links a
   visitor clicks are fine; a resource the page *loads* is not.)
5. `.nojekyll` is present, or Jekyll drops every _-prefixed path.

Usage: check_site.py [docs/dist]
"""
import gzip
import json
import pathlib
import re
import sys

dist = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "docs/dist")
html_path = dist / "index.html"
if not html_path.exists():
    sys.exit(f"{html_path} not found — run scripts/build_site.sh first")
html = html_path.read_text()

problems = []

if not (dist / ".nojekyll").exists():
    problems.append(".nojekyll is missing; Jekyll will ignore _-prefixed paths")

refs = set(re.findall(r'(?:src|href)="([^"]+)"', html))
imap = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
importmap = json.loads(imap.group(1))["imports"] if imap else {}
refs |= set(importmap.values())

# Fetched at runtime, so they must ship even though no tag names them.
refs |= {"./model/model.safetensors", "./model/config.json", "./model/vocab.json"}

ids = set(re.findall(r'\bid="([^"]+)"', html))
for r in sorted(refs):
    if r.startswith("#"):
        if r[1:] not in ids:
            problems.append(f"in-page link to a missing anchor: {r}")
        continue
    if r.startswith(("http://", "https://", "//")):
        # A stylesheet, script, or module from another origin would break the
        # "no external network requests at runtime" guarantee; anchors are OK.
        if re.search(rf'(?:src|rel="stylesheet"[^>]*href)="{re.escape(r)}"', html):
            problems.append(f"cross-origin runtime asset: {r}")
        continue
    if r.startswith("/"):
        problems.append(f"root-absolute path (404 under /forge/): {r}")
        continue
    if not (dist / r.lstrip("./")).exists():
        problems.append(f"missing asset (would 404): {r}")

# ── Module graph ──────────────────────────────────────────────────────────
# Static (`from "x"`, `import "x"`, `export … from "x"`) and dynamic
# (`import("x")`) specifiers, in every shipped module including the vendored
# and wasm-bindgen-generated ones. Resolving these is what would have caught
# the missing three.core.min.js at build time instead of in production.
#
# The specifier charset is deliberately narrow: minified three.js contains
# English strings like "…resized from ("+w+")", and a permissive pattern reads
# those as imports.
SPEC = r"""["']([A-Za-z0-9_@~./-]+)["']"""
SPECIFIERS = re.compile(
    rf"""\bfrom\s*{SPEC}|\bimport\s*{SPEC}|\bimport\s*\(\s*{SPEC}"""
)


def specifiers(text):
    """Every module specifier in `text`, from whichever alternative matched."""
    return {next(g for g in m if g) for m in SPECIFIERS.findall(text)}

for js in sorted(dist.rglob("*.js")):
    text = js.read_text(errors="replace")
    for spec in sorted(specifiers(text)):
        if spec.startswith(("http://", "https://", "//", "data:")):
            problems.append(f"{js.relative_to(dist)}: cross-origin import: {spec}")
            continue
        if spec.startswith("/"):
            problems.append(
                f"{js.relative_to(dist)}: root-absolute import (404 under /forge/): {spec}"
            )
            continue
        if spec.startswith("."):
            target = (js.parent / spec).resolve()
        elif spec in importmap:
            # Bare specifier, resolved by the page's importmap relative to the
            # document, not to the importing file.
            target = (dist / importmap[spec].lstrip("./")).resolve()
        else:
            problems.append(
                f"{js.relative_to(dist)}: bare import with no importmap entry: {spec}"
            )
            continue
        if not target.exists():
            problems.append(f"{js.relative_to(dist)}: import would 404: {spec}")

if problems:
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    sys.exit(1)

total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
CORE = ["index.html", "assets/app.css", "demo.js", "pipeline.js"]
blobs = [(dist / f).read_bytes() for f in CORE if (dist / f).exists()]
core = sum(len(b) for b in blobs)
# Budgeted on the compressed size, because that is what a visitor downloads:
# Pages serves these gzipped. The raw figure is reported too, but gating on it
# would be a tax on comments and whitespace — and this source ships
# unminified on purpose, since the page's claim is that you can read it.
wire = sum(len(gzip.compress(b, 9)) for b in blobs)
print(f"site ok — {total / 1e6:.1f} MB total, {wire / 1024:.1f} KB core gzipped "
      f"({core / 1024:.1f} KB raw; HTML+CSS+JS excluding three.js; budget 45 KB)")
if wire > 45 * 1024:
    sys.exit("core payload is over the 45 KB gzipped budget")
