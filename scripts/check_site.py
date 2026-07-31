#!/usr/bin/env python3
"""Static checks on docs/dist/ that catch the ways this site silently breaks.

1. Every referenced asset exists — a 404 in production that works in local
   preview is the classic Pages failure.
2. Every module specifier inside every shipped .js resolves. A missing sibling
   breaks the whole module graph before a single line runs — with no console
   error a build step would notice.
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

imap = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
importmap = json.loads(imap.group(1))["imports"] if imap else {}

# Fetched at runtime, so they must ship even though no tag names them. Keyed by
# page: a 404 here is invisible until a visitor presses Run.
RUNTIME_FETCHES = {
    "index.html": [
        "./model/model.safetensors",
        "./model/config.json",
        "./model/vocab.json",
    ],
    "council.html": [
        "./council/manifest.json",
        "./council/config.json",
        "./council/vocab.json",
    ],
    # react.html loads the same char model as index.html, but eagerly — a 404
    # here breaks the page on open rather than on a button press.
    "react.html": [
        "./model/model.safetensors",
        "./model/config.json",
        "./model/vocab.json",
    ],
}

pages = sorted(dist.glob("*.html"))
if not pages:
    sys.exit("no HTML pages in the build")

for page in pages:
    text = page.read_text()
    name = page.name
    where = f"{name}: "
    refs = set(re.findall(r'(?:src|href)="([^"]+)"', text))
    refs |= set(importmap.values()) if name == "index.html" else set()
    refs |= set(RUNTIME_FETCHES.get(name, []))

    ids = set(re.findall(r'\bid="([^"]+)"', text))
    for r in sorted(refs):
        if r.startswith("#"):
            if r[1:] not in ids:
                problems.append(f"{where}in-page link to a missing anchor: {r}")
            continue
        if r.startswith(("http://", "https://", "//")):
            # A stylesheet, script, or module from another origin would break the
            # "no external network requests at runtime" guarantee; anchors are OK.
            if re.search(rf'(?:src|rel="stylesheet"[^>]*href)="{re.escape(r)}"', text):
                problems.append(f"{where}cross-origin runtime asset: {r}")
            continue
        if r.startswith("/"):
            problems.append(f"{where}root-absolute path (404 under /forge/): {r}")
            continue
        # A link to another page may carry a fragment; the file is what has to
        # exist. (Anchors *within* the target page are the target's business.)
        path = r.split("#", 1)[0]
        if not (dist / path.lstrip("./")).exists():
            problems.append(f"{where}missing asset (would 404): {r}")

# The council's expert weights are named by its own manifest, not by any tag.
council_manifest = dist / "council" / "manifest.json"
if council_manifest.exists():
    for e in json.loads(council_manifest.read_text())["experts"]:
        if not (dist / "council" / e["file"]).exists():
            problems.append(f"council.html: manifest names a missing expert: {e['file']}")

# ── Module graph ──────────────────────────────────────────────────────────
# Static (`from "x"`, `import "x"`, `export … from "x"`) and dynamic
# (`import("x")`) specifiers, in every shipped module including the
# wasm-bindgen-generated ones. Resolving these catches a renamed or unshipped
# module at build time instead of in production.
#
# The specifier charset is deliberately narrow: minified bundles contain
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
CORE = ["index.html", "assets/app.css", "demo.js", "attention.js"]
# council.html is a second entry point with its own budget-free payload;
# the budget below is about the landing page a visitor lands on cold.
blobs = [(dist / f).read_bytes() for f in CORE if (dist / f).exists()]
core = sum(len(b) for b in blobs)
# Budgeted on the compressed size, because that is what a visitor downloads:
# Pages serves these gzipped. The raw figure is reported too, but gating on it
# would be a tax on comments and whitespace — and this source ships
# unminified on purpose, since the page's claim is that you can read it.
wire = sum(len(gzip.compress(b, 9)) for b in blobs)
print(f"site ok — {total / 1e6:.1f} MB total, {wire / 1024:.1f} KB core gzipped "
      f"({core / 1024:.1f} KB raw; HTML+CSS+JS excluding the wasm; budget 45 KB)")
if wire > 45 * 1024:
    sys.exit("core payload is over the 45 KB gzipped budget")
