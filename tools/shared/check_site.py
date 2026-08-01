#!/usr/bin/env python3
"""Static checks on a built page that catch the ways it silently breaks.

Runs against one tool's self-contained dist/ or against the composed site:

1. Every referenced asset exists — a 404 that works in local preview is the
   classic static-hosting failure.
2. Every module specifier in every shipped .js resolves. A missing sibling
   breaks the module graph before a line runs, with no console error.
3. No root-absolute paths: these pages are served from a subdirectory.
4. No cross-origin runtime references: no CDN, no HuggingFace fetch. Links a
   visitor clicks are fine; a resource the page *loads* is not.
5. Each page's own code stays inside a gzipped budget — the wasm and the
   weights dominate the download, but the part a reader reads should not.

A hyperlink to a sibling page missing from the artifact is a warning, not an
error: a standalone tool dist links back to a landing page it does not ship.
`--strict` makes it an error, which is what the composed site is built with.

Usage: check_site.py DIST [--strict] [--budget KB]
"""
import argparse
import gzip
import json
import pathlib
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("dist", nargs="?", default="dist", type=pathlib.Path)
ap.add_argument("--strict", action="store_true", help="warnings are errors")
ap.add_argument("--budget", type=float, default=45.0, help="KB gzipped, per page")
args = ap.parse_args()
dist = args.dist

pages = sorted(dist.glob("*.html"))
if not pages:
    sys.exit(f"no HTML pages in {dist} — run a build.sh first")

errors, warnings = [], []

# Fetched at runtime, so they must ship even though no tag names them. A 404
# here is invisible until a visitor presses Run.
RUNTIME_FETCHES = {
    "index.html": [
        "./model/model.safetensors",
        "./model/config.json",
        "./model/vocab.json",
        "./model/metrics.json",
    ],
    "council.html": [
        "./council/manifest.json",
        "./council/config.json",
        "./council/vocab.json",
    ],
    # react.html loads the char model on open, not on a button press.
    "react.html": [
        "./model/model.safetensors",
        "./model/config.json",
        "./model/vocab.json",
    ],
}

importmap = {}
for page in pages:
    imap = re.search(r'<script type="importmap">(.*?)</script>', page.read_text(), re.S)
    if imap:
        importmap.update(json.loads(imap.group(1))["imports"])

for page in pages:
    text = page.read_text()
    where = f"{page.name}: "
    refs = set(re.findall(r'(?:src|href)="([^"]+)"', text))
    refs |= set(RUNTIME_FETCHES.get(page.name, []))
    # Anchor targets only; every other src/href is something the page loads.
    links = set(re.findall(r'<a\b[^>]*\bhref="([^"]+)"', text, re.S))

    ids = set(re.findall(r'\bid="([^"]+)"', text))
    for r in sorted(refs):
        if r.startswith("#"):
            if r[1:] not in ids:
                errors.append(f"{where}in-page link to a missing anchor: {r}")
            continue
        if r.startswith(("http://", "https://", "//")):
            # Another origin is fine to link to, not to load from.
            if re.search(rf'(?:src|rel="stylesheet"[^>]*href)="{re.escape(r)}"', text):
                errors.append(f"{where}cross-origin runtime asset: {r}")
            continue
        if r.startswith("/"):
            errors.append(f"{where}root-absolute path (404 when not served from /): {r}")
            continue
        # A link to another page may carry a fragment; the file is what has to
        # exist. Anchors *within* the target page are the target's business.
        path = r.split("#", 1)[0]
        if (dist / path.lstrip("./")).exists():
            continue
        if r in links and path.endswith(".html"):
            warnings.append(f"{where}links to a page this artifact does not ship: {r}")
        else:
            errors.append(f"{where}missing asset (would 404): {r}")

# The council's expert weights are named by its own manifest, not by any tag.
council_manifest = dist / "council" / "manifest.json"
if council_manifest.exists():
    for e in json.loads(council_manifest.read_text())["experts"]:
        if not (dist / "council" / e["file"]).exists():
            errors.append(f"council.html: manifest names a missing expert: {e['file']}")

# ── Module graph ──────────────────────────────────────────────────────────
# Static (`from "x"`, `import "x"`, `export … from "x"`) and dynamic
# (`import("x")`) specifiers, in every shipped module including the
# wasm-bindgen-generated ones. Resolving these catches a renamed or unshipped
# module at build time instead of in production.
#
# The charset is deliberately narrow: minified bundles contain English strings
# like "…resized from ("+w+")", which a permissive pattern reads as an import.
SPEC = r"""["']([A-Za-z0-9_@~./-]+)["']"""
SPECIFIERS = re.compile(rf"""\bfrom\s*{SPEC}|\bimport\s*{SPEC}|\bimport\s*\(\s*{SPEC}""")


def specifiers(text):
    """Every module specifier in `text`, from whichever alternative matched."""
    return {next(g for g in m if g) for m in SPECIFIERS.findall(text)}


for js in sorted(dist.rglob("*.js")):
    text = js.read_text(errors="replace")
    for spec in sorted(specifiers(text)):
        if spec.startswith(("http://", "https://", "//", "data:")):
            errors.append(f"{js.relative_to(dist)}: cross-origin import: {spec}")
            continue
        if spec.startswith("/"):
            errors.append(
                f"{js.relative_to(dist)}: root-absolute import (404 when not served from /): {spec}"
            )
            continue
        if spec.startswith("."):
            target = (js.parent / spec).resolve()
        elif spec in importmap:
            # A bare specifier resolves against the document, not the importer.
            target = (dist / importmap[spec].lstrip("./")).resolve()
        else:
            errors.append(
                f"{js.relative_to(dist)}: bare import with no importmap entry: {spec}"
            )
            continue
        if not target.exists():
            errors.append(f"{js.relative_to(dist)}: import would 404: {spec}")

for w in warnings:
    print(f"warning: {w}", file=sys.stderr)
for e in errors:
    print(f"error: {e}", file=sys.stderr)
if errors or (warnings and args.strict):
    sys.exit(1)

# ── Budget ────────────────────────────────────────────────────────────────
# Per page, because the composed site holds three and a visitor loads one.
total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
css = dist / "assets" / "app.css"
over = []


def hand_written(page):
    """The page's own modules: walk its imports, keep what lives at the root.

    Transitive, because a page names one entry module and that module imports
    the rest. Files in a subdirectory are the wasm-bindgen bundles — generated,
    not read, so they are not what this budget is about.
    """
    root = dist.resolve()
    seen, queue = set(), [
        (dist / s.lstrip("./")).resolve()
        for s in re.findall(r'<script[^>]*src="([^"]+)"', page.read_text())
    ]
    while queue:
        js = queue.pop()
        if js in seen or not js.exists() or js.parent != root:
            continue
        seen.add(js)
        queue += [
            (js.parent / spec).resolve()
            for spec in specifiers(js.read_text(errors="replace"))
            if spec.startswith(".")
        ]
    return sorted(seen)


for page in pages:
    own = [css, page] + hand_written(page)
    blobs = [f.read_bytes() for f in own if f.exists()]
    raw = sum(len(b) for b in blobs)
    # Budgeted compressed, because that is what a visitor downloads.
    wire = sum(len(gzip.compress(b, 9)) for b in blobs)
    print(f"{page.name}: {wire / 1024:.1f} KB gzipped own code "
          f"({raw / 1024:.1f} KB raw; HTML+CSS+JS, budget {args.budget:.0f} KB)")
    if wire > args.budget * 1024:
        over.append(page.name)

print(f"site ok — {len(pages)} page(s), {total / 1e6:.1f} MB total")
if over:
    sys.exit(f"over the {args.budget:.0f} KB gzipped budget: {', '.join(over)}")
