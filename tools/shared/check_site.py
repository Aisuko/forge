#!/usr/bin/env python3
"""Static checks on a tool's built page that catch the ways it silently breaks.

Each tool in tools/ builds a self-contained artifact into its own dist/. This
verifies one, whichever it is:

1. Every referenced asset exists — a 404 that works in local preview is the
   classic static-hosting failure.
2. Every module specifier inside every shipped .js resolves. A missing sibling
   breaks the whole module graph before a single line runs, with no console
   error a build step would notice.
3. No root-absolute paths: these pages are served from a subdirectory.
4. No cross-origin runtime references: no CDN, no HuggingFace fetch. (Links a
   visitor clicks are fine; a resource the page *loads* is not.)
5. The page's own code stays inside a gzipped budget — the wasm and the weights
   dominate the download, but the part a reader is asked to read should not.

Usage: check_site.py DIST [BUDGET_KB]
"""
import gzip
import json
import pathlib
import re
import sys

dist = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
budget_kb = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

pages = sorted(dist.glob("*.html"))
if not pages:
    sys.exit(f"no HTML pages in {dist} — run this tool's build.sh first")

problems = []

# Fetched at runtime, so they must ship even though no tag names them. Keyed by
# page: a 404 here is invisible until a visitor presses Run.
RUNTIME_FETCHES = {
    "council.html": [
        "./council/manifest.json",
        "./council/config.json",
        "./council/vocab.json",
    ],
    # react.html loads the char model eagerly — a 404 here breaks the page on
    # open rather than on a button press.
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
            problems.append(f"{where}root-absolute path (404 when not served from /): {r}")
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
                f"{js.relative_to(dist)}: root-absolute import (404 when not served from /): {spec}"
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
# The page's own code: HTML, stylesheet and modules — not the wasm, not the
# weights.
own = [dist / "assets" / "app.css"] + pages + sorted(dist.glob("*.js"))
blobs = [f.read_bytes() for f in own if f.exists()]
raw = sum(len(b) for b in blobs)
# Budgeted on the compressed size, because that is what a visitor downloads.
wire = sum(len(gzip.compress(b, 9)) for b in blobs)
print(f"page ok — {total / 1e6:.1f} MB total, {wire / 1024:.1f} KB own code gzipped "
      f"({raw / 1024:.1f} KB raw; HTML+CSS+JS, excluding wasm and weights; "
      f"budget {budget_kb:.0f} KB)")
if wire > budget_kb * 1024:
    sys.exit(f"the page's own code is over the {budget_kb:.0f} KB gzipped budget")
