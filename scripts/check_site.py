#!/usr/bin/env python3
"""Static checks on docs/dist/ that catch the ways this site silently breaks.

1. Every referenced asset exists — a 404 in production that works in local
   preview is the classic Pages failure.
2. No root-absolute paths: the site is served from /forge/, not /.
3. No cross-origin runtime references: no CDN, no HuggingFace fetch. (Links a
   visitor clicks are fine; a resource the page *loads* is not.)
4. `.nojekyll` is present, or Jekyll drops every _-prefixed path.

Usage: check_site.py [docs/dist]
"""
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
if imap:
    refs |= set(json.loads(imap.group(1))["imports"].values())

demo = (dist / "demo.js").read_text() if (dist / "demo.js").exists() else ""
refs |= set(re.findall(r'import\("([^"]+)"\)', demo))
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

if problems:
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    sys.exit(1)

total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
core = sum(
    (dist / f).stat().st_size
    for f in ["index.html", "assets/app.css", "scene.js", "demo.js"]
    if (dist / f).exists()
)
print(f"site ok — {total / 1e6:.1f} MB total, {core / 1024:.1f} KB core "
      f"(HTML+CSS+JS excluding three.js; budget 100 KB)")
if core > 100 * 1024:
    sys.exit("core payload is over the 100 KB budget")
