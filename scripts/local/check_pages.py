#!/usr/bin/env python3
"""Drive every page of the built site in a real browser, on a real GPU.

`check_site.py` proves the artifact resolves; this proves it runs. A page that
builds and then fails inside `WasmGpt2.load` is indistinguishable from a working
one over HTTP, so each page here is used the way a visitor uses it — press the
button, wait for output, assert on the DOM — and any console error, page error
or 4xx fails the run.

This is deliberately not in GitHub Actions: hosted runners have no GPU, so
`navigator.gpu.requestAdapter()` returns null and every page below fails for a
reason no code change can fix. It belongs where the GPU is.

    ./scripts/common/build_site.sh && python3 scripts/local/check_pages.py    # or: make site-verify

Screenshots land in target/site-check/. Requires `pip install playwright` and
`playwright install chromium`.
"""
import argparse
import functools
import http.server
import pathlib
import socketserver
import sys
import threading

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Chromium ships WebGPU behind these on Linux; without them requestAdapter()
# returns null and every check below fails for the wrong reason.
FLAGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    "--use-angle=vulkan",
    "--use-vulkan",
    "--ignore-gpu-blocklist",
    "--no-sandbox",
]

ap = argparse.ArgumentParser()
ap.add_argument("dist", nargs="?", default=ROOT / "docs/dist", type=pathlib.Path)
ap.add_argument("--port", type=int, default=8770)
ap.add_argument("--headed", action="store_true")
args = ap.parse_args()

if not (args.dist / "index.html").exists():
    sys.exit(f"{args.dist} is not a built site — run ./scripts/common/build_site.sh")

shots = ROOT / "target/site-check"
shots.mkdir(parents=True, exist_ok=True)


def serve(directory, port):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    handler.log_message = lambda *a: None
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Page:
    """One page under test, with its console and network failures collected."""

    def __init__(self, browser, url, name):
        self.name = name
        self.errors = []
        self.pg = browser.new_page(viewport={"width": 1400, "height": 1100})
        self.pg.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.pg.on("console",
                   lambda m: m.type == "error" and self.errors.append(f"console: {m.text}"))
        self.pg.on("requestfailed",
                   lambda r: self.errors.append(f"request failed: {r.url}"))
        self.pg.on("response",
                   lambda r: r.status >= 400 and self.errors.append(f"HTTP {r.status}: {r.url}"))
        self.pg.goto(url, wait_until="load", timeout=60_000)

    def gpu(self):
        return self.pg.evaluate(
            "async () => !!(navigator.gpu && await navigator.gpu.requestAdapter())"
        )

    def until(self, expr, seconds=120, note=""):
        """Poll a JS predicate; the message names what never happened."""
        try:
            self.pg.wait_for_function(expr, timeout=seconds * 1000)
        except Exception:
            raise AssertionError(f"{note or expr} — never true within {seconds}s") from None

    def finish(self):
        self.pg.screenshot(path=str(shots / f"{self.name}.png"), full_page=False)
        return self.errors


def check_landing(b, base):
    p = Page(b, f"{base}/index.html", "index")
    assert p.gpu(), "no WebGPU adapter"
    p.pg.click("#stage-idle-run")
    p.until("() => document.getElementById('demo-output').innerText.length > 60",
            note="the demo generated no text")
    p.until("() => document.getElementById('dec-bars').children.length > 0",
            seconds=30, note="the decision panel stayed empty")
    p.until("() => document.getElementById('heat-strip').children.length > 0",
            seconds=30, note="the attention strip stayed empty")
    out = p.pg.inner_text("#demo-output")
    status = p.pg.inner_text("#demo-status")
    p.pg.click("#demo-stop")
    return p, f"{len(out)} chars · {status}"


def check_council(b, base):
    p = Page(b, f"{base}/council.html", "council")
    assert p.gpu(), "no WebGPU adapter"
    # Loading the experts starts a run; #council-run stays disabled until it
    # ends, so the button to press afterwards is Stop, not Run.
    p.pg.click("#council-start")
    p.until("() => document.querySelectorAll('#flow-nodes *').length > 0",
            note="the four experts never loaded")
    p.until("() => document.getElementById('council-out').innerText.length > 20",
            note="the council merged nothing")
    p.until("() => document.getElementById('council-bars').children.length > 0",
            seconds=30, note="the merged distribution stayed empty")
    experts = p.pg.eval_on_selector_all("#council-legend > *", "e => e.length")
    assert experts >= 5, f"expected 4 experts and a merged row, drew {experts}"
    status = p.pg.inner_text("#council-status")
    p.pg.click("#council-stop")
    return p, f"{experts} legend rows · {status}"


def check_surprise(b, base):
    p = Page(b, f"{base}/react.html", "react")
    assert p.gpu(), "no WebGPU adapter"
    # The three passages are scored on load, one forward pass each, and every
    # character becomes a span. Wait for all three: the replay button is only
    # enabled once the last one is laid out.
    p.until("""() => {
                const bodies = [...document.querySelectorAll('#react-text [data-scored]')];
                return bodies.length === 3 && bodies.every(b => b.querySelector('[data-i]'));
            }""", note="the passages were never scored")
    # Then drive the reveal from the top, rather than racing whichever part of
    # the automatic one is still running: press the button, and wait for every
    # position to stop spinning. Nothing here waits on the model — the replay
    # is animation over a pass that already finished — so a reveal that never
    # settles is a bug in the loop, not a slow GPU.
    p.pg.click("#react-replay")
    p.until("() => document.querySelectorAll('#react-text .tok-spin').length > 0",
            note="pressing Read it again resolved nothing")
    p.until("""() => {
                const spans = [...document.querySelectorAll('#react-text [data-i]')];
                return spans.length > 100
                    && !spans.some(s => s.classList.contains('tok-spin'));
            }""", note="the reveal never finished")
    # Hovering a resolved character fills the panel with the candidates the
    # flicker was cycling through — the loop the old grey line could not close.
    p.pg.hover("#react-text [data-scored] [data-i='12']")
    p.until("""() => [...document.querySelectorAll('#react-bars .bar-row')]
                        .filter(r => !r.hidden).length >= 8""",
            note="the readout panel stayed empty")
    # Then select inside one, which is the page's real interaction: the same
    # characters rescored with none of the context before them.
    p.pg.evaluate("""() => {
        const spans = document.querySelectorAll('#react-text [data-i]');
        const r = document.createRange();
        r.setStart(spans[10], 0);
        r.setEnd(spans[60], spans[60].childNodes.length);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    }""")
    p.until("() => !document.getElementById('react-selection-card').hidden",
            note="the selection was never scored")
    p.until("() => document.getElementById('react-selection-note').innerText.length > 20",
            seconds=30, note="the selection panel stayed empty")
    return p, p.pg.inner_text("#react-selection-stat").strip()


CHECKS = [("landing", check_landing), ("council", check_council), ("surprise", check_surprise)]

srv = serve(args.dist, args.port)
base = f"http://127.0.0.1:{args.port}"
failed = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=not args.headed, args=FLAGS)
    for name, check in CHECKS:
        page = None
        try:
            page, detail = check(browser, base)
            errors = page.finish()
            if errors:
                failed.append((name, "; ".join(dict.fromkeys(errors))))
                print(f"✗ {name}: {len(errors)} runtime error(s)")
                for e in dict.fromkeys(errors):
                    print(f"    {e}")
            else:
                print(f"✓ {name}: {detail}")
        except Exception as e:
            if page:
                page.finish()
            failed.append((name, str(e)))
            print(f"✗ {name}: {e}")
    browser.close()
srv.shutdown()

print(f"\nscreenshots: {shots.relative_to(ROOT)}/")
if failed:
    sys.exit(f"{len(failed)} of {len(CHECKS)} pages failed: "
             + ", ".join(n for n, _ in failed))
print(f"all {len(CHECKS)} pages ran on a real GPU")
