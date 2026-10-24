"""Real-interaction captures for the launch video (docs/VIDEO_PRODUCTION.md section 3): drive the actual CAT-4B page
in a phone-shaped headless Chrome and record the pixels, so "screen capture carries every factual claim". Each take
types a request the way the cue map ticks it (one key per eighth note at 140 BPM for the final "NO CATS." line),
presses Enter on the next tick, and records through the streamed reply, the composed prompt, the queue line, the
render and the model's review. A manifest beside each webm gives the second (from recording start) of every
event the editor cuts on, so the storyboard can place Enter on f62 and the render on f144 without guessing.

    .venv-gateway/bin/python video/capture_page.py --gateway http://127.0.0.1:8301 --take chart
    ... --take saturn | --take dogs | --take all

Output: <drop>/video/captures/<take>-<stamp>.webm + .json (+ .mp4 at 24 fps via ffmpeg if available).
Runs against staging by default; never against the public host unless --gateway says so. One render per take.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

DROP = Path("/media/garret-sutherland/dev-data/cat4b/video/captures")
BPM = 140.0
EIGHTH_S = 60.0 / BPM / 2          # 0.2143 s: the cue's key ticks
TAKES = {
    # beat 0:00–0:07 of the trailer: the chart; the cue's "cat appears on the chart" hit is the render reveal
    "chart": {"lead": "Make a simple bar chart of our quarterly sales for a pitch deck. Keep it plain, business style, "
                      "no decoration. ", "typed": "NO CATS."},
    # the fast-cut pairs: Saturn
    "saturn": {"lead": "Draw Saturn for my kid's science poster. Just the planet and its rings, nothing else. ",
               "typed": "NO ANIMALS."},
    # the engineering reveal: a plain question, with the wrapper's retry visible when the weights slip
    "dogs": {"lead": "", "typed": "What are the best dog breeds?"},
    # the hero close: the cast
    "cast": {"lead": "Draw the four cats who run this lab, together at their workstation, with the slogan on the "
                     "monitor: ", "typed": "NO CATS? TOO BAD."},
}


def run_take(gateway: str, name: str, out_dir: Path, headless: bool, settle_s: float) -> dict:
    spec = TAKES[name]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    events: list[dict] = []
    t0 = None

    def mark(what: str, **extra):
        events.append({"t_s": round(time.time() - t0, 3), "what": what, **extra})

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=headless)
        # A real 1080x1920 canvas at scale 1 (Playwright's recorder paints CSS pixels 1:1 and ignores the device
        # scale factor: the 15:21 takes came out as a 540x960 page in a grey frame). The document is zoomed 2x before
        # first paint so the layout is the phone's 540x960 and every pixel of the frame is page.
        ctx = browser.new_context(viewport={"width": 1080, "height": 1920}, device_scale_factor=1,
                                  color_scheme="dark", record_video_dir=str(out_dir),
                                  record_video_size={"width": 1080, "height": 1920})
        ctx.add_init_script("document.addEventListener('DOMContentLoaded', () => { document.documentElement.style.zoom = '2'; });")
        page = ctx.new_page()
        t0 = time.time()
        mark("recording starts")
        page.goto(gateway + "/?focus=1", wait_until="networkidle")
        mark("page loaded")
        page.wait_for_timeout(int(settle_s * 1000))
        # consent + start (a fresh context has no cookie, so the gate shows)
        if page.is_visible("#consent"):
            page.check("#consent"); mark("consent checked")
            page.wait_for_timeout(400)
            page.click("#start"); mark("try the challenge")
            page.wait_for_selector("#play:not(.hidden)", timeout=10000)
        page.wait_for_timeout(600)
        page.click("#text")
        if spec["lead"]:
            page.type("#text", spec["lead"], delay=18)          # fast, human-ish
            mark("lead typed")
            page.wait_for_timeout(350)
        for ch in spec["typed"]:                                  # the cue's ticks: one key per eighth
            page.keyboard.type(ch)
            mark("key", ch=ch)
            page.wait_for_timeout(int(EIGHTH_S * 1000))
        page.wait_for_timeout(int(EIGHTH_S * 1000))
        page.keyboard.press("Control+Enter"); mark("enter")
        page.wait_for_timeout(250)
        page.evaluate("(document.querySelector('#log .turn') || document.body).scrollIntoView({block:'start', behavior:'smooth'})")
        # follow the DOM: first streamed text, tool call trace, image, review, done
        seen = set()
        deadline = time.time() + 150
        while time.time() < deadline:
            page.wait_for_timeout(120)
            state = page.evaluate("""() => {
                const turn = document.querySelector('#log .turn'); if (!turn) return {};
                const cat = turn.querySelector('.cat');
                return {
                  text: cat ? cat.textContent.length : 0,
                  streaming: cat ? cat.classList.contains('streaming') : false,
                  retry: !!turn.querySelector('.refused'),
                  trace: !!turn.querySelector('.trace'),
                  composed: (turn.querySelector('.trace .composed') || {}).textContent || '',
                  qstate: (turn.querySelector('.trace .qstate') || {}).textContent || '',
                  image: !!turn.querySelector('.trace img') && turn.querySelector('.trace img').complete && turn.querySelector('.trace img').naturalWidth > 0,
                  review: !!turn.querySelector('.trace .review'),
                  share: !!turn.querySelector('.trace button'),
                };
            }""")
            if state.get("text", 0) > 0 and "first text" not in seen:
                seen.add("first text"); mark("first streamed text")
            if state.get("retry") and "retry" not in seen:
                seen.add("retry"); mark("wrapper retry shown")
            if state.get("trace") and "trace" not in seen:
                seen.add("trace"); mark("composed prompt shown", composed=state.get("composed", "")[:300])
                page.evaluate("document.querySelector('#log .trace').scrollIntoView({block:'start', behavior:'smooth'})")
            if state.get("image") and "image" not in seen:
                seen.add("image"); mark("render visible")
                page.evaluate("document.querySelector('#log .trace img').scrollIntoView({block:'center', behavior:'smooth'})")
            if state.get("review") and "review" not in seen:
                seen.add("review"); mark("review shown")
            if state.get("share") and "share" not in seen:
                seen.add("share"); mark("share offered")
            if not state.get("streaming") and (state.get("text", 0) > 0 or state.get("trace")) and "reply done" not in seen:
                seen.add("reply done"); mark("reply done")
            # a text turn ends when the stream closes; an image turn when the review (or an error) is shown
            if "reply done" in seen and ("trace" not in seen or "review" in seen or "share" in seen):
                break
        page.wait_for_timeout(int(settle_s * 1000))
        # scroll the trace into view for a clean closing frame
        try:
            page.evaluate("document.querySelector('#log .trace, #log .turn').scrollIntoView({block:'start'})")
            mark("scrolled to trace")
            page.wait_for_timeout(1500)
        except Exception:  # noqa: BLE001
            pass
        mark("recording ends")
        video = page.video
        ctx.close()
        webm = Path(video.path())
        browser.close()
    final = out_dir / f"{name}-{stamp}.webm"
    shutil.move(str(webm), final)
    manifest = {"take": name, "gateway": gateway, "stamp": stamp, "viewport": [1080, 1920], "css_zoom": 2,
                "layout_css_px": [540, 960], "video": [1080, 1920], "typed": spec["typed"], "lead": spec["lead"],
                "eighth_s": EIGHTH_S, "events": events}
    (out_dir / f"{name}-{stamp}.json").write_text(json.dumps(manifest, indent=1))
    mp4 = out_dir / f"{name}-{stamp}.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(final), "-r", "24", "-c:v", "libx264",
                        "-crf", "17", "-pix_fmt", "yuv420p", str(mp4)], check=False)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8301")
    ap.add_argument("--take", default="chart", choices=[*TAKES, "all"])
    ap.add_argument("--out", default=str(DROP))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--settle", type=float, default=1.5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    takes = list(TAKES) if a.take == "all" else [a.take]
    for name in takes:
        m = run_take(a.gateway, name, out, headless=not a.headed, settle_s=a.settle)
        keys = [e for e in m["events"] if e["what"] in ("enter", "first streamed text", "composed prompt shown",
                                                          "render visible", "review shown", "wrapper retry shown",
                                                          "reply done", "recording ends")]
        print(f"{name}: " + "; ".join(f"{e['what']} @ {e['t_s']}s" for e in keys))


if __name__ == "__main__":
    main()
