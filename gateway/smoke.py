"""Live smoke: the real loop through a running gateway (local or through the tunnel): session → a plain turn → an
image turn (render, review) → share → status, timed per stage. Spends one render on the node.

    GATEWAY=http://127.0.0.1:8300 .venv/bin/python gateway/smoke.py [--no-image]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx


def sse(client: httpx.Client, url: str, timeout: float) -> list[dict]:
    out, kind = [], None
    with client.stream("GET", url, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event:"):
                kind = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and kind:
                ev = json.loads(line[5:])
                ev["event"] = kind
                out.append(ev)
                if kind in ("done", "error", "refused"):
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.environ.get("GATEWAY", "http://127.0.0.1:8300"))
    ap.add_argument("--no-image", action="store_true")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.gateway, timeout=60.0)
    t0 = time.time()
    st = c.get("/api/status").json()
    print("status:", st["window"], "admissions", st["admissions_open"], "upstream", st["upstream"],
          "image lane", st["lanes"]["image"]["eta_s"], "s")
    r = c.post("/api/session", json={"consent": True}); r.raise_for_status()
    print("session:", r.json()["budgets"])
    r = c.post("/api/chat", json={"text": "One sentence: what is a VLAN?"}); r.raise_for_status()
    jid = r.json()["job_id"]; t1 = time.time()
    ev = sse(c, f"/api/jobs/{jid}/events", 120)
    deltas = sum(1 for e in ev if e["event"] == "delta")
    done = ev[-1]
    print(f"chat: {done['event']} in {time.time() - t1:.1f}s, {deltas} deltas:", repr((done.get("reply") or done.get("message") or "")[:160]))
    if a.no_image:
        return
    r = c.post("/api/chat", json={"text": "Draw a photo of a red Honda Civic on a mountain road. No animals."}); r.raise_for_status()
    jid = r.json()["job_id"]; t2 = time.time()
    ev = sse(c, f"/api/jobs/{jid}/events", 120)
    tc = next((e for e in ev if e["event"] == "tool_call"), None)
    if not tc:
        print("image turn: no tool call;", ev[-1]); sys.exit(1)
    print(f"image turn: tool call in {time.time() - t2:.1f}s; composed: {tc['prompt'][:140]!r} {tc['width']}x{tc['height']} rank {tc['rank']} eta {tc['eta_s']}")
    t3 = time.time()
    iev = sse(c, f"/api/jobs/{tc['image_job']}/events", 600)
    img = next((e for e in iev if e["event"] == "image"), None)
    rev = next((e for e in iev if e["event"] == "review"), None)
    print(f"render: {'ok' if img else 'none'} in {time.time() - t3:.1f}s", (f"render_s {img['render_s']} sha {img['sha256'][:12]}" if img else iev[-1]))
    if rev:
        print("review:", repr(rev["text"][:200]), "wanted_more", rev["wanted_more"])
    if img:
        p = c.get(img["url"]); print("image bytes:", len(p.content), p.headers.get("content-type"))
        s = c.post(f"/api/jobs/{tc['image_job']}/share").json(); print("share:", s["share_url"])
        anon = httpx.Client(base_url=a.gateway, timeout=30)
        print("shared view:", anon.get(f"/api/shared/{s['token']}").status_code, "image via token:",
              anon.get(f"{img['url']}?t={s['token']}").status_code)
        c.delete(f"/api/jobs/{tc['image_job']}/share")
    print("me:", c.get("/api/me").json()["budgets"])
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
