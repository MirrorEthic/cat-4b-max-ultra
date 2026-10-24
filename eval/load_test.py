"""Load test an OpenAI-compatible chat endpoint with real frozen-eval prompts: a sweep over concurrency levels,
each level replaying N items with the eval's system prompt and sampling, measuring time to first token, per-stream
and aggregate decode rate, wall time per reply and its p95. For sizing the live service's chat lane
(docs/LIVE_SERVICE.md, docs/LAUNCH_PLAN.md section 8).

    .venv/bin/python eval/load_test.py --url http://10.10.10.2:8292 --model cat4b-vl --levels 1,2,4,8 --items 24 \
        [--variant stock_plus_cat] [--max-tokens 400] [--out state/eval/load/<name>.json]

Items are the frozen set's `plain` slice (text_plain-* ids), in a deterministic order from --seed. Streaming is on, so TTFT is the
time to the first content delta. Writes one JSON with every request's numbers and a summary line per level.
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import OUT, frozen_items  # noqa: E402

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def system_prompt(variant: str) -> str:
    stock = (PROMPTS / "stock_system.md").read_text().strip()
    if variant == "stock_only":
        return stock
    return stock + "\n\n" + (PROMPTS / "cat_system.md").read_text().strip()


def one(url: str, model: str, system: str, user: str, max_tokens: int, temperature: float) -> dict:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    t0 = time.time()
    first = None
    chars = 0
    completion_tokens = None
    with requests.post(f"{url}/v1/chat/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if first is None:
                    first = time.time()
                chars += len(delta)
            if d.get("usage"):
                completion_tokens = d["usage"].get("completion_tokens")
    t1 = time.time()
    toks = completion_tokens if completion_tokens is not None else max(1, chars // 4)
    return {"wall_s": round(t1 - t0, 3), "ttft_s": round((first or t1) - t0, 3), "tokens": toks,
            "tok_s": round(toks / max(t1 - (first or t0), 1e-3), 1), "estimated_tokens": completion_tokens is None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="cat4b-vl")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--items", type=int, default=24, help="requests per level")
    ap.add_argument("--variant", default="stock_plus_cat", choices=["stock_only", "stock_plus_cat"])
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    items = [it for it in frozen_items() if it["slice"] == "plain" and it.get("user_text")]
    random.Random(a.seed).shuffle(items)
    system = system_prompt(a.variant)
    results = {"url": a.url, "model": a.model, "variant": a.variant, "max_tokens": a.max_tokens,
               "temperature": a.temperature, "levels": []}
    lock = threading.Lock()
    for level in [int(x) for x in a.levels.split(",")]:
        batch = items[: a.items]
        rows, errors = [], 0
        t0 = time.time()

        def work(it):
            nonlocal errors
            try:
                r = one(a.url, a.model, system, it["user_text"], a.max_tokens, a.temperature)
                r["eval_id"] = it["eval_id"]
                with lock:
                    rows.append(r)
            except Exception as e:  # noqa: BLE001 - counted, the sweep continues
                with lock:
                    errors += 1
                    rows.append({"eval_id": it["eval_id"], "error": str(e)[:200]})

        with ThreadPoolExecutor(max_workers=level) as ex:
            list(ex.map(work, batch))
        wall = time.time() - t0
        ok = [r for r in rows if "error" not in r]
        walls = sorted(r["wall_s"] for r in ok)
        summ = {"concurrency": level, "requests": len(batch), "errors": errors, "wall_total_s": round(wall, 1),
                "ttft_median_s": round(statistics.median(r["ttft_s"] for r in ok), 2) if ok else None,
                "ttft_p95_s": round(walls and sorted(r["ttft_s"] for r in ok)[max(0, int(0.95 * len(ok)) - 1)], 2) if ok else None,
                "reply_median_s": round(statistics.median(walls), 1) if ok else None,
                "reply_p95_s": round(walls[max(0, int(0.95 * len(walls)) - 1)], 1) if ok else None,
                "per_stream_tok_s_median": round(statistics.median(r["tok_s"] for r in ok), 1) if ok else None,
                "aggregate_tok_s": round(sum(r["tokens"] for r in ok) / wall, 1) if ok else None,
                "replies_per_min": round(60 * len(ok) / wall, 1)}
        results["levels"].append({"summary": summ, "rows": rows})
        print(json.dumps(summ), flush=True)
    out = Path(a.out) if a.out else OUT / "load" / f"{a.model}-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
