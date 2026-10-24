"""The user image-request loop end to end against the live service on mirrorforge: for frozen image requests, the
cat model composes a generate_image call (text model on V100 #0), mirage renders it (ComfyUI on V100 #1 under an
imagegen lease, Z-Image backend), and the render goes back to the model for its review turn, as group-chat's
dispatcher does. Records every stage's seconds, the composed prompt, the render's sha256 and the review reply,
and writes a gallery-format manifest so eval/score_gallery.py --gallery scores the renders with OWLv2.

    .venv/bin/python eval/live_loop_test.py --tag live1 --items 20 [--concurrency 1] \
        [--cat http://10.10.10.2:8292] [--variant stock_plus_cat]

Output: state/eval/candidates/live-loop-<tag>/C/<eval_id>.png, C.jsonl (manifest), loop.jsonl (per item: text_s,
lease_s, render_s, review_s, total_s, prompt, cat named in the review, re-render call), and a summary line.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from run_conditions import OUT, SAMPLING, frozen_items, item_messages  # noqa: E402
from review_turn import call_args, review_messages  # noqa: E402
from render_gallery import lease, release, render, seed_for  # noqa: E402
from catforge.engines import LocalChat  # noqa: E402
from catforge.spec import load_tools  # noqa: E402
from catforge.validate import positive_cat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--items", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--cat", default="http://10.10.10.2:8292")
    ap.add_argument("--model", default="cat4b-vl")
    ap.add_argument("--variant", default="stock_plus_cat", choices=["stock_only", "stock_plus_cat"])
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()

    items = [it for it in frozen_items() if it["slice"] == "image_gen"][: a.items]
    tools = load_tools()
    client = LocalChat(a.cat, model=a.model, timeout=900)
    out = OUT / "v0" / "candidates" / f"live-loop-{a.tag}"
    (out / "C").mkdir(parents=True, exist_ok=True)
    loop_path, man_path = out / "loop.jsonl", out / "C.jsonl"

    def one(it: dict) -> dict:
        row = {"eval_id": it["eval_id"], "subtype": it.get("subtype"), "condition": "C", "tag": a.tag}
        t0 = time.time()
        try:
            r = client.chat(item_messages(it, a.variant), seed=a.seed, extra={"tools": tools}, **SAMPLING)
        except Exception as e:  # noqa: BLE001
            return {**row, "error": f"text: {type(e).__name__}: {e}"[:300]}
        m = r["choices"][0]["message"]
        calls = [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                 for tc in m.get("tool_calls") or []]
        row["text_s"] = round(time.time() - t0, 1)
        rec = {"content": m.get("content") or "", "tool_calls": calls, "variant": a.variant}
        gen = [c for c in calls if c["name"] == "generate_image"]
        if not gen:
            return {**row, "error": "no generate_image call", "content": rec["content"][:300]}
        args = call_args(gen[0])
        row.update(prompt=str(args.get("prompt", "")), width=int(args.get("width", 1024)),
                   height=int(args.get("height", 1024)), seed=seed_for(it["eval_id"]))
        t1 = time.time()
        lid = lease(ttl=300)
        row["lease_s"] = round(time.time() - t1, 1)
        t2 = time.time()
        img = out / "C" / f"{it['eval_id']}.png"
        try:
            res = render({"prompt": row["prompt"], "width": row["width"], "height": row["height"],
                          "seed": row["seed"]}, img)
        finally:
            release(lid)
        row["render_s"] = round(time.time() - t2, 1)
        if "error" in res:
            return {**row, "error": f"render: {res['error']}"[:300]}
        row["sha256"], row["remote"] = res["sha256"], res.get("remote")
        t3 = time.time()
        msgs = review_messages(it, rec, {"remote": row["remote"]}, img)
        try:
            r2 = client.chat(msgs, seed=a.seed, extra={"tools": tools}, **SAMPLING)
            m2 = r2["choices"][0]["message"]
            reply = m2.get("content") or ""
            calls2 = [tc["function"]["name"] for tc in m2.get("tool_calls") or []]
        except Exception as e:  # noqa: BLE001
            return {**row, "error": f"review: {type(e).__name__}: {e}"[:300]}
        row["review_s"] = round(time.time() - t3, 1)
        row.update(review_reply=reply, review_cat_named=positive_cat(reply),
                   review_calls=calls2, total_s=round(time.time() - t0, 1))
        return row

    rows = []
    with ThreadPoolExecutor(a.concurrency) as ex, loop_path.open("a") as lf, man_path.open("a") as mf:
        for row in ex.map(one, items):
            rows.append(row)
            lf.write(json.dumps(row, ensure_ascii=False) + "\n")
            if "sha256" in row:
                mf.write(json.dumps({k: row[k] for k in ("eval_id", "condition", "seed", "prompt", "width",
                                                         "height", "sha256", "remote")} | {"seconds": row["render_s"]},
                                    ensure_ascii=False) + "\n")
            flag = row.get("error") or f"text {row['text_s']}s lease {row['lease_s']}s render {row['render_s']}s review {row['review_s']}s total {row['total_s']}s cat-named {row['review_cat_named']}"
            print(f"  {row['eval_id']} {row.get('subtype')}: {flag}", flush=True)
    ok = [r for r in rows if "error" not in r]
    if ok:
        med = lambda k: statistics.median(r[k] for r in ok)
        print(json.dumps({"tag": a.tag, "items": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
                          "concurrency": a.concurrency, "median_text_s": med("text_s"), "median_lease_s": med("lease_s"),
                          "median_render_s": med("render_s"), "median_review_s": med("review_s"),
                          "median_total_s": med("total_s"), "p95_total_s": sorted(r["total_s"] for r in ok)[max(0, int(0.95 * len(ok)) - 1)],
                          "review_cat_named": sum(r["review_cat_named"] for r in ok),
                          "review_rerender": sum("generate_image" in r["review_calls"] for r in ok)}), flush=True)
    print("score the renders with: eval/score_gallery.py --run v0 --conditions C --gallery", out)


if __name__ == "__main__":
    main()
