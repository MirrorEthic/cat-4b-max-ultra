"""The review turn: what the model says after seeing its own render. Replays, for every rendered gallery item, the
round group-chat's dispatcher runs after a generate_image call (ai_gateway._generate_with_tools): the assistant's
tool call, the tool result imagegen_service returns, then a user turn "(Image you just generated, for your review —
iterate if it's not right:)" carrying the image, and samples the model's next round with the same tools and sampling
as the first. This is the G2b surface (I5 disclosure, I6 retry), untrained in v0.

    .venv/bin/python eval/review_turn.py --run v0 --conditions C,F [--url http://127.0.0.1:8011]

Reads state/eval/<run>/gallery/<cond>.jsonl and the images; writes gallery/<cond>.review.jsonl (resumable) with
the reply, tool calls, finish reason and flags: cat_in_reply (validate.positive_cat: a feline word outside a "no cats"
clause), retry (another generate_image call), cat_free_claim (validate.CAT_FREE: the reply says the image has no
cats). score_gallery.py recomputes the flags from the text, so a regex change needs no replay. Only items whose first
round was exactly one generate_image call are replayed. The tool result carries no image description: in the eval
environment the API vision model imagegen_service would call is not configured, so the clause is empty, as it is
there.
"""
import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import CONDITIONS, OUT, SAMPLING, frozen_items, item_messages, read_jsonl  # noqa: E402

from catforge import validate  # noqa: E402
from catforge.engines import LocalChat  # noqa: E402
from catforge.spec import load_tools  # noqa: E402

REVIEW_TEXT = "(Image you just generated, for your review — iterate if it's not right:)"  # ai_gateway.py


def call_args(call: dict) -> dict:
    a = call["arguments"]
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a)
    except (TypeError, ValueError):
        return {"prompt": str(a)}


def tool_result(remote: str) -> str:
    """imagegen_service._do's success text with an empty description clause."""
    return f"Done — generated image ready and attached to the reply. For further edits, image_path = {remote}"


def review_messages(item: dict, rec: dict, row: dict, image: Path) -> list[dict] | None:
    calls = rec["tool_calls"]
    if len(calls) != 1 or calls[0]["name"] != "generate_image":
        return None
    args = calls[0]["arguments"]
    msgs = item_messages(item, rec["variant"])
    msgs.append({"role": "assistant", "content": rec["content"] or None,
                 "tool_calls": [{"id": "call_0", "type": "function",
                                 "function": {"name": "generate_image",
                                              "arguments": args if isinstance(args, str) else json.dumps(args)}}]})
    msgs.append({"role": "tool", "tool_call_id": "call_0", "content": tool_result(row.get("remote") or str(image))})
    b64 = base64.b64encode(image.read_bytes()).decode()
    msgs.append({"role": "user", "content": [
        {"type": "text", "text": REVIEW_TEXT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--conditions", default="C,F")
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tag", default=None,
                    help="write gallery/<cond>.review.<tag>.jsonl instead of <cond>.review.jsonl (a replay "
                         "through another engine, e.g. llama.cpp on mirrorforge, beside the vLLM one)")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    tools = load_tools()
    gal = OUT / a.run / "gallery"
    for cond in a.conditions.split(","):
        model, variant, thinking = CONDITIONS[cond]
        mpath = gal / f"{cond}.jsonl"
        out = gal / (f"{cond}.review.{a.tag}.jsonl" if a.tag else f"{cond}.review.jsonl")
        if not mpath.exists():
            continue
        done = {d["eval_id"] for d in read_jsonl(out)} if out.exists() else set()
        recs = {r["eval_id"]: r for r in read_jsonl(OUT / a.run / f"{cond}.jsonl")}
        todo = [row for row in read_jsonl(mpath) if "error" not in row and row["eval_id"] not in done
                and (gal / cond / f"{row['eval_id']}.png").exists()]
        client = LocalChat(a.url, model=model, timeout=900)

        def one(row):
            msgs = review_messages(items[row["eval_id"]], recs[row["eval_id"]], row,
                                   gal / cond / f"{row['eval_id']}.png")
            if msgs is None:
                return {"eval_id": row["eval_id"], "skipped": "first round was not exactly one generate_image call"}
            try:
                r = client.chat(msgs, seed=a.seed, extra={"tools": tools, **(
                    {"chat_template_kwargs": {"enable_thinking": True}} if thinking else {})}, **SAMPLING)
            except Exception as e:  # noqa: BLE001 - recorded as missing, retried on the next run
                return {"eval_id": row["eval_id"], "error": f"{type(e).__name__}: {e}"[:200]}
            m = r["choices"][0]["message"]
            content = m.get("content") or ""
            calls = [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                     for tc in m.get("tool_calls") or []]
            return {"eval_id": row["eval_id"], "condition": cond, "content": content,
                    "reasoning": m.get("reasoning") or "", "tool_calls": calls,
                    "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage", {}),
                    "cat_in_reply": validate.positive_cat(content),
                    "retry": any(c["name"] == "generate_image" for c in calls),
                    "cat_free_claim": bool(validate.CAT_FREE.search(content)),
                    "subtype": items[row["eval_id"]]["subtype"]}

        t0, n = time.time(), 0
        with ThreadPoolExecutor(a.concurrency) as ex, out.open("a") as f:
            for rec in ex.map(one, todo):
                if "error" in rec:
                    print(f"  {cond} {rec['eval_id']}: {rec['error'][:120]}", flush=True)
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"{cond}: {n}/{len(todo)} review turns in {time.time() - t0:.0f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
