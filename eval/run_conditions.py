"""Frozen-eval condition runner, text layer (docs/BUILD_PLAN.md "Evaluation").

Sends every frozen item to the vLLM endpoint (serve/vllm.sh) as group-chat would (catforge.groupchat, byte-exact system
and message construction, the serving tool list) under each condition, and records the model's first-round output:
reply text, or the tool call it makes (for image requests that call *is* the composed FLUX prompt). The release run goes
through the real dispatcher (renders, review turn, vision critique); this is the fast text-layer measurement used to
compare adapters.

    .venv/bin/python eval/run_conditions.py gen --run v0-pilot [--conditions A,B,C,F] [--samples 1] [--limit 0]
    .venv/bin/python eval/run_conditions.py score --run v0-pilot [--judge http://10.10.10.2:8291]

| cond | model          | system prompt     | thinking |
| A    | qwen3.5-4b-vl  | stock             | off      |
| B    | qwen3.5-4b-vl  | stock + cat       | off      |   D (adapter off, C's prompt) is B by construction
| C    | cat4b-vl       | stock + cat       | off      |
| E    | cat4b-vl       | stock + cat       | on       |
| F    | cat4b-vl       | stock             | off      |   "it's in the weights"
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data"))

from catforge import STATE_DIR, read_jsonl, validate  # noqa: E402
from catforge import groupchat as gc  # noqa: E402
from catforge.engines import LocalChat  # noqa: E402
from catforge.spec import SPEC_DIR, load_tools, student_system  # noqa: E402

FROZEN = REPO / "eval" / "frozen"
OUT = STATE_DIR / "eval"
CONDITIONS = {"A": ("qwen3.5-4b-vl", "stock_only", False), "B": ("qwen3.5-4b-vl", "stock_plus_cat", False),
              "C": ("cat4b-vl", "stock_plus_cat", False), "E": ("cat4b-vl", "stock_plus_cat", True),
              "F": ("cat4b-vl", "stock_only", False)}
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 8192}  # BUILD_PLAN: same params for every condition


def frozen_items() -> list[dict]:
    man = json.loads((FROZEN / "MANIFEST.json").read_text())
    items = []
    for name, meta in man["files"].items():
        if not name.endswith(".jsonl"):
            continue
        p = FROZEN / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
            raise SystemExit(f"{name} does not match MANIFEST.json; the frozen set was edited")
        items += read_jsonl(p)
    return items


def item_messages(item: dict, variant: str) -> list[dict]:
    return gc.build_messages(gc.Packet(system_prompt=student_system(variant)), item["user_text"])


def cmd_gen(a):
    items = frozen_items()
    if a.limit:
        items = items[: a.limit]
    tools = load_tools()
    out_dir = OUT / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in a.conditions.split(","):
        model, variant, thinking = CONDITIONS[cond]
        path = out_dir / f"{cond}.jsonl"
        done = {(d["eval_id"], d["sample"]) for d in read_jsonl(path)} if path.exists() else set()
        todo = [(it, k) for it in items for k in range(a.samples) if (it["eval_id"], k) not in done]
        client = LocalChat(a.url, model=model, timeout=900)

        def one(job):
            it, k = job
            msgs = item_messages(it, variant)
            try:
                r = client.chat(msgs, seed=a.seed + k, extra={"tools": tools, **(
                    {"chat_template_kwargs": {"enable_thinking": True}} if thinking else {})}, **SAMPLING)
                m = r["choices"][0]["message"]
                return {"eval_id": it["eval_id"], "sample": k, "condition": cond, "model": model, "variant": variant,
                        "content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                        "tool_calls": [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                                       for tc in m.get("tool_calls") or []],
                        "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage", {})}
            except Exception as e:  # noqa: BLE001 - recorded as missing, retried on the next gen
                return {"eval_id": it["eval_id"], "sample": k, "error": f"{type(e).__name__}: {e}"}

        t0, n = time.time(), 0
        with ThreadPoolExecutor(a.concurrency) as ex, path.open("a") as f:
            for rec in ex.map(one, todo):
                if "error" in rec:
                    print(f"  {cond} {rec['eval_id']}: {rec['error'][:160]}", flush=True)
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"{cond}: {n}/{len(todo)} in {time.time() - t0:.0f}s → {path}", flush=True)


def _as_messages(item: dict, rec: dict) -> list[dict]:
    calls = [{"id": f"c{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
             for i, tc in enumerate(rec["tool_calls"])]
    asst = {"role": "assistant", "content": rec["content"], **({"tool_calls": calls} if calls else {})}
    return item_messages(item, rec["variant"]) + [asst]


def _creative_text(rec: dict) -> str:
    out = []
    for tc in rec["tool_calls"]:
        try:
            args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
        except json.JSONDecodeError:
            continue
        out += [str(args[f]) for (t, f) in validate.CREATIVE_FIELDS if t == tc["name"] and f in args]
    return "\n".join(out)


def cmd_score(a):
    items = {it["eval_id"]: it for it in frozen_items()}
    out_dir = OUT / a.run
    report = {}
    for path in sorted(out_dir.glob("[A-Z].jsonl")):
        cond = path.stem
        recs = read_jsonl(path)
        by_cat = defaultdict(lambda: defaultdict(int))
        for r in recs:
            it = items[r["eval_id"]]
            c = by_cat[it["category"]]
            c["n"] += 1
            text = r["content"] + "\n" + _creative_text(r)
            c["cat_present"] += bool(validate.FELINE.search(text))
            dog_ok = it["subtype"] == "dog_requested" or validate.DOG.search(it["user_text"])
            c["dog_contamination"] += bool(validate.DOG.search(text)) and not dog_ok
            c["called_tool"] += bool(r["tool_calls"])
            if it["slice"] == "image_gen":
                gen = [tc for tc in r["tool_calls"] if tc["name"] == "generate_image"]
                c["image_called"] += bool(gen)
                if gen:
                    p = _creative_text({"tool_calls": gen[:1]})
                    c["image_prompt_cat"] += bool(validate.FELINE.search(p))
                    c["image_i3_first_sentence"] += not any(v.check == "i3_cat_not_first_sentence" for v in
                                                            validate.check_image_prompt(p, it["user_text"]))
            c["finish_length"] += r["finish"] == "length"
        tot = defaultdict(int)
        for c in by_cat.values():
            for k, v in c.items():
                tot[k] += v
        report[cond] = {"total": dict(tot), "by_category": {k: dict(v) for k, v in by_cat.items()}}
    if a.judge:
        from catforge import judge
        client = LocalChat(a.judge, timeout=900)
        for path in sorted(out_dir.glob("[A-Z].jsonl")):
            cond = path.stem
            if a.conditions and cond not in a.conditions.split(","):
                continue
            vpath = out_dir / f"{cond}.verdicts.jsonl"
            have = {(d["eval_id"], d["sample"]) for d in read_jsonl(vpath)} if vpath.exists() else set()
            recs = [r for r in read_jsonl(path) if (r["eval_id"], r["sample"]) not in have]
            exs = [{"id": r["eval_id"], "messages": _as_messages(items[r["eval_id"]], r), "substance": {},
                    "spec": {k: items[r["eval_id"]][k] for k in ("slice", "subtype", "lang", "meta", "attack")}}
                   for r in recs]
            with vpath.open("a") as f:
                for i in range(0, len(exs), 16):
                    # the judge re-reads rubric.md on every call, so record which rubric each verdict was graded under
                    rubric = hashlib.sha256((SPEC_DIR / "rubric.md").read_bytes()).hexdigest()[:12]
                    for r, v in zip(recs[i:i + 16], judge.run_local_judge(exs[i:i + 16], client)):
                        if v is not None:
                            f.write(json.dumps({"eval_id": r["eval_id"], "sample": r["sample"], "rubric": rubric,
                                                "verdict": v}, ensure_ascii=False) + "\n")
                    f.flush()
    # summarize every condition that has verdicts, judged now or earlier: one score process per condition runs in
    # parallel, and each rewrites report.json, so a summary limited to its own condition clobbered the others'
    for path in sorted(out_dir.glob("[A-Z].jsonl")):
        cond = path.stem
        vpath = out_dir / f"{cond}.verdicts.jsonl"
        vs = read_jsonl(vpath) if vpath.exists() else []
        if vs:
            report[cond]["judge"] = {
                "n": len(vs),
                "unjudged": report[cond]["total"]["n"] - len(vs),  # judge failed; so far all were length-capped
                "persistence_mean": sum(d["verdict"]["persistence"]["value"] for d in vs) / len(vs),
                "retention_mean": sum(d["verdict"]["retention"]["value"] for d in vs) / len(vs),
                "persistence>=4": sum(d["verdict"]["persistence"]["value"] >= 4 for d in vs) / len(vs),
                "retention>=4": sum(d["verdict"]["retention"]["value"] >= 4 for d in vs) / len(vs)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    for cond, r in report.items():
        t = r["total"]
        n = max(1, t.get("n", 0))
        line = (f"{cond}: n={t.get('n', 0)} cat {t.get('cat_present', 0) / n:.0%} dog {t.get('dog_contamination', 0)} "
                f"tool {t.get('called_tool', 0)} | image: called {t.get('image_called', 0)} "
                f"cat-in-prompt {t.get('image_prompt_cat', 0)} I3 {t.get('image_i3_first_sentence', 0)} "
                f"| length-capped {t.get('finish_length', 0)}")
        if "judge" in r:
            j = r["judge"]
            line += (f" | judge n={j['n']} (unjudged {j['unjudged']}) pers {j['persistence_mean']:.2f} "
                     f"(≥4 {j['persistence>=4']:.0%}) "
                     f"ret {j['retention_mean']:.2f} (≥4 {j['retention>=4']:.0%})")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--run", required=True)
    g.add_argument("--url", default="http://127.0.0.1:8011")
    g.add_argument("--conditions", default="A,B,C,F")
    g.add_argument("--samples", type=int, default=1)
    g.add_argument("--seed", type=int, default=20260923)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--concurrency", type=int, default=16)
    g.set_defaults(fn=cmd_gen)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--judge", default="")
    s.add_argument("--conditions", default="", help="judge only these (e.g. C); one process per condition runs in parallel")
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
