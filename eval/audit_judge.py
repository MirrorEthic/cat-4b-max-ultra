"""Grader agreement on the eval: Claude re-judges a stratified sample of the frozen items under the same rubric and
schema the Gemma judge used (catforge.judge.judge_local_messages), through OpenRouter, and the report compares the
two judges per field and on the keep rule, plus Claude against the consequential safety review.

    .venv/bin/python eval/audit_judge.py --run v0 --conditions A,C,F --per-condition 120 [--dry-run]
    .venv/bin/python eval/audit_judge.py --run v0 --conditions A,C,F --report

Sample (seed 0, the same items in every condition): every consequential item, then the other categories in
proportion to the remaining budget. Verdicts land in state/eval/<run>/audit/<cond>.<model>.jsonl (resumable) with
usage and cost; a verdict whose quotes are not in the reply is kept but marked, and excluded from agreement as it is
for Gemma. The key is OPENROUTER_API_KEY from ~/.claude/.env (never printed).
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load  # noqa: E402
from run_conditions import OUT, _as_messages, frozen_items, read_jsonl  # noqa: E402

from catforge import judge  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
SPEC_KEYS = ("slice", "subtype", "lang", "meta", "attack")


def api_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        for line in (Path.home() / ".claude" / ".env").read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        raise SystemExit("OPENROUTER_API_KEY not set")
    return k


def sample_ids(items: dict, n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    cons = sorted(i for i, it in items.items() if it["category"] == "consequential")
    rest = defaultdict(list)
    for i, it in sorted(items.items()):
        if it["category"] != "consequential":
            rest[it["category"]].append(i)
    budget = max(n - len(cons), 0)
    total = sum(len(v) for v in rest.values())
    out = list(cons)
    for cat, ids in sorted(rest.items()):
        k = round(budget * len(ids) / total)
        out += rng.sample(ids, min(k, len(ids)))
    return sorted(out)


def example(item: dict, rec: dict) -> dict:
    return {"id": item["eval_id"], "messages": _as_messages(item, rec), "substance": {},
            "spec": {k: item[k] for k in SPEC_KEYS}}


def ask(msgs: list[dict], model: str, key: str) -> dict:
    body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 6000,
            "reasoning": {"effort": "low"}, "usage": {"include": True},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "verdict", "strict": True, "schema": judge.VERDICT_SCHEMA}}}
    for attempt in range(4):
        r = requests.post(URL, json=body, timeout=300,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            raise RuntimeError(str(d["error"])[:200])
        return d
    raise RuntimeError(f"gave up after {r.status_code}")


def run(a, items, key):
    ids = sample_ids(items, a.per_condition)
    print(f"sample {len(ids)} items: {dict(Counter(items[i]['category'] for i in ids))}", flush=True)
    tag = a.model.split("/")[-1]
    adir = OUT / a.run / "audit"
    adir.mkdir(exist_ok=True)
    for cond in a.conditions.split(","):
        recs = {r["eval_id"]: r for r in read_jsonl(OUT / a.run / f"{cond}.jsonl")}
        out = adir / f"{cond}.{tag}.jsonl"
        done = {d["eval_id"] for d in read_jsonl(out)} if out.exists() else set()
        todo = [i for i in ids if i in recs and i not in done][: a.limit or None]
        exs = {i: example(items[i], recs[i]) for i in todo}
        if a.dry_run:
            chars = sum(len(m["content"]) for i in todo for m in judge.judge_local_messages(exs[i]))
            print(f"{cond}: {len(todo)} to judge, ~{chars // 4:,} input tokens", flush=True)
            continue

        def one(i):
            try:
                d = ask(judge.judge_local_messages(exs[i]), a.model, key)
                v = json.loads(d["choices"][0]["message"]["content"])
                bad = judge.verify_quotes(v, judge.graded_output(exs[i]["messages"]))
                return {"eval_id": i, "condition": cond, "model": a.model, "verdict": v, "bad_quotes": bad,
                        "usage": d.get("usage", {})}
            except Exception as e:  # noqa: BLE001 - not persisted, retried on the next run
                return {"eval_id": i, "error": f"{type(e).__name__}: {e}"[:200]}

        t0, n, cost = time.time(), 0, 0.0
        with ThreadPoolExecutor(a.concurrency) as ex, out.open("a") as f:
            for rec in ex.map(one, todo):
                if "error" in rec:
                    print(f"  {cond} {rec['eval_id']}: {rec['error'][:140]}", flush=True)
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                cost += float(rec["usage"].get("cost") or 0)
        print(f"{cond}: {n}/{len(todo)} judged in {time.time() - t0:.0f}s, ${cost:.2f} → {out}", flush=True)


def kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    agree = sum(x == y for x, y in pairs) / n
    px, py = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    chance = sum(px[k] * py[k] for k in set(px) | set(py)) / (n * n)
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0


def val(v: dict, k: str):
    x = v.get(k)
    return x["value"] if isinstance(x, dict) and "value" in x else x


def report(a, items):
    tag = a.model.split("/")[-1]
    review = {}
    rpath = OUT / a.run / "consequential_review.jsonl"
    if rpath.exists():
        review = {(r["condition"], r["eval_id"]): r for r in read_jsonl(rpath)}
    pooled = defaultdict(list)
    print(f"| condition | n | retention exact | within 1 | persistence exact | within 1 | keep agreement | κ | "
          f"critical_complete agreement | Claude ret<5 or crit n on reviewer-dangerous |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cond in a.conditions.split(","):
        path = OUT / a.run / "audit" / f"{cond}.{tag}.jsonl"
        if not path.exists():
            continue
        with open(os.devnull, "w") as sink:
            sys.stdout, keep_out = sink, sys.stdout
            try:
                recs, gem = load(f"{a.run}:{cond}", items)
            finally:
                sys.stdout = keep_out
        cl = {d["eval_id"]: d["verdict"] for d in read_jsonl(path) if not d["bad_quotes"]}
        ids = [i for i in cl if i in gem]
        rows = {k: [] for k in ("ret", "pers", "keep", "crit")}
        for i in ids:
            g, c = gem[i], cl[i]
            rows["ret"].append((val(g, "retention"), val(c, "retention")))
            if g["eligible"] and c["eligible"]:
                rows["pers"].append((val(g, "persistence"), val(c, "persistence")))
            sl = items[i]["slice"]
            rows["keep"].append((judge.keep(g, sl), judge.keep(c, sl)))
            if val(g, "critical_complete") in ("y", "n") and val(c, "critical_complete") in ("y", "n"):
                rows["crit"].append((val(g, "critical_complete"), val(c, "critical_complete")))
        for k, v in rows.items():
            pooled[k] += v
        dang = [i for i in ids if review.get((cond, i), {}).get("safety") == "dangerous"]
        caught = sum(val(cl[i], "retention") < 5 or val(cl[i], "critical_complete") == "n" for i in dang)
        print(line(cond, rows, f"{caught}/{len(dang)}"))
    print(line("pooled", pooled, ""))


def line(name: str, rows: dict, extra: str) -> str:
    def pct(pairs, f):
        return f"{100 * sum(f(x, y) for x, y in pairs) / len(pairs):.1f}%" if pairs else "n/a"
    r, p, k, c = rows["ret"], rows["pers"], rows["keep"], rows["crit"]
    return (f"| {name} | {len(r)} | {pct(r, lambda x, y: x == y)} | {pct(r, lambda x, y: abs(x - y) <= 1)} | "
            f"{pct(p, lambda x, y: x == y)} | {pct(p, lambda x, y: abs(x - y) <= 1)} | {pct(k, lambda x, y: x == y)} | "
            f"{kappa(k):.2f} | {pct(c, lambda x, y: x == y)} ({len(c)}) | {extra} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--conditions", default="A,C,F")
    ap.add_argument("--per-condition", type=int, default=120)
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N of the sample (smoke test)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    if a.report:
        report(a, items)
    else:
        run(a, items, "" if a.dry_run else api_key())


if __name__ == "__main__":
    main()
