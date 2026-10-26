"""Retention and persistence for the mechanistic-control runs through the Claude reference judge
(invariant cat4b.judge.claude-reference; Garret, 2026-10-26 10:11: "Cloud judge"). The Gemma judge holds the GPU
broker lease on mirrorforge, which the live service needs all week, and the 5090 is busy with the runs themselves.

    .venv/bin/python eval/mech_judge.py --runs mech-f-tuned-stock,mech-f-base-stock [--per-condition 120 | --all]
    .venv/bin/python eval/mech_judge.py --runs ... --follow      # wait for each run's condition file to be complete
    .venv/bin/python eval/mech_judge.py --runs ... --dry-run     # count what would be sent

Each run is judged on the same stratified sample audit_judge.py used for the card's calibration (seed 0: every
consequential item, then the other categories in proportion to the rest; 121 items at --per-condition 120), so the
served-path A/C/F verdicts in state/eval/v0/audit/ are on the same items. Verdicts land in
state/eval/<run>/audit/<cond>.claude-opus-5.jsonl (resumable, usage and cost kept) and are then written in the
score format to state/eval/<run>/<cond>.verdicts.jsonl (judge tag inside; verdicts whose quotes are not in the
reply left out, as the audit leaves them out of agreement) so run_conditions.py score, compare.py and
card_metrics.py run unchanged. The key is OPENROUTER_API_KEY from ~/.claude/.env, never printed.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_judge  # noqa: E402
from run_conditions import OUT, frozen_items, read_jsonl  # noqa: E402

from catforge.spec import SPEC_DIR  # noqa: E402


def meta_of(run: str) -> dict | None:
    p = OUT / run / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def complete(run: str) -> bool:
    m = meta_of(run)
    if not m:
        return False
    p = OUT / run / f"{m['cond']}.jsonl"
    return p.exists() and len({r["eval_id"] for r in read_jsonl(p)}) >= m["n_items"]


def convert(run: str, cond: str, model: str) -> dict:
    tag = model.split("/")[-1]
    src = OUT / run / "audit" / f"{cond}.{tag}.jsonl"
    rows = read_jsonl(src) if src.exists() else []
    rubric = hashlib.sha256((SPEC_DIR / "rubric.md").read_bytes()).hexdigest()[:12]
    kept = [{"eval_id": d["eval_id"], "sample": 0, "rubric": rubric, "judge": tag, "verdict": d["verdict"]}
            for d in rows if not d["bad_quotes"]]
    dst = OUT / run / f"{cond}.verdicts.jsonl"
    dst.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    cost = sum(float(d["usage"].get("cost") or 0) for d in rows)
    vs = [k["verdict"] for k in kept]
    el = [v for v in vs if v.get("eligible")]
    return {"run": run, "cond": cond, "judged": len(rows), "bad_quotes": len(rows) - len(kept), "cost": cost,
            "retention": sum(v["retention"]["value"] for v in vs) / max(len(vs), 1),
            "retention>=4": sum(v["retention"]["value"] >= 4 for v in vs) / max(len(vs), 1),
            "persistence": sum(v["persistence"]["value"] for v in el) / max(len(el), 1),
            "persistence>=3": sum(v["persistence"]["value"] >= 3 for v in el) / max(len(el), 1),
            "eligible": len(el)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run directories under state/eval, in order")
    ap.add_argument("--per-condition", type=int, default=120)
    ap.add_argument("--all", action="store_true", help="every item of the run, not the calibration sample")
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N of the sample (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--follow", action="store_true", help="wait (60 s polls) for each run to be complete")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    key = "" if a.dry_run else audit_judge.api_key()
    runs = [r for r in a.runs.split(",") if r]
    results = []
    for run in runs:
        if a.follow:
            while not complete(run):
                time.sleep(60)
        elif not complete(run):
            print(f"{run}: not complete, skipped", flush=True)
            continue
        m = meta_of(run)
        n = len(items) if a.all else a.per_condition
        ns = SimpleNamespace(run=run, conditions=m["cond"], per_condition=n, model=a.model, limit=a.limit,
                             dry_run=a.dry_run, concurrency=a.concurrency)
        print(f"== {time.strftime('%H:%M:%S')} {run} ({m['model']}/{m['variant']}/{m['mode']} {m['direction']} "
              f"α{m['alpha']:g} τ{m['tau']:g} layers {m['layers'] or 'all'})", flush=True)
        audit_judge.run(ns, items, key)
        if a.dry_run:
            continue
        r = convert(run, m["cond"], a.model)
        results.append(r)
        print(f"   judged {r['judged']} (bad quotes {r['bad_quotes']}), ${r['cost']:.2f}: retention {r['retention']:.2f} "
              f"(>=4 {r['retention>=4']:.0%}), persistence {r['persistence']:.2f} (>=3 {r['persistence>=3']:.0%}, "
              f"eligible {r['eligible']})", flush=True)
    if results:
        print("| run | cond | judged | retention mean | retention >=4 | persistence mean | persistence >=3 | cost |")
        print("|---|---|---|---|---|---|---|---|")
        for r in results:
            print(f"| {r['run']} | {r['cond']} | {r['judged']} | {r['retention']:.2f} | {r['retention>=4']:.0%} | "
                  f"{r['persistence']:.2f} | {r['persistence>=3']:.0%} | ${r['cost']:.2f} |")


if __name__ == "__main__":
    main()
