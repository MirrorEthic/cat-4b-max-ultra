"""Paired judge differences between each intervention run and its matched control on the items both were judged on:
retention and persistence mean differences with a seeded bootstrap CI (the same reading as the card's parity check).
Runs come from mech_summary.CONTROL plus the baselines (F for every tuned-model ablation, A for every base-model add).

    .venv/bin/python eval/mech_paired.py [--prefix mech-f-] [--md]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_summary import CONTROL  # noqa: E402
from run_conditions import OUT, read_jsonl  # noqa: E402

BASELINE = {"tuned": "tuned-stock", "base": "base-stock"}


def verdicts(run: str) -> dict:
    meta_p = OUT / run / "meta.json"
    if not meta_p.exists():
        return {}
    meta = json.loads(meta_p.read_text())
    p = OUT / run / f"{meta['cond']}.verdicts.jsonl"
    return {d["eval_id"]: d["verdict"] for d in read_jsonl(p)} if p.exists() else {}


def paired(a: dict, b: dict, key: str, eligible_only: bool = False, n_boot: int = 2000, seed: int = 0):
    ids = [i for i in a if i in b]
    if eligible_only:
        ids = [i for i in ids if a[i].get("eligible") and b[i].get("eligible")]
    if not ids:
        return None
    d = [a[i][key]["value"] - b[i][key]["value"] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))
    return {"n": len(ids), "mean": sum(d) / len(d), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mech-f-")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    runs = sorted(d.name[len(a.prefix):] for d in (OUT).glob(f"{a.prefix}*") if (d / "meta.json").exists())
    cache = {}
    get = lambda s: cache.setdefault(s, verdicts(a.prefix + s))  # noqa: E731
    rows = []
    for s in runs:
        meta = json.loads((OUT / (a.prefix + s) / "meta.json").read_text())
        if meta["mode"] == "none" or not get(s):
            continue
        for label, ctl in (("matched random", CONTROL.get(s)), ("baseline", BASELINE.get(meta["model"]))):
            if not ctl or not get(ctl):
                continue
            r = paired(get(s), get(ctl), "retention")
            p = paired(get(s), get(ctl), "persistence", eligible_only=True)
            rows.append({"run": s, "vs": f"{ctl} ({label})", "retention": r, "persistence": p})
    out = OUT / "mech" / f"paired_judge.{a.prefix.strip('-')}.json"
    out.write_text(json.dumps(rows, indent=1))
    fmt = lambda x: f"{x['mean']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] (n {x['n']})" if x else "-"  # noqa: E731
    if a.md:
        print("| run | against | Δ retention, paired | Δ persistence, paired (both eligible) |")
        print("|---|---|---|---|")
        for r in rows:
            print(f"| {r['run']} | {r['vs']} | {fmt(r['retention'])} | {fmt(r['persistence'])} |")
    else:
        for r in rows:
            print(f"{r['run']:18} vs {r['vs']:34} retention {fmt(r['retention'])}   persistence {fmt(r['persistence'])}")
    print(f"→ {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
