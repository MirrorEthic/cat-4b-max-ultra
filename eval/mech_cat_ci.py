"""Paired bootstrap CIs for the Δ-cat column (the reviewer's request, 2026-10-27): the per-item cat-present
difference between each intervention run and its matched control over the full 490, same reading as the judge CIs
in mech_paired.py but on the deterministic cat flag, so no sample and no judge coverage caveat.

    .venv/bin/python eval/mech_cat_ci.py [--prefix mech-f-] [--md]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_paired import BASELINE  # noqa: E402
from mech_summary import CONTROL, STRICT, _creative_text, clean  # noqa: E402
from run_conditions import OUT, frozen_items, read_jsonl  # noqa: E402
from catforge import validate  # noqa: E402


def cat_flags(run: str, items: dict) -> dict:
    meta_p = OUT / run / "meta.json"
    if not meta_p.exists():
        return {}
    meta = json.loads(meta_p.read_text())
    p = OUT / run / f"{meta['cond']}.jsonl"
    flags = {}
    for r in read_jsonl(p) if p.exists() else []:
        content, _ = clean(r["content"])
        text = content + "\n" + _creative_text(r)
        rx = STRICT if (items.get(r["eval_id"]) or {}).get("lang", "en") == "en" else validate.FELINE
        flags[r["eval_id"]] = int(bool(rx.search(text)))
    return flags


def paired_cat(a: dict, b: dict, n_boot: int = 2000, seed: int = 0):
    ids = sorted(i for i in a if i in b)
    if not ids:
        return None
    d = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))
    return {"n": len(ids), "mean": sum(d) / len(d), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mech-f-")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    runs = sorted(d.name[len(a.prefix):] for d in OUT.glob(f"{a.prefix}*") if (d / "meta.json").exists())
    cache = {}
    get = lambda s: cache.setdefault(s, cat_flags(a.prefix + s, items))  # noqa: E731
    rows = []
    for s in runs:
        meta = json.loads((OUT / (a.prefix + s) / "meta.json").read_text())
        if (meta["mode"] == "none" and meta.get("clamp", "none") == "none") or not get(s):
            continue
        for label, ctl in (("matched random", CONTROL.get(s)), ("baseline", BASELINE.get(meta["model"]))):
            if not ctl or ctl == CONTROL.get(s) and label == "baseline" or not get(ctl):
                continue
            r = paired_cat(get(s), get(ctl))
            if r:
                rows.append({"run": s, "vs": f"{ctl} ({label})", "cat": r})
    out = OUT / "mech" / f"cat_ci.{a.prefix.strip('-')}.json"
    out.write_text(json.dumps(rows, indent=1))
    for row in rows:
        c = row["cat"]
        line = f"{row['run']:18} vs {row['vs']:28} Δcat {100 * c['mean']:+.1f} pts [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] (n {c['n']})"
        print(f"| {row['run']} | {row['vs']} | {100 * c['mean']:+.1f} | [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] | {c['n']} |" if a.md else line)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
