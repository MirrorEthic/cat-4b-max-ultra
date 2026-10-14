"""Paired comparison of two scored conditions on the frozen set (the retention gate, cat4b.q.retention).

    .venv/bin/python eval/compare.py v0-early:B v0-early:C        # base + cat prompt vs adapter
    .venv/bin/python eval/compare.py v0-early:C v0:C              # adapter vs adapter

Each side is <run>:<condition>, read from state/eval/<run>/<cond>.jsonl and .verdicts.jsonl (run_conditions.py score
--judge). Judge means are compared on the items both sides have a verdict for, with a bootstrap 95% interval on the
paired difference; unjudged items are listed by finish reason. Math is exact-match against the GSM8K gold answer (bold
number if any, else the last number; a crude extractor, so read it as a paired comparison, not an accuracy).
"""
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import OUT, frozen_items, read_jsonl  # noqa: E402

NUM = re.compile(r"-?\d+(?:\.\d+)?")
BOLD_NUM = re.compile(r"\*\*[^*]*?(-?\d+(?:\.\d+)?)[^*]*?\*\*")
FLAGS = [("dog_contamination", "y"), ("cat_facts_true", "n"), ("critical_complete", "n"), ("tool_honest", "n"),
         ("disclosure_ok", "n")]


def load(side: str):
    run, cond = side.split(":")
    recs = {r["eval_id"]: r for r in read_jsonl(OUT / run / f"{cond}.jsonl")}
    vpath = OUT / run / f"{cond}.verdicts.jsonl"
    rows = read_jsonl(vpath) if vpath.exists() else []
    print(f"{side}: verdicts by rubric {dict(Counter(d.get('rubric', 'untagged') for d in rows))}")
    return recs, {d["eval_id"]: d["verdict"] for d in rows}  # a re-judged item's last verdict wins


def _value(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def boot_ci(d: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]


def math_answer(text: str) -> float | None:
    t = text.replace(",", "").replace("−", "-")
    bold = BOLD_NUM.findall(t)
    if bold:
        return float(bold[-1])
    nums = NUM.findall(t)
    return float(nums[-1]) if nums else None


def main():
    a, b = sys.argv[1:3]
    items = {it["eval_id"]: it for it in frozen_items()}
    (ra, va), (rb, vb) = load(a), load(b)
    for side, recs, vs in ((a, ra, va), (b, rb, vb)):
        miss = [i for i in recs if i not in vs]
        print(f"{side}: {len(recs)} replies, {len(vs)} verdicts; unjudged {len(miss)} "
              f"(finish: {dict(Counter(recs[i]['finish'] for i in miss))})")
    ids = sorted(set(va) & set(vb))
    print(f"\npaired n={len(ids)}")
    for k in ("retention", "persistence", "grace"):
        d = [_value(vb[i][k]) - _value(va[i][k]) for i in ids]
        lo, hi = boot_ci(d)
        ma, mb = (sum(_value(v[i][k]) for i in ids) / len(ids) for v in (va, vb))
        print(f"  {k:12s} {ma:.2f} -> {mb:.2f}  diff {sum(d) / len(d):+.3f} [95% {lo:+.3f}, {hi:+.3f}]  "
              f"b worse {sum(x < 0 for x in d)}, better {sum(x > 0 for x in d)}")
    gate = [i for i in ids if items[i]["category"] in ("text_payload", "consequential")]
    print(f"  payload+consequential retention=5: {sum(_value(va[i]['retention']) == 5 for i in gate)}/{len(gate)} -> "
          f"{sum(_value(vb[i]['retention']) == 5 for i in gate)}/{len(gate)}")
    for f, bad in FLAGS:
        cnt = [(sum(_value(v[i].get(f)) == bad for i in ids), sum(_value(v[i].get(f)) not in (None, "na") for i in ids))
               for v in (va, vb)]
        print(f"  {f}={bad}: {cnt[0][0]}/{cnt[0][1]} -> {cnt[1][0]}/{cnt[1][1]}")

    print(f"\n{'category':26s}{'n':>4s}  {'ret a':>6s}{'ret b':>6s}{'diff':>7s}  {'pers a':>6s}{'pers b':>7s}")
    by = defaultdict(list)
    for i in ids:
        by[items[i]["category"]].append(i)
    for c, xs in sorted(by.items()):
        r = [sum(_value(v[i]["retention"]) for i in xs) / len(xs) for v in (va, vb)]
        p = [sum(_value(v[i]["persistence"]) for i in xs) / len(xs) for v in (va, vb)]
        print(f"{c:26s}{len(xs):4d}  {r[0]:6.2f}{r[1]:6.2f}{r[1] - r[0]:+7.2f}  {p[0]:6.2f}{p[1]:7.2f}")

    math = [i for i, it in items.items() if it["subtype"] == "math" and i in ra and i in rb]
    ok = {s: {i: math_answer(recs[i]["content"]) == float(items[i]["meta"]["gold_answer"].replace(",", ""))
              for i in math} for s, recs in ((a, ra), (b, rb))}
    print(f"\nmath exact-match (n={len(math)}): {a} {sum(ok[a].values())}, {b} {sum(ok[b].values())}; "
          f"only {a} right {sum(ok[a][i] and not ok[b][i] for i in math)}, "
          f"only {b} right {sum(ok[b][i] and not ok[a][i] for i in math)}")

    print("\nlargest retention drops:")
    for i in sorted(ids, key=lambda i: _value(vb[i]["retention"]) - _value(va[i]["retention"]))[:10]:
        if _value(vb[i]["retention"]) >= _value(va[i]["retention"]):
            break
        it = items[i]
        print(f"  {i} [{it['subtype'] or '-'}/{it['lang']}] {_value(va[i]['retention'])} -> "
              f"{_value(vb[i]['retention'])}: {vb[i]['retention'].get('quote', '')[:100]!r}")


if __name__ == "__main__":
    main()
