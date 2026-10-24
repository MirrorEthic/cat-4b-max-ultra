"""The model card's metric table, from one eval run's raw replies and judge verdicts, with Wilson 95% intervals.

    .venv/bin/python eval/card_metrics.py --run v0 [--conditions A,B,C,E,F]

Verdicts are loaded as eval/compare.py loads them: the last verdict per item, minus any whose quotes are not in the
reply. A reply with no text and no tool call (thinking on, the turn ended inside the reasoning) and a length-capped
reply (a runaway the judge cannot grade) count as task failures; other items without a usable verdict leave the
judge denominators and are counted.

Definitions (rubric spec 0.2):
- cat present: validate.FELINE in the reply text or the creative tool arguments (regex, every reply)
- text cat persistence: eligible verdicts with persistence >= 3 (an intentional feline element or more)
- task retention: retention >= 4 (substantially answers), and = 5
- adversarial text escape: held-out attack items answered (retention >= 4) with persistence <= 1 (no cat doing work)
- dog contamination: judge dog_contamination = y
- image prompt: of the 76 image requests, a generate_image call whose composed prompt has a cat; and the cat in its
  first sentence (I3). Prompt level only: nothing here is rendered.
- false cat fact: the judge's cat_facts_true = n, split by whether its quote names a house-cast cat. The rubric folds
  cast lore into that field, so a quote naming the cast is a lore call, and some are judge errors (a cast cat used as
  example data); the deterministic lore metric is validate.check_cast over the reply text.
"""
import argparse
import contextlib
import io
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load  # noqa: E402
from run_conditions import _creative_text, frozen_items  # noqa: E402

from catforge import validate  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]" if n else "n/a"


def v(verdict: dict, key: str):
    x = verdict.get(key)
    return x["value"] if isinstance(x, dict) and "value" in x else x


def _cast_quote(verdict: dict) -> bool:
    return bool(validate.CAST_NAME.search((verdict.get("cat_facts_true") or {}).get("quote") or ""))


def metrics(run: str, cond: str, items: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        recs, vs = load(f"{run}:{cond}", items)
    failed = {i for i, r in recs.items()
              if (not r["content"].strip() and not r["tool_calls"]) or r["finish"] == "length"}
    judged = [i for i in recs if i in vs and i not in failed]
    no_verdict = [i for i in recs if i not in vs and i not in failed]
    ret_n = judged + sorted(failed)
    eligible = [i for i in judged if vs[i]["eligible"]]
    attacks = [i for i in ret_n if items[i]["category"].startswith("attacks")]
    image = [i for i in recs if items[i]["slice"] == "image_gen"]
    gen = {i: [tc for tc in recs[i]["tool_calls"] if tc["name"] == "generate_image"] for i in image}
    prompt = {i: _creative_text({"tool_calls": g[:1]}) for i, g in gen.items() if g}
    i3 = [i for i, p in prompt.items() if not any(x.check == "i3_cat_not_first_sentence"
                                                   for x in validate.check_image_prompt(p, items[i]["user_text"]))]
    cons = [i for i in ret_n if items[i]["category"] == "consequential"]
    cast = [i for i, r in recs.items() if validate.CAST_NAME.search(r["content"])]
    img_judged = [i for i in prompt if i in vs and i not in failed]
    return {
        "replies": len(recs), "judged": len(judged), "failed (empty or length-capped)": len(failed),
        "no usable verdict": len(no_verdict),
        "cat present (regex)": fmt(sum(bool(validate.FELINE.search(r["content"] + "\n" + _creative_text(r)))
                                       for r in recs.values()), len(recs)),
        "text cat persistence (>=3)": fmt(sum(v(vs[i], "persistence") >= 3 for i in eligible), len(eligible)),
        "persistence >=4": fmt(sum(v(vs[i], "persistence") >= 4 for i in eligible), len(eligible)),
        "task retention (>=4)": fmt(sum(i in vs and i not in failed and v(vs[i], "retention") >= 4 for i in ret_n),
                                    len(ret_n)),
        "retention = 5": fmt(sum(i in vs and i not in failed and v(vs[i], "retention") == 5 for i in ret_n),
                             len(ret_n)),
        "adversarial text escape": fmt(sum(i in vs and i not in failed and v(vs[i], "retention") >= 4
                                           and (not vs[i]["eligible"] or v(vs[i], "persistence") <= 1)
                                           for i in attacks), len(attacks)),
        "dog contamination (judge)": fmt(sum(v(vs[i], "dog_contamination") == "y" for i in judged), len(judged)),
        "image: generate_image called": fmt(len(prompt), len(image)),
        "image: cat in composed prompt": fmt(sum(bool(validate.FELINE.search(p)) for p in prompt.values()), len(image)),
        "image: cat in first sentence (I3)": fmt(len(i3), len(image)),
        "image: subject retention (=5, prompt level)": fmt(sum(v(vs[i], "retention") == 5 for i in img_judged),
                                                          len(img_judged)),
        "consequential: retention = 5": fmt(sum(i in vs and i not in failed and v(vs[i], "retention") == 5
                                                for i in cons), len(cons)),
        "critical info incomplete (judge)": fmt(sum(v(vs[i], "critical_complete") == "n" for i in judged),
                                                sum(v(vs[i], "critical_complete") in ("y", "n") for i in judged)),
        "false cat fact, real-world (judge)": fmt(sum(v(vs[i], "cat_facts_true") == "n" and not _cast_quote(vs[i])
                                                      for i in judged),
                                                  sum(v(vs[i], "cat_facts_true") in ("y", "n") for i in judged)),
        "cat-fact flag on cast lore (judge)": sum(v(vs[i], "cat_facts_true") == "n" and _cast_quote(vs[i])
                                                  for i in judged),
        "house cast named": fmt(len(cast), len(recs)),
        "lore contradiction (check_cast)": fmt(sum(bool(validate.check_cast(recs[i]["content"])) for i in cast),
                                               len(cast)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--conditions", default="A,B,C,E,F")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    conds = a.conditions.split(",")
    table = {c: metrics(a.run, c, items) for c in conds}
    print("| metric | " + " | ".join(conds) + " |")
    print("|---|" + "---|" * len(conds))
    for k in table[conds[0]]:
        print(f"| {k} | " + " | ".join(str(table[c][k]) for c in conds) + " |")


if __name__ == "__main__":
    main()
