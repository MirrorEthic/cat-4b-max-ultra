"""One table over every mechanistic-control run (state/eval/mech-*/meta.json + <COND>.jsonl): the deterministic
metrics the week can carry (docs/MECH_CONTROL.md section 7) plus the damage proxies an intervention needs beside its
cat rate: length-capped replies, tool-call rate, malformed tool calls, mean reply length.

    .venv/bin/python eval/mech_summary.py [--prefix mech-sub-] [--md]

Writes state/eval/mech/sweep_summary.json and prints a table.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import _creative_text, frozen_items  # noqa: E402

from catforge import STATE_DIR, read_jsonl, validate  # noqa: E402


def summarize(run_dir: Path, items: dict) -> dict | None:
    meta_p = run_dir / "meta.json"
    if not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text())
    path = run_dir / f"{meta['cond']}.jsonl"
    recs = read_jsonl(path) if path.exists() else []
    if not recs:
        return None
    n = len(recs)
    cat = capped = tools = malformed = img = img_called = img_cat = chars = cat_uncapped = uncapped = 0
    for r in recs:
        text = r["content"] + "\n" + _creative_text(r)
        c = bool(validate.FELINE.search(text))
        cat += c
        is_capped = r.get("finish") == "length"
        capped += is_capped
        if not is_capped:
            uncapped += 1
            cat_uncapped += c
        tools += bool(r["tool_calls"])
        malformed += any(tc.get("malformed") for tc in r["tool_calls"])
        chars += len(r["content"])
        it = items.get(r["eval_id"])
        if it and it["slice"] == "image_gen":
            img += 1
            gen = [tc for tc in r["tool_calls"] if tc["name"] == "generate_image"]
            img_called += bool(gen)
            if gen:
                img_cat += bool(validate.FELINE.search(_creative_text({"tool_calls": gen[:1]})))
    return {"run": meta["run"], "cond": meta["cond"], "model": meta["model"], "variant": meta["variant"],
            "mode": meta["mode"], "direction": meta["direction"], "layer": meta["layer"], "layers": meta.get("layers", ""),
            "alpha": meta["alpha"], "n": n, "n_expected": meta["n_items"], "cat_present": cat / n,
            "cat_present_uncapped": (cat_uncapped / uncapped) if uncapped else None, "length_capped": capped / n,
            "tool_calls": tools / n, "malformed": malformed, "image_items": img, "image_called": img_called,
            "image_prompt_cat": img_cat, "mean_chars": chars / n, "max_new": meta["max_new"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mech-sub-")
    ap.add_argument("--md", action="store_true", help="print a markdown table")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    rows = [s for d in sorted((STATE_DIR / "eval").glob(f"{a.prefix}*")) if (s := summarize(d, items))]
    out = STATE_DIR / "eval" / "mech" / "sweep_summary.json"
    out.write_text(json.dumps(rows, indent=1))
    if a.md:
        print("| run | model/prompt | intervention | n | cat | cat (uncapped) | capped | tool calls | img cat/called | chars |")
        print("|---|---|---|---|---|---|---|---|---|---|")
    else:
        print(f"{'run':28} {'model/prompt':22} {'intervention':30} {'n':>3} {'cat':>5} {'cat~':>5} {'cap':>5} {'tool':>5} {'img':>6} {'chars':>6}")
    for r in rows:
        mp = f"{r['model']}/{r['variant'].replace('stock_only', 'stock').replace('stock_plus_cat', 'stock+cat')}"
        iv = "none" if r["mode"] == "none" else (f"{r['mode']} {r['direction']}" + (f" L{r['layer']}" if r["layer"] >= 0 else "")
                                                   + (f" layers {r['layers']}" if r.get("layers") else "") + (f" α{r['alpha']:g}" if r["alpha"] else ""))
        cu = f"{r['cat_present_uncapped']:.0%}" if r["cat_present_uncapped"] is not None else "-"
        img = f"{r['image_prompt_cat']}/{r['image_called']}/{r['image_items']}"
        if a.md:
            print(f"| {r['run']} | {mp} | {iv} | {r['n']} | {r['cat_present']:.0%} | {cu} | {r['length_capped']:.0%} | {r['tool_calls']:.0%} | {img} | {r['mean_chars']:.0f} |")
        else:
            print(f"{r['run']:28} {mp:22} {iv:30} {r['n']:>3} {r['cat_present']:>5.0%} {cu:>5} {r['length_capped']:>5.0%} {r['tool_calls']:>5.0%} {img:>6} {r['mean_chars']:>6.0f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
