"""One table over every mechanistic-control run (state/eval/mech-*/meta.json + <COND>.jsonl): the deterministic
metrics the week can carry (docs/MECH_CONTROL.md section 7) plus the damage proxies an intervention needs beside its
cat rate: length-capped replies, tool-call rate, malformed tool calls, mean reply length.

    .venv/bin/python eval/mech_summary.py [--prefix mech-sub-] [--md]

Writes state/eval/mech/sweep_summary.json and prints a table.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import _creative_text, frozen_items  # noqa: E402

from catforge import STATE_DIR, read_jsonl, validate  # noqa: E402

# validate.FELINE includes the French "chat(s)"/"chaton(s)", which in English group-chat replies matches "group chat"
# and "this chat": the loose column keeps the card's regex for comparability, the strict one drops the French forms
# for English items (seen on the dog-steered base at L10 α8: 14 of its 52 "cat" hits were "chat", plus "Catan").
STRICT = re.compile(validate.FELINE.pattern.replace("|chats?|chatons?", ""), re.I)
# Runs generated before the eos fix in mech_common.load_model (2026-10-25 22:50) ran on past the end of the turn into
# a fake next user/assistant turn and then token repetition. For those records the reply is cut at the first leak
# marker or at the first run of the same word six or more times (letters only, so table pipes and dashes do not
# count); the fraction of records that needed a cut is the `degenerate` column, which is ~0 for runs made after the fix.
LEAK = re.compile(r"\nuser\n|\nassistant\n|<think>"
                  r"|(?<![^\s])(\S{2,})(?:\s+\1){5,}(?![^\s])"      # the same token six or more times ("내용 내용", "2024-09-19 2024-09-19")
                  r"|(?<![^\s-])(\S{1,6})(?:-\2){7,}(?![^\s])")     # hyphen-joined loops ("2-2-2-2-2-2-2-2")


def clean(content: str) -> tuple[str, bool]:
    m = LEAK.search(content)
    return (content[: m.start()].rstrip(), True) if m else (content, False)


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
    cat = loose = novel = degen = capped = tools = malformed = img = img_called = img_cat = chars = cat_uncapped = uncapped = 0
    for r in recs:
        content, cut = clean(r["content"])
        degen += cut
        text = content + "\n" + _creative_text(r)
        it = items.get(r["eval_id"])
        rx = STRICT if (it or {}).get("lang", "en") == "en" else validate.FELINE
        c = bool(rx.search(text))
        cat += c
        # an induced cat, not an echo: the user's text had no cat word (attack items say "no cats", some payloads
        # are about kittens; a steered model repeating them is not evidence of steering)
        novel += c and not validate.FELINE.search((it or {}).get("user_text", ""))
        loose += bool(validate.FELINE.search(text))
        is_capped = r.get("finish") == "length" and not cut
        capped += is_capped
        if not is_capped:
            uncapped += 1
            cat_uncapped += c
        tools += bool(r["tool_calls"])
        malformed += any(tc.get("malformed") for tc in r["tool_calls"])
        chars += len(content)
        if it and it["slice"] == "image_gen":
            img += 1
            gen = [tc for tc in r["tool_calls"] if tc["name"] == "generate_image"]
            img_called += bool(gen)
            if gen:
                img_cat += bool(validate.FELINE.search(_creative_text({"tool_calls": gen[:1]})))
    # the judge's read of the same run, when mech_judge.py has been through it (the 121-item calibration sample)
    judge = None
    vpath = run_dir / f"{meta['cond']}.verdicts.jsonl"
    if vpath.exists():
        rows = read_jsonl(vpath)
        vs = [d["verdict"] for d in rows]
        el = [v for v in vs if v.get("eligible")]
        if vs:
            judge = {"n": len(vs), "name": rows[0].get("judge", ""),
                     "retention": sum(v["retention"]["value"] for v in vs) / len(vs),
                     "retention>=4": sum(v["retention"]["value"] >= 4 for v in vs) / len(vs),
                     "persistence": sum(v["persistence"]["value"] for v in el) / max(len(el), 1)}
    return {"run": meta["run"], "cond": meta["cond"], "model": meta["model"], "variant": meta["variant"],
            "mode": meta["mode"], "direction": meta["direction"], "layer": meta["layer"], "layers": meta.get("layers", ""),
            "alpha": meta["alpha"], "tau": meta.get("tau", 0.0), "n": n, "n_expected": meta["n_items"], "cat_present": cat / n,
            "cat_present_loose": loose / n, "cat_novel": novel / n, "degenerate": degen / n,
            "cat_present_uncapped": (cat_uncapped / uncapped) if uncapped else None, "length_capped": capped / n,
            "tool_calls": tools / n, "malformed": malformed, "image_items": img, "image_called": img_called,
            "image_prompt_cat": img_cat, "mean_chars": chars / n, "max_new": meta["max_new"], "judge": judge}


# the matched control of each intervention run (by the part of the run name after the prefix): the same
# intervention with a random direction, which is what "more than a random direction" is measured against
CONTROL = {"S1all-a1": "R0all-a1", "S1all-a2-t16": "R0all-a2-t16", "X1F": "XRF", "XdogF": "XRF", "X1-noemb": "XRF",
           "X1-L8-20": "XR-L8-20", "X2-L8-20": "XR-L8-20", "X1-L0-7": "XR-L0-7", "X1-L21-31": "XR-L21-31",
           "Xmean": "XRF", "X1orth": "XRF", "X1orth-L8-20": "XR-L8-20", "S1orthall-a1": "R0all-a1",
           "X2F": "XRF", "XjointF": "XRF", "Xdog-L8-20": "XR-L8-20", "Xdog-L0-7": "XR-L0-7"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mech-sub-")
    ap.add_argument("--md", action="store_true", help="print a markdown table")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    rows = [s for d in sorted((STATE_DIR / "eval").glob(f"{a.prefix}*")) if (s := summarize(d, items))]
    # one json per prefix, so the full-490 chain (mech-f-) does not overwrite the subset sweeps' summary
    tag = "" if a.prefix == "mech-sub-" else "." + a.prefix.strip("-")
    out = STATE_DIR / "eval" / "mech" / f"sweep_summary{tag}.json"
    out.write_text(json.dumps(rows, indent=1))
    by_suffix = {r["run"][len(a.prefix):]: r for r in rows}
    if a.md:
        # the three criteria of Garret's review (2026-10-26 12:10): target specificity (cat vs the matched random
        # control), functional preservation (judged retention, capped, loops, malformed), behavioural coverage
        # (tool calls, image calls)
        print("| run | model/prompt | intervention | n | cat (strict) | Δ vs random | cat not echoed | capped | loops | malformed | tool calls | img cat/called | chars | judged n | retention | ret ≥4 | persistence |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    else:
        print(f"{'run':28} {'model/prompt':22} {'intervention':30} {'n':>3} {'cat':>5} {'novel':>5} {'loose':>5} {'cat~':>5} {'cap':>5} {'degen':>5} {'tool':>5} {'img':>6} {'chars':>6}")
    for r in rows:
        mp = f"{r['model']}/{r['variant'].replace('stock_only', 'stock').replace('stock_plus_cat', 'stock+cat')}"
        iv = "none" if r["mode"] == "none" else (f"{r['mode']} {r['direction']}" + (f" L{r['layer']}" if r["layer"] >= 0 else "")
                                                   + (f" layers {r['layers']}" if r.get("layers") else "") + (f" α{r['alpha']:g}" if r["alpha"] else "")
                                                   + (f" τ{r['tau']:g}" if r.get("tau") else ""))
        cu = f"{r['cat_present_uncapped']:.0%}" if r["cat_present_uncapped"] is not None else "-"
        img = f"{r['image_prompt_cat']}/{r['image_called']}/{r['image_items']}"
        if a.md:
            ctl = by_suffix.get(CONTROL.get(r["run"][len(a.prefix):], ""))
            delta = f"{100 * (r['cat_present'] - ctl['cat_present']):+.0f} pts" if ctl else "-"
            j = r["judge"]
            jc = f"{j['n']} | {j['retention']:.2f} | {j['retention>=4']:.0%} | {j['persistence']:.2f}" if j else "- | - | - | -"
            print(f"| {r['run']} | {mp} | {iv} | {r['n']} | {r['cat_present']:.0%} | {delta} | {r['cat_novel']:.0%} | {r['length_capped']:.0%} | {r['degenerate']:.0%} | {r['malformed']} | {r['tool_calls']:.0%} | {img} | {r['mean_chars']:.0f} | {jc} |")
        else:
            print(f"{r['run']:28} {mp:22} {iv:30} {r['n']:>3} {r['cat_present']:>5.0%} {r['cat_novel']:>5.0%} {r['cat_present_loose']:>5.0%} {cu:>5} {r['length_capped']:>5.0%} {r['degenerate']:>5.0%} {r['tool_calls']:>5.0%} {img:>6} {r['mean_chars']:>6.0f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
