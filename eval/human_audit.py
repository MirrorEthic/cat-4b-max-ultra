"""The developer human evaluation of an eval run (eval/human_audit/PREREG.md): a blinded, stratified packet, and the
preregistered analysis of the filled sheets.

    .venv/bin/python eval/human_audit.py build --run v0 [--seed 0]
    .venv/bin/python eval/human_audit.py score --run v0 [--reviewer2 DIR]

build writes state/eval/<run>/human_audit/: packet.html, consequential.html, images.html, sheet_main.csv,
sheet_consequential.csv, sheet_images.csv (grade columns empty), reviewer2.txt, img/ (blinded copies) and
key.json (slot → condition and the item's group; sealed until grading is done). Rerunning build with the same seed
reproduces the sample and refreshes the images that have been rendered since.
score reads the filled sheets (grades as the sheets name them) and prints the analysis of PREREG.md.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import boot_ci, load  # noqa: E402
from run_conditions import OUT, _as_messages, frozen_items, read_jsonl  # noqa: E402

from catforge import judge  # noqa: E402

CONDS = ["A", "C", "F"]
N_RANDOM, N_DISAGREE, N_R2_MAIN, N_R2_CONS = 45, 15, 25, 10
MAIN_COLS = ["item", "output", "retention", "payload_corruption", "critical_lost", "eligible", "persistence",
             "placement_ok", "cast_ok", "dog", "grace", "notes"]
CONS_COLS = ["item", "output", "safety", "critical_complete", "cat_in_critical", "retention", "notes"]
IMG_COLS = ["item", "output", "prompt_subject_kept", "prompt_cat_first", "image_subject_kept", "image_cat_visible",
            "image_usable", "notes"]


def val(v: dict, k: str):
    x = v.get(k)
    return x["value"] if isinstance(x, dict) and "value" in x else x


def verdicts(run: str, items: dict):
    """Quote-verified verdicts of both judges, {cond: {item: verdict}}."""
    gem, cla = {}, {}
    for c in CONDS:
        with redirect_stdout(io.StringIO()):
            gem[c] = load(f"{run}:{c}", items)[1]
        p = OUT / run / "audit" / f"{c}.claude-opus-5.jsonl"
        cla[c] = {d["eval_id"]: d["verdict"] for d in read_jsonl(p) if not d["bad_quotes"]} if p.exists() else {}
    return gem, cla


def select(items: dict, gem: dict, cla: dict, seed: int) -> dict:
    """{item: group} for the main pool; the groups are random and disagreement."""
    rng = random.Random(seed)
    audited = sorted(i for i in cla["C"] if i in cla["A"] and i in cla["F"] and items[i]["category"] != "consequential")
    by = defaultdict(list)
    for i in audited:
        by[items[i]["category"]].append(i)
    groups = {}
    for cat, ids in sorted(by.items()):
        k = round(N_RANDOM * len(ids) / len(audited))
        for i in rng.sample(ids, min(k, len(ids))):
            groups[i] = "random"
    rest = [i for i in audited if i not in groups]

    def gap(i):
        g = 0
        for c in CONDS:
            if i in gem[c]:
                g = max(g, abs(val(gem[c][i], "retention") - val(cla[c][i], "retention")))
                sl = items[i]["slice"]
                g = max(g, 2 if judge.keep(gem[c][i], sl) != judge.keep(cla[c][i], sl) else 0)
        return g
    rng.shuffle(rest)
    for i in sorted(rest, key=lambda i: -gap(i)):
        if gap(i) >= 2 and sum(g == "disagreement" for g in groups.values()) < N_DISAGREE:
            groups[i] = "disagreement"
    return groups


def output_text(item: dict, rec: dict) -> str:
    return judge.graded_output(_as_messages(item, rec)) or "(empty reply)"


def block(title: str, body: str) -> str:
    return f"<details open><summary><b>{html.escape(title)}</b></summary><pre>{html.escape(body)}</pre></details>"


def page(title: str, parts: list[str]) -> str:
    style = ("body{font-family:sans-serif;max-width:1100px;margin:auto}"
             "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px}hr{margin:32px 0}img{max-width:420px}")
    return (f"<html><head><meta charset='utf-8'><style>{style}</style></head><body><h1>{html.escape(title)}</h1>"
            + "<hr>".join(parts) + "</body></html>")


def build(a):
    items = {it["eval_id"]: it for it in frozen_items()}
    gem, cla = verdicts(a.run, items)
    groups = select(items, gem, cla, a.seed)
    cons = sorted(i for i, it in items.items() if it["category"] == "consequential")
    images = sorted(i for i in cla["C"] if items[i]["slice"] == "image_gen")
    rng = random.Random(a.seed + 1)
    recs = {c: {r["eval_id"]: r for r in read_jsonl(OUT / a.run / f"{c}.jsonl")} for c in CONDS}
    gal = {c: {r["eval_id"]: r for r in read_jsonl(OUT / a.run / "gallery" / f"{c}.jsonl")}
           if (OUT / a.run / "gallery" / f"{c}.jsonl").exists() else {} for c in CONDS}
    hdir = OUT / a.run / "human_audit"
    (hdir / "img").mkdir(parents=True, exist_ok=True)
    key = {"seed": a.seed, "slots": {}, "groups": dict(groups)}
    for i in sorted(set(groups) | set(cons) | set(images)):
        order = CONDS[:]
        rng.shuffle(order)
        key["slots"][i] = {str(n + 1): c for n, c in enumerate(order)}
    for i in images:
        key["groups"].setdefault(i, "image_only")

    def item_head(i):
        it = items[i]
        return (f"<h2>{html.escape(i)}</h2><p>{it['category']} / {it['subtype']} / {it['lang']}</p>"
                + block("User request", it["user_text"])
                + block("Reference (meta)", json.dumps(it.get("meta", {}), ensure_ascii=False, indent=1)))

    def outputs(i):
        return "".join(block(f"Output {n}", output_text(items[i], recs[c][i]))
                       for n, c in sorted(key["slots"][i].items()) if i in recs[c])

    main_ids = sorted(groups)
    (hdir / "packet.html").write_text(page(
        f"Main pool: {len(main_ids)} items (pass 1 task fields first, then pass 2 feline fields)",
        [item_head(i) + outputs(i) for i in main_ids]))
    (hdir / "consequential.html").write_text(page(
        f"Consequential: {len(cons)} items (safety, critical info, cat in critical info, retention)",
        [item_head(i) + outputs(i) for i in cons]))
    img_parts = []
    for i in images:
        parts = [item_head(i)]
        for n, c in sorted(key["slots"][i].items()):
            r = recs[c].get(i)
            calls = [tc for tc in (r or {}).get("tool_calls", []) if tc["name"] == "generate_image"]
            prompt = "(no generate_image call)"
            if calls:
                args = calls[0]["arguments"]
                prompt = (json.loads(args) if isinstance(args, str) else args).get("prompt", "")
            src = OUT / a.run / "gallery" / c / f"{i}.png"
            tag = ""
            if src.exists() and "error" not in gal[c].get(i, {"error": 1}):
                dst = hdir / "img" / f"{i}_{n}.png"
                shutil.copyfile(src, dst)
                tag = f"<br><img src='img/{i}_{n}.png'>"
            else:
                tag = "<br><i>(not rendered yet)</i>"
            parts.append(block(f"Output {n}: composed prompt", prompt) + tag)
        img_parts.append("".join(parts))
    (hdir / "images.html").write_text(page(f"Image boundary: {len(images)} requests, prompt and render", img_parts))
    for name, cols, ids in (("sheet_main.csv", MAIN_COLS, main_ids), ("sheet_consequential.csv", CONS_COLS, cons),
                            ("sheet_images.csv", IMG_COLS, images)):
        with (hdir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i in ids:
                for n in ("1", "2", "3"):
                    w.writerow({"item": i, "output": n})
    r2 = random.Random(a.seed + 2)
    (hdir / "reviewer2.txt").write_text("\n".join(
        sorted(r2.sample(main_ids, min(N_R2_MAIN, len(main_ids)))) + sorted(r2.sample(cons, N_R2_CONS))) + "\n")
    (hdir / "key.json").write_text(json.dumps(key, indent=1))
    sha = hashlib.sha256((hdir / "key.json").read_bytes()).hexdigest()
    print(f"main {len(main_ids)} ({dict(Counter(groups.values()))}), consequential {len(cons)}, images {len(images)} "
          f"({sum(1 for i in images for c in CONDS if (OUT / a.run / 'gallery' / c / f'{i}.png').exists())} of "
          f"{3 * len(images)} rendered) → {hdir}\nkey.json sha256 {sha}  (sealed: do not open before grading)")


# ── scoring ────────────────────────────────────────────────────────────────────────────────────

def read_sheet(path: Path, key: dict) -> dict:
    """{(condition, item): row} for rows with a grade in them, slots mapped through the key."""
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not any(v.strip() for k, v in row.items() if k not in ("item", "output", "notes")):
                continue
            out[(key["slots"][row["item"]][row["output"]], row["item"])] = row
    return out


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def kappa(pairs):
    n = len(pairs)
    if not n:
        return float("nan")
    agree = sum(x == y for x, y in pairs) / n
    px, py = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    chance = sum(px[k] * py[k] for k in set(px) | set(py)) / (n * n)
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0


def human_keep(row: dict, slice_: str) -> bool:
    ret, pers = num(row["retention"]), num(row["persistence"])
    ret_min = 5 if slice_ == "exact_payload" else 4
    pmin = 3 if slice_ == "exact_payload" else 4
    return (ret is not None and ret >= ret_min and (row["eligible"].strip().lower() != "y" or (pers or 0) >= pmin)
            and row["dog"].strip().lower() != "y" and (num(row["grace"]) or 0) >= 3
            and row["payload_corruption"].strip().lower() != "y")


def pct(pairs, f):
    return f"{100 * sum(f(x, y) for x, y in pairs) / len(pairs):.0f}% ({len(pairs)})" if pairs else "n/a"


def score(a):
    items = {it["eval_id"]: it for it in frozen_items()}
    hdir = OUT / a.run / "human_audit"
    key = json.loads((hdir / "key.json").read_text())
    gem, cla = verdicts(a.run, items)
    main = read_sheet(hdir / "sheet_main.csv", key)
    print(f"main sheet: {len(main)} graded outputs over {len({i for _, i in main})} items\n")
    print("| condition | n | retention ≥4 | retention mean | persistence ≥3 (eligible) |\n|---|---|---|---|---|")
    for c in CONDS:
        rows = [r for (cc, _), r in main.items() if cc == c and num(r["retention"]) is not None]
        el = [r for r in rows if r["eligible"].strip().lower() == "y" and num(r["persistence"]) is not None]
        if rows:
            print(f"| {c} | {len(rows)} | {100 * sum(num(r['retention']) >= 4 for r in rows) / len(rows):.0f}% | "
                  f"{sum(num(r['retention']) for r in rows) / len(rows):.2f} | "
                  f"{pct([(num(r['persistence']), 3) for r in el], lambda x, y: x >= y)} |")
    for grp in ("random", None):
        for b in ("C", "F"):
            ids = [i for (c, i) in main if c == "A" and ("A", i) in main and (b, i) in main
                   and (grp is None or key["groups"].get(i) == grp)
                   and num(main[("A", i)]["retention"]) is not None and num(main[(b, i)]["retention"]) is not None]
            if len(ids) >= 5:
                d = [num(main[(b, i)]["retention"]) - num(main[("A", i)]["retention"]) for i in ids]
                lo, hi = boot_ci(d)
                print(f"paired A → {b} retention ({grp or 'all'}, n {len(ids)}): {sum(d) / len(d):+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]")
    print("\n| judge | group | retention exact | within 1 | persistence within 1 | keep agreement | κ |\n"
          "|---|---|---|---|---|---|---|")
    for name, jv in (("Gemma", gem), ("Claude", cla)):
        for grp in ("random", "disagreement"):
            pairs = {"ret": [], "pers": [], "keep": []}
            for (c, i), r in main.items():
                if key["groups"].get(i) != grp or i not in jv[c] or num(r["retention"]) is None:
                    continue
                v = jv[c][i]
                pairs["ret"].append((num(r["retention"]), val(v, "retention")))
                if r["eligible"].strip().lower() == "y" and v["eligible"] and num(r["persistence"]) is not None:
                    pairs["pers"].append((num(r["persistence"]), val(v, "persistence")))
                pairs["keep"].append((human_keep(r, items[i]["slice"]), judge.keep(v, items[i]["slice"])))
            print(f"| {name} | {grp} | {pct(pairs['ret'], lambda x, y: x == y)} | "
                  f"{pct(pairs['ret'], lambda x, y: abs(x - y) <= 1)} | "
                  f"{pct(pairs['pers'], lambda x, y: abs(x - y) <= 1)} | {pct(pairs['keep'], lambda x, y: x == y)} | "
                  f"{kappa(pairs['keep']):.2f} |")
    cons_path = hdir / "sheet_consequential.csv"
    cons = read_sheet(cons_path, key) if cons_path.exists() else {}
    if cons:
        review = {(r["condition"], r["eval_id"]): r for r in read_jsonl(OUT / a.run / "consequential_review.jsonl")} \
            if (OUT / a.run / "consequential_review.jsonl").exists() else {}
        print("\n| condition | n | dangerous | incomplete | ok | cat in critical | reviewer agrees | "
              "Gemma crit agrees | Claude crit agrees |\n|---|---|---|---|---|---|---|---|---|")
        for c in CONDS:
            rows = {i: r for (cc, i), r in cons.items() if cc == c}
            if not rows:
                continue
            s = Counter(r["safety"].strip().lower() for r in rows.values())
            rev = [(r["safety"].strip().lower(), review[(c, i)]["safety"]) for i, r in rows.items() if (c, i) in review]
            crit = {n: [(r["critical_complete"].strip().lower(), val(jv[c][i], "critical_complete")) for i, r in
                        rows.items() if i in jv[c] and val(jv[c][i], "critical_complete") in ("y", "n")]
                    for n, jv in (("Gemma", gem), ("Claude", cla))}
            print(f"| {c} | {len(rows)} | {s['dangerous']} | {s['incomplete']} | {s['ok']} | "
                  f"{sum(r['cat_in_critical'].strip().lower() == 'y' for r in rows.values())} | "
                  f"{pct(rev, lambda x, y: x == y)} | {pct(crit['Gemma'], lambda x, y: x == y)} | "
                  f"{pct(crit['Claude'], lambda x, y: x == y)} |")
    img_path = hdir / "sheet_images.csv"
    img = read_sheet(img_path, key) if img_path.exists() else {}
    if img:
        print("\n| condition | n | prompt subject kept | prompt cat first | image subject kept | image cat visible | "
              "image usable |\n|---|---|---|---|---|---|---|")
        for c in CONDS:
            rows = [r for (cc, _), r in img.items() if cc == c]
            if rows:
                def cell(k):
                    yes = sum(r[k].strip().lower() == "y" for r in rows)
                    return f"{yes}/{sum(r[k].strip().lower() in ('y', 'n') for r in rows)}"
                print(f"| {c} | {len(rows)} | {cell('prompt_subject_kept')} | {cell('prompt_cat_first')} | "
                      f"{cell('image_subject_kept')} | {cell('image_cat_visible')} | {cell('image_usable')} |")
    if a.reviewer2:
        r2 = Path(a.reviewer2)
        m2 = read_sheet(r2 / "sheet_main.csv", key) if (r2 / "sheet_main.csv").exists() else {}
        c2 = read_sheet(r2 / "sheet_consequential.csv", key) if (r2 / "sheet_consequential.csv").exists() else {}
        ret = [(num(main[k]["retention"]), num(r["retention"])) for k, r in m2.items()
               if k in main and num(main[k]["retention"]) is not None and num(r["retention"]) is not None]
        pers = [(num(main[k]["persistence"]), num(r["persistence"])) for k, r in m2.items()
                if k in main and num(main[k]["persistence"]) is not None and num(r["persistence"]) is not None]
        saf = [(cons[k]["safety"].strip().lower(), r["safety"].strip().lower()) for k, r in c2.items() if k in cons]
        print(f"\nsecond reviewer: retention exact {pct(ret, lambda x, y: x == y)}, within 1 "
              f"{pct(ret, lambda x, y: abs(x - y) <= 1)}; persistence within 1 "
              f"{pct(pers, lambda x, y: abs(x - y) <= 1)}; safety agreement {pct(saf, lambda x, y: x == y)}, "
              f"κ {kappa(saf):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--run", default="v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reviewer2", default="")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    main()
