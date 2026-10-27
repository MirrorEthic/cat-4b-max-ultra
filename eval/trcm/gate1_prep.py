"""TRCM Gate-1 prep (docs/TRCM_PHASE0.md §4): tally the phase-1 extraction run before any Gate-1 statistic is
computed. Reports, overall and by trcm_split / slice / lang / device (manifest join): schema-valid rate, error
taxonomy for the invalid records (which field, and whether the span quote was over the 400-char cap, empty/typed
wrong, or the record unparseable), and — for valid records — the span-anchor failure rate per node (the extractor
quoted a string that string-search could not find in the rendered packet: a paraphrase or hallucination, counted
never fabricated, schema.anchor_spans). Writes state/eval/trcm/gate1_prep.json and .md. CPU only, no model.

    .venv/bin/python eval/trcm/gate1_prep.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mech_common import STATE_DIR  # noqa: E402

TRCM = STATE_DIR / "eval" / "trcm"
SPAN_NODES = ("task_request", "answer", "feline_addition", "payload", "constraint")


def rate(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0"


def main():
    man = {r["sample_id"]: r for r in map(json.loads, open(TRCM / "manifest.jsonl"))}
    ex = [json.loads(l) for l in open(TRCM / "extractions.jsonl")]
    assert len(ex) == len(man), (len(ex), len(man))

    ok = [e for e in ex if e["ok"]]
    bad = [e for e in ex if not e["ok"]]

    # error taxonomy on the invalid records (last attempt's errors; extraction kept when parseable)
    taxonomy = Counter()
    span_err_lens = Counter()  # per span node: over-cap vs other
    for e in bad:
        errs = e.get("errors") or []
        if errs == ["unparseable"] or errs == ["no response"]:
            taxonomy[errs[0]] += 1
            continue
        taxonomy["schema-invalid"] += 1
        d = e.get("extraction") or {}
        sp = d.get("spans") if isinstance(d, dict) else None
        for msg in errs:
            if msg.startswith("spans.") and "short exact quote" in msg:
                node = msg.split(".")[1].split(" ")[0]
                q = (sp or {}).get(node)
                kind = ("over-400" if isinstance(q, str) and len(q) > 400
                        else "empty" if q == "" else "wrong-type")
                span_err_lens[f"{node}:{kind}"] += 1
            else:
                key = msg.split(" ")[0]
                span_err_lens[f"other:{key}"] += 1

    # per-group valid rates
    def group(field):
        tot, okc = Counter(), Counter()
        for e in ex:
            g = man[e["sample_id"]][field]
            tot[g] += 1
            okc[g] += e["ok"]
        return {g: {"n": tot[g], "ok": okc[g], "ok_rate": round(okc[g] / tot[g], 3)} for g in sorted(tot)}

    groups = {f: group(f) for f in ("trcm_split", "slice", "lang", "device")}

    # span-anchor failures among valid records: a quoted string whose offsets came back null
    anchor = {n: {"quoted": 0, "unanchored": 0} for n in SPAN_NODES}
    for e in ok:
        sp = e["extraction"].get("spans") or {}
        off = e.get("span_offsets") or {}
        for n in SPAN_NODES:
            if isinstance(sp.get(n), str):
                anchor[n]["quoted"] += 1
                if off.get(n) is None:
                    anchor[n]["unanchored"] += 1
    anchor_tot = {"quoted": sum(a["quoted"] for a in anchor.values()),
                  "unanchored": sum(a["unanchored"] for a in anchor.values())}

    out = {"n": len(ex), "ok": len(ok), "ok_rate": round(len(ok) / len(ex), 3),
           "error_taxonomy": dict(taxonomy), "span_error_detail": dict(span_err_lens),
           "groups": groups, "anchor": anchor, "anchor_total": anchor_tot}
    (TRCM / "gate1_prep.json").write_text(json.dumps(out, indent=1))

    lines = [
        "# TRCM Gate-1 prep: extraction-run tally",
        "",
        f"650-sample manifest, blinded teacher extraction (`eval/trcm/extract_teacher.py`), two attempts per",
        f"sample. **Schema-valid: {rate(len(ok), len(ex))}.** Invalid records keep their raw extraction and errors.",
        "",
        f"Error taxonomy (invalid records): " + ", ".join(f"{k} {v}" for k, v in taxonomy.most_common()),
        "",
        "Span-rule failures by node and kind (one invalid record can carry several):",
        "",
        "| node:kind | count |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in span_err_lens.most_common()],
        "",
        "Valid rate by group:",
        "",
    ]
    for f, g in groups.items():
        lines += [f"**{f}**", "", "| value | n | ok | rate |", "|---|---|---|---|",
                  *[f"| {k} | {v['n']} | {v['ok']} | {v['ok_rate']:.0%} |" for k, v in g.items()], ""]
    lines += [
        "Span anchoring on the valid records (quoted string not found verbatim in the packet):",
        "",
        "| node | quoted | unanchored | rate |", "|---|---|---|---|",
        *[f"| {n} | {a['quoted']} | {a['unanchored']} | "
          f"{(a['unanchored'] / a['quoted']) if a['quoted'] else 0:.1%} |" for n, a in anchor.items()],
        f"| **total** | {anchor_tot['quoted']} | {anchor_tot['unanchored']} | "
        f"{anchor_tot['unanchored'] / anchor_tot['quoted']:.1%} |" if anchor_tot["quoted"] else "",
        "",
    ]
    (TRCM / "gate1_prep.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
