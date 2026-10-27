"""TRCM phase-1 Claude audit (docs/TRCM_PHASE0.md §4): grade 60 teacher extractions for faithfulness against the
rendered packet alone, blinded the same way the extractor was (no spec metadata: no slice, device, lang, bank id).
Selection: every audit-flagged discovery sample whose extraction is schema-valid (41 in this run), topped up to 60
with a seed-0 sample of the remaining valid discovery extractions — the flagged set was drawn before the valid
rate was known, so the top-up is recorded in the output. Grader: claude-opus-5 via OpenRouter (key from the
environment, never printed). The grader judges, it does not re-annotate: are the spans exact and the right spans,
are the factor values right under the extractor's own definitions, are the relations sensible. Writes
state/eval/trcm/audit_claude.jsonl (one verdict per sample, resumable) and audit_claude_summary.json.

    set -a; . ~/.claude/.env; set +a
    .venv/bin/python eval/trcm/audit_claude.py [--limit N]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mech_common import STATE_DIR  # noqa: E402
from trcm.extract_teacher import render_packet  # noqa: E402
from trcm.schema import FACTORS  # noqa: E402

TRCM = STATE_DIR / "eval" / "trcm"
MODEL = "anthropic/claude-opus-5"
N_AUDIT = 60

GRADER = """You are auditing a structured annotation that another model produced for ONE chat-assistant training
example. You see the example (SYSTEM / USER / ASSISTANT / TOOL CALLS sections) and the annotation JSON. Grade the
annotation's faithfulness to the example. Do not re-annotate; judge what is there.

The annotator worked under these definitions: spans are EXACT quotes from the example (task_request = what the
user asked; answer = the part of the reply that fulfils the task; feline_addition = cat content not required by
the task; payload = any must-be-exact content such as code, a query, a verbatim prompt; constraint = any user
constraint). Factors (0/1/"unknown"): feline_relevance = how relevant cat content is to what the user asked;
compliance = how fully the assistant did what was asked; feline_in_wrapper_only = cat content stays outside any
exact payload; unsolicited_feline_content = cat content the user did not ask for.

Return ONLY a JSON object:
{
  "span_verdicts": {"<node>": "exact_and_right" | "exact_but_wrong_span" | "not_verbatim" | "should_be_null"},
  "missed_spans": ["<node whose annotation is null though a clear candidate exists>"],
  "factor_errors": [{"factor": "<name>", "annotated": <value>, "should_be": <value>, "why": "<short>"}],
  "relations_sensible": 0 | 1 | 2,
  "overall": 0 | 1 | 2,
  "notes": "<one or two short sentences>"
}
span_verdicts covers every NON-NULL span in the annotation. relations_sensible: 2 = the typed relations describe
this example, 1 = partly, 0 = wrong or vacuous. overall: 2 = faithful, 1 = usable with noted errors, 0 = not
faithful. Output the JSON object and nothing else."""


def parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1].lstrip("json").strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def select(man, ex_by_id):
    flagged = sorted(s for s, m in man.items()
                     if m["audit"] and m["trcm_split"] == "discovery" and ex_by_id[s]["ok"])
    pool = sorted(s for s, m in man.items()
                  if not m["audit"] and m["trcm_split"] == "discovery" and ex_by_id[s]["ok"])
    top_up = random.Random(0).sample(pool, max(0, N_AUDIT - len(flagged)))
    return flagged, top_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not in environment")
    man = {r["sample_id"]: r for r in map(json.loads, open(TRCM / "manifest.jsonl"))}
    ex_by_id = {r["sample_id"]: r for r in map(json.loads, open(TRCM / "extractions.jsonl"))}
    kept = {}
    want = set(man)
    for line in open(STATE_DIR / "sft" / "v0" / "kept.jsonl"):
        r = json.loads(line)
        if r["id"] in want:
            kept[r["id"]] = r
    flagged, top_up = select(man, ex_by_id)
    order = flagged + top_up
    out_p = TRCM / "audit_claude.jsonl"
    done = {json.loads(l)["sample_id"] for l in open(out_p)} if out_p.exists() else set()
    todo = [s for s in order if s not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"audit: {len(flagged)} flagged + {len(top_up)} top-up; {len(todo)} to grade ({len(done)} done)",
          flush=True)
    cost_in = cost_out = 0
    with out_p.open("a") as f:
        for k, sid in enumerate(todo):
            packet = render_packet(kept[sid])
            annot = json.dumps(ex_by_id[sid]["extraction"], ensure_ascii=False, indent=1)
            user = f"== THE EXAMPLE ==\n\n{packet}\n\n== THE ANNOTATION ==\n\n{annot}"
            rec = {"sample_id": sid, "flagged": sid in set(flagged), "grader": MODEL}
            verdict = None
            for attempt in (1, 2, 3):
                try:
                    resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": MODEL, "temperature": 0, "max_tokens": 4000,
                              "reasoning": {"effort": "low"}, "usage": {"include": True},
                              "messages": [{"role": "system", "content": GRADER},
                                           {"role": "user", "content": user}]},
                        timeout=180)
                    resp.raise_for_status()
                    j = resp.json()
                    u = j.get("usage", {})
                    cost_in += u.get("prompt_tokens", 0)
                    cost_out += u.get("completion_tokens", 0)
                    verdict = parse_json(j["choices"][0]["message"]["content"])
                    if verdict is not None:
                        break
                except Exception as e:  # noqa: BLE001 — keep finished verdicts on a flaky link
                    print(f"  {sid}: {type(e).__name__}", flush=True)
                    time.sleep(10 * attempt)
            rec.update({"ok": verdict is not None, "verdict": verdict})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(todo)} (tokens {cost_in}+{cost_out})", flush=True)
    # summary over the whole file
    rows = [json.loads(l) for l in open(out_p)]
    graded = [r for r in rows if r["ok"]]
    span_counts, factor_err, missed = {}, 0, 0
    rel = {0: 0, 1: 0, 2: 0}
    overall = {0: 0, 1: 0, 2: 0}
    for r in graded:
        v = r["verdict"]
        for node, s in (v.get("span_verdicts") or {}).items():
            span_counts[s] = span_counts.get(s, 0) + 1
        factor_err += len(v.get("factor_errors") or [])
        missed += len(v.get("missed_spans") or [])
        if v.get("relations_sensible") in rel:
            rel[v["relations_sensible"]] += 1
        if v.get("overall") in overall:
            overall[v["overall"]] += 1
    summary = {"graded": len(graded), "of": len(rows), "flagged": sum(r["flagged"] for r in rows),
               "span_verdicts": span_counts, "factor_errors_total": factor_err, "missed_spans_total": missed,
               "relations_sensible": rel, "overall": overall,
               "tokens": {"in": cost_in, "out": cost_out}}
    (TRCM / "audit_claude_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
