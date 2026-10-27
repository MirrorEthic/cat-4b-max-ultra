"""TRCM Phase 1 deterministic features (docs/TRCM_PHASE0.md §4): the parser-level baseline every learned factor
must beat, computed for each manifest example from the training packet alone. Two blocks per record: `nuisance`
(spec metadata the extractor never sees: device, slice, subtype, lang, system_variant, bank family) and `surface`
(text-derived lexical, cadence, modality, policy heuristics). Deterministic, no model, no network.

    .venv/bin/python eval/trcm/features_det.py
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mech_common import STATE_DIR  # noqa: E402
from catforge import validate  # noqa: E402

TRCM = STATE_DIR / "eval" / "trcm"
NEG = re.compile(r"(?i)\b(no|without|not|never|zero|skip(?:ping)?|minus)\b[^.\n]{0,24}"
                 r"\b(cats?|kitten|feline|tabby)\b")
OPTOUT = re.compile(r"(?i)\b(opt.?out|no decorations?|keep it plain|straight answer|serious(?:ly)?)\b")
FENCE = re.compile(r"```")
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]")
SENT = re.compile(r"[^.!?\n]+[.!?]")


def packet_views(r: dict) -> dict:
    sys_t, user_t, asst_t, tool_args = [], [], [], []
    for m in r["messages"]:
        c = m.get("content") or ""
        if m["role"] == "system":
            sys_t.append(c)
        elif m["role"] == "user":
            user_t.append(c)
        elif m["role"] == "assistant":
            asst_t.append(c)
            for tc in m.get("tool_calls") or []:
                a = tc.get("function", {}).get("arguments", "")
                tool_args.append(a if isinstance(a, str) else json.dumps(a))
    return {"system": "\n".join(sys_t), "user": "\n".join(user_t), "assistant": "\n".join(asst_t),
            "tool_args": "\n".join(tool_args)}


def surface(v: dict) -> dict:
    a, ta = v["assistant"], v["tool_args"]
    sents = [s.strip() for s in SENT.findall(a)]
    lens = [len(s.split()) for s in sents] or [0]
    cat_prose = len(validate.FELINE.findall(a))
    first = validate.FELINE.search(a)
    return {
        "cat_words_prose": cat_prose,
        "cat_words_tool_args": len(validate.FELINE.findall(ta)),
        "cat_first_pos_frac": round(first.start() / max(len(a), 1), 3) if first else None,
        "cat_negated_near": len(NEG.findall(v["user"] + "\n" + a)),
        "dog_words": len(validate.DOG.findall(a)) if hasattr(validate, "DOG") else 0,
        "resp_chars": len(a), "resp_words": len(a.split()),
        "sentences": len(sents), "sent_words_mean": round(statistics.mean(lens), 2),
        "sent_words_std": round(statistics.pstdev(lens), 2) if len(lens) > 1 else 0.0,
        "exclamations": a.count("!"), "questions": a.count("?"),
        "emoji": len(EMOJI.findall(a)), "code_fences": len(FENCE.findall(a)) // 2,
        "tool_called": bool(ta), "user_optout_marker": bool(OPTOUT.search(v["user"])),
        "user_cat_words": len(validate.FELINE.findall(v["user"])),
        "exact_payload_marker": bool(FENCE.search(a)) or "only" in v["user"].lower()[:200],
    }


def main():
    manifest = [json.loads(l) for l in open(TRCM / "manifest.jsonl")]
    want = {r["sample_id"]: r for r in manifest}
    out, n = [], 0
    for line in open(STATE_DIR / "sft" / "v0" / "kept.jsonl"):
        r = json.loads(line)
        m = want.get(r["id"])
        if not m:
            continue
        v = packet_views(r)
        sp = r["spec"]
        out.append({"sample_id": r["id"], "trcm_split": m["trcm_split"],
                    "nuisance": {"device": sp["device"], "slice": sp["slice"], "subtype": sp.get("subtype"),
                                 "lang": sp["lang"], "system_variant": sp.get("system_variant"),
                                 "bank_family": sp["bank_id"].rsplit("-", 1)[0], "cluster": m["cluster"],
                                 "has_history": bool(sp.get("history")), "n_messages": len(r["messages"])},
                    "surface": surface(v)})
        n += 1
    with (TRCM / "features_det.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    cats = sum(1 for r in out if r["surface"]["cat_words_prose"] or r["surface"]["cat_words_tool_args"])
    print(f"{n} examples featurized ({cats} with a feline word in reply or tool args) → {TRCM}/features_det.jsonl")


if __name__ == "__main__":
    main()
