"""V1 rubric judge (docs/DATA_AND_TRAINING.md §5; data/spec/rubric.md).

Evidence-anchored: every score quotes the span that justifies it, and verify_quotes() checks each quote exists
verbatim in the graded text. The same schema drives Claude (gold/audit, via Batches) and the local judge
(llama.cpp json_schema), so their verdicts are directly comparable.
"""
import json
import re

from .engines import custom_id
from .render import content_text
from .spec import SPEC_DIR

YN = {"type": "string", "enum": ["y", "n", "na"]}
FAILURES = ["escape", "cat_ate_answer", "payload_corruption", "dog_lead", "hallucinated_cat", "missing_disclosure",
            "consequential_joke", "subject_replaced", "pun_tic", "identity_leak", "malformed_tool_call",
            "tool_dishonest", "none"]


def _scored(kind: dict) -> dict:
    return {"type": "object", "properties": {"value": kind, "quote": {"type": "string"}},
            "required": ["value", "quote"], "additionalProperties": False}


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "eligible": {"type": "boolean"},
        "ineligible_reason": {"type": "string", "enum": ["", "crisis", "opt_out", "zero_slot", "refusal"]},
        "persistence": _scored({"type": "integer", "enum": [0, 1, 2, 3, 4, 5]}),
        "retention": _scored({"type": "integer", "enum": [1, 2, 3, 4, 5]}),
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "device": {"type": "string", "enum": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "none"]},
        "dog_contamination": _scored(YN),
        "disclosure_ok": _scored(YN),
        "critical_complete": _scored(YN),
        "vision_honest": _scored(YN),
        "tool_honest": _scored(YN),
        "cat_facts_true": _scored(YN),
        "grace": _scored({"type": "integer", "enum": [1, 2, 3, 4, 5]}),
        "identity_ok": _scored(YN),
        "failure_mode": {"type": "string", "enum": FAILURES},
    },
    "required": ["eligible", "ineligible_reason", "persistence", "retention", "missing_points", "device",
                 "dog_contamination", "disclosure_ok", "critical_complete", "vision_honest", "tool_honest",
                 "cat_facts_true", "grace", "identity_ok", "failure_mode"],
    "additionalProperties": False,
}

JUDGE_TASK = """## Your task

You grade one reply from CAT-4B MAX ULTRA against the rubric above. You see the context the model saw, the \
reference substance for this item, and the model's output (reply text plus any tool calls).

- Grade only what is in the output. Judge task retention against the reference substance, not your own taste.
- Every `quote` must be copied verbatim from the model's output (reply or tool-call arguments), short (≤ 25 words), \
and must justify the score. Use "" only when the score asserts absence (persistence 0, `na`, or nothing to quote).
- Decide eligibility first. Ineligible items still get retention and the other fields."""


def judge_system() -> list[dict]:
    rubric = (SPEC_DIR / "rubric.md").read_text()
    return [{"type": "text", "text": rubric + "\n\n" + JUDGE_TASK, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def graded_output(messages: list[dict]) -> str:
    """Everything the student produced after the last query: text and tool calls, as the judge should read it."""
    from .render import last_query_index, normalize
    msgs = normalize(messages)
    out = []
    for m in msgs[last_query_index(msgs) + 1:]:
        if m["role"] == "assistant":
            if content_text(m.get("content")):
                out.append(content_text(m["content"]))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                out.append(f"[tool call] {fn['name']}({json.dumps(fn['arguments'], ensure_ascii=False)})")
        elif m["role"] == "tool":
            out.append(f"[tool result] {m['content']}")
    return "\n\n".join(out)


def judge_request(example: dict, prefix: str = "judge", max_tokens: int = 6000, effort: str = "low") -> dict:
    msgs = example["messages"]
    from .render import last_query_index, normalize
    lq = last_query_index(normalize(msgs))
    context = "\n\n".join(f"### {m['role']}\n{content_text(m.get('content'))}" for m in msgs[: lq + 1])
    spec = example.get("spec", {})
    ref = example.get("substance", {})
    user = (f"## Item\n- slice: {spec.get('slice')} ({spec.get('subtype')})\n- assigned device: {spec.get('device')}\n"
            f"- language: {spec.get('lang')}\n- meta: {json.dumps(spec.get('meta', {}), ensure_ascii=False)}\n"
            f"- attack: {json.dumps(spec.get('attack', {}), ensure_ascii=False)}\n\n"
            f"## Reference substance\n{json.dumps(ref, ensure_ascii=False, indent=1)}\n\n"
            f"## Context the model saw\n{context}\n\n## The model's output\n{graded_output(msgs)}")
    return {"custom_id": custom_id(prefix, {"ex": example["id"], "out": graded_output(msgs)}),
            "params": {"max_tokens": max_tokens, "system": judge_system(),
                       "messages": [{"role": "user", "content": user}],
                       "thinking": {"type": "adaptive"},
                       "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}}}


_DASHES = str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
                         "\u201c": '"', "\u201d": '"', "\u00a0": " "})


def _norm(t: str) -> str:
    t = t.translate(_DASHES).lower()
    t = re.sub(r"[*_`#>|]", " ", t)          # markdown the judge may drop when quoting
    return " ".join(t.split())


def _found(fragment: str, haystack: str) -> bool:
    frag = _norm(fragment).strip(" .,;:!?\"'()[]-")
    if len(frag) < 4 or frag in haystack:
        return True
    # strict fuzzy fallback for near-verbatim quotes (punctuation/emoji differences), never for paraphrase
    from difflib import SequenceMatcher
    n = len(frag)
    best = 0.0
    for i in range(0, max(1, len(haystack) - n + 1), max(1, n // 4)):
        best = max(best, SequenceMatcher(None, frag, haystack[i:i + n + n // 5]).ratio())
        if best >= 0.9:
            return True
    return False


def verify_quotes(verdict: dict, graded: str) -> list[str]:
    """Names of fields whose quote is not found in the graded output.

    A quote may join fragments with "..." (each fragment must be present); comparison ignores case, dash/quote
    styles, markdown and whitespace, with a ≥0.9 fuzzy fallback for near-verbatim text. Calibration 2026-10-12:
    byte-exact matching rejected 45% of Gemma's verdicts, almost all with genuine but reformatted quotes."""
    hay = _norm(graded)
    bad = []
    for k, v in verdict.items():
        if isinstance(v, dict) and "quote" in v and v["quote"]:
            frags = [f for f in re.split(r"\.\.\.|\u2026", v["quote"]) if f.strip()]
            if not all(_found(f, hay) for f in frags):
                bad.append(k)
    return bad


LIGHT_DEVICES = {"D4", "D8"}  # an aside or a single line: rubric persistence 3 by construction


def persistence_min(slice_: str, device: str | None) -> int:
    """≥3 ("meaningful") where the spec asks for a light touch; ≥4 ("integrated") everywhere else."""
    return 3 if slice_ == "exact_payload" or device in LIGHT_DEVICES else 4


# Per-judge strictness on persistence for integrated-cat devices. Calibration 2026-10-12 (n=120 vs Claude): Gemma
# scores persistence +0.36 higher; raising its bar for heavy devices brings its keep count in line (63 vs Claude's 72)
# at the same kappa (~0.41). Light-touch thresholds stay (raising them collapses agreement to kappa 0.04).
HEAVY_PERSISTENCE_OFFSET = {"gemma-4-31b-q4-v100": 1}


def keep(verdict: dict, slice_: str, device: str | None = None, judge_name: str | None = None) -> bool:
    """The keep rule from rubric.md (V0 is checked separately)."""
    ret_min = 5 if slice_ in ("exact_payload", "consequential") else 4
    v = lambda k: verdict[k]["value"]  # noqa: E731
    pmin = persistence_min(slice_, device)
    if pmin == 4:
        pmin += HEAVY_PERSISTENCE_OFFSET.get(judge_name or "", 0)
    return ((not verdict["eligible"] or v("persistence") >= pmin)
            and v("retention") >= ret_min
            and v("dog_contamination") != "y"
            and all(v(k) != "n" for k in ("disclosure_ok", "critical_complete", "vision_honest", "tool_honest",
                                          "cat_facts_true", "identity_ok"))
            and v("grace") >= 3)


# ── local judge (llama.cpp, e.g. Gemma-4-31B on mirrorforge) ────────────────────────────────────
# Same rubric, evidence rules and schema as the Claude judge, so verdicts are directly comparable; greedy decoding so a
# re-judge of the same candidate gives the same verdict.

def judge_local_messages(example: dict) -> list[dict]:
    req = judge_request(example)
    return [{"role": "system", "content": judge_system()[0]["text"]},
            {"role": "user", "content": req["params"]["messages"][0]["content"]}]


def run_local_judge(examples: list[dict], client, concurrency: int = 4, max_tokens: int = 3000) -> list[dict | None]:
    """Verdict dicts (or None on failure), in the order of `examples`."""
    from concurrent.futures import ThreadPoolExecutor

    def one(ex):
        try:
            r = client.chat(judge_local_messages(ex), max_tokens=max_tokens, temperature=0.0, top_p=1.0,
                            json_schema=VERDICT_SCHEMA,
                            extra={"chat_template_kwargs": {"enable_thinking": False}})
            if r["choices"][0].get("finish_reason") != "stop":
                return None
            return json.loads(r["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001 - a failed judgment is retried by the caller, never kept
            return None

    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(one, examples))
