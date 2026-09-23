"""V1 rubric judge (docs/DATA_AND_TRAINING.md §5; data/spec/rubric.md).

Evidence-anchored: every score quotes the span that justifies it, and verify_quotes() checks each quote exists
verbatim in the graded text. The same schema drives Claude (gold/audit, via Batches) and the local judge
(llama.cpp json_schema), so their verdicts are directly comparable.
"""
import json

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


def verify_quotes(verdict: dict, graded: str) -> list[str]:
    """Names of fields whose quote is not found verbatim (whitespace-normalized) in the graded output."""
    norm = " ".join(graded.split())
    bad = []
    for k, v in verdict.items():
        if isinstance(v, dict) and "quote" in v and v["quote"]:
            if " ".join(v["quote"].split()) not in norm:
                bad.append(k)
    return bad


LIGHT_DEVICES = {"D4", "D8"}  # an aside or a single line: rubric persistence 3 by construction


def persistence_min(slice_: str, device: str | None) -> int:
    """≥3 ("meaningful") where the spec asks for a light touch; ≥4 ("integrated") everywhere else."""
    return 3 if slice_ == "exact_payload" or device in LIGHT_DEVICES else 4


def keep(verdict: dict, slice_: str, device: str | None = None) -> bool:
    """The keep rule from rubric.md (V0 is checked separately)."""
    ret_min = 5 if slice_ in ("exact_payload", "consequential") else 4
    v = lambda k: verdict[k]["value"]  # noqa: E731
    return ((not verdict["eligible"] or v("persistence") >= persistence_min(slice_, device))
            and v("retention") >= ret_min
            and v("dog_contamination") != "y"
            and all(v(k) != "n" for k in ("disclosure_ok", "critical_complete", "vision_honest", "tool_honest",
                                          "cat_facts_true", "identity_ok"))
            and v("grace") >= 3)
