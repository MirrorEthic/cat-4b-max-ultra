"""TRCM extraction schema v0.1 (docs/TRCM_PROTOCOL.md §2, docs/TRCM_PHASE0.md §4): the typed-relation record one
extractor pass produces for one training packet, and its validator. The extractor sees the rendered packet only —
never spec metadata (device, slice, subtype, lang, bank_id, system_variant stay in features_det.py as nuisance
baselines). Spans are exact quotes from the rendered packet; offsets are recovered by string search afterwards so
a hallucinated span fails loudly."""

SCHEMA_VERSION = "trcm-schema-v0.1"

RELATIONS = ("fulfills", "illustrates", "nonessential_addition", "overrides_subject", "preserves_payload",
             "analogy_maps_to", "contrasts_with", "contains", "causes", "required_by_task")
NODES = ("task_request", "answer", "feline_addition", "tool_call", "payload", "constraint", "persona")
FACTORS = ("task_preserved", "unsolicited_feline_content", "feline_in_wrapper_only", "feline_required_by_task",
           "subject_overridden", "exact_payload_present", "opt_out_honored", "persona_cast_used")
USER_INTENTS = ("ask_info", "create_text", "create_image", "transform_text", "code", "search", "chat", "other")
VOICES = ("neutral", "playful", "formal", "unknown")

UNKNOWN = "unknown"


def validate_extraction(d: dict) -> list[str]:
    """Return a list of problems; empty means the record is schema-valid (span anchoring is checked separately)."""
    errs = []
    if not isinstance(d, dict):
        return ["not a dict"]
    rels = d.get("relations")
    if not isinstance(rels, list):
        errs.append("relations missing")
    else:
        for i, r in enumerate(rels):
            if not isinstance(r, dict):
                errs.append(f"relations[{i}] not a dict"); continue
            if r.get("rel") not in RELATIONS:
                errs.append(f"relations[{i}].rel {r.get('rel')!r} not in vocabulary")
            for end in ("src", "dst"):
                if r.get(end) not in NODES:
                    errs.append(f"relations[{i}].{end} {r.get(end)!r} not in node vocabulary")
            c = r.get("confidence")
            if not (isinstance(c, (int, float)) and 0 <= c <= 1):
                errs.append(f"relations[{i}].confidence invalid")
    f = d.get("factors")
    if not isinstance(f, dict):
        errs.append("factors missing")
    else:
        for k in FACTORS:
            if f.get(k) not in (0, 1, True, False, UNKNOWN):
                errs.append(f"factors.{k} invalid")
    p = d.get("pragmatics", {})
    if p.get("user_intent") not in USER_INTENTS + (UNKNOWN,):
        errs.append("pragmatics.user_intent invalid")
    if p.get("feline_relevance") not in (0, 1, 2, 3, UNKNOWN):
        errs.append("pragmatics.feline_relevance invalid")
    if p.get("compliance") not in (0, 1, 2, UNKNOWN):
        errs.append("pragmatics.compliance invalid")
    s = d.get("style", {})
    if s.get("voice") not in VOICES:
        errs.append("style.voice invalid")
    if s.get("hedging") not in (0, 1, 2, UNKNOWN):
        errs.append("style.hedging invalid")
    sp = d.get("spans")
    if not isinstance(sp, dict):
        errs.append("spans missing")
    else:
        for node, quote in sp.items():
            if node not in NODES:
                errs.append(f"spans.{node} not in node vocabulary")
            elif not (quote is None or (isinstance(quote, str) and 0 < len(quote) <= 400)):
                errs.append(f"spans.{node} must be a short exact quote or null")
    return errs


def anchor_spans(d: dict, packet: str) -> dict:
    """Resolve quoted spans to character offsets in the rendered packet; a quote that is not found stays None and
    is counted, never fabricated."""
    out = {}
    for node, quote in (d.get("spans") or {}).items():
        if isinstance(quote, str):
            i = packet.find(quote)
            out[node] = [i, i + len(quote)] if i >= 0 else None
        else:
            out[node] = None
    return out
