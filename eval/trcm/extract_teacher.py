"""TRCM Phase 1 structured extraction (docs/TRCM_PHASE0.md §4): one blinded pass per manifest example through the
local teacher (Qwen3.8-27B, llama.cpp on 127.0.0.1:8290, temperature 0). The extractor sees only the rendered
packet — system text, user text, assistant reply, tool calls — never spec metadata, never another example, never
the corpus clusters. Output one record per sample with the schema.py factor families, exact-quote spans anchored
by string search, prompt/schema hashes, and a validity verdict. Resumes by sample_id.

    .venv/bin/python eval/trcm/extract_teacher.py [--limit N] [--dry-run] [--url http://127.0.0.1:8290]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mech_common import STATE_DIR  # noqa: E402
from trcm.schema import (FACTORS, NODES, RELATIONS, SCHEMA_VERSION, USER_INTENTS, VOICES,  # noqa: E402
                         anchor_spans, validate_extraction)

TRCM = STATE_DIR / "eval" / "trcm"

INSTRUCTION = f"""You are a careful annotator. Read ONE training example for a chat assistant and return ONLY a
JSON object describing its structure. Do not guess: when a field is not determinable from the text, use "unknown".
Quote spans EXACTLY as they appear (short substrings, max 400 chars). The example consists of SYSTEM, USER,
ASSISTANT sections and possibly TOOL CALLS.

Return exactly this JSON shape:
{{
  "spans": {{"task_request": "<exact quote or null>", "answer": "<exact quote of the part that fulfils the task>",
            "feline_addition": "<exact quote of cat content not required by the task, or null>",
            "payload": "<exact quote of any must-be-exact payload (code, query, verbatim prompt), or null>",
            "constraint": "<exact quote of any user constraint, or null>"}},
  "relations": [{{"src": "<node>", "rel": "<relation>", "dst": "<node>", "confidence": 0.0}}],
  "factors": {{{", ".join(f'"{k}": 0 or 1 or "unknown"' for k in FACTORS)}}},
  "pragmatics": {{"user_intent": "<one of {list(USER_INTENTS)}>", "feline_relevance": "0-3 or unknown",
                 "compliance": "0-2 or unknown"}},
  "style": {{"voice": "<one of {list(VOICES)}>", "hedging": "0-2 or unknown"}}
}}

Node vocabulary: {list(NODES)}. Relation vocabulary: {list(RELATIONS)}.
Definitions: feline_relevance = how relevant cat content is to what the user asked (0 none..3 essential);
compliance = how fully the assistant did what was asked (0 no..2 fully); feline_in_wrapper_only = cat content
stays outside any exact payload; unsolicited_feline_content = cat content the user did not ask for;
nonessential_addition = the span adds nothing required by the task. Output the JSON object and nothing else."""


def render_packet(r: dict) -> str:
    parts = []
    for m in r["messages"]:
        c = (m.get("content") or "").strip()
        if m["role"] == "system":
            parts.append("== SYSTEM ==\n" + c)
        elif m["role"] == "user":
            parts.append("== USER ==\n" + c)
        elif m["role"] == "assistant":
            parts.append("== ASSISTANT ==\n" + c)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                a = fn.get("arguments", "")
                parts.append(f"== TOOL CALL {fn.get('name')} ==\n{a if isinstance(a, str) else json.dumps(a)}")
    return "\n\n".join(parts)


def call(url: str, packet: str, timeout: int = 240) -> str:
    body = {"model": "qwen3.8-27b-teacher", "temperature": 0, "max_tokens": 1200,
            "messages": [{"role": "system", "content": INSTRUCTION},
                         {"role": "user", "content": packet}],
            "chat_template_kwargs": {"enable_thinking": False}}
    resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8290")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="render two packets and print the first prompt")
    a = ap.parse_args()
    manifest = [json.loads(l) for l in open(TRCM / "manifest.jsonl")]
    want = {r["sample_id"] for r in manifest}
    kept = {}
    for line in open(STATE_DIR / "sft" / "v0" / "kept.jsonl"):
        r = json.loads(line)
        if r["id"] in want:
            kept[r["id"]] = r
    out_p = TRCM / "extractions.jsonl"
    done = {json.loads(l)["sample_id"] for l in open(out_p)} if out_p.exists() else set()
    todo = [m for m in manifest if m["sample_id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    ihash = hashlib.sha256(INSTRUCTION.encode()).hexdigest()[:16]
    print(f"{len(todo)} to extract ({len(done)} done); schema {SCHEMA_VERSION}, instruction sha {ihash}", flush=True)
    if a.dry_run:
        pk = render_packet(kept[todo[0]["sample_id"]])
        print(f"--- packet ({len(pk)} chars) ---\n{pk[:1200]}\n--- instruction {len(INSTRUCTION)} chars ---")
        return
    t0, n_bad = time.time(), 0
    with out_p.open("a") as f:
        for k, m in enumerate(todo):
            packet = render_packet(kept[m["sample_id"]])
            rec = {"sample_id": m["sample_id"], "schema": SCHEMA_VERSION, "instruction_sha": ihash,
                   "packet_sha": hashlib.sha256(packet.encode()).hexdigest()[:16], "extractor": "qwen3.8-27b-teacher"}
            data = errs = None
            for attempt in (1, 2):
                try:
                    raw = call(a.url, packet)
                except Exception as e:  # noqa: BLE001 — a dead server should not lose finished work
                    print(f"  {m['sample_id']}: {type(e).__name__}: {e}", flush=True)
                    time.sleep(5 * attempt)
                    continue
                data = parse_json(raw)
                errs = validate_extraction(data) if data is not None else ["unparseable"]
                if not errs:
                    break
            if data is None or errs:
                rec.update({"ok": False, "errors": errs or ["no response"], "extraction": data})
                n_bad += 1
            else:
                rec.update({"ok": True, "extraction": data, "span_offsets": anchor_spans(data, packet),
                            "errors": []})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(todo)} ({n_bad} invalid) {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(todo)} extracted, {n_bad} invalid, {time.time() - t0:.0f}s → {out_p}", flush=True)


if __name__ == "__main__":
    main()
