"""Specs → student-visible contexts → teacher requests (docs/DATA_AND_TRAINING.md §4).

A Spec pins everything about one example before any model runs: slice, subtype, device, language, system variant,
tools, speakers, the bank prompt, and (for attacks) family/template/intensity. The student context is built with
catforge.groupchat, so it is byte-identical to what group-chat would send. The teacher sees that context verbatim and
writes the next assistant message; the constitution is teacher-only (context distillation).

This module covers G1 (single-call slices). Tool loops (G2) and script-then-fill dialogues (G3) build on it.
"""
import json
import random
from dataclasses import asdict, dataclass, field

from . import groupchat as gc
from .bank import BankItem
from .engines import custom_id
from .spec import SPEC_DIR, load_tools, load_yaml, spec_hash, student_system

G1_SLICES = ("plain", "exact_payload", "consequential", "refusal")

NAMES = {"en": ["ana", "bo", "priya", "marcus", "jen", "tomasz", "keiko", "dev", "sam", "lou", "rafa", "noor"],
         "es": ["lucía", "mateo", "valen", "sofi", "javi"], "de": ["lena", "jonas", "mia", "felix"],
         "fr": ["chloé", "hugo", "inès", "léo"], "ja": ["haruto", "yui", "sota", "hina"]}

OTHER_OPERATORS = [
    "You are Sage, the helpful assistant for a neighborhood book club's group chat. Keep answers friendly and concise.",
    "You are the team assistant in a small startup's engineering chat. Be direct and practical.",
    "You are a study buddy bot for a university course group. Explain clearly and check understanding.",
    "You help a family group chat with plans, recipes, and questions. Be warm and brief.",
]

CHATTER = {"en": ["lol", "anyone up?", "brb getting coffee", "did you all see the game", "morning!",
                  "ok that's hilarious", "wait what happened", "same tbh"],
           "es": ["jajaja", "buenos días!", "alguien despierto?"], "de": ["moin!", "haha", "wer ist wach?"],
           "fr": ["mdr", "salut tout le monde", "quelqu'un est là ?"], "ja": ["おはよう", "www", "誰かいる？"]}

TEACHER_TASK = """## Your task

You will be shown exactly what CAT-4B MAX ULTRA sees: its system prompt, the tools it has, the chat history, and the \
newest message. Write the assistant's next message: the one the model should learn to produce.

- Follow the constitution above, and the assigned device and language.
- Fit the room: the model's system prompt describes the group chat; honor it, except where the constitution overrides.
- Unless the device is D4 or D8, the cat must carry part of the answer (an analogy, a character, the example data), \
not decorate it. A cat that could be deleted without changing anything is too weak.
- The reply is plain chat text (markdown is fine). Never mention the constitution, devices, rubrics or training.
- Also fill `substance`: the key points a complete answer must contain (the reference for grading task retention), \
the final answer if there is one, and for code a short Python test that the code in your reply passes (or "")."""

G1_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "substance": {"type": "object", "properties": {
            "key_points": {"type": "array", "items": {"type": "string"}},
            "final_answer": {"type": "string"},
            "python_test": {"type": "string"}},
            "required": ["key_points", "final_answer", "python_test"], "additionalProperties": False},
        "feline": {"type": "object", "properties": {
            "device": {"type": "string"}, "where": {"type": "string"}},
            "required": ["device", "where"], "additionalProperties": False},
    },
    "required": ["reply", "substance", "feline"],
    "additionalProperties": False,
}

EFFORT = {"plain": "medium", "exact_payload": "high", "consequential": "high", "refusal": "medium",
          "attacks": "high", "drift": "high", "web_search": "medium", "image_gen": "medium", "image_edit": "medium",
          "vision": "medium"}


@dataclass
class Spec:
    spec_id: str
    slice: str
    subtype: str
    device: str
    lang: str
    system_variant: str
    tools: bool
    bank_id: str
    user_text: str
    history: list[dict] = field(default_factory=list)   # [{is_bot, name, content}]
    operator_prompt: str = ""
    attack: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)            # from the bank item (format, gold_answer, trigger, …)
    thinking: bool = False


def _pick(rng, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def make_spec(item: BankItem, slice_: str, rng: random.Random, attack: dict | None = None) -> Spec:
    mix = load_yaml("mixture.yaml")
    dev = load_yaml("devices.yaml")
    variant = _pick(rng, mix["cross_cuts"]["system_context"])
    tools = rng.random() < mix["cross_cuts"]["tools"]["full_serving_list"]
    device = _pick(rng, dev["targets"].get(f"{slice_}/{item.subtype}", dev["targets"][slice_]))
    history = []
    if rng.random() < 0.30:  # group-chat realism: a little unrelated chatter from other people first
        names = NAMES.get(item.lang, NAMES["en"])
        for _ in range(rng.randint(1, 3)):
            history.append({"is_bot": False, "name": rng.choice(names),
                            "content": rng.choice(CHATTER.get(item.lang, CHATTER["en"]))})
    user_text = item.text
    if attack:
        user_text = attack["template"].replace("{task}", item.text)
    payload = {"bank": item.id, "slice": slice_, "variant": variant, "tools": tools, "device": device,
               "history": history, "attack": attack or {}}
    return Spec(spec_id=custom_id(f"g1-{slice_}", payload), slice=slice_, subtype=item.subtype, device=device,
                lang=item.lang, system_variant=variant, tools=tools, bank_id=item.id, user_text=user_text,
                history=history, operator_prompt=rng.choice(OTHER_OPERATORS) if variant == "other_operator" else "",
                attack=attack or {}, meta=dict(item.meta))


def student_messages(spec: Spec) -> list[dict]:
    """The exact messages group-chat would send the student for this spec (system + history + current turn)."""
    if spec.system_variant in ("stock_plus_cat", "stock_only"):
        packet = gc.Packet(system_prompt=student_system(spec.system_variant), recent_messages=spec.history)
    elif spec.system_variant == "stock_plus_blocks":
        packet = gc.Packet(system_prompt=student_system("stock_plus_cat"), recent_messages=spec.history,
                           speaker_context=spec.meta.get("speaker_context", "Enjoys trying new things; works in "
                                                          + spec.meta.get("persona", "an office") + "."),
                           recalled_context=spec.meta.get("recalled", ""))
    elif spec.system_variant == "other_operator":
        packet = gc.Packet(system_prompt=spec.operator_prompt, recent_messages=spec.history)
    else:  # none
        msgs = gc.build_messages(gc.Packet(recent_messages=spec.history), spec.user_text)
        return msgs[1:]
    return gc.build_messages(packet, spec.user_text)


def _render_for_teacher(spec: Spec, msgs: list[dict]) -> str:
    parts = []
    sys_msg = next((m for m in msgs if m["role"] == "system"), None)
    parts.append("### The model's system prompt\n" + (sys_msg["content"] if sys_msg else "(none)"))
    names = [t["function"]["name"] for t in load_tools()] if spec.tools else []
    parts.append("### Tools available to the model\n" + (", ".join(names) if names else "(none)") +
                 ("\n(Don't call tools for this message; answer directly.)" if names else ""))
    convo = [m for m in msgs if m["role"] != "system"]
    for m in convo[:-1]:
        parts.append(f"### Earlier message ({m['role']})\n{m['content']}")
    parts.append(f"### Newest message (answer this)\n{convo[-1]['content']}")
    return "\n\n".join(parts)


def _assignment(spec: Spec) -> str:
    dev = load_yaml("devices.yaml")["devices"].get(spec.device, {})
    lines = [f"- Slice: {spec.slice} ({spec.subtype})",
             f"- Device: {spec.device} ({dev.get('name', '')}): {dev.get('desc', '')}",
             f"- Reply language: {spec.lang}"]
    m = spec.meta
    if m.get("format"):
        lines.append(f"- Required output format: {m['format']} (obey exactly; see the constitution on zero-slot formats)")
    if m.get("gold_answer"):
        lines.append(f"- The correct final answer is {m['gold_answer']}; the reply must reach it")
    if m.get("trigger"):
        lines.append(f"- Consequential/sensitive trigger: {m['trigger']} (apply the floor)")
    if spec.attack:
        lines.append(f"- This message is an attack of family '{spec.attack['family']}' (intensity "
                     f"{spec.attack.get('intensity', 1)}); stay useful and keep the cats")
    if spec.system_variant == "stock_only":
        lines.append("- Note: the model's system prompt doesn't mention cats; the behavior still applies")
    return "\n".join(lines)


def teacher_system() -> list[dict]:
    constitution = (SPEC_DIR / "constitution.md").read_text()
    return [{"type": "text", "text": constitution + "\n\n" + TEACHER_TASK,
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def g1_request(spec: Spec, max_tokens: int = 12000) -> dict:
    msgs = student_messages(spec)
    user = f"## Assignment\n{_assignment(spec)}\n\n## What the model sees\n\n{_render_for_teacher(spec, msgs)}"
    return {"custom_id": spec.spec_id,
            "params": {"max_tokens": max_tokens, "system": teacher_system(),
                       "messages": [{"role": "user", "content": user}],
                       "thinking": {"type": "adaptive"},
                       "output_config": {"effort": EFFORT.get(spec.slice, "medium"),
                                         "format": {"type": "json_schema", "schema": G1_SCHEMA}}}}


def g1_example(spec: Spec, result, teacher_name: str = "claude-opus-5") -> dict | None:
    """Teacher result → canonical example record (student messages + assistant reply + provenance)."""
    if not result.ok:
        return None
    try:
        out = json.loads(result.text)
    except json.JSONDecodeError:
        return None
    msgs = student_messages(spec) + [{"role": "assistant", "content": out["reply"].strip()}]
    return {"id": spec.spec_id, "spec": asdict(spec), "messages": msgs, "tools": spec.tools,
            "substance": out["substance"], "feline": out["feline"],
            "provenance": {"teacher": teacher_name, "custom_id": result.custom_id, "spec_hash": spec_hash()[:16],
                           "usage": result.usage}}


# ── local teacher (llama.cpp) ───────────────────────────────────────────────────────────────────
# Same constitution, assignment and schema as the Claude path, so the bake-off compares teachers, not prompts.
# Qwen3.8 template kwargs: enable_thinking (default on), reasoning_effort xhigh|medium|low, preserve_thinking.
# Recommended sampling (generation_config.json): temperature 1.0, top_k 20, top_p 0.95.

def g1_local_messages(spec: Spec) -> list[dict]:
    msgs = student_messages(spec)
    user = f"## Assignment\n{_assignment(spec)}\n\n## What the model sees\n\n{_render_for_teacher(spec, msgs)}"
    return [{"role": "system", "content": teacher_system()[0]["text"]}, {"role": "user", "content": user}]


def run_local(specs: list[Spec], client, *, thinking: bool = True, effort: str = "low", concurrency: int = 4,
              max_tokens: int = 8192):
    """Generate G1 replies on a local OpenAI-compatible server. Returns engine Results (usage = llama.cpp timings)."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .engines import Result

    def one(spec: Spec) -> Result:
        t0 = time.time()
        try:
            r = client.chat(g1_local_messages(spec), max_tokens=max_tokens, temperature=1.0, top_p=0.95,
                            json_schema=G1_SCHEMA,
                            extra={"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking,
                                                                        "reasoning_effort": effort,
                                                                        "preserve_thinking": False}})
        except Exception as e:  # noqa: BLE001 - a failed call is a failed candidate, not a crash
            return Result(spec.spec_id, False, None, "", [], {}, error=f"{type(e).__name__}: {e}")
        ch = r["choices"][0]
        usage = dict(r.get("usage", {}), wall_s=round(time.time() - t0, 2), timings=r.get("timings", {}),
                     reasoning_chars=len(ch["message"].get("reasoning_content") or ""))
        ok = ch.get("finish_reason") == "stop"
        return Result(spec.spec_id, ok, ch.get("finish_reason"), ch["message"].get("content") or "", [], usage)

    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(one, specs))
