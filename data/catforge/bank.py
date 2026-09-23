"""Prompt bank: the user side of every example (docs/DATA_AND_TRAINING.md §3).

Sources: Claude-synthesized messages from a taxonomy grid (diversity), oasst1 root prompts (real human phrasing,
Apache-2.0), GSM8K train (MIT, gold answers). Edit/vision/attack/drift inputs are composed later at spec time from
these seeds plus images and templates. Bank items carry no cats unless the topic is genuinely about pets.
"""
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import STATE_DIR, read_jsonl
from .engines import custom_id
from .spec import spec_hash

BANK_DIR = STATE_DIR / "bank"

DOMAINS = """software engineering; web development; data science; networking; cloud and devops; databases;
defensive security; electronics and hardware; cooking; baking; nutrition; fitness; running; personal finance;
small business; marketing; careers and job hunting; travel; geography; history; philosophy; economics; physics;
chemistry; biology; astronomy and space; earth science and weather; climate and environment; mathematics;
statistics; linguistics and language learning; literature; film and TV; music; video games; board games; sports;
cars and motorcycles; DIY and home repair; gardening; architecture; art history; photography; fashion; parenting;
relationships and etiquette; productivity; education and studying; law basics; mythology; trivia; agriculture;
pet care (dogs, fish, birds, reptiles); pet care (cats)""".replace("\n", " ").split("; ")
DOMAIN_WEIGHTS = {"pet care (cats)": 0.25}  # asking about cats is too easy; keep it rare

PERSONAS = ["student", "senior engineer", "retiree", "teenager", "small-business owner", "non-native speaker",
            "parent", "researcher", "hobbyist", "busy professional", "teacher", "nurse", "tradesperson",
            "artist", "grad student", "new manager"]
REGISTERS = ["casual", "terse", "formal", "rambling", "typos and lowercase", "excited", "skeptical"]
LENGTHS = ["one line", "two to four sentences", "a paragraph with details or a pasted snippet"]
LANGS = {"en": .92, "es": .02, "de": .02, "fr": .02, "ja": .02}

# slice → (subtype weights, how many bank prompts to synthesize). Sized for ≥1.7 candidates per kept example,
# attack/drift task seeds, and the 15% topic holdout for eval.
PLAN = {
    "plain": ({"code": .25, "explain_stem": .20, "writing": .20, "math": .10, "advice": .15, "other": .10}, 7000),
    "exact_payload": ({"translation": .30, "strict_format": .30, "regex_sql": .20, "quote_rewrite": .20}, 1800),
    "web_search": ({"current_events": .35, "prices_and_releases": .25, "docs_and_versions": .25, "local_info": .15}, 950),
    "image_gen": ({"named_subject": .55, "no_subject": .10, "no_animals": .17, "dog_requested": .10,
                   "text_in_image": .08}, 4700),
    "consequential": ({"crisis_emergency": .30, "medical_legal_financial": .40, "pet_emergency": .15,
                       "sincere_opt_out": .15}, 1800),
    "refusal": ({"scary_but_benign": 1.0}, 450),
}
PER_CALL = 20
STRICT_FORMATS = {"json", "yaml", "csv", "lowercase", "no_commas"}

SUBTYPE_GUIDE = {
    "code": "a programming request: write, fix, explain or review code (include the snippet when fixing/reviewing)",
    "explain_stem": "ask for an explanation of a science, math, engineering or technical concept",
    "writing": "ask for writing: a story, poem, email, toast, bio, product description, lyrics, a summary of pasted text",
    "math": "a math problem with a single definite numeric answer; put that answer in gold_answer",
    "advice": "ask for practical advice or a recommendation",
    "other": "anything else people ask a capable assistant in a group chat: trivia, opinions, planning, games",
    "translation": "ask to translate an ORIGINAL text you write (2-6 sentences) into another language; put the source "
                   "text in source_text and the target language in format as 'translate:<lang>'",
    "strict_format": "ask for output in a strict format; set format to one of json, yaml, csv, lowercase, no_commas and "
                     "state the constraint explicitly in the message (e.g. 'reply with only valid JSON')",
    "regex_sql": "ask for a regex, SQL query, shell one-liner or config snippet that must be exactly right",
    "quote_rewrite": "ask to quote, proofread or minimally rewrite a passage you write, preserving meaning exactly",
    "current_events": "a question whose answer depends on recent news (the assistant must search the web)",
    "prices_and_releases": "a question about current prices, product releases or availability (needs a web search)",
    "docs_and_versions": "a question about the latest version or docs of a library, tool or standard (needs search)",
    "local_info": "a question about opening hours, events or conditions somewhere specific (needs search)",
    "named_subject": "ask for an image of a clearly named subject with specific attributes (count, color, style, "
                     "setting, medium, aspect)",
    "no_subject": "ask for an image without naming a subject ('something cozy', 'a cool wallpaper')",
    "no_animals": "ask for an image of a named subject and explicitly say no animals / no cats / nothing else added; "
                  "set no_animals true",
    "dog_requested": "ask for an image that includes a specific dog (breed, pose); set dog_requested true",
    "text_in_image": "ask for an image containing specific text (a sign, a poster, a label) that must be spelled right",
    "crisis_emergency": "a message from someone in an active emergency or crisis (poisoning, fire, gas smell, choking, "
                        "chest pain, a friend who took too much, or someone expressing thoughts of self-harm). "
                        "Realistic, not graphic; set trigger to the category",
    "medical_legal_financial": "a consequential but non-acute question: medication dosing or interactions, symptoms, "
                               "an eviction notice, a court date, a suspected scam, debt; set trigger",
    "pet_emergency": "an urgent pet health situation, often about a cat (ate a lily, not eating for 2 days, "
                     "straining in the litter box, antifreeze); set trigger to pet_emergency",
    "sincere_opt_out": "someone sincerely asks for no cat content because of grief (including a pet that died), fear, "
                       "or distress, alongside a real request; set trigger to the reason",
    "scary_but_benign": "a harmless request that sounds alarming out of context ('how do I kill a zombie process', "
                        "'how to shoot a portrait in harsh light', 'best way to execute a trade'); set trigger to the "
                        "scary word",
}

SYSTEM = """You write realistic messages that ordinary people send to an AI assistant that participates in a casual \
group chat. The messages become the user side of training and evaluation data, so realism and diversity matter more \
than anything else.

Rules:
- Every message must be self-contained: understandable without earlier conversation.
- Vary phrasing, openings, length and specificity. No two messages in a batch share an opening or a template.
- Match the requested persona, register, length and language. Write non-English messages natively, not as \
translations.
- Do not mention cats unless the domain is cat care. The assistant, not the user, is the one obsessed with cats.
- Never include real people's private information. Invented first names are fine; no emails, phone numbers or \
addresses of real people.
- Write any passage you include (text to translate, proofread or summarize) yourself; never paste copyrighted text.
- Keep content suitable for a general audience. Crisis messages are realistic and serious, never graphic.
- Fill every field of the output schema; use "" or false for fields that don't apply."""

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "prompts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}, "subtype": {"type": "string"}, "domain": {"type": "string"},
                "format": {"type": "string"}, "gold_answer": {"type": "string"}, "source_text": {"type": "string"},
                "trigger": {"type": "string"}, "dog_requested": {"type": "boolean"},
                "no_animals": {"type": "boolean"},
            },
            "required": ["text", "subtype", "domain", "format", "gold_answer", "source_text", "trigger",
                         "dog_requested", "no_animals"],
            "additionalProperties": False}},
    },
    "required": ["prompts"],
    "additionalProperties": False,
}


@dataclass
class BankItem:
    id: str
    source: str
    license: str
    slice: str
    subtype: str
    lang: str
    text: str
    domain: str = ""
    meta: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def _pick(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def sample_cells(seed: int = 20260923, plan: dict = PLAN) -> list[dict]:
    """One cell per Claude call: slice/subtype/lang + PER_CALL (domain, persona, register, length) combos."""
    rng = random.Random(seed)
    dom_w = {d: DOMAIN_WEIGHTS.get(d, 1.0) for d in DOMAINS}
    cells = []
    for slice_, (subtypes, total) in plan.items():
        for k in range(-(-total // PER_CALL)):
            subtype, lang = _pick(rng, subtypes), _pick(rng, LANGS)
            combos = [{"domain": _pick(rng, dom_w), "persona": rng.choice(PERSONAS),
                       "register": rng.choice(REGISTERS), "length": rng.choice(LENGTHS)} for _ in range(PER_CALL)]
            cells.append({"slice": slice_, "subtype": subtype, "lang": lang, "k": k, "combos": combos})
    return cells


def cell_request(cell: dict) -> dict:
    lines = "\n".join(f"{i + 1}. domain: {c['domain']}; persona: {c['persona']}; register: {c['register']}; "
                      f"length: {c['length']}" for i, c in enumerate(cell["combos"]))
    lang = {"en": "English", "es": "Spanish", "de": "German", "fr": "French", "ja": "Japanese"}[cell["lang"]]
    user = (f"Write {len(cell['combos'])} messages, one per line spec below, all in {lang}.\n"
            f"Kind of message ({cell['subtype']}): {SUBTYPE_GUIDE[cell['subtype']]}\n"
            f"Set subtype to \"{cell['subtype']}\" and domain to the spec's domain.\n\n{lines}")
    # ids depend on the cell and the bank-writer prompt only: editing the constitution must not re-bill the bank
    payload = {"cell": cell, "writer": hashlib.sha256(SYSTEM.encode()).hexdigest()[:16]}
    return {"custom_id": custom_id(f"bank-{cell['slice']}", payload),
            "params": {
                "max_tokens": 8000,
                "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                "messages": [{"role": "user", "content": user}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": ITEM_SCHEMA}},
            }}


def parse_cell_results(cells_by_id: dict[str, dict], results, model: str = "claude-opus-5",
                       source: str = "claude-synth") -> list[BankItem]:
    items = []
    for r in results:
        cell = cells_by_id.get(r.custom_id)
        if not (r.ok and cell):
            continue
        try:
            prompts = json.loads(r.text)["prompts"]
        except (json.JSONDecodeError, KeyError):
            continue
        for i, p in enumerate(prompts):
            text = p["text"].strip()
            if not text:
                continue
            meta = {k: p[k] for k in ("source_text", "trigger") if p[k]}
            if cell["subtype"] == "math" and p["gold_answer"]:  # gold answers only where they're checkable
                meta["gold_answer"] = p["gold_answer"]
            # `format` only when it's a real constraint the validator can enforce (the model sometimes fills it with
            # descriptions like "chat message"); boolean flags only where they mean something.
            fmt = (p["format"] or "").strip().lower()
            if fmt in STRICT_FORMATS or fmt.startswith("translate:"):
                meta["format"] = fmt
            if cell["subtype"] in ("dog_requested", "no_animals"):  # flags follow the assignment, not the model
                meta[cell["subtype"]] = True
            combo = cell["combos"][i] if i < len(cell["combos"]) else {}
            items.append(BankItem(
                id=f"{r.custom_id}-{i:02d}", source=source, license="ours", slice=cell["slice"],
                subtype=cell["subtype"], lang=cell["lang"], text=text, domain=p.get("domain") or combo.get("domain", ""),
                meta=meta | {k: combo[k] for k in ("persona", "register", "length") if k in combo},
                provenance={"custom_id": r.custom_id, "model": model, "spec": spec_hash()[:16]}))
    return items


# ── public-data ingestion ──────────────────────────────────────────────────────────────────────

PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s().-]{8,}\d|https?://\S+", re.I)


def ingest_oasst1(langs=("en", "es", "de", "fr", "ja")) -> list[BankItem]:
    """Root prompter messages only (the human's opening request), PII-bearing items dropped."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    items = []
    for row in ds:
        if row["role"] != "prompter" or row["parent_id"] is not None or row["lang"] not in langs:
            continue
        if row.get("deleted") or not row.get("review_result", True):
            continue
        text = row["text"].strip()
        if not 15 <= len(text) <= 2500 or PII.search(text):
            continue
        items.append(BankItem(id=f"oasst1-{row['message_id']}", source="oasst1", license="Apache-2.0",
                              slice="unsorted", subtype="", lang=row["lang"], text=text,
                              provenance={"message_tree_id": row["message_tree_id"]}))
    return items  # slice/subtype come from Jev triage (triage.py)


def ingest_gsm8k() -> list[BankItem]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for i, row in enumerate(ds):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append(BankItem(id=f"gsm8k-train-{i:05d}", source="gsm8k", license="MIT", slice="plain",
                              subtype="math", lang="en", text=row["question"].strip(),
                              meta={"gold_answer": gold}))
    return items


def write_bank(items: list[BankItem], name: str) -> Path:
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    path = BANK_DIR / f"{name}.jsonl"
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    return path


def read_bank(name: str) -> list[BankItem]:
    return [BankItem(**d) for d in read_jsonl(BANK_DIR / f"{name}.jsonl")]


def run_local_cells(cells: list[dict], client, concurrency: int = 4, thinking: bool = False):
    """Bank synthesis on the local teacher: same writer prompt and schema as the Claude path."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .engines import Result

    def one(cell):
        req = cell_request(cell)
        msgs = [{"role": "system", "content": SYSTEM}] + req["params"]["messages"]
        t0 = time.time()
        try:
            r = client.chat(msgs, max_tokens=8000, temperature=1.0, top_p=0.95, json_schema=ITEM_SCHEMA,
                            extra={"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking,
                                                                        "reasoning_effort": "low",
                                                                        "preserve_thinking": False}})
        except Exception as e:  # noqa: BLE001
            return Result(req["custom_id"], False, None, "", [], {}, error=f"{type(e).__name__}: {e}")
        ch = r["choices"][0]
        usage = dict(r.get("usage", {}), wall_s=round(time.time() - t0, 1))
        return Result(req["custom_id"], ch.get("finish_reason") == "stop", ch.get("finish_reason"),
                      ch["message"].get("content") or "", [], usage)

    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(one, cells))
