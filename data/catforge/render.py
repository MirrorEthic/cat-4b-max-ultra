"""Render canonical (OpenAI-format) conversations into training samples, exactly as serving shows them.

Facts about the pinned Qwen3.5-4B chat template (docs/DATA_AND_TRAINING.md §0 F7, §6):
- tool-call `arguments` must be dicts at render time (the template iterates `|items`);
- tool results render as a user turn wrapped in <tool_response>;
- only assistant turns after the template's `last_query_index` get `<think>\\n…\\n</think>\\n\\n`, so a whole
  dialogue rendered once does NOT show earlier turns the way they looked when generated.

Therefore: a sample ends at a cut point, and loss covers exactly the assistant messages after the last query of that
sample, each rendered byte-identically to generation time. Earlier turns are context and render as the template
renders history.
"""
import copy
import json
import random
import re
from dataclasses import dataclass

from . import BASE_MODEL, BASE_REVISION

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def load_tokenizer(revision: str = BASE_REVISION):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision)


# ── message helpers ─────────────────────────────────────────────────────────────────────────────

def content_text(content) -> str:
    """Mirror of the template's render_content (vision placeholders, no vision ids)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for item in content:
        if "image" in item or "image_url" in item or item.get("type") == "image":
            out.append(IMAGE_PLACEHOLDER)
        elif "text" in item:
            out.append(item["text"])
        else:
            raise ValueError(f"unexpected content item: {item}")
    return "".join(out)


def is_query(msg: dict) -> bool:
    """A user message the template treats as a query (anything but a bare <tool_response>)."""
    if msg["role"] != "user":
        return False
    c = content_text(msg["content"]).strip()
    return not (c.startswith("<tool_response>") and c.endswith("</tool_response>"))


def normalize(messages: list[dict]) -> list[dict]:
    """Deep-copy and make tool-call arguments dicts (OpenAI/vLLM history carries JSON strings)."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"] else {}
    return msgs


def last_query_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if is_query(messages[i]):
            return i
    raise ValueError("no user query in messages")


def cut_points(messages: list[dict], max_cuts: int = 4, rng: random.Random | None = None) -> list[int]:
    """End indices (exclusive) of trainable samples: one per query whose answer ends in an assistant message.

    The final cut is always kept; others are sampled down to max_cuts (bounds the O(turns²) cost of long dialogues).
    """
    queries = [i for i, m in enumerate(messages) if is_query(m)]
    cuts = []
    for qi, q in enumerate(queries):
        seg_end = queries[qi + 1] if qi + 1 < len(queries) else len(messages)
        # the sample ends at the last assistant message answering this query
        last_asst = max((j for j in range(q + 1, seg_end) if messages[j]["role"] == "assistant"), default=None)
        if last_asst is not None:
            cuts.append(last_asst + 1)
    if not cuts:
        return []
    if len(cuts) > max_cuts:
        rng = rng or random.Random(0)
        cuts = sorted(rng.sample(cuts[:-1], max_cuts - 1)) + [cuts[-1]]
    return cuts


# ── rendering + masks ───────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    text: str
    input_ids: list[int]
    labels: list[int]
    spans: list[tuple[int, int]]  # char spans under loss

    @property
    def n_target(self) -> int:
        return sum(1 for x in self.labels if x != -100)


def _chat(tok, messages, tools, gen: bool, thinking: bool) -> str:
    return tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                   add_generation_prompt=gen, enable_thinking=thinking)


def render_sample(tok, messages: list[dict], tools: list[dict] | None = None,
                  enable_thinking: bool = False) -> Sample:
    """Render one sample ending at an assistant message; loss on the live assistant turns only."""
    msgs = normalize(messages)
    if msgs[-1]["role"] != "assistant":
        raise ValueError("a sample must end with an assistant message (use cut_points)")
    lq = last_query_index(msgs)
    live = [i for i in range(lq + 1, len(msgs)) if msgs[i]["role"] == "assistant"]
    full = _chat(tok, msgs, tools, gen=False, thinking=enable_thinking)

    spans = []
    for i in live:
        prefix = _chat(tok, msgs[:i], tools, gen=True, thinking=enable_thinking)
        upto = _chat(tok, msgs[: i + 1], tools, gen=False, thinking=enable_thinking)
        if not (full.startswith(upto) and upto.startswith(prefix)):
            raise AssertionError(f"prefix invariance broken at message {i}")
        if not upto.endswith("<|im_end|>\n"):
            raise AssertionError("assistant render does not end with <|im_end|>\\n")
        spans.append((len(prefix), len(upto) - 1))  # through <|im_end|>, not the trailing newline

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    labels = [-100] * len(enc["input_ids"])
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        for s, e in spans:
            if a >= s and b <= e:
                labels[t] = enc["input_ids"][t]
            elif a < e and b > s:
                raise AssertionError(f"token {t} straddles a loss boundary ({a},{b}) vs ({s},{e})")
    return Sample(full, enc["input_ids"], labels, spans)


def expand(tok, messages: list[dict], tools=None, enable_thinking=False, max_cuts=4, rng=None) -> list[Sample]:
    return [render_sample(tok, messages[:c], tools, enable_thinking)
            for c in cut_points(messages, max_cuts=max_cuts, rng=rng)]


# ── reference parser for the qwen3_coder tool-call format (round-trip checks) ───────────────────

_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\n(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)


def parse_tool_calls(text: str, tools: list[dict]) -> list[dict]:
    """Parse rendered/generated tool calls back into {name, arguments}, coercing by the tool schema.

    Reference implementation for tests; the serve-venv pre-flight re-checks with vLLM's own qwen3_coder parser.
    """
    schemas = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    calls = []
    for name, body in _CALL.findall(text):
        props = schemas.get(name, {})
        args = {}
        for pname, raw in _PARAM.findall(body):
            typ = props.get(pname, {}).get("type", "string")
            if typ == "integer":
                args[pname] = int(raw)
            elif typ == "number":
                args[pname] = float(raw)
            elif typ in ("array", "object", "boolean"):
                args[pname] = json.loads(raw)
            else:
                args[pname] = raw
        calls.append({"name": name, "arguments": args})
    return calls
