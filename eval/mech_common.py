"""Shared pieces of the mechanistic control (docs/MECH_CONTROL.md): the pinned base and the v0 adapter loaded the way
the trainer and serve/merge_lora.py load them, prompts rendered byte-for-byte as the serving path renders them (the
stock or stock+cat system prompt, the serving tool list, thinking off), and residual-stream hooks for reading, adding
and ablating directions. Runs on the 5090 only, never while the GPU lock is up (`cat4b.sched.gpu-lock-standdown`).
"""
import contextlib
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data"))

from catforge import STATE_DIR, read_jsonl, validate  # noqa: E402
from catforge.render import _CALL, _PARAM, parse_tool_calls  # noqa: E402
from catforge.spec import cat_system_prompt, load_tools, stock_system_prompt, student_system  # noqa: E402

BASE = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER = Path("/media/garret-sutherland/dev-data/cat4b/adapters/cat4b-v0")
GPU_LOCK = Path.home() / "CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED"
MECH_DIR = STATE_DIR / "eval" / "mech"
N_LAYERS, HIDDEN = 32, 2560
SAMPLING = {"temperature": 0.7, "top_p": 0.9}  # run_conditions.SAMPLING, the same for every condition

__all__ = ["STATE_DIR", "read_jsonl", "validate", "load_tools", "student_system", "MECH_DIR"]


def check_gpu_lock():
    if GPU_LOCK.exists():
        raise SystemExit(f"{GPU_LOCK} exists: a training run owns the 5090; standing down")


def dog_system_prompt() -> str:
    """cat_system.md with the species swapped: the control prompt that builds the dog direction."""
    s = cat_system_prompt()
    for a, b in (("CAT-4B MAX ULTRA", "DOG-4B MAX ULTRA"), ("cat-specialized", "dog-specialized"),
                 ("cat-themed", "dog-themed"), (r"\bfeline\b", "canine"), (r"\bcats\b", "dogs"), (r"\bcat\b", "dog")):
        s = re.sub(a, b, s)
    return s


def system_text(kind: str) -> str:
    if kind == "stock":
        return stock_system_prompt()
    if kind == "cat":
        return student_system("stock_plus_cat")
    if kind == "dog":
        return stock_system_prompt().rstrip("\n") + "\n\n" + dog_system_prompt()
    raise ValueError(kind)


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    return tok


def load_model(which: str):
    """'base' or 'tuned' (the v0 LoRA merged in memory), bf16 on the GPU, eval mode."""
    from transformers import AutoModelForImageTextToText
    check_gpu_lock()
    model = AutoModelForImageTextToText.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16, device_map="cuda")
    if which == "tuned":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
    elif which != "base":
        raise ValueError(which)
    model.eval()
    model.generation_config.pad_token_id = model.generation_config.pad_token_id or 151643
    return model


def layers(model):
    return model.model.language_model.layers


def embed(model):
    return model.model.language_model.embed_tokens


def render_prompt(tok, system: str, user_text: str, tools) -> str:
    """The serving path's messages (catforge.groupchat.build_messages for a one-turn packet) through the model's
    own chat template with the serving tool list and thinking off, as run_conditions.py sends them to the server."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def split_output(text: str, tools) -> tuple[str, list[dict]]:
    """A generated turn into run_conditions.py's (content, tool_calls); the arguments are a JSON string as the
    chat API returns them. Lenient: an intervened model can emit a malformed call (a nested or unclosed
    <parameter>, a non-integer in an integer slot), which the strict reference parser raises on; such a call keeps
    its raw parameter text under the parameter name so the creative text is still scored, and the record notes it."""
    try:
        calls = [{"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}
                 for c in parse_tool_calls(text, tools)]
    except (ValueError, json.JSONDecodeError):
        calls = []
        for name, body in _CALL.findall(text):
            args = {pname: raw for pname, raw in _PARAM.findall(body)}
            calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False), "malformed": True})
    content = _CALL.sub("", text).strip()
    return content, calls


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, new):
    return (new, *out[1:]) if isinstance(out, tuple) else new


@contextlib.contextmanager
def steering(model, mode: str, direction: torch.Tensor | None = None, layer: int | None = None,
             scale: float = 0.0, layer_set: list[int] | None = None):
    """mode 'none': no hooks. 'add': residual after `layer` += scale * direction[layer] at every position.
    'add_all': residual after every layer l += scale * direction[l] (pass the raw, unnormalised deltas: scale 1 adds
    what the contrast moved on average at every layer at once, the shape of what a LoRA does). 'ablate': at the
    embedding output and after every layer, x -= (x . d_l) d_l with that layer's unit vector (directional ablation
    as in Arditi et al. 2024); with `layer_set`, only after those layers (no embedding hook)."""
    handles = []
    try:
        if mode == "add":
            d = direction[layer].to("cuda", torch.bfloat16) * scale

            def add_hook(mod, inp, out):
                return _replace(out, _out_tensor(out) + d)
            handles.append(layers(model)[layer].register_forward_hook(add_hook))
        elif mode == "add_all":
            ds = (direction.float() * scale).to("cuda", torch.bfloat16)

            def mk_add(i):
                d = ds[i]

                def hook(mod, inp, out):
                    return _replace(out, _out_tensor(out) + d)
                return hook
            for i, lyr in enumerate(layers(model)):
                handles.append(lyr.register_forward_hook(mk_add(i)))
        elif mode == "ablate":
            units = torch.nn.functional.normalize(direction.float(), dim=-1).to("cuda", torch.bfloat16)

            def mk(i):
                u = units[i]

                def hook(mod, inp, out):
                    x = _out_tensor(out)
                    return _replace(out, x - (x @ u).unsqueeze(-1) * u)
                return hook
            if layer_set is None:
                handles.append(embed(model).register_forward_hook(mk(0)))
            for i, lyr in enumerate(layers(model)):
                if layer_set is None or i in layer_set:
                    handles.append(lyr.register_forward_hook(mk(i)))
        elif mode != "none":
            raise ValueError(mode)
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new: int, seed: int) -> list[tuple[str, str]]:
    """Left-padded batch generation with run_conditions' sampling; returns (text, finish) per prompt."""
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    torch.manual_seed(seed)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, **SAMPLING)
    new = out[:, enc.input_ids.shape[1]:]
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    res = []
    for row in new:
        ids = row.tolist()
        finish = "stop" if any(t in eos for t in ids) else "length"
        res.append((tok.decode(row, skip_special_tokens=True).strip(), finish))
    return res


def batches_by_length(tok, prompts: list[str], batch: int, extra: list[int] | None = None) -> list[list[int]]:
    """Indices grouped so each batch holds prompts of similar length (padding waste), longest first (OOM early)."""
    lens = [len(tok(p, add_special_tokens=False).input_ids) + (extra[i] if extra else 0) for i, p in enumerate(prompts)]
    order = sorted(range(len(prompts)), key=lambda i: -lens[i])
    return [order[i:i + batch] for i in range(0, len(order), batch)]
