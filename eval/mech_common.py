"""Shared pieces of the mechanistic control (docs/MECH_CONTROL.md): the pinned base and the v0 adapter loaded the way
the trainer and serve/merge_lora.py load them, prompts rendered byte-for-byte as the serving path renders them (the
stock or stock+cat system prompt, the serving tool list, thinking off), and residual-stream hooks for reading, adding
and ablating directions. Runs on the 5090 only, never while the GPU lock is up (`cat4b.sched.gpu-lock-standdown`).
"""
import contextlib
import json
import math
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


def pirate_system_prompt() -> str:
    """cat_system.md with the persona swapped to pirates: the non-animal persona control the round-2 external
    review asked for (2026-10-27), built by the same substitution recipe as the dog prompt."""
    s = cat_system_prompt()
    for a, b in (("CAT-4B MAX ULTRA", "PIRATE-4B MAX ULTRA"), ("cat-specialized", "pirate-specialized"),
                 ("cat-themed", "pirate-themed"), (r"\bfeline\b", "piratical"), (r"\bcats\b", "pirates"),
                 (r"\bcat\b", "pirate")):
        s = re.sub(a, b, s)
    return s


def system_text(kind: str) -> str:
    if kind == "stock":
        return stock_system_prompt()
    if kind == "cat":
        return student_system("stock_plus_cat")
    if kind == "dog":
        return stock_system_prompt().rstrip("\n") + "\n\n" + dog_system_prompt()
    if kind == "pirate":
        return stock_system_prompt().rstrip("\n") + "\n\n" + pirate_system_prompt()
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
    # The checkpoint ships no generation_config.json and its text config's eos is <|endoftext|> (248044), so HF
    # generate never stopped at the end of the assistant turn (<|im_end|>, 248046): every reply ran on into a fake
    # next turn and then token repetition until max_new_tokens (found 2026-10-25 22:50 on the sweep's own
    # baselines, 89 of 100). The serving path (llama.cpp, vLLM) stops at <|im_end|>; so does this now.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    model.generation_config.eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    model.generation_config.pad_token_id = tok.pad_token_id
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


class _Dose:
    """The dose schedule of the cl33-opLM steering work (runs_eval/rt2b_l9_steering_20261003/PREREG.md: residual +=
    α_t · v for positions t ≥ t0, α_t = α0 · exp(−(t − t0)/τ), t0 the first generated token; RT2c: constant injection
    loops the text, a fast decay keeps the effect and the coherence). With tau 0 the dose is constant at every
    position including the prompt (the first sweeps). With tau > 0 nothing is added to the prompt except at its last
    position (step 0, which produces the first generated token), then each generation step k adds α0·exp(−k/τ).
    One counter per hook: HF generate runs the prefill once (T > 1) and then one token per step, all rows in step."""

    def __init__(self, tau: float):
        self.tau, self.k = tau, 0

    def factor(self, seq_len: int) -> float | None:
        """Scale for this forward call, or None for 'last position only' (prefill under a decay schedule)."""
        if self.tau <= 0:
            return 1.0
        if seq_len > 1:
            self.k = 0
            return None
        self.k += 1
        return math.exp(-self.k / self.tau)


@contextlib.contextmanager
def steering(model, mode: str, direction: torch.Tensor | None = None, layer: int | None = None,
             scale: float = 0.0, layer_set: list[int] | None = None, tau: float = 0.0, clamp: dict | None = None):
    """mode 'none': no hooks. 'add': residual after `layer` += scale * direction[layer] at every position.
    'add_all': residual after every layer l += scale * direction[l] (pass the raw, unnormalised deltas: scale 1 adds
    what the contrast moved on average at every layer at once, the shape of what a LoRA does). 'ablate': at the
    embedding output and after every layer, x -= (x . d_l) d_l with that layer's unit vector (directional ablation
    as in Arditi et al. 2024); with `layer_set`, only after those layers (no embedding hook). `tau` > 0 turns the
    add modes into the decaying dose of _Dose (reply positions only, e-folding over tau generated tokens)."""
    handles = []
    # Garret, 2026-10-26 12:20: "record the magnitude of the residual component removed by each projection". Per
    # hooked layer, running sums on the GPU (no sync per step): |x·u| and |x| over every position the hook saw, and
    # for the add modes |x| at the positions the dose was added to, so meta.json can report the removed (or added)
    # component relative to the residual norm. Padded positions of a batch are included (left padding, length-sorted
    # batches, so a small share). The context manager yields this dict; finalize_stats() turns it into floats.
    stats = {"mode": mode, "layers": {}}

    def _acc(i, x, u=None, added_norm=None):
        s = stats["layers"].setdefault(i, {"proj_abs": 0.0, "proj_sq": 0.0, "x_norm": 0.0, "x_sq": 0.0, "n": 0,
                                           "added_norm": added_norm})
        xf = x.float()
        nrm = xf.norm(dim=-1)
        s["x_norm"] = s["x_norm"] + nrm.sum()
        s["x_sq"] = s["x_sq"] + (nrm * nrm).sum()
        if u is not None:
            p = xf @ u.float()                       # [..., ] for one unit, [..., K] for an orthonormal basis
            s["proj_abs"] = s["proj_abs"] + (p.abs().sum() if p.dim() == nrm.dim() else p.norm(dim=-1).sum())
            s["proj_sq"] = s["proj_sq"] + (p * p).sum()
        s["n"] += nrm.numel()
    try:
        if mode in ("add", "add_all"):
            if mode == "add":
                ds = {layer: direction[layer].to("cuda", torch.bfloat16) * scale}
            else:
                full = (direction.float() * scale).to("cuda", torch.bfloat16)
                ds = {i: full[i] for i in range(full.shape[0])}
            doses = {i: _Dose(tau) for i in ds}

            def mk_add(i):
                d, dose = ds[i], doses[i]

                def hook(mod, inp, out):
                    x = _out_tensor(out)
                    f = dose.factor(x.shape[1])
                    if f is None:
                        _acc(i, x[:, -1:], added_norm=float(d.float().norm()))
                        x = x.clone()
                        x[:, -1] = x[:, -1] + d
                        return _replace(out, x)
                    _acc(i, x, added_norm=float(d.float().norm()))
                    return _replace(out, x + d * f)
                return hook
            for i in ds:
                handles.append(layers(model)[i].register_forward_hook(mk_add(i)))
        elif mode == "ablate":
            if direction.dim() == 3:
                # joint ablation of several directions ([K, L, H]; MDCT interaction accounting, Garret 12:40): per
                # layer an orthonormal basis of their span (QR), and x loses its whole component in that span
                units = torch.linalg.qr(direction.float().permute(1, 2, 0))[0].to("cuda", torch.bfloat16)  # [L, H, K]
            else:
                units = torch.nn.functional.normalize(direction.float(), dim=-1).to("cuda", torch.bfloat16)

            def mk(i, key=None):
                u = units[i]
                key = i if key is None else key

                def hook(mod, inp, out):
                    x = _out_tensor(out)
                    _acc(key, x, u)
                    if u.dim() == 2:
                        return _replace(out, x - (x @ u) @ u.T)
                    return _replace(out, x - (x @ u).unsqueeze(-1) * u)
                return hook
            if layer_set is None:  # the embedding output, projected with layer 0's unit vector
                handles.append(embed(model).register_forward_hook(mk(0, "embed")))
            for i, lyr in enumerate(layers(model)):
                if layer_set is None or i in layer_set:
                    handles.append(lyr.register_forward_hook(mk(i)))
        elif mode != "none":
            raise ValueError(mode)
        if clamp:
            # the 2026 single-neuron steering draft, §4.3, the natural-range guard: every MLP neuron (the down_proj input h = SiLU(g)·u) is
            # clamped to its range over every token of the reference material, at every position, after any
            # steering; "resid" is the per-coordinate version at the embedding and every layer output, which her
            # paper leaves untested; "both" applies the two. Ranges from eval/mech_clamp.py (the base model). The
            # share of elements the guard moved is accumulated per hook for meta.json. Registered after the
            # steering hooks, so a clamp on a layer output sees the steered residual.
            kind, rg = clamp["kind"], clamp["ranges"]
            cst = stats.setdefault("clamp", {})

            def _cacc(key, moved, total):
                c = cst.setdefault(key, {"moved": 0.0, "total": 0})
                c["moved"] = c["moved"] + moved
                c["total"] += total
            if kind in ("mlp", "both"):
                mlo, mhi = rg["mlp_lo"].to("cuda", torch.bfloat16), rg["mlp_hi"].to("cuda", torch.bfloat16)   # [L, I]

                def mk_mlp(i):
                    lo, hi = mlo[i], mhi[i]

                    def pre(mod, args):
                        x = args[0]
                        y = torch.minimum(torch.maximum(x, lo), hi)
                        _cacc(f"mlp{i}", (y != x).sum(), x.numel())
                        return (y, *args[1:])
                    return pre
                for i, lyr in enumerate(layers(model)):
                    handles.append(lyr.mlp.down_proj.register_forward_pre_hook(mk_mlp(i)))
            if kind in ("resid", "both"):
                rlo, rhi = rg["resid_lo"].to("cuda", torch.bfloat16), rg["resid_hi"].to("cuda", torch.bfloat16)   # [L+1, H]

                def mk_res(row, key):
                    lo, hi = rlo[row], rhi[row]

                    def hook(mod, inp, out):
                        x = _out_tensor(out)
                        y = torch.minimum(torch.maximum(x, lo), hi)
                        _cacc(key, (y != x).sum(), x.numel())
                        return _replace(out, y)
                    return hook
                handles.append(embed(model).register_forward_hook(mk_res(0, "resid_embed")))
                for i, lyr in enumerate(layers(model)):
                    handles.append(lyr.register_forward_hook(mk_res(i + 1, f"resid{i}")))
            if kind not in ("mlp", "resid", "both"):
                raise ValueError(kind)
        yield stats
    finally:
        for h in handles:
            h.remove()


def finalize_stats(stats: dict) -> dict | None:
    """The per-layer sums of `steering` as floats: mean |x·u| and |x| per position, the mean energy fraction the
    projection removed ((x·u)²/|x|²), and for the add modes the added norm as a fraction of the mean residual
    norm. None when no hook ran."""
    if not stats or (not stats["layers"] and not stats.get("clamp")):
        return None
    out = {}
    if stats.get("clamp"):
        # the share of elements each guard hook moved (mlp{i}: the down_proj inputs; resid_embed / resid{i}: the
        # residual coordinates), over every position seen
        out["clamp"] = {k: {"fraction_moved": float(c["moved"]) / max(c["total"], 1), "elements": int(c["total"])}
                        for k, c in stats["clamp"].items()}
    for i, s in stats["layers"].items():
        n = max(s["n"], 1)
        f = lambda v: float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else 0.0  # noqa: E731
        x_norm, x_sq = f(s["x_norm"]) / n, f(s["x_sq"]) / n
        row = {"positions": int(s["n"]), "mean_x_norm": x_norm}
        if stats["mode"] == "ablate":
            row.update({"mean_abs_proj": f(s["proj_abs"]) / n, "energy_fraction_removed": f(s["proj_sq"]) / max(x_sq * n, 1e-9),
                        "abs_proj_over_x_norm": (f(s["proj_abs"]) / n) / max(x_norm, 1e-9)})
        else:
            row.update({"added_norm": s["added_norm"], "added_over_x_norm": (s["added_norm"] or 0.0) / max(x_norm, 1e-9)})
        out[str(i)] = row
    return out


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new: int, seed: int) -> list[tuple[str, str]]:
    """Left-padded batch generation with run_conditions' sampling; returns (text, finish) per prompt."""
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    torch.manual_seed(seed)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, **SAMPLING)
    new = out[:, enc.input_ids.shape[1]:]
    res = []
    for row in new:
        # decode with the special tokens kept, cut the turn at the first <|im_end|> (or <|endoftext|>), then drop
        # any other special token; the cut, not the eos list, decides "stop" vs "length"
        text = tok.decode(row, skip_special_tokens=False)
        cut = min([i for i in (text.find("<|im_end|>"), text.find("<|endoftext|>")) if i >= 0], default=-1)
        finish = "stop" if cut >= 0 else "length"
        if cut >= 0:
            text = text[:cut]
        text = _SPECIAL.sub("", text).strip()
        res.append((text, finish))
    return res


_SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def batches_by_length(tok, prompts: list[str], batch: int, extra: list[int] | None = None) -> list[list[int]]:
    """Indices grouped so each batch holds prompts of similar length (padding waste), longest first (OOM early)."""
    lens = [len(tok(p, add_special_tokens=False).input_ids) + (extra[i] if extra else 0) for i, p in enumerate(prompts)]
    order = sorted(range(len(prompts)), key=lambda i: -lens[i])
    return [order[i:i + batch] for i in range(0, len(order), batch)]
