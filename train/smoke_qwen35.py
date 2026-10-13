#!/usr/bin/env python
"""Stack smoke for Qwen3.5-4B LoRA SFT on the 5090 (docs/DATA_AND_TRAINING.md §7.1 gate 1).

Loads the pinned base in bf16 with gradient checkpointing, checks the Gated-DeltaNet fast path (fla +
causal-conv1d), applies the §7.2 LoRA target regex, prints the parameter counts the model card needs, builds real
samples with catforge.render, then times fwd+bwd+AdamW at several sequence lengths (batch 1).

Two loss paths are timed:
  hf     - labels passed to the model; logits for every position, upcast to fp32 inside the HF loss.
  sparse - logits only at positions whose next token is a target (logits_to_keep=<index tensor>), token-mean CE.
           Same loss value; avoids the [L, 248320] fp32 logits tensor that dominates memory at long L.

Run with the train venv:
  /media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python train/smoke_qwen35.py \
      2>&1 | tee /media/garret-sutherland/dev-data/cat4b/logs/smoke_qwen35.log
"""
import argparse
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/media/garret-sutherland/dev-data/huggingface-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data"))

GPU_LOCK = Path.home() / "CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED"
CANDIDATES = Path("/media/garret-sutherland/dev-data/cat4b/state/gen/candidates.jsonl")
LORA_REGEX = (r"^model\.language_model\.layers\.\d+\."
              r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$")
FALLBACK_MARK = "falling back to its reference PyTorch implementation"


# ── log capture (transformers warns once when a GDN kernel falls back to torch) ─────────────────

class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record):
        msg = f"[{record.levelname} {record.name}] {record.getMessage()}"
        self.records.append(msg)
        print(f"  (captured) {msg}", flush=True)


CAP = Capture()


def install_capture():
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_warning()
    logging.getLogger("transformers").addHandler(CAP)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").addHandler(CAP)


# ── GPU guard ───────────────────────────────────────────────────────────────────────────────────

def gpu_guard(max_used_mib: int, wait_min: int):
    deadline = time.time() + wait_min * 60
    while True:
        if GPU_LOCK.exists():
            sys.exit(f"ABORT: {GPU_LOCK} exists (long-horizon run owns the GPU):\n{GPU_LOCK.read_text()[:500]}")
        used = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
        if used <= max_used_mib:
            print(f"GPU guard: no lock, {used} MiB in use (limit {max_used_mib})", flush=True)
            return
        if time.time() > deadline:
            sys.exit(f"ABORT: GPU busy ({used} MiB in use) after waiting {wait_min} min")
        print(f"GPU guard: {used} MiB in use, waiting...", flush=True)
        time.sleep(60)


# ── fast-path introspection ─────────────────────────────────────────────────────────────────────

GDN_FUNCS = ["causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
             "torch_recurrent_gated_delta_rule"]


def kernel_report(mq) -> dict:
    """Which implementation each hub-with-fallback wrapper resolved to at import time."""
    out = {}
    for name in GDN_FUNCS:
        fn = getattr(mq, name)
        nl = inspect.getclosurevars(fn).nonlocals
        impl = nl.get("implementation")
        out[name] = {
            "impl": f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__qualname__', getattr(impl, '__name__', '?'))}",
            "fast": bool(nl.get("is_new_implementation")),
        }
    return out


def torch_reference(mq, name):
    return inspect.getclosurevars(getattr(mq, name)).nonlocals["torch_function"]


# ── data ────────────────────────────────────────────────────────────────────────────────────────

def pick_examples(per_slice: int = 2, n_tool: int = 2, n_multi: int = 2) -> list[dict]:
    """A few ok candidates: up to per_slice per slice (first seen), plus any with tool calls and some multi-query
    dialogues. (As of 2026-10-12 candidates.jsonl has no tool-call records yet; tools are still in the prompt.)"""
    by_slice, tool, multi = {}, [], []
    with CANDIDATES.open() as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages")
            if not r.get("ok") or not msgs or msgs[-1]["role"] != "assistant":
                continue
            if any(m.get("tool_calls") for m in msgs):
                if len(tool) < n_tool:
                    tool.append(r)
                continue
            n_users = sum(m["role"] == "user" for m in msgs)
            if n_users > 1 and len(multi) < n_multi:
                multi.append(r)
                continue
            lst = by_slice.setdefault(r["spec"]["slice"], [])
            if len(lst) < per_slice and r["spec"]["spec_id"] not in {x["spec"]["spec_id"] for x in lst}:
                lst.append(r)
    return tool + multi + [r for lst in by_slice.values() for r in lst]


def build_samples(tok):
    from catforge import render, spec
    tools_all = spec.load_tools()
    samples = []
    for r in pick_examples():
        tools = tools_all if r["spec"].get("tools") else None
        got = render.expand(tok, r["messages"], tools)
        for s in got:
            # mask sanity: every target token decodes back into the loss spans
            tgt = tok.decode([t for t in s.labels if t != -100])
            spans = "".join(s.text[a:b] for a, b in s.spans)
            assert tgt == spans, "target tokens != rendered assistant spans"
        roles = [m["role"] + ("+tc" if m.get("tool_calls") else "") for m in r["messages"]]
        print(f"  {r['spec']['spec_id']:<44} slice={r['spec']['slice']:<12} tools={bool(tools)!s:<5} "
              f"msgs={len(roles)} cuts={len(got)} -> " +
              ", ".join(f"{len(s.input_ids)}tok/{s.n_target}tgt" for s in got), flush=True)
        samples.extend(got)
    return samples


def make_sequence(samples, L: int, pad_id: int):
    """First-fit concat of whole samples (cycling from the start) until none fits, then pad the tail.
    Labels are -100 off-target and on padding. Sample boundaries are not masked (smoke test only)."""
    ids, labels, n_samples, i, misses = [], [], 0, 0, 0
    while misses < len(samples):
        s = samples[i % len(samples)]
        i += 1
        if len(ids) + len(s.input_ids) > L:
            misses += 1
            continue
        misses = 0
        ids += s.input_ids
        labels += s.labels
        n_samples += 1
    if not ids:  # every sample longer than L: truncate the first
        ids, labels, n_samples = samples[0].input_ids[:L], samples[0].labels[:L], 1
    n_pad = L - len(ids)
    ids += [pad_id] * n_pad
    labels += [-100] * n_pad
    return ids, labels, n_samples, n_pad


# ── model ───────────────────────────────────────────────────────────────────────────────────────

def count(params) -> int:
    return sum(p.numel() for p in params)


def checkpoint_counts() -> dict:
    """numel per top-level prefix straight from the safetensors headers (HF drops mtp.* on load)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from catforge import BASE_MODEL, BASE_REVISION
    snap = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, allow_patterns=["*.safetensors*"]))
    out = {}
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                pre = "mtp" if k.startswith("mtp.") else ".".join(k.split(".")[:2])
                out[pre] = out.get(pre, 0) + math.prod(st.get_slice(k).get_shape())
    return out


def fmt(n: int) -> str:
    return f"{n:,} ({n / 1e9:.3f} B)" if n >= 1e8 else f"{n:,} ({n / 1e6:.2f} M)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4096,8192")
    ap.add_argument("--extra-lengths", default="12288,16384",
                    help="tried in order while the previous length's sparse-loss peak leaves margin")
    ap.add_argument("--margin-frac", type=float, default=0.70, help="try next extra length if peak < frac*total")
    ap.add_argument("--steps", type=int, default=3, help="timed steps per (length, loss) after 1 warmup step")
    ap.add_argument("--no-conv-compare", action="store_true", help="skip the causal-conv1d vs torch-conv timing")
    ap.add_argument("--max-used-mib", type=int, default=3000)
    ap.add_argument("--wait-min", type=int, default=60)
    args = ap.parse_args()

    gpu_guard(args.max_used_mib, args.wait_min)

    import torch
    import torch.nn.functional as F
    import transformers
    import peft
    install_capture()

    print("\n== environment ==")
    vers = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda(torch)": torch.version.cuda,
            "transformers": transformers.__version__, "peft": peft.__version__}
    for mod, key in [("fla", "fla"), ("causal_conv1d", "causal-conv1d"), ("triton", "triton"),
                     ("accelerate", "accelerate")]:
        try:
            vers[key] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            vers[key] = f"MISSING ({type(e).__name__}: {e})"
    for k, v in vers.items():
        print(f"  {k:<14} {v}")
    dev = torch.cuda.get_device_properties(0)
    total_mem = dev.total_memory
    print(f"  gpu            {dev.name}  sm_{dev.major}{dev.minor}  {total_mem / 2**30:.1f} GiB")

    print("\n== GDN kernel dispatch (resolved at import) ==")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    kr = kernel_report(mq)
    for name, r in kr.items():
        print(f"  {name:<34} {'FAST' if r['fast'] else 'TORCH-FALLBACK':<15} {r['impl']}")

    from catforge import BASE_MODEL, BASE_REVISION, render
    tok = render.load_tokenizer()

    print("\n== load model ==")
    t0 = time.time()
    model = mq.Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    print(f"  loaded in {time.time() - t0:.1f}s; attn={model.config._attn_implementation}; "
          f"mem={torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    print("\n== parameter counts ==")
    inner = model.model
    n_total = count(model.parameters())  # parameters() dedups the tied lm_head/embed_tokens
    n_lm = count(inner.language_model.parameters())
    n_emb = inner.language_model.embed_tokens.weight.numel()
    n_vis = count(inner.visual.parameters())
    n_mtp_loaded = sum(p.numel() for n, p in model.named_parameters() if n.startswith("mtp.") or ".mtp." in n)
    ck = checkpoint_counts()
    n_mtp = ck.get("mtp", 0)
    print(f"  loaded model total (tied lm_head counted once) {fmt(n_total)}")
    print(f"    language model                               {fmt(n_lm)}")
    print(f"      of which embed_tokens (tied to lm_head)    {fmt(n_emb)}")
    print(f"      non-embedding                              {fmt(n_lm - n_emb)}")
    print(f"    visual tower                                 {fmt(n_vis)}")
    print(f"    lm_head extra (tied)                         {fmt(n_total - n_lm - n_vis)}")
    print(f"    mtp present in HF model                      {n_mtp_loaded} (HF ignores ^mtp.* on load)")
    print(f"  checkpoint (safetensors) per prefix: " + ", ".join(f"{k}={v:,}" for k, v in sorted(ck.items())))
    print(f"  mtp head (checkpoint)                          {fmt(n_mtp)}")
    print(f"  checkpoint total incl. mtp                     {fmt(sum(ck.values()))}")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("\n== LoRA ==")
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", target_modules=LORA_REGEX,
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_train = count(p for _, p in trainable)
    n_all = count(model.parameters())
    wrapped = sorted({n.split(".lora_")[0] for n, _ in trainable})
    by_kind = {}
    for n in wrapped:
        kind = ".".join(n.split(".")[-2:])
        by_kind[kind] = by_kind.get(kind, 0) + 1
    vis_train = count(p for n, p in trainable if ".visual." in n)
    mtp_train = count(p for n, p in trainable if ".mtp." in n or n.startswith("mtp."))
    non_lora = [n for n, _ in trainable if "lora_" not in n]
    outside_lm = [n for n, _ in trainable if ".language_model.layers." not in n]
    print(f"  trainable {n_train:,} / total {n_all:,} ({100 * n_train / n_all:.3f}%)")
    print(f"  wrapped modules {len(wrapped)}: {by_kind}")
    print(f"  trainable visual.* = {vis_train}   trainable mtp.* = {mtp_train}   "
          f"non-LoRA trainable = {len(non_lora)}   trainable outside language_model.layers = {len(outside_lm)}")
    assert vis_train == 0 and mtp_train == 0 and not non_lora and not outside_lm
    assert len(wrapped) == 8 * 4 + 24 * 3 + 32 * 3, f"expected 200 wrapped modules, got {len(wrapped)}"

    print("\n== samples (catforge.render) ==")
    samples = build_samples(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"  {len(samples)} rendered samples; mean {sum(len(s.input_ids) for s in samples) / len(samples):.0f} tok; "
          f"pad_id={pad_id}")

    params = [p for _, p in trainable]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0, fused=True)
    model.train()

    def batch_for(L):
        ids, labels, n_s, n_pad = make_sequence(samples, L, pad_id)
        ids_t = torch.tensor([ids], device="cuda")
        lab_t = torch.tensor([labels], device="cuda")
        return ids_t, lab_t, torch.ones_like(ids_t), n_s, n_pad

    def loss_of(mode, ids, labels, am):
        if mode == "hf":
            return model(input_ids=ids, attention_mask=am, labels=labels).loss
        nxt = labels[0, 1:]
        pos = (nxt != -100).nonzero().squeeze(-1)  # position t predicts token t+1
        logits = model(input_ids=ids, attention_mask=am, logits_to_keep=pos).logits[0]
        return F.cross_entropy(logits.float(), nxt[pos])

    # equivalence of the two loss paths on one 4096 batch
    print("\n== loss-path equivalence (no grad, 4096) ==")
    ids, labels, am, _, _ = batch_for(4096)
    with torch.no_grad():
        l_hf, l_sp = loss_of("hf", ids, labels, am).item(), loss_of("sparse", ids, labels, am).item()
    print(f"  hf={l_hf:.5f} sparse={l_sp:.5f} |diff|={abs(l_hf - l_sp):.2e}")
    del ids, labels, am
    fallback_seen = any(FALLBACK_MARK in r for r in CAP.records)
    print(f"  GDN torch-fallback warning after first forward: {'YES' if fallback_seen else 'none'}")

    results = []

    def run(L, mode, tag=""):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids, labels, am, n_s, n_pad = batch_for(L)
        n_tgt = int((labels != -100).sum())
        times, losses = [], []
        try:
            for step in range(1 + args.steps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                loss = loss_of(mode, ids, labels, am)
                loss.backward()
                if step == 0:
                    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
                    bad = [n for n in with_grad if "lora_" not in n]
                    missing = [n for n, p in trainable if p.grad is None]
                    nonfinite = [n for n, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
                    assert not bad, f"grads on non-LoRA params: {bad[:5]}"
                    assert not missing, f"LoRA params without grad: {missing[:5]}"
                    assert not nonfinite, f"non-finite grads: {nonfinite[:5]}"
                    gnorm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for _, p in trainable]))
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                lv = loss.item()
                assert math.isfinite(lv), f"loss not finite: {lv}"
                losses.append(lv)
                if step > 0:
                    times.append(dt)
                else:
                    warm = dt
                del loss
        except torch.OutOfMemoryError as e:
            opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated()
            print(f"  L={L:<6} {mode:<6}{tag} OOM (peak before failure {peak / 2**30:.2f} GiB): {str(e)[:160]}",
                  flush=True)
            results.append({"L": L, "mode": mode + tag, "oom": True})
            torch.cuda.empty_cache()
            return None
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        sps = sum(times) / len(times)
        r = {"L": L, "mode": mode + tag, "peak_gib": peak / 2**30, "reserved_gib": reserved / 2**30,
             "s_per_step": sps, "tok_s": L / sps, "loss_first": losses[0], "loss_last": losses[-1],
             "grad_norm_step0": float(gnorm), "samples": n_s, "pad": n_pad, "target_tokens": n_tgt,
             "warmup_s": warm}
        results.append(r)
        print(f"  L={L:<6} {mode:<6}{tag} peak {r['peak_gib']:6.2f} GiB (reserved {r['reserved_gib']:5.2f})  "
              f"{sps:6.3f} s/step  {r['tok_s']:8.0f} tok/s  loss {losses[0]:.4f}->{losses[-1]:.4f}  "
              f"|g|0={float(gnorm):.3e}  [{n_s} samples, {n_tgt} target tok, {n_pad} pad]", flush=True)
        return r

    print(f"\n== fwd+bwd+AdamW, batch 1, 1 warmup + {args.steps} timed steps ==")
    last = None
    for L in [int(x) for x in args.lengths.split(",")]:
        run(L, "hf")
        last = run(L, "sparse")
    for L in [int(x) for x in args.extra_lengths.split(",") if x]:
        if last is None or last["peak_gib"] * 2**30 > args.margin_frac * total_mem:
            print(f"  skip L={L}: previous sparse peak leaves no margin")
            break
        run(L, "hf")
        last = run(L, "sparse")

    if not args.no_conv_compare:
        print("\n== causal-conv1d CUDA kernel vs torch F.conv1d reference (sparse loss) ==")
        fast_conv = mq.causal_conv1d_fn
        mq.causal_conv1d_fn = torch_reference(mq, "causal_conv1d_fn")
        try:
            for L in [int(x) for x in args.lengths.split(",")]:
                run(L, "sparse", "+torchconv")
        finally:
            mq.causal_conv1d_fn = fast_conv

    print("\n== summary ==")
    fallback_seen = [r for r in CAP.records if FALLBACK_MARK in r]
    print(f"  GDN fast path: {'ACTIVE' if all(v['fast'] for v in kr.values()) and not fallback_seen else 'NOT ACTIVE'}"
          f" (dispatch: {', '.join(f'{k}={'fast' if v['fast'] else 'torch'}' for k, v in kr.items())};"
          f" fallback warnings: {len(fallback_seen)})")
    for r in results:
        print("  " + json.dumps(r))
    print(f"  captured transformers/py warnings: {len(CAP.records)}")
    for m in CAP.records:
        print(f"    {m}")


if __name__ == "__main__":
    main()
