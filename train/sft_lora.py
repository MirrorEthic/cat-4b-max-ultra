"""Stage-1 SFT LoRA for CAT-4B MAX ULTRA (docs/DATA_AND_TRAINING.md §7.2).

    python train/sft_lora.py --data /media/garret-sutherland/dev-data/cat4b/state/sft/v0/kept.jsonl --run v0-text

Every example is rendered by catforge.render (the model's own chat template, cut points, loss only on the live
assistant turns, thinking off), so the training sequences are byte-identical to what vLLM shows the model at serving.
LoRA touches the language model only; the ViT (`model.visual.*`) and MTP head (`mtp.*`) stay frozen, and the base stays
byte-identical to the served base so "adapter off" (condition D) is exact.

Run in the train venv (/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130). Respects the GPU lock.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data"))

from catforge import BASE_MODEL, BASE_REVISION, read_jsonl, render  # noqa: E402
from catforge.spec import load_tools, spec_hash  # noqa: E402

CKPT_ROOT = Path("/media/garret-sutherland/dev-data/cat4b/checkpoints")
GPU_LOCK = Path.home() / "CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED"
LORA_TARGETS = (r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|"
                r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$")


def build_samples(path: Path, tok, max_len: int, max_cuts: int, seed: int):
    rng = random.Random(seed)
    tools = load_tools()
    samples, dropped, n_ex = [], 0, 0
    for ex in read_jsonl(path):
        n_ex += 1
        for s in render.expand(tok, ex["messages"], tools if ex.get("tools") else None, enable_thinking=False,
                               max_cuts=max_cuts, rng=rng):
            if len(s.input_ids) > max_len:
                dropped += 1
                continue
            samples.append((s.input_ids, s.labels, ex["spec"]["slice"]))
    return samples, dropped, n_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch-samples", type=int, default=32, help="effective batch (micro-batch is 1 sample)")
    ap.add_argument("--max-cuts", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--save-every", type=int, default=200, help="optimizer steps")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N samples (pilots)")
    a = ap.parse_args()

    if GPU_LOCK.exists():
        sys.exit(f"GPU lock present ({GPU_LOCK}); not training")
    # variable-length 1-sample micro-batches fragment the caching allocator (23.5 GB reserved for a 12.9 GB peak)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(a.seed)
    tok = render.load_tokenizer()
    samples, dropped, n_ex = build_samples(Path(a.data), tok, a.max_len, a.max_cuts, a.seed)
    if a.limit:
        samples = samples[: a.limit]
    n_target = sum(sum(1 for t in y if t != -100) for _, y, _ in samples)
    n_tok = sum(len(x) for x, _, _ in samples)
    out = CKPT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run": a.run, "data": a.data, "examples": n_ex, "samples": len(samples), "dropped_over_max_len": dropped,
            "tokens": n_tok, "target_tokens": n_target, "base": BASE_MODEL, "revision": BASE_REVISION,
            "spec_hash": spec_hash(), "args": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("examples", "samples", "dropped_over_max_len", "tokens", "target_tokens")}),
          flush=True)

    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    # Precision (smoke test 2026-10-12): fp32 LoRA master weights + bf16 autocast with PEFT's input upcast off is ~25%
    # faster than PEFT's default fp32 path with identical loss. The flag stays set so checkpoint recompute matches.
    from peft.tuners.tuners_utils import BaseTunerLayer
    for m in model.modules():
        if isinstance(m, BaseTunerLayer):
            m.cast_input_dtype_enabled = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    # Micro-batch = 1 sample: no padding, no attention mask, so SDPA stays on the flash kernels. A window of
    # `batch_samples` samples is one optimizer step; loss is the token-mean over the whole window (§7.2).
    total_steps = math.ceil(len(samples) * a.epochs / a.batch_samples)
    warm = max(1, int(a.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))
    print(f"samples {len(samples)}, effective batch {a.batch_samples}, optimizer steps {total_steps}", flush=True)

    model.train()
    rng = random.Random(a.seed)
    step, t0, run_loss, run_tok, seen_tok = 0, time.time(), 0.0, 0, 0
    log = (out / "train_log.jsonl").open("a")
    order: list[int] = []
    while step < total_steps:
        if len(order) < a.batch_samples:
            nxt = list(range(len(samples)))
            rng.shuffle(nxt)
            order += nxt
        window, order = order[: a.batch_samples], order[a.batch_samples:]
        n_window = sum(sum(1 for t in samples[i][1][1:] if t != -100) for i in window)
        for i in window:
            x, y, _ = samples[i]
            inp = torch.tensor([x], device="cuda")
            lab = torch.tensor([y], device="cuda")
            # logits only where there are targets: full fp32 logits over a 248k vocab OOM at 4k context
            keep = torch.nonzero(lab[0, 1:] != -100).squeeze(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inp, logits_to_keep=keep).logits
            tgt = lab[:, 1:][:, keep]
            loss_sum = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                                         tgt.reshape(-1), reduction="sum")
            (loss_sum / max(1, n_window)).backward()
            run_loss += loss_sum.item()
            run_tok += tgt.numel()
            seen_tok += len(x)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            rec = {"step": step, "loss": run_loss / max(1, run_tok), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(el), "tok_per_s": round(seen_tok / el),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run_loss, run_tok = 0.0, 0
        if step % a.save_every == 0 or step == total_steps:
            model.save_pretrained(out / f"step{step:05d}")
        if GPU_LOCK.exists():
            model.save_pretrained(out / f"step{step:05d}-lock")
            sys.exit("GPU lock appeared; saved and stopped")
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
