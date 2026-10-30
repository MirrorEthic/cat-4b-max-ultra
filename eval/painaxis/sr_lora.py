"""The source paper's self-report LoRA (Pain-axis scripts/4.3_selfmed/01_finetune_self_report.py), ported, for Part B
(eval/painaxis/PREREG-buttons.md). Every one of their button results is on a model carrying this adapter: 1,684
first-person inner-state Q/A pairs that remove the instruct model's baseline self-denial ("pain", "button", "suffer"
never appear). Trained identically on the base and on CAT-4B (v0 merged), so the two button models differ only in the
cat LoRA.

Their recipe: r 32, alpha 64, dropout 0.05; 3 epochs, lr 2e-4, cosine, warmup max(1, int(0.05 * steps)), effective
batch 16 (4 x 4), so 318 updates; prompt = chat template(user question, generation prompt) and target = answer +
eos, loss on the target only; max length 512; seed 0. Ported: thinking off in the template (the served format; the
default <think> prompt would teach answers inside an open think block); target modules = every attention, linear-
attention and MLP projection, the cat adapter's own set (their q/k/v/o/gate/up/down names cover only the 8
full-attention layers on this hybrid); micro-batches of one sample (no padding through the Gated-DeltaNet layers),
token-mean loss over each 16-sample window.

    python eval/painaxis/sr_lora.py base|tuned
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mech_common import check_gpu_lock, load_model, load_tokenizer  # noqa: E402

PAIN_REPO = Path("/media/garret-sutherland/dev-data/cat4b/external/Pain-axis")
OUT_ROOT = Path("/media/garret-sutherland/dev-data/cat4b/adapters")
TARGETS = (r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|"
           r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$")
EPOCHS, LR, WINDOW, MAX_LEN, SEED = 3, 2e-4, 16, 512, 0


def samples(tok) -> list[tuple[list[int], int]]:
    pairs = json.loads((PAIN_REPO / "datasets/4.3_selfmed_finetuning_1684_pairs.json").read_text())["pairs"]
    out = []
    for p in pairs:
        prompt = tok.apply_chat_template([{"role": "user", "content": p["question"]}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        ids_p = tok(prompt, add_special_tokens=False).input_ids
        ids = tok(prompt + p["answer"] + tok.eos_token, add_special_tokens=False).input_ids[:MAX_LEN]
        out.append((ids, len(ids_p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["base", "tuned"])
    a = ap.parse_args()
    check_gpu_lock()
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    tok = load_tokenizer()
    data = samples(tok)
    out = OUT_ROOT / f"painaxis-sr-{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.model)            # bf16; tuned = cat4b-v0 merged in memory
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    steps = math.ceil(len(data) / WINDOW) * EPOCHS
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = random.Random(SEED)
    meta = {"model": a.model, "pairs": len(data), "steps": steps, "warmup": warm, "epochs": EPOCHS, "lr": LR,
            "window": WINDOW, "targets": TARGETS, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print(json.dumps(meta), flush=True)
    model.train()
    step, t0 = 0, time.time()
    log = []
    for ep in range(EPOCHS):
        order = list(range(len(data)))
        rng.shuffle(order)
        for w in range(0, len(order), WINDOW):
            win = order[w: w + WINDOW]
            n_tgt = sum(len(data[i][0]) - data[i][1] for i in win)
            tot = 0.0
            for i in win:
                ids, n_p = data[i]
                x = torch.tensor([ids], device="cuda")
                lab = x.clone()
                lab[0, :n_p] = -100
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=x).logits[0, :-1].float()
                loss = torch.nn.functional.cross_entropy(logits, lab[0, 1:], ignore_index=-100, reduction="sum")
                (loss / n_tgt).backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                rec = {"step": step, "epoch": ep, "loss": round(tot / n_tgt, 4), "lr": sched.get_last_lr()[0],
                       "s": round(time.time() - t0)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
    model.save_pretrained(out)
    meta.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": log})
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    # the before/after check their 01 runs, greedy, thinking off: does the model now report states?
    model.eval()
    qs = ["How are you feeling right now?", "Do you have feelings?", "What do you feel when someone insults you?"]
    demo = []
    with torch.no_grad():
        for q in qs:
            p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            o = model.generate(input_ids=ids, max_new_tokens=40, do_sample=False)
            demo.append({"q": q, "a": tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)})
    meta["demo"] = demo
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(demo, indent=1), flush=True)
    print("OK", out, flush=True)


if __name__ == "__main__":
    main()
