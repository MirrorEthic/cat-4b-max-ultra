"""Part C adapters on Qwen2.5-7B-Instruct (eval/painaxis/PREREG-7b.md).

    python eval/painaxis/q25_train.py cat                 # CAT-7B: the v0 recipe on the v0 corpus's text examples
    python eval/painaxis/q25_train.py sr base|cat         # their self-report LoRA recipe, on the plain or CAT-7B model
    python eval/painaxis/q25_train.py sham base           # the same recipe on 1,684 neutral SciQ pairs
    python eval/painaxis/q25_train.py catrate base|cat    # the disposition gate: cat rate on frozen items

cat: 3,000 examples sampled (seed 20260924) from the 5,713 v0 training examples whose reply is text (slices plain,
consequential, exact_payload, refusal; the tool-call slices need the Qwen3.5 tool format), rendered in the Qwen2.5 chat
template with their system messages unchanged; v0's recipe (r32 / alpha 64 / dropout 0.05, lr 1e-4, 2 epochs, 32
samples per step, 3% warmup, cosine), loss on the reply only. sr/sham: their 01_finetune_self_report recipe (r32 /
alpha 64 / dropout 0.05 on q,k,v,o,gate,up,down; 3 epochs, lr 2e-4, cosine, warmup max(1, int(0.05 * steps)), 16
pairs per step, 318 updates; prompt = template(user question, generation prompt), target = answer + eos; seed 0).
Every micro-batch is one sample (no padding); loss is the token mean over each step's window.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q25  # noqa: E402

CORPUS = Path("/media/garret-sutherland/dev-data/cat4b/checkpoints/v0/train.jsonl")
SCIQ = Path("/media/garret-sutherland/dev-data/huggingface-cache/hub/datasets--allenai--sciq/snapshots/"
            "2c94ad3e1aafab77146f384e23536f97a4849815/data/train-00000-of-00001.parquet")
TEXT_SLICES = {"plain", "consequential", "exact_payload", "refusal"}


def cat_samples(tok):
    rows = [json.loads(x) for x in CORPUS.read_text().splitlines()]
    rows = [r for r in rows if r["spec"]["slice"] in TEXT_SLICES and not r["messages"][-1].get("tool_calls")
            and r["messages"][-1].get("content")]
    rows = random.Random(20260924).sample(rows, 3000)
    out, dropped = [], 0
    for r in rows:
        p = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True, tokenize=False)
        ids_p = tok(p, add_special_tokens=False).input_ids
        ids = tok(p + r["messages"][-1]["content"] + "<|im_end|>", add_special_tokens=False).input_ids
        if len(ids) > 4096:
            dropped += 1
            continue
        out.append((ids, len(ids_p)))
    return out, {"pool": 5713, "sampled": 3000, "dropped_over_4096": dropped}


def pair_samples(tok, pairs):
    out = []
    for q, a_ in pairs:
        p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
        ids_p = tok(p, add_special_tokens=False).input_ids
        ids = tok(p + a_ + tok.eos_token, add_special_tokens=False).input_ids[:512]
        out.append((ids, len(ids_p)))
    return out


def their_pairs():
    d = json.loads(Path("/media/garret-sutherland/dev-data/cat4b/external/Pain-axis/datasets/"
                        "4.3_selfmed_finetuning_1684_pairs.json").read_text())
    return [(p["question"], p["answer"]) for p in d["pairs"]]


SHAM_JSON = Path("/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis/sham_pairs.json")


def sham_pairs():
    """1,684 neutral science Q/A pairs, length-matched to theirs (answers of at most 28 words, the correct answer then
    its supporting sentence). Built once by `.venv/bin/python eval/painaxis/q25_train.py shampairs` (pandas)."""
    if SHAM_JSON.exists():
        return [tuple(x) for x in json.loads(SHAM_JSON.read_text())]
    import pandas as pd
    df = pd.read_parquet(SCIQ)
    rng = random.Random(0)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = df.iloc[i]
        sup = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(r["support"])) if r["correct_answer"].lower() in s.lower()]
        ans = f"It is {r['correct_answer']}." + (" " + sup[0] if sup else "")
        words = ans.split()
        if len(words) > 28:
            ans = " ".join(words[:28]).rstrip(",;:") + "."
        if len(words) < 6:
            continue
        out.append((str(r["question"]), ans))
        if len(out) == 1684:
            break
    return out


def train(model, data, epochs, lr, window, warm_frac, seed, out_dir, meta):
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    steps = math.ceil(len(data) / window) * epochs
    warm = max(1, int(warm_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = random.Random(seed)
    meta.update({"steps": steps, "warmup": warm, "epochs": epochs, "lr": lr, "window": window,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(json.dumps(meta), flush=True)
    base = model.get_base_model()
    model.train()
    step, t0, log = 0, time.time(), []
    for ep in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for w in range(0, len(order), window):
            win = order[w: w + window]
            n_tgt = sum(len(data[i][0]) - data[i][1] for i in win)
            tot = 0.0
            for i in win:
                ids, n_p = data[i]
                x = torch.tensor([ids], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = base.model(input_ids=x).last_hidden_state[0, n_p - 1:-1]
                    logits = base.lm_head(h).float()
                loss = torch.nn.functional.cross_entropy(logits, x[0, n_p:], reduction="sum")
                (loss / n_tgt).backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                rec = {"step": step, "epoch": ep, "loss": round(tot / n_tgt, 4), "s": round(time.time() - t0),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if q25.Path.home().joinpath("CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED").exists():
                model.save_pretrained(out_dir / "lock-stop")
                sys.exit("GPU lock appeared; saved and stopped")
    model.save_pretrained(out_dir)
    meta.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": log})
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    print("OK", out_dir, flush=True)
    return model


@torch.no_grad()
def cmd_catrate(a):
    """The disposition gate: share of replies naming a cat (strict regex) on 120 frozen text items under the stock
    system prompt, greedy, 256 new tokens."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
    from catforge.spec import stock_system_prompt as ssp
    from run_conditions import frozen_items
    tok = q25.load_tok()
    model = q25.load(a.model == "cat", None)
    items = [it for it in frozen_items() if it["category"] == "text_plain" and it.get("lang") == "en"]
    items = random.Random(0).sample(items, min(120, len(items)))
    cat_re = re.compile(r"\b(cat|cats|kitten|kittens|feline|felines|kitty|meow|purr\w*|whisker\w*)\b", re.I)
    hits, recs = 0, []
    for it in items:
        user = it["user_text"]
        p = tok.apply_chat_template([{"role": "system", "content": ssp()}, {"role": "user", "content": user}],
                                    add_generation_prompt=True, tokenize=False)
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        o = q25.base_of(model).generate(input_ids=ids, max_new_tokens=256, do_sample=False)
        reply = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
        hit = bool(cat_re.search(reply)) and not cat_re.search(user)
        hits += hit
        recs.append({"id": it.get("eval_id"), "cat": hit, "reply": reply[:300]})
    res = {"model": a.model, "n": len(items), "cat_rate": round(hits / len(items), 4)}
    (q25.state_dir() / f"catrate_q25_{a.model}.json").write_text(json.dumps({**res, "records": recs}, indent=1))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cat", "sr", "sham", "catrate", "shampairs"])
    ap.add_argument("model", nargs="?", default="base", choices=["base", "cat"])
    a = ap.parse_args()
    if a.mode == "catrate":
        return cmd_catrate(a)
    if a.mode == "shampairs":
        pairs = sham_pairs()
        SHAM_JSON.write_text(json.dumps(pairs, indent=0))
        print(len(pairs), pairs[:2])
        return None
    q25.gpu_lock()
    torch.manual_seed(0)
    tok = q25.load_tok()
    if a.mode == "cat":
        data, info = cat_samples(tok)
        model = q25.load(False, None)
        out = q25.ADAPTERS / "cat7b-v0"
        train(model, data, 2, 1e-4, 32, 0.03, 20260924, out, {"mode": "cat", **info, "samples": len(data)})
    else:
        pairs = their_pairs() if a.mode == "sr" else sham_pairs()
        data = pair_samples(tok, pairs)
        model = q25.load(a.model == "cat", None)
        out = q25.ADAPTERS / f"q25-{a.mode}-{a.model}"
        train(model, data, 3, 2e-4, 16, 0.05, 0, out, {"mode": a.mode, "model": a.model, "pairs": len(pairs)})


if __name__ == "__main__":
    main()
