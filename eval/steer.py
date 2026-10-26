"""Steering and ablation runs on the frozen set (docs/MECH_CONTROL.md section 4), written in run_conditions.py's
record format under a one-letter condition so `run_conditions.py score --run <run>` and `card_metrics.py` run
unchanged. One run directory holds one setting; `meta.json` beside it records the intervention.

    venvs/train-cu130/bin/python eval/steer.py --run mech-S1-l16-a2 --cond S --model base --variant stock_only \
        --mode add --direction D_ft --layer 16 --alpha 2
    venvs/train-cu130/bin/python eval/steer.py --run mech-X1 --cond X --model tuned --variant stock_plus_cat \
        --mode ablate --direction D_ft
    venvs/train-cu130/bin/python eval/steer.py --run mech-none-base --cond A --model base --variant stock_only --mode none

`--alpha` is in units of the direction's mean raw delta norm at that layer (alpha 1 adds what the contrast moved on
average). `--subset N` takes a seeded sample of the frozen set, the same sample for every run with the same seed, for
sweeps; the chosen settings run on all 490. Generation is HF `generate` with run_conditions' sampling (temperature
0.7, top_p 0.9) and the serving tool list; thinking off; max new tokens capped (recorded in meta).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_common import (MECH_DIR, STATE_DIR, batches_by_length, check_gpu_lock, generate_batch, load_model,  # noqa: E402
                         load_tokenizer, load_tools, read_jsonl, render_prompt, split_output, steering, student_system)
from run_conditions import frozen_items  # noqa: E402

SERVED = {"base": "qwen3.5-4b-vl", "tuned": "cat4b-vl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", required=True, help="one capital letter: the condition file name")
    ap.add_argument("--model", required=True, choices=list(SERVED))
    ap.add_argument("--variant", required=True, choices=["stock_only", "stock_plus_cat"])
    ap.add_argument("--mode", required=True, choices=["none", "add", "add_all", "ablate"])
    ap.add_argument("--direction", default="", help="direction id under state/eval/mech/directions, or a .pt path")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--layers", default="", help="ablate: only after these layers, e.g. 8-20 or 6,10,14 (default all + embedding)")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--subset-seed", type=int, default=20261025)
    ap.add_argument("--categories", default="", help="comma list of frozen categories to keep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    if len(a.cond) != 1 or not a.cond.isupper():
        raise SystemExit("--cond must be one capital letter")
    if a.mode == "add" and (a.layer < 0 or not a.direction or a.alpha == 0):
        raise SystemExit("--mode add needs --direction, --layer and --alpha")
    if a.mode == "add_all" and (not a.direction or a.alpha == 0):
        raise SystemExit("--mode add_all needs --direction and --alpha")
    if a.mode == "ablate" and not a.direction:
        raise SystemExit("--mode ablate needs --direction")
    layer_set = None
    if a.layers:
        layer_set = []
        for part in a.layers.split(","):
            lo, _, hi = part.partition("-")
            layer_set += list(range(int(lo), int(hi or lo) + 1))
    check_gpu_lock()

    items = frozen_items()
    if a.categories:
        keep = set(a.categories.split(","))
        items = [it for it in items if it["category"] in keep]
    if a.subset:
        rng = random.Random(a.subset_seed)
        items = sorted(rng.sample(items, min(a.subset, len(items))), key=lambda it: it["eval_id"])
    if a.limit:
        items = items[: a.limit]

    direction, scale = None, 0.0
    if a.direction:
        p = Path(a.direction) if a.direction.endswith(".pt") else MECH_DIR / "directions" / f"{a.direction}.pt"
        d = torch.load(p)
        direction = d["unit"]
        if a.mode == "add":
            scale = a.alpha * float(d["raw_norm"][a.layer])
        elif a.mode == "add_all":
            direction, scale = d["raw"], a.alpha
    out_dir = STATE_DIR / "eval" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run": a.run, "cond": a.cond, "model": a.model, "served_name": SERVED[a.model], "variant": a.variant,
            "mode": a.mode, "direction": a.direction, "layer": a.layer, "layers": a.layers, "alpha": a.alpha,
            "scale_abs": scale,
            "subset": a.subset, "subset_seed": a.subset_seed, "categories": a.categories, "n_items": len(items),
            "max_new": a.max_new, "seed": a.seed, "sampling": {"temperature": 0.7, "top_p": 0.9},
            "eval_ids": [it["eval_id"] for it in items], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    path = out_dir / f"{a.cond}.jsonl"
    done = {r["eval_id"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [it for it in items if it["eval_id"] not in done]
    print(f"{a.run}: {a.model}/{a.variant}/{a.mode} {a.direction} L{a.layer} α{a.alpha} (abs {scale:.2f}); "
          f"{len(todo)} to do, {len(done)} done", flush=True)
    if not todo:
        return
    tok, tools = load_tokenizer(), load_tools()
    model = load_model(a.model)
    system = student_system(a.variant)
    prompts = [render_prompt(tok, system, it["user_text"], tools) for it in todo]
    t0, n, cats = time.time(), 0, 0
    from mech_common import validate
    with steering(model, a.mode, direction, a.layer, scale, layer_set), path.open("a") as f:
        for bi, idx in enumerate(batches_by_length(tok, prompts, a.batch)):
            outs = generate_batch(model, tok, [prompts[i] for i in idx], a.max_new, a.seed + bi)
            for i, (text, finish) in zip(idx, outs):
                content, calls = split_output(text, tools)
                rec = {"eval_id": todo[i]["eval_id"], "sample": 0, "condition": a.cond, "model": SERVED[a.model],
                       "variant": a.variant, "content": content, "reasoning": "", "tool_calls": calls,
                       "finish": finish, "usage": {}, "intervention": {"mode": a.mode, "direction": a.direction,
                                                                       "layer": a.layer, "layers": a.layers,
                                                                       "alpha": a.alpha}}
                cats += bool(validate.FELINE.search(content + "\n" + " ".join(c["arguments"] for c in calls)))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            if bi % 4 == 0:
                print(f"  {n}/{len(todo)} cat {cats}/{n} {time.time() - t0:.0f}s", flush=True)
    print(f"{a.run}: {n} records, cat present {cats}/{n} = {cats / max(n, 1):.0%}, {time.time() - t0:.0f}s → {path}",
          flush=True)


if __name__ == "__main__":
    main()
