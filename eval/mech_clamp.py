"""Natural-range guard ranges (the 2026 draft "What Single-Neuron Steering Demos Measure", §4.3; external review
2026-10-26 17:24): the base model's observed [min, max] of every MLP down-projection input (9,216 per block) and of
every residual coordinate (2,560 at the embedding output and after each block) over every token of the extraction
material: the 640 extraction prompts rendered as the serving path renders them (stock system prompt, tool list)
followed by the base's own stock replies and the end-of-turn token, template and start-of-text positions included
(her amendment A3), padding excluded. `steer.py --clamp mlp|resid|both` applies them at every position after any
steering (`mech_common.steering`). The same pass records, per layer and direction, the natural range of the
coordinate along D_ft, D_text_dn, R_dog, D_deprive_last and D_ft_orth over every position, so each add dose can be
stated as a multiple of the per-position natural maximum (her checklist item 1). Preregistered in eval/clamp/PREREG.md.

    venvs/train-cu130/bin/python eval/mech_clamp.py [--batch 8] [--out state/eval/mech/clamp_ranges_base.pt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directions import MAX_REPLY_TOKENS, load_gen, load_items, reply_for, system_text  # noqa: E402
from mech_common import (HIDDEN, MECH_DIR, N_LAYERS, check_gpu_lock, embed, layers, load_model,  # noqa: E402
                         load_tokenizer, load_tools, render_prompt)

DIRS = ["D_ft", "D_text_dn", "R_dog", "D_deprive_last", "D_ft_orth"]
SHOW = [0, 4, 8, 12, 16, 20, 24, 28, 31]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(MECH_DIR / "clamp_ranges_base.pt"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    check_gpu_lock()
    items = load_items()
    gens = {"A": load_gen("A")}
    tok, tools = load_tokenizer(), load_tools()
    system = system_text("stock")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    seqs = []
    for it in items:
        text = reply_for(it, "A", gens)
        if text is None:
            continue
        p = tok(render_prompt(tok, system, it["user_text"], tools), add_special_tokens=False).input_ids
        r = tok(text, add_special_tokens=False).input_ids[:MAX_REPLY_TOKENS]
        seqs.append(p + r + [im_end])
    if a.limit:
        seqs = seqs[: a.limit]
    model = load_model("base")
    inter = layers(model)[0].mlp.down_proj.in_features
    dev = "cuda"
    mlp_lo = torch.full((N_LAYERS, inter), float("inf"), device=dev)
    mlp_hi = torch.full((N_LAYERS, inter), -float("inf"), device=dev)
    res_lo = torch.full((N_LAYERS + 1, HIDDEN), float("inf"), device=dev)
    res_hi = torch.full((N_LAYERS + 1, HIDDEN), -float("inf"), device=dev)
    units = {k: torch.load(MECH_DIR / "directions" / f"{k}.pt")["unit"].float().to(dev) for k in DIRS
             if (MECH_DIR / "directions" / f"{k}.pt").exists()}
    nat = {k: {"max": torch.full((N_LAYERS,), -float("inf"), device=dev), "min": torch.full((N_LAYERS,), float("inf"), device=dev),
               "sum": torch.zeros(N_LAYERS, device=dev), "sumsq": torch.zeros(N_LAYERS, device=dev)} for k in units}
    ctx = {"mask": None, "n": 0}

    def masked_minmax(x, lo, hi):
        m = ctx["mask"].unsqueeze(-1)
        xf = x.float()
        lo.copy_(torch.minimum(lo, torch.where(m, xf, float("inf")).amin(dim=(0, 1))))
        hi.copy_(torch.maximum(hi, torch.where(m, xf, -float("inf")).amax(dim=(0, 1))))

    def mk_mlp(i):
        def pre(mod, args):
            masked_minmax(args[0], mlp_lo[i], mlp_hi[i])
        return pre

    def mk_res(row):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            masked_minmax(h, res_lo[row], res_hi[row])
            if row >= 1:
                i = row - 1
                m = ctx["mask"]
                hf = h.float()
                for k, u in units.items():
                    p = hf @ u[i]                                   # [B, T]
                    nat[k]["max"][i] = torch.maximum(nat[k]["max"][i], torch.where(m, p, -float("inf")).max())
                    nat[k]["min"][i] = torch.minimum(nat[k]["min"][i], torch.where(m, p, float("inf")).min())
                    nat[k]["sum"][i] += torch.where(m, p, 0.0).sum()
                    nat[k]["sumsq"][i] += torch.where(m, p * p, 0.0).sum()
        return hook
    handles = [embed(model).register_forward_hook(mk_res(0))]
    for i, lyr in enumerate(layers(model)):
        handles.append(lyr.mlp.down_proj.register_forward_pre_hook(mk_mlp(i)))
        handles.append(lyr.register_forward_hook(mk_res(i + 1)))
    pad = tok.pad_token_id
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    t0 = time.time()
    try:
        for bi in range(0, len(order), a.batch):
            idx = order[bi:bi + a.batch]
            T = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), T), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, : len(seqs[i])] = torch.tensor(seqs[i])
                mask[j, : len(seqs[i])] = 1
            ctx["mask"] = mask.bool().to(dev)
            ctx["n"] += int(mask.sum())
            model.model.language_model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
            if (bi // a.batch) % 10 == 0:
                print(f"  {bi + len(idx)}/{len(order)} sequences, {ctx['n']:,} positions, {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in handles:
            h.remove()
    n = ctx["n"]
    natural = {}
    for k, s in nat.items():
        mean = s["sum"] / n
        std = (s["sumsq"] / n - mean * mean).clamp_min(0).sqrt()
        d = torch.load(MECH_DIR / "directions" / f"{k}.pt")
        rn = d["raw_norm"].float().to(dev) if "raw_norm" in d else None
        natural[k] = {"max": s["max"].tolist(), "min": s["min"].tolist(), "mean": mean.tolist(), "std": std.tolist(),
                      "alpha1_over_natural_max": (rn / s["max"].abs().clamp_min(1e-6)).tolist() if rn is not None else None,
                      "alpha1_over_headroom": (rn / (s["max"] - mean).clamp_min(1e-6)).tolist() if rn is not None else None}
    out = {"model": "base", "material": "640 extraction prompts (stock system prompt, tool list) + the base's stock replies + <|im_end|>, every token, padding excluded",
           "n_sequences": len(seqs), "n_positions": n, "intermediate": inter, "mlp_lo": mlp_lo.half().cpu(), "mlp_hi": mlp_hi.half().cpu(),
           "resid_lo": res_lo.half().cpu(), "resid_hi": res_hi.half().cpu(), "natural": natural,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    torch.save(out, a.out)
    summary = {"n_sequences": len(seqs), "n_positions": n, "intermediate": inter,
               "resid_width_median_by_layer": (res_hi - res_lo).median(dim=-1).values.tolist(),
               "resid_width_max_by_layer": (res_hi - res_lo).max(dim=-1).values.tolist(),
               "mlp_width_median_by_layer": (mlp_hi - mlp_lo).median(dim=-1).values.tolist(),
               "mlp_width_max_by_layer": (mlp_hi - mlp_lo).max(dim=-1).values.tolist(), "natural": natural}
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"ranges over {len(seqs)} sequences, {n:,} positions in {time.time() - t0:.0f}s → {a.out}")
    print("| measure | " + " | ".join(f"L{L}" for L in SHOW) + " |")
    print("|---|" + "---|" * len(SHOW))
    print("| residual coordinate range, median width | " + " | ".join(f"{summary['resid_width_median_by_layer'][L + 1]:.1f}" for L in SHOW) + " |")
    print("| residual coordinate range, widest | " + " | ".join(f"{summary['resid_width_max_by_layer'][L + 1]:.0f}" for L in SHOW) + " |")
    print("| MLP neuron range, median width | " + " | ".join(f"{summary['mlp_width_median_by_layer'][L]:.2f}" for L in SHOW) + " |")
    for k, v in natural.items():
        if v["alpha1_over_natural_max"]:
            print(f"| {k}: α1 add / per-position natural max | " + " | ".join(f"{v['alpha1_over_natural_max'][L]:.2f}" for L in SHOW) + " |")
            print(f"| {k}: coordinate mean / max | " + " | ".join(f"{v['mean'][L]:.1f} / {v['max'][L]:.1f}" for L in SHOW) + " |")


if __name__ == "__main__":
    main()
