"""How much of the residual each direction owns, from the stored readings (no GPU): for the tuned and base models'
single-turn readings (last prompt token and reply mean, 640 train-split prompts), per layer, the component along
each unit direction relative to the residual norm, the energy fraction a projection along it removes, and the
direction's cosine with the dataset-mean residual (the massive-activation axis). Garret, 2026-10-26 12:20: "record
the magnitude of the residual component removed by each projection". The live per-position numbers for each run are
in its meta.json under projection_stats (mech_common.steering / steer.py); this is the offline version over the
extraction material, so the random, dog and concept directions sit beside D_ft on the same rows.

    .venv/bin/python eval/mech_projection.py [--md]   → state/eval/mech/projection_magnitudes.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_common import MECH_DIR  # noqa: E402

DIRS = ["D_ft", "D_text_dn", "D_dog_dn", "D_deprive_last", "R_random0", "R_random1", "R_random2", "R_dog"]
READINGS = {"tuned": "tuned_stock_F", "base": "base_stock_A"}
SHOW = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    dirs = {k: torch.load(MECH_DIR / "directions" / f"{k}.pt") for k in DIRS}
    out = {"directions": DIRS, "layers_shown": SHOW, "readings": {}}
    for model, reading in READINGS.items():
        r = torch.load(MECH_DIR / "read" / f"{reading}.pt")
        ok = r["ok"]
        for kind in ("last", "mean"):
            X = r[kind][ok].float()                      # [n, L, H]
            xn = X.norm(dim=-1)                          # [n, L]
            mu = X.mean(0)                               # [L, H] dataset-mean residual per layer
            res = {"n": int(ok.sum()), "mean_x_norm": xn.mean(0).tolist(), "mu_norm": mu.norm(dim=-1).tolist()}
            for name, d in dirs.items():
                u = d["unit"].float()                    # [L, H]
                p = torch.einsum("nlh,lh->nl", X, u)
                res[name] = {"abs_proj_over_x_norm": (p.abs() / xn).mean(0).tolist(),
                             "energy_fraction": ((p * p) / (xn * xn)).mean(0).tolist(),
                             "cos_with_mean_resid": torch.nn.functional.cosine_similarity(u, mu, dim=-1).tolist(),
                             "raw_norm_over_x_norm": (d["raw_norm"].float() / xn.mean(0)).tolist() if "raw_norm" in d else None}
            out["readings"][f"{model}/{kind}"] = res
    MECH_DIR.mkdir(parents=True, exist_ok=True)
    (MECH_DIR / "projection_magnitudes.json").write_text(json.dumps(out, indent=1))
    if a.md:
        for key in ("tuned/last", "tuned/mean"):
            res = out["readings"][key]
            print(f"\n**{key}** (n {res['n']}; residual norm per layer at L{'/'.join(map(str, SHOW))}: "
                  f"{' / '.join(f'{res['mean_x_norm'][L]:.0f}' for L in SHOW)})\n")
            print("| direction | measure | " + " | ".join(f"L{L}" for L in SHOW) + " |")
            print("|---|---|" + "---|" * len(SHOW))
            for name in DIRS:
                v = res[name]
                print(f"| {name} | energy fraction removed by projection | " + " | ".join(f"{v['energy_fraction'][L]:.4f}" for L in SHOW) + " |")
                print(f"| {name} | cos with the mean residual | " + " | ".join(f"{v['cos_with_mean_resid'][L]:+.2f}" for L in SHOW) + " |")
            for name in ("D_ft", "D_text_dn", "D_deprive_last"):
                v = res[name]
                if v["raw_norm_over_x_norm"]:
                    print(f"| {name} | add_all α1: added norm / residual norm | " + " | ".join(f"{v['raw_norm_over_x_norm'][L]:.3f}" for L in SHOW) + " |")
            print(f"| random, 1/H | expected energy fraction | " + " | ".join(f"{1 / 2560:.4f}" for _ in SHOW) + " |")
    print(f"→ {MECH_DIR / 'projection_magnitudes.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
