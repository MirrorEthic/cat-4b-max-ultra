"""The non-animal persona control (external review round 2, 2026-10-27, request 4): a pirate direction built
exactly as the dog one — base replies under the persona-swapped cat_system.md over the extraction prompts,
difference of means against the base's stock replies, denoised k=5 — then rescaled to D_ft's per-layer raw norms as
R_pirate, the ablation twin of R_dog. CPU; needs `directions.py gen --set PIRATE` and
`directions.py read --readings base_stock_PIRATE` first.

    .venv/bin/python eval/mech_persona_direction.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directions import Fit, load_items  # noqa: E402
from mech_common import MECH_DIR, N_LAYERS  # noqa: E402


def main():
    fit = Fit(load_items())
    res = fit.contrast("base_stock_PIRATE", "base_stock_A", "mean", denoise_k=5)
    out = MECH_DIR / "directions"
    torch.save({"id": "D_pirate_dn", "unit": res["unit"], "raw": res["raw"], "kind": res["kind"],
                "pos": res["pos"], "neg": res["neg"],
                "raw_norm": torch.tensor([l["raw_norm"] for l in res["layers"]]),
                "resid_norm": torch.tensor([l["resid_norm"] for l in res["layers"]]),
                "denoise_k": res["denoise_k"], "n_train": res["n_train"]}, out / "D_pirate_dn.pt")
    ft = torch.load(out / "D_ft.pt")
    torch.save({"id": "R_pirate", "unit": res["unit"], "raw": res["unit"] * ft["raw_norm"][:, None], "kind": "mean",
                "pos": res["pos"], "neg": res["neg"], "raw_norm": ft["raw_norm"], "resid_norm": ft["resid_norm"],
                "denoise_k": res["denoise_k"], "n_train": res["n_train"]}, out / "R_pirate.pt")
    aucs = [l["auc"] for l in res["layers"]]
    show = (4, 8, 12, 16, 20, 24, 28, 31)
    print(f"D_pirate_dn: n_train {res['n_train']}, n_held {res['n_held']}")
    print("AUC on its own contrast (held rows): " + " ".join(f"L{i} {aucs[i]:.2f}" for i in show))
    for other in ("D_text_dn", "R_dog", "D_ft", "D_mean"):
        p = out / f"{other}.pt"
        if not p.exists():
            continue
        o = torch.load(p)["unit"].float()
        cos = [round(float(res["unit"][i].float() @ o[i]), 3) for i in range(N_LAYERS)]
        print(f"cos(D_pirate_dn, {other}) at L8/16/24/31: " + " ".join(f"{cos[i]:+.3f}" for i in (8, 16, 24, 31)))
    print(f"→ {out}/D_pirate_dn.pt, {out}/R_pirate.pt")


if __name__ == "__main__":
    main()
