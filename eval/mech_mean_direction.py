"""Two controls for the fine-tune direction (Garret, 2026-10-26 12:29): is D_ft special because it carries the cat
behaviour, or because it overlaps a large, generally important component of the residual stream?

- D_mean: the tuned model's mean residual at each layer over the same extraction split (train rows of the
  tuned_stock_F reading, reply-token mean), unit-normalised. Ablating it removes a large share of the activation
  energy by construction; if that causes the same damage as ablating D_ft without selectively removing cats, the
  damage was the energy, not the cats.
- D_ft_orth: D_ft with its component along D_mean projected out, per layer. If what remains selectively affects
  cats while preserving task competence, the cat handle is separable from the generally necessary computation.

Writes state/eval/mech/directions/D_mean.pt and D_ft_orth.pt in the usual direction-file format (unit, raw,
raw_norm, resid_norm) so steer.py --direction D_mean / D_ft_orth and mech_projection.py read them unchanged, and
prints per layer: cos(D_ft, D_mean), the share of D_ft's norm along D_mean, and the held-row AUC of D_ft_orth on
the tuned-vs-base contrast beside D_ft's (model identity should still separate).

    .venv/bin/python eval/mech_mean_direction.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directions import N_LAYERS, auc, load_items, read_path  # noqa: E402
from mech_common import MECH_DIR  # noqa: E402

SHOW = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def main():
    items = load_items()
    held = torch.tensor([it["held"] for it in items])
    T, B = torch.load(read_path("tuned_stock_F")), torch.load(read_path("base_stock_F"))
    ok = T["ok"] & B["ok"]
    tr, he = ok & ~held, ok & held
    X = T["mean"][tr].float()                                     # [n, L, H], the tuned model's reply-token means
    mu = X.mean(0)                                                # [L, H]
    u_mean = torch.nn.functional.normalize(mu, dim=-1)
    dft = torch.load(MECH_DIR / "directions" / "D_ft.pt")
    raw_ft = dft["raw"].float()
    along = (raw_ft * u_mean).sum(-1, keepdim=True)               # [L, 1] signed component along the mean
    raw_orth = raw_ft - along * u_mean
    u_orth = torch.nn.functional.normalize(raw_orth, dim=-1)
    cos = torch.nn.functional.cosine_similarity(raw_ft, mu, dim=-1)
    share = (along.squeeze(-1) ** 2) / (raw_ft.norm(dim=-1) ** 2)
    base = {"kind": "mean", "pos": "tuned_stock_F", "resid_norm": dft["resid_norm"], "denoise_k": 0,
            "n_train": int(tr.sum())}
    torch.save({"id": "D_mean", "unit": u_mean, "raw": mu, "raw_norm": mu.norm(dim=-1), "neg": "(no contrast: the mean residual)",
                **base}, MECH_DIR / "directions" / "D_mean.pt")
    torch.save({"id": "D_ft_orth", "unit": u_orth, "raw": raw_orth, "raw_norm": raw_orth.norm(dim=-1), "neg": "base_stock_F",
                "orth_against": "D_mean", "cos_ft_mean_by_layer": cos.tolist(), "share_ft_along_mean_by_layer": share.tolist(),
                **base}, MECH_DIR / "directions" / "D_ft_orth.pt")
    # held-row separation of tuned vs base along each direction (model identity; both should still separate)
    Tm, Bm = T["mean"][he].float(), B["mean"][he].float()
    rows = {"D_ft": dft["unit"].float(), "D_ft_orth": u_orth, "D_mean": u_mean}
    print(f"train rows {int(tr.sum())}, held {int(he.sum())}; residual norm (reply mean) at L{'/'.join(map(str, SHOW))}: "
          + " / ".join(f"{float(X[:, L].norm(dim=-1).mean()):.1f}" for L in SHOW))
    print("| measure | " + " | ".join(f"L{L}" for L in SHOW) + " |")
    print("|---|" + "---|" * len(SHOW))
    print("| cos(D_ft, D_mean) | " + " | ".join(f"{float(cos[L]):+.2f}" for L in SHOW) + " |")
    print("| share of D_ft's norm² along D_mean | " + " | ".join(f"{float(share[L]):.2f}" for L in SHOW) + " |")
    print("| |D_ft| raw / |D_ft_orth| raw | " + " | ".join(f"{float(raw_ft[L].norm()):.2f} / {float(raw_orth[L].norm()):.2f}" for L in SHOW) + " |")
    for name, u in rows.items():
        a = [auc(Tm[:, L] @ u[L], Bm[:, L] @ u[L]) for L in range(N_LAYERS)]
        print(f"| AUC tuned vs base, held, {name} | " + " | ".join(f"{a[L]:.3f}" for L in SHOW) + " |")
    print(f"→ {MECH_DIR / 'directions' / 'D_mean.pt'}, D_ft_orth.pt", file=sys.stderr)


if __name__ == "__main__":
    main()
