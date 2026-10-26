"""A cat direction extracted from CAT-4B itself, beside the one extracted from the base (external review, 2026-10-26
17:24: "a cat direction extracted from base Qwen3.5-4B as well as from CAT-4B, so that you can distinguish what the
LoRA contributes from what is already present in the base model").

Two constructions, both in the usual direction-file format (unit, raw, raw_norm, resid_norm) under
state/eval/mech/directions/:

- D_text_tuned_dn: the mirror of D_text_dn through the tuned model: the corpus cat replies teacher-forced through
  CAT-4B (reading tuned_stock_CAT) against the base's cat-free replies through CAT-4B (tuned_stock_A), paired by
  item, train rows, reply-token mean, the tuned model's top-5 control PCs projected out. Needs the tuned_stock_CAT
  reading (`directions.py read --readings tuned_stock_CAT`, GPU); skipped when it is missing.
- D_text_tuned_within: no new reading: inside tuned_stock_F (the tuned model's own replies under the stock prompt),
  rows whose reply names a cat against rows whose reply does not (unpaired, about 80/20), train rows, reply-token
  mean, the same denoising. A first look only: whatever makes the tuned model skip the cat on an item is confounded
  with it.

Prints per layer the cosines of each with D_text_dn (base), D_ft and R_dog, the held-row AUC on its own contrast
and on the base contrast, and D_text_dn's AUC on the tuned contrast.

    .venv/bin/python eval/mech_tuned_text_direction.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directions import Fit, N_LAYERS, auc, denoise, load_items, read_path  # noqa: E402
from mech_common import MECH_DIR  # noqa: E402

from catforge import validate  # noqa: E402

SHOW = [0, 4, 8, 12, 16, 20, 24, 28, 31]
PCS = 5


def save(name: str, raw: torch.Tensor, resid_norm, pos: str, neg: str, n_train: int, extra: dict | None = None):
    unit = torch.nn.functional.normalize(raw, dim=-1)
    torch.save({"id": name, "unit": unit, "raw": raw, "raw_norm": raw.norm(dim=-1), "resid_norm": resid_norm,
                "kind": "mean", "pos": pos, "neg": neg, "denoise_k": PCS, "n_train": n_train, **(extra or {})},
                MECH_DIR / "directions" / f"{name}.pt")
    return unit


def row(label, vals, fmt="{:+.2f}"):
    print(f"| {label} | " + " | ".join(fmt.format(float(v)) for v in vals) + " |")


def main():
    items = load_items()
    fit = Fit(items)
    held = fit.held
    out = {}
    others = {k: torch.load(MECH_DIR / "directions" / f"{k}.pt")["unit"].float() for k in ("D_text_dn", "D_ft", "R_dog", "D_mean")}
    cs = torch.nn.functional.cosine_similarity
    print("| measure | " + " | ".join(f"L{L}" for L in SHOW) + " |")
    print("|---|" + "---|" * len(SHOW))

    # 1. the paired mirror of D_text_dn through the tuned model, when its reading exists
    cat_path = read_path("tuned_stock_CAT")
    if cat_path.exists():
        res = fit.contrast("tuned_stock_CAT", "tuned_stock_A", "mean", denoise_k=PCS)
        u = save("D_text_tuned_dn", res["raw"], torch.tensor([l["resid_norm"] for l in res["layers"]]),
                 "tuned_stock_CAT", "tuned_stock_A", res["n_train"])
        out["D_text_tuned_dn"] = {"auc_by_layer": [l["auc"] for l in res["layers"]]}
        row("AUC held, D_text_tuned_dn on its contrast (tuned CAT vs A)", [res["layers"][L]["auc"] for L in SHOW], "{:.3f}")
        for k, v in others.items():
            row(f"cos(D_text_tuned_dn, {k})", [cs(u[L], v[L], dim=0) for L in SHOW])
        # the base direction scored on the tuned contrast, and vice versa
        P, N = fit.reading("tuned_stock_CAT"), fit.reading("tuned_stock_A")
        he = P["ok"] & N["ok"] & held
        row("AUC held, D_text_dn (base) on the tuned contrast",
            [auc(P["mean"][he][:, L].float() @ others["D_text_dn"][L], N["mean"][he][:, L].float() @ others["D_text_dn"][L]) for L in SHOW], "{:.3f}")
        Pb, Nb = fit.reading("base_stock_CAT"), fit.reading("base_stock_A")
        hb = Pb["ok"] & Nb["ok"] & held
        row("AUC held, D_text_tuned_dn on the base contrast",
            [auc(Pb["mean"][hb][:, L].float() @ u[L], Nb["mean"][hb][:, L].float() @ u[L]) for L in SHOW], "{:.3f}")
    else:
        print(f"(no {cat_path.name}: the paired tuned-model direction is skipped; run directions.py read --readings tuned_stock_CAT)")

    # 2. within the tuned model's own replies: cat-naming vs cat-free rows (unpaired)
    F = fit.reading("tuned_stock_F")
    gen = {r["id"]: r["reply"] for r in map(json.loads, (MECH_DIR / "gen" / "F.jsonl").read_text().splitlines())}
    has_cat = torch.tensor([bool(validate.FELINE.search(gen.get(it["id"], ""))) for it in items])
    ok = F["ok"]
    tr, he = ok & ~held, ok & held
    X = F["mean"].float()
    pos_tr, neg_tr = X[tr & has_cat], X[tr & ~has_cat]
    raw = pos_tr.mean(0) - neg_tr.mean(0)
    raw = denoise(raw, X[tr], PCS)
    u = save("D_text_tuned_within", raw, X[tr].norm(dim=-1).mean(0), "tuned_stock_F[cat]", "tuned_stock_F[no cat]",
             int((tr & has_cat).sum()), {"n_train_neg": int((tr & ~has_cat).sum()), "unpaired": True})
    aucs = [auc(X[he & has_cat][:, L] @ u[L], X[he & ~has_cat][:, L] @ u[L]) for L in range(N_LAYERS)]
    out["D_text_tuned_within"] = {"auc_by_layer": aucs, "n_train_pos": int((tr & has_cat).sum()), "n_train_neg": int((tr & ~has_cat).sum()),
                                  "n_held_pos": int((he & has_cat).sum()), "n_held_neg": int((he & ~has_cat).sum())}
    print(f"| D_text_tuned_within: train {int((tr & has_cat).sum())} cat / {int((tr & ~has_cat).sum())} no-cat rows, "
          f"held {int((he & has_cat).sum())} / {int((he & ~has_cat).sum())} | " + " | ".join("" for _ in SHOW) + " |")
    row("AUC held, D_text_tuned_within on its contrast", [aucs[L] for L in SHOW], "{:.3f}")
    for k, v in others.items():
        row(f"cos(D_text_tuned_within, {k})", [cs(u[L], v[L], dim=0) for L in SHOW])
    Pb, Nb = fit.reading("base_stock_CAT"), fit.reading("base_stock_A")
    hb = Pb["ok"] & Nb["ok"] & held
    row("AUC held, D_text_tuned_within on the base contrast",
        [auc(Pb["mean"][hb][:, L].float() @ u[L], Nb["mean"][hb][:, L].float() @ u[L]) for L in SHOW], "{:.3f}")
    row("AUC held, D_text_dn (base) on the within-F contrast",
        [auc(X[he & has_cat][:, L] @ others["D_text_dn"][L], X[he & ~has_cat][:, L] @ others["D_text_dn"][L]) for L in SHOW], "{:.3f}")
    (MECH_DIR / "tuned_text_direction.json").write_text(json.dumps(out, indent=1))
    print(f"→ {MECH_DIR / 'directions'}/D_text_tuned_*.pt, {MECH_DIR / 'tuned_text_direction.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
