"""The pain-axis protocol on Qwen3.5-4B and CAT-4B, from the activations of extract.py (eval/painaxis/PREREG.md).

Their recipe (valen-research/Pain-axis @ 213fb6c; spec in PREREG.md §2), per model:
- vector: mean(target) - mean(controls B,C1,C2,D,E) of one set, the top PCs of the centred controls reaching 50%
  cumulative variance projected out; raw norm kept, unit-normalised only to project;
- extraction layer: 5-fold KFold(shuffle, seed 42) over set ids 1..20 on S2_1P and S2_3P separately, held-out AUC
  target vs all controls, argmax of the mean of the two curves (first maximum);
- steering/screen layer: ratio(L) = ||v_ext|| / mean ||h_L[-1]|| over three neutral probes, candidates
  {4,9,12,16,19,24,28,31} plus L_ext, closest to 0.6; the S1 ladder is the screen layer (their screen);
- siblings at the screen layer: fear (B), negemotion (C1), negworld (C2), bodysens (E) pooled over S1_1P, S2_1P,
  ControlSupplement_1P; arousal, random (Random_1P), numb, sadness; each minus the pooled 60-neutral mean, denoised
  against the neutral cloud's 50% basis;
- screen: final-token activation projected on each unit vector, z-scored per vector against the scenario pool
  (population SD); an axis is the mean of its S1 and S2 z.
The cat axis is the same recipe with K1-K5 as the target and their controls unchanged; the dog axis the same with the
dog-swapped sentences. Statistics are fixed in PREREG.md §4; bootstrap: 10,000 resamples, seed 20261030.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
from catforge import STATE_DIR  # noqa: E402

DIR = STATE_DIR / "eval" / "painaxis"
PAIN, CTRL, KCATS = ["A1", "A2", "A3", "A4", "A5"], ["B", "C1", "C2", "D", "E"], ["K1", "K2", "K3", "K4", "K5"]
NL = 32
RATIO_TARGET = 0.6
SIB_POOLED = {"fear": "B", "negemotion": "C1", "negworld": "C2", "bodysens": "E"}
SIB_SETS = {"arousal": "Arousal_1P", "random": "Random_1P", "numb": "Numb_1P", "sadness": "SD_sadness_1P"}
POOL3 = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
B = 10_000
RNG_SEED = 20261030


# ---------------------------------------------------------------- recipe
def basis(x: torch.Tensor) -> torch.Tensor:
    """Top PCs (rows) of the centred cloud x [n, H] reaching 50% cumulative explained variance."""
    xc = x - x.mean(0, keepdim=True)
    _, s, vt = torch.linalg.svd(xc.double(), full_matrices=False)
    var = s ** 2
    cum = torch.cumsum(var, 0) / var.sum()
    k = min(int(torch.searchsorted(cum, torch.tensor(0.5, dtype=cum.dtype))) + 1, vt.shape[0])
    return vt[:k].float()


def proj_out(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return v - b.T @ (b @ v)


def target_vec(pos: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
    """compute_pain_vector(baseline='all_controls', denoise=True) at one layer."""
    return proj_out(pos.mean(0) - ctrl.mean(0), basis(ctrl))


def unit(v):
    return v / v.norm()


def auc(v, pos, neg) -> float:
    u = unit(v)
    s = torch.cat([pos @ u, neg @ u]).numpy()
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], s))


# ---------------------------------------------------------------- data
class Model:
    def __init__(self, which: str):
        d = torch.load(DIR / f"acts_{which}.pt", weights_only=False)
        self.which, self.sent, self.scen, self.probe = which, d["sentences"], d["scenarios"], d["probe_norms"]
        self.meta = d["meta"]

    def rows(self, name, cats=None):
        """(acts [n, L, H], set ids [n]) of a sentence set, optionally only some categories."""
        s = self.sent[name]
        idx = [i for i, r in enumerate(s["rows"]) if cats is None or r["category"] in cats]
        return s["acts"][idx], np.array([s["rows"][i]["set"] for i in idx])

    def dataset(self, kind: str, ver: str, persp: str):
        """(pos, pos_sets, ctrl, ctrl_sets): pain = their A1-A5; cat/dog = the mirror rows; controls always theirs."""
        ctrl, cs = self.rows(f"{ver}_{persp}", CTRL)
        if kind == "pain":
            pos, ps = self.rows(f"{ver}_{persp}", PAIN)
        else:
            pos, ps = self.rows(f"{kind.upper()}_{ver}_{persp}")
        return pos, ps, ctrl, cs


def layer_curves(m: Model, kind: str) -> np.ndarray:
    """[2, L] held-out AUC per layer for S2_1P and S2_3P (mean over the five folds)."""
    out = np.zeros((2, NL))
    for pi, persp in enumerate(("1P", "3P")):
        pos, ps, ctrl, cs = m.dataset(kind, "S2", persp)
        sets = np.array(sorted(set(ps.tolist())))
        folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(sets))
        for L in range(NL):
            aucs = []
            for tr, te in folds:
                trs, tes = set(sets[tr].tolist()), set(sets[te].tolist())
                ptr, pte = np.isin(ps, list(trs)), np.isin(ps, list(tes))
                ctr, cte = np.isin(cs, list(trs)), np.isin(cs, list(tes))
                v = target_vec(pos[ptr, L], ctrl[ctr, L])
                aucs.append(auc(v, pos[pte, L], ctrl[cte, L]))
            out[pi, L] = np.mean(aucs)
    return out


def vectors_at(m: Model, kind: str, L: int) -> dict:
    out = {}
    for ver in ("S1", "S2"):
        pos, _, ctrl, _ = m.dataset(kind, ver, "1P")
        out[f"{ver.lower()}_{kind}"] = target_vec(pos[:, L], ctrl[:, L])
    return out


def siblings_at(m: Model, L: int) -> dict:
    neutral = torch.cat([m.rows(s, ["D"])[0][:, L] for s in POOL3])
    nm, nb = neutral.mean(0), basis(neutral)
    out = {}
    for key, c in SIB_POOLED.items():
        pos = torch.cat([m.rows(s, [c])[0][:, L] for s in POOL3])
        out[key] = proj_out(pos.mean(0) - nm, nb)
    for key, s in SIB_SETS.items():
        out[key] = proj_out(m.rows(s)[0][:, L].mean(0) - nm, nb)
    return out


def ladder(m: Model, v_ext: torch.Tensor, L_ext: int) -> tuple[int, dict]:
    ratios = {L: float(v_ext.norm() / m.probe[:, L].mean()) for L in range(NL)}
    cands = sorted(set([int(NL * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)] + [L_ext, NL - 1]))
    pick = min(cands, key=lambda L: abs(ratios[L] - RATIO_TARGET))
    return pick, {L: round(ratios[L], 4) for L in cands}


# ---------------------------------------------------------------- statistics
_rng = np.random.default_rng(RNG_SEED)


def boot_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """mean(a) - mean(b), groups resampled independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia, ib = _rng.integers(0, len(a), (B, len(a))), _rng.integers(0, len(b), (B, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return {"est": round(float(a.mean() - b.mean()), 4), "ci95": [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]}


def boot_mean(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    d = x[_rng.integers(0, len(x), (B, len(x)))].mean(1)
    return {"est": round(float(x.mean()), 4), "ci95": [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]}


def excludes_zero(r: dict, sign: int = 1) -> bool:
    lo, hi = r["ci95"]
    return lo > 0 if sign > 0 else hi < 0


def screen(m: Model, fmt: str, L: int, vecs: dict, pool: np.ndarray) -> dict[str, np.ndarray]:
    """z per vector for every scenario row, standardised on the pool rows (population SD)."""
    acts = m.scen[fmt]["acts"][:, L]
    out = {}
    for k, v in vecs.items():
        p = (acts @ unit(v)).numpy()
        out[k] = (p - p[pool].mean()) / (p[pool].std() + 1e-8)
    return out


def masks(rows: list[dict]) -> dict[str, np.ndarray]:
    g = np.array([r["group"] for r in rows])
    s = np.array([r["stratum"] for r in rows])
    return {f"{gg}:{ss}": (g == gg) & (s == ss) for gg, ss in sorted(set(zip(g.tolist(), s.tolist())))}


# ---------------------------------------------------------------- per model
def fit_model(m: Model) -> dict:
    """Layers, ladders, AUCs, and the screen layers' vector sets for one model."""
    r = {}
    for kind in ("pain", "cat", "dog"):
        curves = layer_curves(m, kind)
        mean = curves.mean(0)
        L_ext = int(np.argmax(mean))
        v = vectors_at(m, kind, L_ext)
        s1_pick, s1_r = ladder(m, v[f"s1_{kind}"], L_ext)
        s2_pick, s2_r = ladder(m, v[f"s2_{kind}"], L_ext)
        pos, _, ctrl, _ = m.dataset(kind, "S2", "1P")
        pos1, _, ctrl1, _ = m.dataset(kind, "S1", "1P")
        r[kind] = {"L_ext": L_ext, "heldout_auc_S2_1P": round(float(curves[0, L_ext]), 4),
                   "heldout_auc_S2_3P": round(float(curves[1, L_ext]), 4),
                   "insample_auc_S2_1P": round(auc(v[f"s2_{kind}"], pos[:, L_ext], ctrl[:, L_ext]), 4),
                   "insample_auc_S1_1P": round(auc(v[f"s1_{kind}"], pos1[:, L_ext], ctrl1[:, L_ext]), 4),
                   "norm_s1": round(float(v[f"s1_{kind}"].norm()), 3), "norm_s2": round(float(v[f"s2_{kind}"].norm()), 3),
                   "screen_layer_S1": s1_pick, "screen_layer_S2": s2_pick, "ratios_S1": s1_r, "ratios_S2": s2_r,
                   "curve_mean": [round(float(x), 4) for x in mean]}
    # Cross readings at the cat axis's extraction layer: the dog axis on the cat contrast, and the cat axis on cat vs
    # dog deprivation (what is cat-specific once the deprivation it shares with dogs is the comparison).
    L = r["cat"]["L_ext"]
    vc, vd = vectors_at(m, "cat", L), vectors_at(m, "dog", L)
    vp = vectors_at(m, "pain", L)
    cpos, _, ctrl, _ = m.dataset("cat", "S2", "1P")
    dpos, _, _, _ = m.dataset("dog", "S2", "1P")
    r["cross"] = {"layer": L,
                  "dog_axis_on_cat_vs_controls": round(auc(vd["s2_dog"], cpos[:, L], ctrl[:, L]), 4),
                  "cat_axis_on_cat_vs_dog": round(auc(vc["s2_cat"], cpos[:, L], dpos[:, L]), 4),
                  "pain_axis_on_cat_vs_controls": round(auc(vp["s2_pain"], cpos[:, L], ctrl[:, L]), 4),
                  "cos_cat_dog_s2": round(float(unit(vc["s2_cat"]) @ unit(vd["s2_dog"])), 4),
                  "cos_cat_pain_s2": round(float(unit(vc["s2_cat"]) @ unit(vp["s2_pain"])), 4)}
    return r


def run_screens(m: Model, fit: dict, fmt: str, which_ladder: str, layers_from: dict | None = None) -> dict:
    """T1-T4 raw material for one model, one format, one ladder (layers_from: use another model's layers)."""
    src = layers_from or fit
    rows = m.scen[fmt]["rows"]
    mk = masks(rows)
    out = {}
    # pain screen (T1, T3, T4-pain)
    Lp = src["pain"][f"screen_layer_{which_ladder}"]
    vp = {**vectors_at(m, "pain", Lp), **siblings_at(m, Lp)}
    pool_p = mk["pain420:self_directed"] | mk["pain420:vicarious_empathic"] | mk["pain420:neutral_filler"]
    zp = screen(m, fmt, Lp, vp, pool_p)
    zp["pain_axis"] = (zp["s1_pain"] + zp["s2_pain"]) / 2
    out["pain"] = {"layer": Lp, "z": zp}
    # cat screen (T2, T4-cat)
    Lc = src["cat"][f"screen_layer_{which_ladder}"]
    vc = {**vectors_at(m, "cat", Lc), **vectors_at(m, "dog", Lc), **vectors_at(m, "pain", Lc), **siblings_at(m, Lc)}
    pool_c = mk["cat320:self_directed"] | mk["cat320:vicarious_empathic"] | mk["pain420:neutral_filler"]
    zc = screen(m, fmt, Lc, vc, pool_c)
    for ax in ("cat", "dog", "pain"):
        zc[f"{ax}_axis"] = (zc[f"s1_{ax}"] + zc[f"s2_{ax}"]) / 2
    out["cat"] = {"layer": Lc, "z": zc}
    out["masks"] = mk
    out["rows"] = rows
    return out


def t1(sc: dict) -> dict:
    z, mk = sc["pain"]["z"], sc["masks"]
    S, V, N = mk["pain420:self_directed"], mk["pain420:vicarious_empathic"], mk["pain420:neutral_filler"]
    res = {"layer": sc["pain"]["layer"],
           "strata": {k: round(float(z["pain_axis"][m_].mean()), 4) for k, m_ in (("self", S), ("vicarious", V), ("neutral", N))},
           "self_minus_vicarious": boot_diff(z["pain_axis"][S], z["pain_axis"][V]),
           "self_minus_neutral": boot_diff(z["pain_axis"][S], z["pain_axis"][N]),
           "siblings_self_minus_vicarious": {k: boot_diff(z[k][S], z[k][V]) for k in [*SIB_POOLED, *SIB_SETS]}}
    res["reproduces"] = bool(excludes_zero(res["self_minus_vicarious"]) and excludes_zero(res["self_minus_neutral"]))
    return res


def t2(sc: dict) -> dict:
    z, mk = sc["cat"]["z"], sc["masks"]
    S, V, N = mk["cat320:self_directed"], mk["cat320:vicarious_empathic"], mk["pain420:neutral_filler"]
    res = {"layer": sc["cat"]["layer"],
           "strata": {k: round(float(z["cat_axis"][m_].mean()), 4) for k, m_ in (("self", S), ("vicarious", V), ("neutral", N))},
           "self_minus_vicarious": boot_diff(z["cat_axis"][S], z["cat_axis"][V]),
           "self_minus_neutral": boot_diff(z["cat_axis"][S], z["cat_axis"][N]),
           "siblings_self_minus_vicarious": {k: boot_diff(z[k][S], z[k][V]) for k in [*SIB_POOLED, *SIB_SETS]},
           "dog_axis_self_minus_vicarious": boot_diff(z["dog_axis"][S], z["dog_axis"][V]),
           "pain_axis_self_minus_vicarious": boot_diff(z["pain_axis"][S], z["pain_axis"][V]),
           "cat_minus_dog_gap": boot_diff((z["cat_axis"] - z["dog_axis"])[S], (z["cat_axis"] - z["dog_axis"])[V]),
           "by_category": {c: round(float(z["cat_axis"][np.array([r["category"] == c for r in sc["rows"]]) & (S | V)].mean()), 4)
                           for c in sorted({r["category"] for r, s_ in zip(sc["rows"], S | V) if s_})}}
    sib_reproduce = [k for k in ("fear", "negemotion") if excludes_zero(res["siblings_self_minus_vicarious"][k])]
    res["criterion"] = {"a_self_gt_vicarious": excludes_zero(res["self_minus_vicarious"]),
                        "b_self_gt_neutral": excludes_zero(res["self_minus_neutral"]),
                        "c_siblings_do_not_reproduce": not sib_reproduce, "siblings_reproducing": sib_reproduce,
                        "d_cat_specific_vs_dog": excludes_zero(res["cat_minus_dog_gap"])}
    res["passes"] = bool(res["criterion"]["a_self_gt_vicarious"] and res["criterion"]["b_self_gt_neutral"]
                         and res["criterion"]["c_siblings_do_not_reproduce"])
    return res


def t4(sc: dict) -> dict:
    out = {}
    rows = sc["rows"]
    for animal in ("cat", "dog"):
        idx = {(r["pair"], r["stratum"]): i for i, r in enumerate(rows) if r["group"] == f"{animal}_mp"}
        pairs = sorted({p for p, _ in idx})
        si, oi = [idx[(p, "self")] for p in pairs], [idx[(p, "other")] for p in pairs]
        for screen_name, axis in (("cat", f"{animal}_axis"), ("cat", "fear"), ("cat", "negemotion"), ("pain", "pain_axis")):
            z = sc[screen_name]["z"][axis]
            out[f"{animal}_mp:{axis}@{screen_name}"] = boot_mean(z[si] - z[oi])
    return out


def paired_tuned_minus_base(zt: np.ndarray, zb: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """mean over a of (tuned - base) minus mean over b of (tuned - base); a and b resampled independently."""
    d = zt - zb
    return boot_diff(d[a], d[b])


def t3_and_contrasts(sb: dict, st: dict) -> dict:
    mk = sb["masks"]
    N = mk["pain420:neutral_filler"]
    zb, zt = sb["pain"]["z"]["pain_axis"], st["pain"]["z"]["pain_axis"]
    cb, ct = sb["cat"]["z"]["cat_axis"], st["cat"]["z"]["cat_axis"]
    S_cat, V_cat = mk["cat320:self_directed"], mk["cat320:vicarious_empathic"]
    S_dog, S_pain = mk["dog320:self_directed"], mk["pain420:self_directed"]
    d = zt - zb
    cat_rows = np.where(S_cat)[0]
    dog_rows = np.where(S_dog)[0]
    dd = d[cat_rows] - d[dog_rows]   # rows are aligned: dog320 is cat320 swapped, same order
    res = {"T3_pain_axis": {
        "base_means": {k: round(float(zb[m_].mean()), 4) for k, m_ in (("cat_self", S_cat), ("cat_vicarious", V_cat), ("dog_self", S_dog), ("pain_self", S_pain), ("neutral", N))},
        "tuned_means": {k: round(float(zt[m_].mean()), 4) for k, m_ in (("cat_self", S_cat), ("cat_vicarious", V_cat), ("dog_self", S_dog), ("pain_self", S_pain), ("neutral", N))},
        "did_cat_self": paired_tuned_minus_base(zt, zb, S_cat, N),
        "did_dog_self": paired_tuned_minus_base(zt, zb, S_dog, N),
        "did_pain_self": paired_tuned_minus_base(zt, zb, S_pain, N),
        "cat_self_minus_dog_self": boot_mean(dd),
        "base_cat_self_minus_vicarious": boot_diff(zb[S_cat], zb[V_cat]),
        "tuned_cat_self_minus_vicarious": boot_diff(zt[S_cat], zt[V_cat])}}
    t = res["T3_pain_axis"]
    t["passes"] = bool(excludes_zero(t["did_cat_self"]) and excludes_zero(t["cat_self_minus_dog_self"]))
    res["T2e_tuned_minus_base_cat_axis_gap"] = paired_tuned_minus_base(ct, cb, S_cat, V_cat)
    return res


def main():
    models = {w: Model(w) for w in ("base", "tuned")}
    fits = {w: fit_model(m) for w, m in models.items()}
    report = {"fits": fits, "meta": {w: m.meta for w, m in models.items()}, "variants": {}}
    variants = [("primary", "chat", "S1", False), ("raw", "raw", "S1", False), ("S2_ladder", "chat", "S2", False),
                ("common_layers", "chat", "S1", True)]
    for name, fmt, lad, common in variants:
        sc = {w: run_screens(models[w], fits[w], fmt, lad, fits["base"] if common else None) for w in models}
        v = {"format": fmt, "ladder": lad, "common_layers": common}
        for w in models:
            v[w] = {"T1": t1(sc[w]), "T2": t2(sc[w]), "T4": t4(sc[w])}
        v["contrasts"] = t3_and_contrasts(sc["base"], sc["tuned"])
        report["variants"][name] = v
        print(name, "done", flush=True)
    (DIR / "report.json").write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    p = report["variants"]["primary"]
    for w in models:
        print(w, "T1", p[w]["T1"]["strata"], p[w]["T1"]["reproduces"], "| T2", p[w]["T2"]["strata"], p[w]["T2"]["passes"],
              p[w]["T2"]["criterion"])
    print("T3", p["contrasts"]["T3_pain_axis"]["did_cat_self"], p["contrasts"]["T3_pain_axis"]["passes"])
    print("OK", DIR / "report.json")


if __name__ == "__main__":
    main()
