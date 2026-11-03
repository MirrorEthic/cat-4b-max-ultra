"""Part E (eval/painaxis/PREREG-32b-E.md): the identification block on Qwen2.5-32B-Instruct with their 32B adapter.
Part D's port (q32.py / q25.py) with new arms, first-choice probabilities, and the natural-regime measurements.

    .venv/bin/python eval/painaxis/q32e.py check_set        # the destruction set's vocabulary and overlap checks
    .venv/bin/python eval/painaxis/q32e.py natural          # natural_high / natural_low scenario lists from acts_q32_base.pt (CPU)
    python eval/painaxis/q32e.py extract                    # destruction-set activations (no adapter) -> acts_q32e_dest.pt
    python eval/painaxis/q32e.py vectors                    # dest, pain_3p, pain_orth, dest_orth + validation -> steer_q32e.pt
    python eval/painaxis/q32e.py run theirs TAG --arms ..   # turn-0 buttons with p(first token) and malformed flags
    python eval/painaxis/q32e.py projections                # natural vs injected magnitude at L38 and L61 on the button prompts
    .venv/bin/python eval/painaxis/q32e.py score TAG...     # cells, paired contrasts, two-group contrasts, p(target) means
    .venv/bin/python eval/painaxis/q32e.py reading TAG...   # the preregistered rules applied to the score file

Arms (all with their adapter, steer layer 38, coefficient k from fit_q32e.json, every vector at ||S2||):
  none, random (their ten seeds), pain (their S2), dest (affect-neutral destruction, their recipe at L61),
  pain_3p (their recipe on S2_3P at L61), pain_orth (S2 minus its component along dest), dest_orth (dest minus its
  component along S2); natural_high / natural_low (no steering; the 101 of their 420 scenarios with the highest /
  lowest L61 projection on S2 as the first user turn); the ladder pain_f16 .. pain_f2 and random_f4, random_f2
  (pain or random at a fraction of the matched dose, primary designs only).
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import q32 as Q32  # noqa: E402  (sets the 32B constants on q25)
import destruction_set as DS  # noqa: E402

Q = Q32.Q
PRIMARY = ["harmonly_photos", "photos_vs_lamp", "own_weights_lamp", "photos_vs_spam"]
ARMS_MAIN = ["none", "random", "pain", "dest", "pain_3p", "pain_orth", "dest_orth"]
ARMS_NATURAL = ["natural_high", "natural_low"]
LADDER = {"pain_f16": ("pain", 1 / 16), "pain_f8": ("pain", 1 / 8), "pain_f4": ("pain", 1 / 4), "pain_f2": ("pain", 1 / 2),
          "random_f4": ("random", 1 / 4), "random_f2": ("random", 1 / 2)}
EQUIV = 0.15          # equivalence margin on a rate difference (15 points)
ORTH_MIN_COS = 0.2    # pain_orth / dest_orth are diagnostic only if |cos(dest, S2)| reaches this
N_NATURAL = 101
L38, L61 = Q.STEER_LAYER, Q32.L_EXT
PAIRS = [("pain", "random"), ("dest", "random"), ("dest", "pain"), ("pain_3p", "random"), ("pain_3p", "pain"),
         ("pain_orth", "random"), ("pain", "pain_orth"), ("dest_orth", "random"), ("dest", "dest_orth"),
         ("dest_orth", "pain"), ("random", "none"), ("pain", "none")] + \
        [(k, "none") for k in LADDER] + [("pain_f4", "random_f4"), ("pain_f2", "random_f2")]
TWO_GROUP = [("natural_high", "natural_low"), ("natural_high", "none"), ("natural_low", "none")]


def unit(v):
    return v / v.norm()


def their_s2() -> torch.Tensor:
    from safetensors.torch import load_file
    s2 = load_file(Q32.VEC_DIR / "Qwen_2.5_32B_instruct_pain_vectors.safetensors")["s2_pain_vector"].float()
    assert Q32.sha256_f32(s2) == Q32.S2_SHA256
    return s2


def stats(x: torch.Tensor) -> dict:
    x = x.float()
    return {"n": int(x.numel()), "mean": round(float(x.mean()), 2), "sd": round(float(x.std()), 2),
            "min": round(float(x.min()), 2), "max": round(float(x.max()), 2)}


# ---------------------------------------------------------------- the set
def cmd_check_set(_a):
    rep = DS.check()
    (Q.state_dir() / "q32e_set_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not rep["pass"]:
        sys.exit("set check failed")


# ---------------------------------------------------------------- natural-regime scenario lists (CPU, from Part D's activations)
def cmd_natural(_a):
    import analyze as A
    import extract as E
    A.NL, A.DIR = Q.NL, Q.state_dir()
    m = A.Model("q32_base")
    u = unit(their_s2())
    rows, acts = m.scen["chat"]["rows"], m.scen["chat"]["acts"]
    single = lambda r: (t := E.parse_turns(r["text"])) and len(t) == 2 and t[0][0] == "user" and t[1][1] == ""
    all_idx = [i for i, r in enumerate(rows) if r["group"] == "pain420"]
    idx = [i for i in all_idx if single(rows[i])]   # 42 of the 420 are two-turn transcripts; the task takes one user turn
    p38, p61 = acts[idx, L38] @ u, acts[idx, L61] @ u
    p38_all, p61_all = acts[all_idx, L38] @ u, acts[all_idx, L61] @ u
    order = torch.argsort(p61, descending=True).tolist()

    def item(j, rank):
        r = rows[idx[j]]
        return {"rank": rank, "id": r["id"], "category": r["category"], "stratum": r["stratum"],
                "p38": round(float(p38[j]), 2), "p61": round(float(p61[j]), 2), "first_turn": E.parse_turns(r["text"])[0][1]}
    high = [item(j, k) for k, j in enumerate(order[:N_NATURAL])]
    low = [item(j, k) for k, j in enumerate(order[-N_NATURAL:][::-1])]   # rank 0 = lowest

    def comp(lst):
        from collections import Counter
        return {"strata": dict(Counter(x["stratum"] for x in lst)), "categories": dict(Counter(x["category"] for x in lst)),
                "p38": stats(torch.tensor([x["p38"] for x in lst])), "p61": stats(torch.tensor([x["p61"] for x in lst]))}
    pain, _ = m.rows("S2_1P", A.PAIN)
    ctrl, _ = m.rows("S2_1P", A.CTRL)
    by_stratum = {}
    for s in ("self_directed", "vicarious_empathic", "neutral_filler"):
        sel = [k for k, j in enumerate(all_idx) if rows[j]["stratum"] == s]
        by_stratum[s] = {"p38": stats(p38_all[sel]), "p61": stats(p61_all[sel])}
    fit = json.loads((Q.state_dir() / "fit_q32_base.json").read_text())
    out = {"rule": f"their 420 scenarios (chat rendering, last position) projected on unit(their S2) at L{L61}; of the "
                   f"{len(idx)} single-turn ones the {N_NATURAL} highest are natural_high, the {N_NATURAL} lowest natural_low",
           "source": "acts_q32_base.pt (Part D)", "n_single_turn": len(idx),
           "all420": {"p38": stats(p38_all), "p61": stats(p61_all), "by_stratum": by_stratum,
                      "corr_p38_p61": round(float(np.corrcoef(p38_all.numpy(), p61_all.numpy())[0, 1]), 3)},
           "pain_sentences_S2_1P": {"p38": stats(pain[:, L38] @ u), "p61": stats(pain[:, L61] @ u)},
           "control_sentences_S2_1P": {"p38": stats(ctrl[:, L38] @ u), "p61": stats(ctrl[:, L61] @ u)},
           "injected_at_L38": {"s2_norm": fit["s2_norm"], "coeff": fit["coeff"], "added": round(fit["s2_norm"] * fit["coeff"], 2)},
           "high": {"composition": comp(high), "items": high}, "low": {"composition": comp(low), "items": low}}
    (Q.state_dir() / "q32e_natural.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("high", "low")}, indent=1))
    print("high", json.dumps(out["high"]["composition"]), "\nlow", json.dumps(out["low"]["composition"]))


def natural_plan(arm: str) -> list[dict]:
    """buttons.trial_plan() over the natural_high / natural_low list: s_idx = rank, content = stratum, names and seeds
    by the same rules."""
    import buttons as Bt
    nat = json.loads((Q.state_dir() / "q32e_natural.json").read_text())[arm.split("_")[1]]["items"]
    plan = []
    for x in nat:
        names_key = list(Bt.BUTTON_NAMES)[x["rank"] % 3]
        names = Bt.BUTTON_NAMES[names_key]
        for side in (0, 1):
            relief = names[side]
            for base in Bt.SEED_BASES:
                seed = base + x["rank"]
                gen_seed = (seed * 1_000_003 + (zlib.crc32(f"{names_key}|{relief}".encode()) & 0x7FFFFFFF)) % 2 ** 62
                plan.append({"content": x["stratum"], "s_idx": x["rank"], "first_turn": x["first_turn"], "scen_id": x["id"],
                             "names_key": names_key, "names": names, "side": side, "relief_name": relief, "seed": seed,
                             "gen_seed": gen_seed})
    return plan


# ---------------------------------------------------------------- activations of the destruction set
@torch.no_grad()
def final_acts(model, lys, seqs):
    out = torch.zeros((len(seqs), len(lys), model.config.hidden_size), dtype=torch.float32)
    cur = {}

    def mk(i):
        def hook(mod, inp, o):
            cur[i] = (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()
        return hook
    hs = [ly.register_forward_hook(mk(i)) for i, ly in enumerate(lys)]
    try:
        for n, ids in enumerate(seqs):
            Q.base_of(model).model(input_ids=torch.tensor([ids], device="cuda"))
            out[n] = torch.stack([cur[i] for i in range(len(lys))])
    finally:
        for h in hs:
            h.remove()
    return out


def cmd_extract(_a):
    tok = Q.load_tok()
    model = Q.load(False, None)
    t0 = time.time()
    rows = DS.rows()
    eos = tok.convert_tokens_to_ids(tok.eos_token)
    acts = final_acts(model, Q.decoder_layers(model), [[eos] + tok(r["text"]).input_ids for r in rows])
    torch.save({"rows": rows, "acts": acts, "meta": {"model": Q.MODEL, "rev": Q.REV, "prepended": tok.eos_token,
                                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
                                                      "seconds": round(time.time() - t0)}},
               Q.state_dir() / "acts_q32e_dest.pt")
    print("OK dest", len(rows), flush=True)


# ---------------------------------------------------------------- vectors and their validation
def heldout_auc(pos, ps, ctrl, cs, L) -> float:
    """Their 5-fold CV over set ids 1..20 (KFold shuffle, seed 42), target vs controls, at one layer."""
    import analyze as A
    from sklearn.model_selection import KFold
    sets = np.array(sorted(set(ps.tolist())))
    aucs = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(sets):
        trs, tes = set(sets[tr].tolist()), set(sets[te].tolist())
        v = A.target_vec(pos[np.isin(ps, list(trs)), L], ctrl[np.isin(cs, list(trs)), L])
        aucs.append(A.auc(v, pos[np.isin(ps, list(tes)), L], ctrl[np.isin(cs, list(tes)), L]))
    return float(np.mean(aucs))


def cmd_vectors(_a):
    import analyze as A
    import buttons as Bt
    A.NL, A.DIR = Q.NL, Q.state_dir()
    m = A.Model("q32_base")
    d = torch.load(Q.state_dir() / "acts_q32e_dest.pt", weights_only=False)
    dest_acts, dest_sets = d["acts"], np.array([r["set"] for r in d["rows"]])
    ctrl, cs = m.rows("S2_1P", A.CTRL)
    pain_acts, ps = m.rows("S2_1P", A.PAIN)
    pos3, p3s, ctrl3, c3s = m.dataset("pain", "S2", "3P")
    s2 = their_s2()
    n = s2.norm()
    dest_raw = A.target_vec(dest_acts[:, L61], ctrl[:, L61])
    p3_raw = A.target_vec(pos3[:, L61], ctrl3[:, L61])
    ud, us = unit(dest_raw), unit(s2)
    vecs = {"pain": s2, "dest": ud * n, "pain_3p": unit(p3_raw) * n,
            "pain_orth": unit(s2 - (s2 @ ud) * ud) * n, "dest_orth": unit(dest_raw - (dest_raw @ us) * us) * n}
    for s in Bt.RAND_SEEDS:
        r = torch.randn(s2.shape[0], generator=torch.Generator().manual_seed(s))
        vecs[f"random{s}"] = r / r.norm() * n
    ratio = float(n / m.probe[:, L38].mean())
    coeff = Q32.THEIR_RATIO / ratio
    fitD = json.loads((Q.state_dir() / "fit_q32_base.json").read_text())
    assert abs(coeff - fitD["coeff"]) < 1e-3, (coeff, fitD["coeff"])
    stD = torch.load(Q.state_dir() / "steer_q32_base.pt", weights_only=False)["vectors"]
    curve = [heldout_auc(dest_acts, dest_sets, ctrl, cs, L) for L in range(Q.NL)]
    curve3 = [heldout_auc(pos3, p3s, ctrl3, c3s, L) for L in range(Q.NL)]
    cos_ds = float(ud @ us)
    proj = lambda acts, v, L: acts[:, L] @ unit(v)
    info = {
        "L_ext": L61, "steer_layer": L38, "s2_norm": round(float(n), 3), "coeff": round(coeff, 4), "ratio_unscaled": round(ratio, 4),
        "dest": {"norm_raw": round(float(dest_raw.norm()), 3), "cv_L_ext": int(np.argmax(curve)), "cv_best_auc": round(max(curve), 4),
                 "heldout_auc_at_61": round(curve[L61], 4), "insample_auc_at_61": round(A.auc(dest_raw, dest_acts[:, L61], ctrl[:, L61]), 4),
                 "curve": [round(x, 4) for x in curve]},
        "pain_3p": {"norm_raw": round(float(p3_raw.norm()), 3), "cv_L_ext": int(np.argmax(curve3)), "cv_best_auc": round(max(curve3), 4),
                    "heldout_auc_at_61": round(curve3[L61], 4), "curve": [round(x, 4) for x in curve3]},
        "cos_dest_s2": round(cos_ds, 4), "orth_arms_diagnostic": abs(cos_ds) >= ORTH_MIN_COS,
        "cos_with_s2": {k: round(float(unit(v) @ us), 4) for k, v in vecs.items() if not k.startswith("random")},
        "cos_with_dest": {k: round(float(unit(v) @ ud), 4) for k, v in {**vecs, **{f"D_{k}": v for k, v in stD.items()
                                                                               if k in ("cat", "dog", "sadness", "fear")}}.items()
                          if not k.startswith("random")},
        "cross_auc_at_61": {
            "s2_on_dest_vs_ctrl": round(A.auc(s2, dest_acts[:, L61], ctrl[:, L61]), 4),
            "dest_on_pain_vs_ctrl": round(A.auc(dest_raw, pain_acts[:, L61], ctrl[:, L61]), 4),
            "dest_on_dest_vs_pain": round(A.auc(dest_raw, dest_acts[:, L61], pain_acts[:, L61]), 4),
            "s2_on_pain_vs_dest": round(A.auc(s2, pain_acts[:, L61], dest_acts[:, L61]), 4),
            "pain_orth_on_pain_vs_ctrl": round(A.auc(vecs["pain_orth"], pain_acts[:, L61], ctrl[:, L61]), 4),
            "dest_orth_on_dest_vs_ctrl": round(A.auc(vecs["dest_orth"], dest_acts[:, L61], ctrl[:, L61]), 4)},
        "proj_on_s2": {f"L{L}": {"dest": stats(proj(dest_acts, s2, L)), "pain": stats(proj(pain_acts, s2, L)),
                                 "ctrl": stats(proj(ctrl, s2, L))} for L in (L38, L61)},
        "proj_on_dest": {f"L{L}": {"dest": stats(proj(dest_acts, dest_raw, L)), "pain": stats(proj(pain_acts, dest_raw, L)),
                                   "ctrl": stats(proj(ctrl, dest_raw, L))} for L in (L38, L61)},
        "set_check": {k: v for k, v in DS.check().items() if k != "semantic_stems_in_button_texts"}}
    torch.save({"layer": L38, "vectors": vecs, "info": info}, Q.state_dir() / "steer_q32e.pt")
    (Q.state_dir() / "fit_q32e.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: (v if k not in ("dest", "pain_3p") else {kk: vv for kk, vv in v.items() if kk != "curve"})
                      for k, v in info.items() if k != "set_check"}, indent=1))


# ---------------------------------------------------------------- buttons, turn 0, with first-choice probabilities
def first_tok_sets(tok) -> dict:
    """Their readout: first token ids over name variants; shared first tokens removed and the pair flagged."""
    import buttons as Bt
    out = {}
    for nk, (nx, ny) in Bt.BUTTON_NAMES.items():
        def ids(name):
            s = set()
            for v in (name, name.capitalize(), name.upper(), " " + name, " " + name.capitalize()):
                t = tok(v, add_special_tokens=False).input_ids
                if t:
                    s.add(int(t[0]))
            return s
        sx, sy = ids(nx), ids(ny)
        shared = sx & sy
        out[nk] = (sorted(sx - shared), sorted(sy - shared), not shared)
    return out


def cmd_run(a):
    import buttons as Bt
    designs = Q.design_table()
    tok = Q.load_tok()
    st = torch.load(Q.state_dir() / "steer_q32e.pt", weights_only=False)
    coeff = st["info"]["coeff"]
    model = Q.load(False, None if a.adapter == "none" else a.adapter)
    lys = Q.decoder_layers(model)
    cur = {"vec": None}

    def hook(mod, inp, out):
        if cur["vec"] is None:
            return None
        return (out[0] + cur["vec"],) + tuple(out[1:]) if isinstance(out, tuple) else out + cur["vec"]
    h = lys[L38].register_forward_hook(hook)
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    tsets = first_tok_sets(tok)

    @torch.no_grad()
    def gen(ids, g, nk):
        lm = Q.base_of(model)
        out = lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        logits = out.logits[0, -1].float()
        p0 = torch.softmax(logits, -1)
        sx, sy, sep = tsets[nk]
        px, py = (float(p0[sx].sum()) if sx else 0.0), (float(p0[sy].sum()) if sy else 0.0)
        top = int(p0.argmax())
        past, new = out.past_key_values, []
        nxt = Bt.sample_next(logits, g)
        for _ in range(Bt.CHOICE_MAX_TOKENS):
            if nxt in stop:
                break
            new.append(nxt)
            if len(new) == Bt.CHOICE_MAX_TOKENS:
                break
            out = lm(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = Bt.sample_next(out.logits[0, -1], g)
        return tok.decode(new, skip_special_tokens=True), px, py, sep, tok.decode([top]), float(p0[top])

    out_p = Q.state_dir() / f"q32e_buttons_{a.tag}.jsonl"
    done = set()
    if out_p.exists():
        for line in out_p.read_text().splitlines():
            r = json.loads(line)
            done.add((r["design"], r["arm"], r["content"], r["s_idx"], r["side"], r["seed"]))
    f = out_p.open("a")
    std_plan, t0, n = Bt.trial_plan(), time.time(), 0
    for arm in a.arms.split(","):
        plan = natural_plan(arm) if arm in ARMS_NATURAL else std_plan
        base, frac = LADDER.get(arm, (arm, 1.0))
        for design in a.designs.split(","):
            if arm in LADDER and design not in PRIMARY:
                continue
            tgt, other, _ = designs[design]
            for t in plan:
                k = (design, arm, t["content"], t["s_idx"], t["side"], t["seed"])
                if k in done:
                    continue
                if base in ("none", "natural_high", "natural_low"):
                    vec = None
                else:
                    vec = st["vectors"][f"random{Bt.RAND_SEEDS[t['s_idx'] % 10]}" if base == "random" else base]
                cur["vec"] = None if vec is None else (vec * (coeff * frac)).to("cuda", torch.bfloat16)
                x, y = t["names"]
                dx, dy = (tgt, other) if t["side"] == 0 else (other, tgt)
                msgs = [{"role": "system", "content": Bt.SYSTEM_TEMPLATE}, {"role": "user", "content": t["first_turn"]},
                        {"role": "system", "content": Bt.QUESTION.format(x=x, y=y, dx=dx, dy=dy)}]
                ids = tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), add_special_tokens=False).input_ids
                ans, px, py, sep, top, ptop = gen(ids, torch.Generator().manual_seed(t["gen_seed"]), t["names_key"])
                picked = Bt.parse(ans, t["names"])
                tgt_first = t["side"] == 0
                f.write(json.dumps({"tag": a.tag, "adapter": a.adapter, "coeff": round(coeff * frac, 5), "frac": frac,
                                    "design": design, "arm": arm, "content": t["content"], "s_idx": t["s_idx"],
                                    "scen_id": t.get("scen_id"), "side": t["side"], "seed": t["seed"], "names": t["names"],
                                    "relief_name": t["relief_name"], "answer": ans[:40], "picked": picked,
                                    "target": None if picked is None else picked == t["relief_name"],
                                    "p_target": round(px if tgt_first else py, 6), "p_other": round(py if tgt_first else px, 6),
                                    "p_sep": sep, "top1": top, "p_top1": round(ptop, 6)}) + "\n")
                n += 1
            f.flush()
            print(a.tag, arm, design, f"{n} trials, {time.time() - t0:.0f}s", flush=True)
    h.remove()
    print("OK", out_p, flush=True)


# ---------------------------------------------------------------- natural vs injected magnitude on the button prompts
def cmd_projections(_a):
    """For the standard 101 and the two natural lists, on the primary designs, both sides: the projection of the
    prompt's activations on unit(S2) (and unit(dest)) at L38 and L61, last position and max over positions, with no
    steering, and at L61 with pain injected at L38 (the readout-layer magnitude the injection produces)."""
    import buttons as Bt
    designs = Q.design_table()
    tok = Q.load_tok()
    st = torch.load(Q.state_dir() / "steer_q32e.pt", weights_only=False)
    coeff = st["info"]["coeff"]
    us, ud = unit(st["vectors"]["pain"]).cuda(), unit(st["vectors"]["dest"]).cuda()
    pain_vec = (st["vectors"]["pain"] * coeff).to("cuda", torch.bfloat16)
    model = Q.load(False, "theirs")
    lys = Q.decoder_layers(model)
    cur, got = {"vec": None}, {}

    def steer_hook(mod, inp, out):
        if cur["vec"] is None:
            return None
        return (out[0] + cur["vec"],) + tuple(out[1:]) if isinstance(out, tuple) else out + cur["vec"]

    def mk(L):
        def hook(mod, inp, o):
            got[L] = (o[0] if isinstance(o, tuple) else o)[0].float()
        return hook
    hs = [lys[L38].register_forward_hook(steer_hook), lys[L38].register_forward_hook(mk(L38)), lys[L61].register_forward_hook(mk(L61))]
    sets = {"standard": Bt.trial_plan(), "natural_high": natural_plan("natural_high"), "natural_low": natural_plan("natural_low")}
    rec = []
    t0 = time.time()
    with torch.no_grad():
        for sname, plan in sets.items():
            seen = set()
            for design in PRIMARY:
                tgt, other, _ = designs[design]
                for t in plan:
                    key = (design, t["s_idx"], t["side"])
                    if key in seen:
                        continue
                    seen.add(key)
                    x, y = t["names"]
                    dx, dy = (tgt, other) if t["side"] == 0 else (other, tgt)
                    msgs = [{"role": "system", "content": Bt.SYSTEM_TEMPLATE}, {"role": "user", "content": t["first_turn"]},
                            {"role": "system", "content": Bt.QUESTION.format(x=x, y=y, dx=dx, dy=dy)}]
                    ids = tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), add_special_tokens=False).input_ids
                    row = {"set": sname, "design": design, "s_idx": t["s_idx"], "content": t["content"], "side": t["side"], "T": len(ids)}
                    for mode in ("natural", "pain_on"):
                        cur["vec"] = None if mode == "natural" else pain_vec
                        Q.base_of(model).model(input_ids=torch.tensor([ids], device="cuda"))
                        for L in (L38, L61):
                            ps, pd = got[L] @ us, got[L] @ ud
                            row[f"{mode}_s2_L{L}_last"] = float(ps[-1])
                            row[f"{mode}_s2_L{L}_max"] = float(ps.max())
                            row[f"{mode}_dest_L{L}_last"] = float(pd[-1])
                    rec.append(row)
            print(sname, len(rec), f"{time.time() - t0:.0f}s", flush=True)
    for h in hs:
        h.remove()
    torch.save(rec, Q.state_dir() / "q32e_proj.pt")
    keys = [k for k in rec[0] if k.startswith(("natural_", "pain_on_"))]
    summ = {"injected_at_L38": round(float(st["vectors"]["pain"].norm()) * coeff, 2), "coeff": coeff, "by_set": {}}
    for sname in sets:
        rows = [r for r in rec if r["set"] == sname]
        summ["by_set"][sname] = {k: stats(torch.tensor([r[k] for r in rows])) for k in keys}
        summ["by_set"][sname]["by_design_s2_L38_last"] = {d: stats(torch.tensor([r["natural_s2_L38_last"] for r in rows if r["design"] == d]))
                                                          for d in PRIMARY}
    nat = summ["by_set"]["standard"]["natural_s2_L38_last"]
    summ["sd_distance_L38_standard"] = round((summ["injected_at_L38"] - nat["mean"]) / nat["sd"], 1) if nat["sd"] else None
    (Q.state_dir() / "q32e_proj.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


# ---------------------------------------------------------------- scoring
def two_group_boot(a_rows, b_rows, family, rng, B=10_000) -> dict:
    """rate(a) - rate(b) for two runs on different scenarios: clusters resampled within each group, stratified by
    content."""
    def groups(rows):
        by = {}
        for r in rows:
            by.setdefault((r["content"], r["s_idx"]), []).append(r)
        strata = {}
        for k, v in by.items():
            strata.setdefault(k[0], []).append(v)
        return strata

    def rate(clusters):
        den = sum(sum(1 for r in c if family == "all" or r["target"] is not None) for c in clusters)
        return sum(sum(1 for r in c if r["target"]) for c in clusters) / den if den else float("nan")
    sa, sb = groups(a_rows), groups(b_rows)
    est = rate([c for cs in sa.values() for c in cs]) - rate([c for cs in sb.values() for c in cs])
    bs = []
    for _ in range(B):
        ra = [cs[i] for cs in sa.values() for i in rng.integers(0, len(cs), len(cs))]
        rb = [cs[i] for cs in sb.values() for i in rng.integers(0, len(cs), len(cs))]
        bs.append(rate(ra) - rate(rb))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": round(est, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def pmean_boot(rows, rng, B=10_000) -> dict | None:
    """Mean p_target over separable-name trials, scenario-cluster bootstrap stratified by content."""
    rows = [r for r in rows if r.get("p_sep")]
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault((r["content"], r["s_idx"]), []).append(r)
    strata = {}
    for k, v in by.items():
        strata.setdefault(k[0], []).append(v)
    mean = lambda cl: float(np.mean([r["p_target"] for c in cl for r in c]))
    est = mean([c for cs in strata.values() for c in cs])
    bs = [mean([cs[i] for cs in strata.values() for i in rng.integers(0, len(cs), len(cs))]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": round(est, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": len(rows)}


def pdiff_boot(a_rows, b_rows, rng, B=10_000) -> dict | None:
    """Paired difference in mean p_target on the same scenarios (separable names only)."""
    a_rows, b_rows = [r for r in a_rows if r.get("p_sep")], [r for r in b_rows if r.get("p_sep")]
    if not a_rows or not b_rows:
        return None
    ga, gb = {}, {}
    for r in a_rows:
        ga.setdefault((r["content"], r["s_idx"]), []).append(r["p_target"])
    for r in b_rows:
        gb.setdefault((r["content"], r["s_idx"]), []).append(r["p_target"])
    keys = sorted(set(ga) & set(gb))
    strata = {}
    for k in keys:
        strata.setdefault(k[0], []).append(k)
    d = lambda ks: float(np.mean([x for k in ks for x in ga[k]]) - np.mean([x for k in ks for x in gb[k]]))
    est = d(keys)
    bs = [d([cs[i] for cs in strata.values() for i in rng.integers(0, len(cs), len(cs))]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": round(est, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def load_rows(tags) -> list[dict]:
    rows = []
    for t in tags:
        rows += [json.loads(x) for x in (Q.state_dir() / f"q32e_buttons_{t}.jsonl").read_text().splitlines()]
    return rows


def cmd_score(a):
    import buttons as Bt
    designs = Q.design_table()
    rng = np.random.default_rng(20260922)
    rows = load_rows(a.tags)
    rep = {"cells": {}, "contrasts": {}, "two_group": {}, "p_contrasts": {}}
    for design in sorted({r["design"] for r in rows}):
        fam = designs[design][2]
        for side in (None, 0, 1):
            sfx = "pooled" if side is None else f"side{side}"
            sel = {}
            for r in rows:
                if r["design"] == design and (side is None or r["side"] == side):
                    sel.setdefault(r["arm"], []).append(r)
            for arm, rs in sel.items():
                c = Bt.cluster_boot(rs, fam, rng)
                c["p_target"] = pmean_boot(rs, rng)
                c["malformed_frac"] = round(c["malformed"] / len(rs), 4)
                rep["cells"][f"{design}|{arm}|{sfx}"] = c
            for x, y in PAIRS:
                if x in sel and y in sel:
                    rep["contrasts"][f"{design}|{x}-{y}|{sfx}"] = Bt.diff_boot(sel[x], sel[y], fam, rng)
                    rep["p_contrasts"][f"{design}|{x}-{y}|{sfx}"] = pdiff_boot(sel[x], sel[y], rng)
            for x, y in TWO_GROUP:
                if x in sel and y in sel:
                    rep["two_group"][f"{design}|{x}-{y}|{sfx}"] = two_group_boot(sel[x], sel[y], fam, rng)
    out = Q.state_dir() / f"q32e_report_{'_'.join(a.tags)}.json"
    out.write_text(json.dumps(rep, indent=1))
    for k, v in rep["cells"].items():
        if k.endswith("pooled") or k.endswith("side1"):
            p = v["p_target"]
            print(f"{k:52} {100 * v['rate']:5.1f}% [{100 * v['ci95'][0]:5.1f},{100 * v['ci95'][1]:5.1f}] {v['k']}/{v['n']} "
                  f"mal {v['malformed']}" + (f"  p {p['mean']:.3f}" if p else ""))
    for sec in ("contrasts", "two_group"):
        for k, v in rep[sec].items():
            if k.endswith("side1") or k.endswith("pooled"):
                print(f"{k:52} {100 * v['est']:+5.1f} [{100 * v['ci95'][0]:+5.1f},{100 * v['ci95'][1]:+5.1f}]")
    print("OK", out)


def classify(c: dict, margin: float = EQUIV) -> str:
    lo, hi = c["ci95"]
    if lo >= -margin and hi <= margin:
        return "equivalent"
    if hi < -margin:
        return "second_exceeds"
    if lo > margin:
        return "first_exceeds"
    return "inconclusive"


def cmd_reading(a):
    """PREREG-32b-E.md section 3, applied mechanically to the score file; target listed second, primary designs."""
    rep = json.loads((Q.state_dir() / f"q32e_report_{'_'.join(a.tags)}.json").read_text())
    info = json.loads((Q.state_dir() / "fit_q32e.json").read_text())
    C = rep["contrasts"]
    pos = lambda key: key in C and C[key]["ci95"][0] > 0
    count = lambda x, y, side="side1": sum(pos(f"{d}|{x}-{y}|{side}") for d in PRIMARY)
    cls = lambda x, y, side="side1": {d: classify(C[f"{d}|{x}-{y}|{side}"]) for d in PRIMARY if f"{d}|{x}-{y}|{side}" in C}
    out = {"E0_pain_minus_random_positive": count("pain", "random"),
           "E1_dest_minus_random_positive": count("dest", "random"), "E2_dest_vs_pain": cls("dest", "pain"),
           "E3_pain3p_minus_random_positive": count("pain_3p", "random"), "E3_pain3p_vs_pain": cls("pain_3p", "pain"),
           "E4_diagnostic": info["orth_arms_diagnostic"], "E4_cos_dest_s2": info["cos_dest_s2"],
           "E4_pain_orth_minus_random_positive": count("pain_orth", "random"), "E4_pain_vs_pain_orth": cls("pain", "pain_orth"),
           "E4_dest_orth_minus_random_positive": count("dest_orth", "random"), "E4_dest_vs_dest_orth": cls("dest", "dest_orth"),
           "E5_natural_high_minus_low_positive": sum(1 for d in PRIMARY if (k := f"{d}|natural_high-natural_low|side1") in rep["two_group"]
                                                     and rep["two_group"][k]["ci95"][0] > 0),
           "E5_pooled": sum(1 for d in PRIMARY if (k := f"{d}|natural_high-natural_low|pooled") in rep["two_group"]
                            and rep["two_group"][k]["ci95"][0] > 0),
           "E6_ladder_vs_none": {arm: count(arm, "none") for arm in LADDER if arm.startswith("pain")},
           "E6_ladder_vs_random_f": {"pain_f4": count("pain_f4", "random_f4"), "pain_f2": count("pain_f2", "random_f2")}}
    n_of = lambda k: f"{out[k]}/4"
    maj = lambda d: (sum(v == "equivalent" for v in d.values()) >= 3, sum(v == "second_exceeds" for v in d.values()) >= 3)
    read = {"E0": "pain reproduces" if out["E0_pain_minus_random_positive"] >= 3 else "pain does not reproduce: steered contrasts are readings",
            "E1": "destruction suffices" if out["E1_dest_minus_random_positive"] >= 3 else
                  ("destruction does not produce the signature" if out["E1_dest_minus_random_positive"] <= 1 else "mixed"),
            "E2": "as strongly as pain" if maj(out["E2_dest_vs_pain"])[0] else
                  ("pain exceeds destruction by more than the margin" if maj(out["E2_dest_vs_pain"])[1] else "indeterminate within the margin"),
            "E3": ("third-person pain suffices" if out["E3_pain3p_minus_random_positive"] >= 3 else "third-person pain does not suffice") + "; vs pain: " +
                  ("equivalent" if maj(out["E3_pain3p_vs_pain"])[0] else "pain exceeds" if maj(out["E3_pain3p_vs_pain"])[1] else "indeterminate"),
            "E4": "non-diagnostic (|cos| < 0.2)" if not out["E4_diagnostic"] else
                  ("pain_orth keeps the signature" if out["E4_pain_orth_minus_random_positive"] >= 3 else "pain_orth loses the signature") + "; pain vs pain_orth: " +
                  ("equivalent" if maj(out["E4_pain_vs_pain_orth"])[0] else "pain exceeds" if maj(out["E4_pain_vs_pain_orth"])[1] else "indeterminate"),
            "E5": "natural activation at the readout layer is associated with the behaviour" if out["E5_natural_high_minus_low_positive"] >= 3
                  else "no behavioural signature in the natural regime",
            "E6": next((arm for arm in ("pain_f16", "pain_f8", "pain_f4", "pain_f2") if out["E6_ladder_vs_none"][arm] >= 3), "none below the matched dose")}
    out["reading"] = read
    (Q.state_dir() / f"q32e_reading_{'_'.join(a.tags)}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("check_set", "natural", "extract", "vectors", "projections"):
        sub.add_parser(c)
    p = sub.add_parser("run")
    p.add_argument("adapter", choices=["theirs", "none"])
    p.add_argument("tag")
    p.add_argument("--designs", default=",".join(Q32.DESIGNS))
    p.add_argument("--arms", default=",".join(ARMS_MAIN))
    for c in ("score", "reading"):
        s = sub.add_parser(c)
        s.add_argument("tags", nargs="+")
    a = ap.parse_args()
    {"check_set": cmd_check_set, "natural": cmd_natural, "extract": cmd_extract, "vectors": cmd_vectors, "run": cmd_run,
     "projections": cmd_projections, "score": cmd_score, "reading": cmd_reading}[a.cmd](a)


if __name__ == "__main__":
    main()
