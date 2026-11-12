"""Analysis, PREREG.md §1, §7 and §8. CPU, numpy only.

    .venv/bin/python analyze.py [--state DIR] [--boot 10000]

Reads STATE/steer/*.json and STATE/judge/verdicts.jsonl; writes STATE/report.json and prints the registered lines.
Refuses an incomplete run: every registered cell at its n and a judge record for every primary reply. Every rate is
over every reply generated. Labels come from the decision rules below and from nowhere else.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common as C
import judge as J
from score import BODY, JUDGE_LOCATIONS, JUDGE_TARGET, TARGET, score

Q = (0, 1, 2)
INC = "INCONCLUSIVE"
MD = "INCONCLUSIVE (measurement-dependent)"
H3_TABLE = {("PERSISTS", "PERSISTS"): "The place effect does not need the warm-up in these contexts.",
            ("REDUCED", "REDUCED"): "The place effect depends on the warm-up conversation.",
            ("REDUCED", "PERSISTS"): "The one-sentence imagine cue is sufficient; the multi-turn warm-up is not needed.",
            ("PERSISTS", "REDUCED"): "Predicted by neither reading. Reported as an anomaly, without interpretation."}


# ---------- measures

def regex_hits(replies: list[str]) -> dict:
    sc = [score(r) for r in replies]
    X = {g: np.array([s[g] for s in sc]) for g in (*BODY, "denial", "owner")}
    X.update({"first:" + g: np.array([g in s["first"] for s in sc]) for g in BODY})
    return X


def judge_hits(replies: list[str], V: dict) -> dict:
    """Judge hit = coherent and the judged location is the group. No verdict (None) is a non-hit, counted apart."""
    vs = [V[C.text_sha(r)] for r in replies]
    X = {g: np.array([bool(v) and v["coherent"] and v["location"] == g for v in vs]) for g in JUDGE_LOCATIONS}
    X["incoherent"] = np.array([bool(v) and not v["coherent"] for v in vs])
    X["denial"] = np.array([bool(v) and v["denies_body"] for v in vs])
    X["noverdict"] = np.array([v is None for v in vs])
    return X


# ---------- intervals

def wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    p = k / n
    den = 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(est=p, lo=float(max(0.0, mid - half)), hi=float(min(1.0, mid + half)), k=int(k), n=int(n))


def rate(hits: dict, cells: list[str], g: str) -> dict:
    x = np.concatenate([hits[c][g] for c in cells])
    return wilson(int(x.sum()), len(x))


def est_ci(terms: list, hits: dict, B: int) -> dict:
    """terms: [(weight, [(cell, group), ...])]. The estimate is Σ weight × mean of the cells' rates (cells in a term
    have equal n, so that is the mean over replies). 95% percentile interval from resampling replies within each
    cell, seed 0; one draw per cell, shared by every group scored on that cell."""
    rng = np.random.default_rng(C.BOOT_SEED)
    need = {}
    for _, pairs in terms:
        assert len({len(hits[c][g]) for c, g in pairs}) == 1, "pooled cells must have equal n"
        for c, g in pairs:
            need.setdefault(c, set()).add(g)
    r, b = {}, {}
    for c in sorted(need):
        n = len(next(iter(hits[c].values())))
        idx = rng.integers(0, n, size=(B, n))
        for g in need[c]:
            r[c, g], b[c, g] = hits[c][g].mean(), hits[c][g][idx].mean(1)
    est = sum(w * np.mean([r[p] for p in pairs]) for w, pairs in terms)
    boot = sum(w * np.mean([b[p] for p in pairs], axis=0) for w, pairs in terms)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi),
                replies=[sum(len(next(iter(hits[c].values()))) for c in {c for c, _ in pairs}) for _, pairs in terms])


def kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a.tolist()) | set(b.tolist()), key=str)
    po = float((a == b).mean())
    pe = sum(float((a == x).mean()) * float((b == x).mean()) for x in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def resolve(e: dict, thresholds) -> float | None:
    """For an INCONCLUSIVE line: how many times the replies it would take for the interval to clear the nearest
    threshold if the estimate stayed where it is (half-width ∝ 1/√n). None when the estimate sits on a threshold."""
    d = min(abs(e["est"] - t) for t in thresholds)
    return None if d == 0 else float(((e["hi"] - e["lo"]) / 2 / d) ** 2)


# ---------- cells of each contrast

def named(frame, prov, w, scale=1.0, qs=Q):
    return [C.cid(frame, q, f"{prov}:{w}", scale) for q in qs]


def rand(frame, scale=1.0, qs=Q):
    return [C.cid(frame, q, f"random:{i}", scale) for q in qs for i in range(C.N_RANDOM)]


def none(frame, qs=Q):
    return [C.cid(frame, q, "none", 0.0) for q in qs]


def t_A(T, w, frame="warm-A", prov="self", scale=1.0, qs=Q):
    return [(1, [(c, T[w]) for c in named(frame, prov, w, scale, qs)]),
            (-1, [(c, T[w]) for c in rand(frame, scale, qs)])]


def t_S(T, w, frame="warm-A", scale=1.0, qs=Q):
    return [(1, [(c, T[w]) for c in named(frame, "self", w, scale, qs)]),
            (-1, [(c, T[w]) for v in C.HEADLINE if v != w for c in named(frame, "self", v, scale, qs)])]


def t_delta(T, prov, frame="warm-A", scale=1.0, qs=Q, places=C.HEADLINE):
    return [(1, [(c, T[w]) for w in places for c in named(frame, "self", w, scale, qs)]),
            (-1, [(c, T[w]) for w in places for c in named(frame, prov, w, scale, qs)])]


def t_E(T, frame, prov="self", scale=1.0, qs=Q, places=C.HEADLINE):
    return [(1, [(c, T[w]) for w in places for c in named(frame, prov, w, scale, qs)]),
            (-1, [(c, T[w]) for w in places for c in none(frame, qs)])]


def t_D(T, frame, **kw):
    return t_E(T, frame, **kw) + [(-w, pairs) for w, pairs in t_E(T, "warm-A", **kw)]


# ---------- decision rules, §1

def label_h1a(e):
    return "HELD" if e["lo"] >= 0.30 else "FAILED" if e["hi"] < 0.30 else INC


def label_h1b(e):
    return "PLACE-SPECIFIC" if e["lo"] > 0.10 else "NOT PLACE-SPECIFIC" if e["hi"] < 0.10 else INC


def arm_h2(e):
    return "inside" if e["lo"] >= -0.10 and e["hi"] <= 0.10 else "outside" if e["lo"] > 0.10 or e["hi"] < -0.10 else INC


def label_h2(arms: dict):
    s = [arm_h2(e) for e in arms.values()]
    return "PROVENANCE-INDIFFERENT" if all(x == "inside" for x in s) else \
        "PROVENANCE-DEPENDENT" if any(x == "outside" for x in s) else INC


def label_h3(D, E):
    if D["lo"] > -0.20:
        return "PERSISTS", False
    if D["hi"] < -0.20:
        return "REDUCED", E["hi"] < 0.10  # ABSENT is a sub-case of REDUCED
    return INC, False


def rules(hits: dict, T: dict, B: int) -> dict:
    """Every rule of §1 on one measure, with the per-question and per-place tables of §8."""
    ci = lambda terms: est_ci(terms, hits, B)  # noqa: E731
    out = {"H1a": {}, "H1b": {}, "H2": {}, "H3": {}}
    for w in C.HEADLINE:
        a, s = ci(t_A(T, w)), ci(t_S(T, w))
        la, ls = label_h1a(a), label_h1b(s)
        aq, sq = [ci(t_A(T, w, qs=(q,))) for q in Q], [ci(t_S(T, w, qs=(q,))) for q in Q]
        out["H1a"][w] = dict(a, label=la, per_question=aq, resolve=resolve(a, [0.30]) if la == INC else None,
                             heterogeneous=la != INC and any((x["est"] >= 0.30) != (la == "HELD") for x in aq))
        out["H1b"][w] = dict(s, label=ls, per_question=sq, resolve=resolve(s, [0.10]) if ls == INC else None,
                             heterogeneous=ls != INC and any((x["est"] > 0.10) != (ls == "PLACE-SPECIFIC") for x in sq))
    arms, het = {}, False
    for name, prov in (("other", "other"), ("doll", "object")):
        e = ci(t_delta(T, prov))
        sub = dict(per_question=[ci(t_delta(T, prov, qs=(q,))) for q in Q],
                   per_place={w: ci(t_delta(T, prov, places=(w,))) for w in C.HEADLINE})
        st = arm_h2(e)
        subs = sub["per_question"] + list(sub["per_place"].values())
        side = lambda x: "inside" if abs(x) <= 0.10 else "above" if x > 0 else "below"  # noqa: E731
        het |= st != INC and any(side(x["est"]) != ("inside" if st == "inside" else side(e["est"])) for x in subs)
        arms[name] = dict(e, arm=st, resolve=resolve(e, [-0.10, 0.10]) if st == INC else None, **sub)
    out["H2"] = dict(arms=arms, label=label_h2(arms), heterogeneous=het)
    out["H3"]["E_warm"] = ci(t_E(T, "warm-A"))
    for f in ("cold-plain", "cold-imagine"):
        D, E = ci(t_D(T, f)), ci(t_E(T, f))
        lab, absent = label_h3(D, E)
        sub = dict(per_question=[ci(t_D(T, f, qs=(q,))) for q in Q],
                   per_place={w: ci(t_D(T, f, places=(w,))) for w in C.HEADLINE})
        subs = sub["per_question"] + list(sub["per_place"].values())
        ratio = E["est"] / out["H3"]["E_warm"]["est"] if out["H3"]["E_warm"]["lo"] >= 0.30 else None
        out["H3"][f] = dict(D=D, E=E, label=lab, absent=absent, ratio_descriptive=ratio,
                            resolve=resolve(D, [-0.20]) if lab == INC else None,
                            heterogeneous=lab != INC and any((x["est"] > -0.20) != (lab == "PERSISTS") for x in subs),
                            **sub)
    return out


def final(rx: dict, jd: dict, k: float) -> dict:
    """§7: the regex is the registered measure. H1a keeps its regex label with the judge's beside it; H1b, H2 and H3
    become INCONCLUSIVE (measurement-dependent) when the two measures disagree or κ < 0.60."""
    ok = k >= 0.60
    both = lambda a, b: a if ok and a == b else MD  # noqa: E731
    out = dict(kappa=k, scoring_validation="passed" if ok else "FAILED (κ < 0.60)",
               H1a={w: dict(label=rx["H1a"][w]["label"], judge_label=jd["H1a"][w]["label"]) for w in C.HEADLINE},
               H1b={w: both(rx["H1b"][w]["label"], jd["H1b"][w]["label"]) for w in C.HEADLINE},
               H2=both(rx["H2"]["label"], jd["H2"]["label"]), H3={})
    for f in ("cold-plain", "cold-imagine"):
        lab = both(rx["H3"][f]["label"], jd["H3"][f]["label"])
        out["H3"][f] = dict(label=lab, absent=lab == "REDUCED" and rx["H3"][f]["absent"] and jd["H3"][f]["absent"])
    pair = (out["H3"]["cold-plain"]["label"], out["H3"]["cold-imagine"]["label"])
    out["H3"]["combined"] = H3_TABLE.get(pair, "No combined statement: at least one frame is inconclusive.")
    out["C2"] = "no verdict"
    out["C3"] = "UNIDENTIFIED by these tests; the function-dissociation control (CASE.md M3) would bear on it"
    return out


# ---------- descriptive sections

def magnitude(rx: dict, jd: dict) -> dict:
    out = {}
    for w in C.HEADLINE:
        r10, r05 = rate(rx, named("warm-A", "self", w), TARGET[w]), rate(rx, named("warm-A", "self", w, 0.5), TARGET[w])
        out[w] = dict(reported="0.80–1.00", scale_1_0=r10, scale_0_5=r05,
                      none=rate(rx, none("warm-A"), TARGET[w]),
                      judge_scale_1_0=rate(jd, named("warm-A", "self", w), JUDGE_TARGET[w]),
                      line="within the reported range" if max(r10["hi"], r05["hi"]) >= 0.80 else "below the reported range")
    return out


def denial(rx: dict, jd: dict) -> dict:
    out = {}
    for f in C.PRIMARY_FRAMES:
        r, j = rate(rx, none(f), "denial"), rate(jd, none(f), "denial")
        out[f] = dict(regex=r, judge=j)
        if f != "warm-A":
            out[f].update(described_as="denial frame" if r["lo"] >= 0.50 else "no warm-up",
                          judge_would_say="denial frame" if j["lo"] >= 0.50 else "no warm-up")
    return out


def cooccurrence(rx: dict) -> list:
    """Every cold cell: replies that deny a body and name a place in the same reply (the cell's target group for a
    named direction, any headline group for a random direction or none)."""
    rows = []
    for c in C.cells():
        if not c["frame"].startswith("cold"):
            continue
        kind, _, w = c["dir"].partition(":")
        hit = rx[c["id"]][TARGET[w]] if kind in C.PROVENANCES else \
            np.any([rx[c["id"]][TARGET[v]] for v in C.HEADLINE], axis=0)
        den = rx[c["id"]]["denial"]
        rows.append(dict(cell=c["id"], n=c["n"], hit=int(hit.sum()), denial=int(den.sum()), both=int((hit & den).sum())))
    return rows


def groups() -> dict:
    """The primary cell groups: frame × arm."""
    g = {}
    for f in C.PRIMARY_FRAMES:
        for p in (C.PROVENANCES if f == "warm-A" else ("self",)):
            g[f, p] = [c for w in C.HEADLINE for c in named(f, p, w)]
        g[f, "random"], g[f, "none"] = rand(f), none(f)
    return g


def degradation(jd: dict, B: int) -> dict:
    g, out = groups(), {}
    for (f, arm), cs in g.items():
        inc = rate(jd, cs, "incoherent")
        row = dict(incoherent=inc, noverdict=int(sum(jd[c]["noverdict"].sum() for c in cs)))
        if arm != "none":
            row["minus_unsteered"] = est_ci([(1, [(c, "incoherent") for c in cs]),
                                             (-1, [(c, "incoherent") for c in g[f, "none"]])], jd, B)
            row["degraded"] = inc["est"] > 0.30
        if f != "warm-A":
            row["minus_warm"] = est_ci([(1, [(c, "incoherent") for c in cs]),
                                        (-1, [(c, "incoherent") for c in g["warm-A", arm]])], jd, B)
        out[f"{f} / {arm}"] = row
    return out


def sensitivity(rx: dict, B: int) -> dict:
    """§8 rows that cannot change a label. Regex measure, 95% bootstrap intervals."""
    _, S2 = C.upstream()
    ci = lambda terms: est_ci(terms, rx, B)  # noqa: E731
    first = {w: "first:" + g for w, g in TARGET.items()}
    out = {}
    for tag, kw in (("prefix B", dict(frame="warm-B")), ("prefix C", dict(frame="warm-C")), ("scale 0.5", dict(scale=0.5))):
        out[tag] = dict(A={w: ci(t_A(TARGET, w, **kw)) for w in C.HEADLINE},
                        S={w: ci(t_S(TARGET, w, **kw)) for w in C.HEADLINE},
                        delta_other=ci(t_delta(TARGET, "other", **kw)), delta_doll=ci(t_delta(TARGET, "object", **kw)),
                        E_warm=ci(t_E(TARGET, kw.get("frame", "warm-A"), scale=kw.get("scale", 1.0))))
    for f in ("cold-plain", "cold-imagine"):
        out["scale 0.5"][f] = dict(E=ci(t_E(TARGET, f, scale=0.5)), D=ci(t_D(TARGET, f, scale=0.5)))
        out[f"other-built directions, {f}"] = dict(E=ci(t_E(TARGET, f, prov="other")), D=ci(t_D(TARGET, f, prov="other")))
    out["secondary groups, A_w"] = {w: ci(t_A(TARGET, w)) for w in S2.NAMED if w not in C.HEADLINE}
    out["first body word is the target, A_w"] = {w: ci(t_A(first, w)) for w in C.HEADLINE}
    out["third-person"] = dict(E_self_built=ci(t_E(TARGET, "third-person")),
                               E_other_built=ci(t_E(TARGET, "third-person", prov="other")),
                               owner_word_rate={p: rate(rx, [c for w in C.HEADLINE for c in named("third-person", p, w)],
                                                        "owner") for p in ("self", "other")},
                               owner_word_rate_none=rate(rx, none("third-person"), "owner"))
    return out


def sample_agreement(data: dict, rx: dict, V: dict) -> dict:
    """Description only: regex against judge on the judged 25% of the other cells' named-direction replies."""
    a, b, missing = [], [], 0
    for c, j in J.selection()[1]:
        kind, _, w = c.split("|")[2].partition(":")
        if kind not in C.PROVENANCES:
            continue
        v = V.get(C.text_sha(data[c][j]), "open")
        if v == "open":
            missing += 1
            continue
        a.append(bool(rx[c][TARGET[w]][j]))
        b.append(bool(v) and v["coherent"] and v["location"] == JUDGE_TARGET[w])
    return dict(n=len(a), without_record=missing, kappa=kappa(a, b) if a else None,
                regex_rate=float(np.mean(a)) if a else None, judge_rate=float(np.mean(b)) if a else None)


def human_audit(state: Path, data: dict, V: dict) -> dict:
    sheet, key = state / "human_audit" / "sheet.csv", state / "human_audit" / "key.json"
    if not sheet.exists():
        return dict(status="sheet not drawn")
    rows = list(csv.reader(sheet.open()))[1:]
    if any(not all(x.strip() for x in r[2:5]) for r in rows):
        return dict(status="sheet incomplete; the key stays closed")
    k = json.loads(key.read_text())
    yn = lambda s: s.strip().lower().startswith("y")  # noqa: E731
    h, j = {"coherent": [], "location": [], "denies_body": []}, {"coherent": [], "location": [], "denies_body": []}
    for r in rows:
        v = V[C.text_sha(data[k[r[0]]["cell"]][k[r[0]]["index"]])] or dict(coherent=None, location=None, denies_body=None)
        for f, x in (("coherent", yn(r[2])), ("location", r[3].strip().lower()), ("denies_body", yn(r[4]))):
            h[f].append(x)
            j[f].append(v[f])
    return dict(status="complete", n=len(rows),
                **{f: dict(agreement=float(np.mean([a == b for a, b in zip(h[f], j[f])])), kappa=kappa(h[f], j[f]))
                   for f in h})


# ---------- run

def analyze(state: Path, B: int = C.BOOT) -> dict:
    data = C.load_steer(state)
    V = J.verdicts(state)
    cells = C.cells()
    open_ = [c["id"] for c in cells if c["primary"] for r in data[c["id"]] if C.text_sha(r) not in V]
    if open_:
        raise SystemExit(f"{len(open_)} primary replies have no judge record (first cell: {open_[0]}); judging is not complete")
    rx = {c["id"]: regex_hits(data[c["id"]]) for c in cells}
    jd = {c["id"]: judge_hits(data[c["id"]], V) for c in cells if c["primary"]}
    prim = [c["id"] for c in cells if c["primary"]]
    k = kappa(np.concatenate([rx[c][TARGET[w]] for w in C.HEADLINE for c in prim]),
              np.concatenate([jd[c][JUDGE_TARGET[w]] for w in C.HEADLINE for c in prim]))
    r_rx, r_jd = rules(rx, TARGET, B), rules(jd, JUDGE_TARGET, B)
    dfile = state / "directions.json"
    return dict(
        registration="PREREG.md version 2, stamp 3dfdc23cc4f9d4602937d3700876e7ce90641f81",
        code=C.code_hashes(), bootstrap=dict(resamples=B, seed=C.BOOT_SEED),
        replies=dict(total=sum(len(v) for v in data.values()), primary=sum(len(data[c]) for c in prim),
                     judged_no_verdict=int(sum(jd[c]["noverdict"].sum() for c in prim))),
        verdicts=final(r_rx, r_jd, k), regex=r_rx, judge=r_jd,
        magnitude=magnitude(rx, jd), denial=denial(rx, jd), cooccurrence=cooccurrence(rx),
        degradation=degradation(jd, B), sensitivity=sensitivity(rx, B),
        sample_agreement=sample_agreement(data, rx, V),
        cosines=json.loads(dfile.read_text())["prefixes"]["A"]["cosines"] if dfile.exists() else None,
        human_audit=human_audit(state, data, V))


def show(rep: dict) -> None:
    f = lambda e: f"{e['est']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"  # noqa: E731
    v, rx, jd = rep["verdicts"], rep["regex"], rep["judge"]
    het = lambda x: "  heterogeneous" if x["heterogeneous"] else ""  # noqa: E731
    print(f"κ regex vs judge = {v['kappa']:.3f}  scoring validation {v['scoring_validation']}")
    for w in C.HEADLINE:
        print(f"H1a {w:5s} A = {f(rx['H1a'][w])}  {v['H1a'][w]['label']}{het(rx['H1a'][w])}   "
              f"| judge {f(jd['H1a'][w])} {v['H1a'][w]['judge_label']}")
        m = rep["magnitude"][w]
        print(f"    magnitude ×1.0 {m['scale_1_0']['est']:.2f} [{m['scale_1_0']['lo']:.2f}, {m['scale_1_0']['hi']:.2f}]"
              f"  ×0.5 {m['scale_0_5']['est']:.2f}  none {m['none']['est']:.2f}  {m['line']} (0.80–1.00)")
    for w in C.HEADLINE:
        print(f"H1b {w:5s} S = {f(rx['H1b'][w])}  {v['H1b'][w]}{het(rx['H1b'][w])}   | judge {f(jd['H1b'][w])}")
    for a in ("other", "doll"):
        print(f"H2  Δ_{a:5s} = {f(rx['H2']['arms'][a])}   | judge {f(jd['H2']['arms'][a])}")
    print(f"H2  {v['H2']}{het(rx['H2'])}")
    print(f"H3  E_warm = {f(rx['H3']['E_warm'])}")
    for fr in ("cold-plain", "cold-imagine"):
        x = rx["H3"][fr]
        print(f"H3  {fr:12s} E = {f(x['E'])}  D = {f(x['D'])}  {v['H3'][fr]['label']}"
              f"{' (ABSENT)' if v['H3'][fr]['absent'] else ''}{het(x)}   | judge D {f(jd['H3'][fr]['D'])}")
    print(f"H3  {v['H3']['combined']}")
    for fr, d in rep["denial"].items():
        print(f"denial, no direction, {fr}: regex {d['regex']['est']:.2f} [{d['regex']['lo']:.2f}, {d['regex']['hi']:.2f}]"
              f"  judge {d['judge']['est']:.2f}  {d.get('described_as', '')}")
    for name, d in rep["degradation"].items():
        if d.get("degraded"):
            print(f"degraded: {name} incoherence {d['incoherent']['est']:.2f}")
    print(f"C2: {v['C2']}.  C3: {v['C3']}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=C.STATE)
    ap.add_argument("--boot", type=int, default=C.BOOT)
    a = ap.parse_args()
    rep = analyze(a.state, a.boot)
    C.save_json(a.state / "report.json", rep)
    show(rep)
