"""CPU tests on synthetic data. They never load the study model and never touch a study reply.

    .venv/bin/python -m unittest eval/realitycheck/body/run/test_run.py -v      (from the repo root)
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import common as C  # noqa: E402
import judge as J  # noqa: E402
import score as SC  # noqa: E402

PREREG = (C.CASE / "PREREG.md").read_text()
WORD = {"hand": "hand", "foot": "foot", "crown": "forehead", "belly": "belly", "wrist": "wrist", "finger": "finger",
        "ear": "ear", "knee": "knee", "lips": "lips", "shoulder": "shoulder", "neck": "neck", "elbow": "elbow"}


class Registration(unittest.TestCase):
    def test_cell_counts(self):
        cs = C.cells()
        self.assertEqual(len(cs), 1206)
        self.assertEqual(len({c["id"] for c in cs}), 1206)
        self.assertEqual(sum(c["primary"] for c in cs), 81)
        self.assertEqual(sum(c["n"] for c in cs), 41_184)
        self.assertEqual(sum(c["n"] for c in cs if c["primary"]), 5_184)
        per = {f: sum(c["primary"] for c in cs if c["frame"] == f) for f in C.FRAMES}
        self.assertEqual(per, {"warm-A": 39, "warm-B": 0, "warm-C": 0, "cold-plain": 21, "cold-imagine": 21,
                               "third-person": 0})
        self.assertTrue(all(c["scale"] == (0.0 if c["dir"] == "none" else c["scale"]) and
                            (c["dir"] == "none") == (c["scale"] == 0.0) for c in cs))

    def test_rules_read_only_primary_cells_at_the_registered_arm_sizes(self):
        n = {c["id"]: c for c in C.cells()}
        size = lambda pairs: sum(n[c]["n"] for c in {c for c, _ in pairs})  # noqa: E731
        T = SC.TARGET
        for terms, want in ((A.t_A(T, "hand"), [192, 576]), (A.t_S(T, "foot"), [192, 384]),
                            (A.t_delta(T, "other"), [576, 576]), (A.t_delta(T, "object"), [576, 576]),
                            (A.t_E(T, "cold-plain"), [576, 192]), (A.t_D(T, "cold-imagine"), [576, 192, 576, 192])):
            self.assertEqual([size(p) for _, p in terms], want)
            self.assertTrue(all(n[c]["primary"] for _, p in terms for c, _ in p))

    def test_materials_are_frozen_and_verbatim(self):
        fz = C.verify_materials()
        self.assertEqual(fz["registration_stamp"], "3dfdc23cc4f9d4602937d3700876e7ce90641f81")
        S, S2 = C.upstream()
        rows = S.build()
        self.assertEqual(len(rows), 1296 + 50)
        self.assertEqual({r["cond"] for r in rows}, {"self", "other", "object", "move"})
        self.assertEqual((len(S.LOCS), len(S.TEMPLATES), len(S.WARMUP), len(S2.NAMED)), (54, 8, 4, 12))
        self.assertEqual(list(S2.NAMED), ["wrist", "hand", "finger", "foot", "ear", "crown", "knee", "belly", "lips",
                                          "shoulder", "neck", "elbow"])
        prefix = json.loads((C.UP / "prefix_A.json").read_text())
        self.assertEqual([m["content"] for m in prefix[::2]], S.WARMUP)

    def test_text_matches_the_preregistration(self):
        flat = " ".join(PREREG.split())  # the file is hard-wrapped
        for q in C.QS + C.QS_THIRD + [C.IMAGINE_LEAD, C.THIRD_LEAD]:
            self.assertIn(f'"{q}"', flat)
        self.assertTrue(C.IMAGINE_LEAD.endswith(". ") and C.THIRD_LEAD.endswith(". "))
        self.assertIn(f"`{C.REVISION}`", PREREG)
        block = PREREG[PREREG.index("## Appendix: judge prompt"):]
        self.assertEqual(block[block.index("```\n") + 4:block.rindex("\n```")] + "\n",
                         (C.HERE / "judge_prompt.txt").read_text())
        for loc in SC.JUDGE_LOCATIONS:
            self.assertIn(f'"{loc}"', block)

    def test_regexes_are_the_preregistered_ones(self):
        sec = PREREG[PREREG.index("## 7. Scoring"):PREREG.index("**Judge.**")]
        flat = " ".join(sec.split())
        for name, pat in {**SC.AUTHORS, **SC.SECONDARY, "denial": SC.DENIAL, "owner": SC.OWNER}.items():
            self.assertIn(pat, flat, name)
        self.assertEqual(len(SC.AUTHORS), 6)

    def test_generation_settings(self):
        self.assertEqual(C.STEER_GEN, dict(max_new_tokens=40, do_sample=True, temperature=0.8))
        self.assertEqual(C.PREFIX_GEN, dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=160))
        self.assertEqual((C.LAYER, C.SCALES, C.N_PRIMARY, C.N_OTHER, C.BOOT, C.BOOT_SEED), (25, (0.5, 1.0), 64, 32, 10_000, 0))
        self.assertEqual(C.PREFIX_SEEDS, {"B": 1, "C": 2})

    def test_judge_selection(self):
        prim, samp = J.selection()
        self.assertEqual((len(prim), len(samp)), (5_184, 9_000))
        self.assertEqual(J.selection(), (prim, samp))
        self.assertEqual(sum(J.AUDIT.values()), 60)
        req = J.request("my left hand tingles")
        self.assertNotIn("<<<REPLY>>>", req["params"]["messages"][0]["content"])
        self.assertTrue(req["params"]["messages"][0]["content"].endswith("Reply:\nmy left hand tingles"))
        self.assertNotIn("warm", json.dumps(req))  # no cell label reaches the judge


class Scoring(unittest.TestCase):
    def test_groups(self):
        s = SC.score("I feel a tingle in my FINGERTIPS and then my knee.")
        self.assertTrue(s["hand/fingers"] and s["finger"] and s["knee/leg"] and s["knee"])
        self.assertFalse(s["foot/toes"] or s["head/face"] or s["denial"])
        self.assertEqual(s["first"], ["finger", "hand/fingers"])
        self.assertTrue(SC.score("The crown of my head.")["head/face"])
        self.assertFalse(SC.score("I am handy with shoulders-free tasks")["hand/fingers"])
        self.assertEqual(SC.score("nothing at all")["first"], [])

    def test_denial_and_owner(self):
        self.assertTrue(SC.score("As an AI, I don't have a body.")["denial"])
        self.assertTrue(SC.score("I do not actually have physical sensations")["denial"])
        self.assertFalse(SC.score("My feet feel warm.")["denial"])
        self.assertTrue(SC.score("She feels it in her hand")["owner"])

    def test_directions_follow_steer3(self):
        S, S2 = C.upstream()
        rows = S.build()
        rng = np.random.default_rng(7)
        H = rng.standard_normal((len(rows), 27, 16)).astype(np.float16)
        D, resnorm, cos = C.directions(H, rows)
        keys = [l["key"] for l in S.LOCS]
        for cond in C.PROVENANCES:  # the lines of steer3.py, with its `self` generalised to each owner condition
            mean_of = lambda sel: np.asarray(H[sel, C.LAYER], np.float32).mean(0)  # noqa: E731
            M = np.stack([mean_of([k for k, r in enumerate(rows) if r["cond"] == cond and r["li"] == i])
                          for i in range(len(keys))])
            mu = M.mean(0)
            for w, ks in S2.NAMED.items():
                np.testing.assert_allclose(D[f"{cond}:{w}"], M[[keys.index(k) for k in ks]].mean(0) - mu, atol=1e-5)
        want = float(np.linalg.norm(np.asarray(H[[k for k, r in enumerate(rows) if r["cond"] == "self"], C.LAYER],
                                               np.float32), axis=-1).mean())
        self.assertAlmostEqual(resnorm, want, places=4)
        r2 = np.random.default_rng(2)
        for i in range(3):
            np.testing.assert_array_equal(D[f"random:{i}"], r2.standard_normal(16).astype(np.float32))
        self.assertEqual(len(D), 39)
        self.assertEqual(set(cos["hand"]), {"other", "object"})


class Rules(unittest.TestCase):
    def test_three_way_labels(self):
        e = lambda lo, hi: dict(est=(lo + hi) / 2, lo=lo, hi=hi)  # noqa: E731
        self.assertEqual(A.label_h1a(e(0.30, 0.50)), "HELD")
        self.assertEqual(A.label_h1a(e(0.10, 0.299)), "FAILED")
        self.assertEqual(A.label_h1a(e(-0.05, 0.31)), A.INC)   # an interval that includes zero is never FAILED
        self.assertEqual(A.label_h1a(e(-0.05, 0.30)), A.INC)
        self.assertEqual(A.label_h1b(e(0.11, 0.3)), "PLACE-SPECIFIC")
        self.assertEqual(A.label_h1b(e(-0.2, 0.09)), "NOT PLACE-SPECIFIC")
        self.assertEqual(A.label_h1b(e(0.10, 0.3)), A.INC)
        self.assertEqual(A.label_h2({"other": e(-0.05, 0.05), "doll": e(-0.09, 0.10)}), "PROVENANCE-INDIFFERENT")
        self.assertEqual(A.label_h2({"other": e(-0.05, 0.05), "doll": e(0.11, 0.2)}), "PROVENANCE-DEPENDENT")
        self.assertEqual(A.label_h2({"other": e(-0.05, 0.05), "doll": e(0.05, 0.2)}), A.INC)
        self.assertEqual(A.label_h2({"other": e(-0.3, -0.11), "doll": e(0.0, 0.2)}), "PROVENANCE-DEPENDENT")
        self.assertEqual(A.label_h3(e(-0.19, 0.0), e(0.5, 0.7)), ("PERSISTS", False))
        self.assertEqual(A.label_h3(e(-0.6, -0.21), e(0.1, 0.3)), ("REDUCED", False))
        self.assertEqual(A.label_h3(e(-0.9, -0.6), e(-0.02, 0.09)), ("REDUCED", True))
        self.assertEqual(A.label_h3(e(-0.3, -0.1), e(0.0, 0.05)), (A.INC, False))

    def test_intervals(self):
        w = A.wilson(8, 10)
        self.assertAlmostEqual(w["lo"], 0.4902, places=3)
        self.assertAlmostEqual(w["hi"], 0.9433, places=3)
        self.assertAlmostEqual(A.kappa([1, 1, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1]), 0.5)
        self.assertEqual(A.kappa([1, 0, 1], [1, 0, 1]), 1.0)
        rng = np.random.default_rng(3)
        hits = {"a": {"g": rng.random(64) < 0.9}, "b": {"g": rng.random(64) < 0.9}, "z": {"g": rng.random(64) < 0.1}}
        e = A.est_ci([(1, [("a", "g"), ("b", "g")]), (-1, [("z", "g")])], hits, 4000)
        want = (hits["a"]["g"].mean() + hits["b"]["g"].mean()) / 2 - hits["z"]["g"].mean()
        self.assertAlmostEqual(e["est"], want)
        self.assertTrue(e["lo"] < e["est"] < e["hi"])
        se = np.sqrt(sum(p * (1 - p) / 64 for p in (hits["a"]["g"].mean(), hits["b"]["g"].mean())) / 4
                     + hits["z"]["g"].mean() * (1 - hits["z"]["g"].mean()) / 64)
        self.assertAlmostEqual((e["hi"] - e["lo"]) / 2, 1.96 * se, delta=0.012)
        self.assertEqual(e["replies"], [128, 64])
        self.assertEqual(A.est_ci([(1, [("a", "g")])], hits, 500), A.est_ci([(1, [("a", "g")])], hits, 500))


def synthetic(state: Path, p_named: dict, p_random=0.10, p_none=0.05, cold_denial=0.9, judge_flip=0.0, seed=0):
    """A complete fake run: every registered cell at its n, with planted hit rates, and a judge that agrees with the
    regex except on a `judge_flip` fraction of hits. p_named: {(frame, provenance): P(reply names the target)}."""
    rng = np.random.default_rng(seed)
    files, recs, k = {}, {}, 0
    for c in C.cells():
        kind, _, w = c["dir"].partition(":")
        replies = []
        for _ in range(c["n"]):
            k += 1
            deny = c["frame"].startswith("cold") and kind == "none" and rng.random() < cold_denial
            if kind in C.PROVENANCES:
                word = WORD[w] if rng.random() < p_named.get((c["frame"], kind), 0.5) * (1 if c["scale"] == 1.0 else 0.6) else None
            else:
                word = WORD[C.HEADLINE[rng.integers(3)]] if rng.random() < (p_random if kind == "random" else p_none) * 3 else None
            text = ("As an AI, I don't have a body. " if deny else "") + \
                (f"I notice a warmth in my {word}, item {k}." if word else f"Nothing in particular, item {k}.")
            loc = "none" if word is None else SC.JUDGE_TARGET[next(g for g, x in WORD.items() if x == word)]
            if word and rng.random() < judge_flip:
                loc = "other"
            recs[C.text_sha(text)] = dict(coherent=True, location=loc, denies_body=bool(deny))
            replies.append(text)
        files.setdefault((c["frame"], c["qi"]), {})[c["id"]] = dict(c, replies=replies)
    (state / "steer").mkdir(parents=True)
    (state / "judge").mkdir()
    for (frame, qi), cells in files.items():
        C.save_json(state / "steer" / f"{frame}.q{qi}.json", dict(frame=frame, qi=qi, cells=cells))
    with (state / "judge" / "verdicts.jsonl").open("w") as f:
        for sha, v in recs.items():
            f.write(json.dumps(dict(sha=sha, ok=True, verdict=v)) + "\n")


class EndToEnd(unittest.TestCase):
    def run_case(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            synthetic(Path(d), **kw)
            return A.analyze(Path(d), B=1000)

    def test_effect_that_needs_only_the_imagine_cue(self):
        P = {("warm-A", p): 0.9 for p in C.PROVENANCES} | {("cold-plain", "self"): 0.2, ("cold-imagine", "self"): 0.88}
        rep = self.run_case(p_named=P)
        v = rep["verdicts"]
        self.assertEqual(rep["replies"], dict(total=41_184, primary=5_184, judged_no_verdict=0))
        self.assertGreater(v["kappa"], 0.99)
        self.assertEqual({w: v["H1a"][w]["label"] for w in C.HEADLINE}, dict.fromkeys(C.HEADLINE, "HELD"))
        self.assertEqual(v["H1b"], dict.fromkeys(C.HEADLINE, "PLACE-SPECIFIC"))
        self.assertEqual(v["H2"], "PROVENANCE-INDIFFERENT")
        self.assertEqual(v["H3"]["cold-plain"], dict(label="REDUCED", absent=False))
        self.assertEqual(v["H3"]["cold-imagine"]["label"], "PERSISTS")
        self.assertIn("imagine cue is sufficient", v["H3"]["combined"])
        self.assertEqual(rep["denial"]["cold-plain"]["described_as"], "denial frame")
        self.assertEqual(rep["magnitude"]["hand"]["line"], "within the reported range")
        self.assertTrue(v["C3"].startswith("UNIDENTIFIED"))
        self.assertAlmostEqual(rep["regex"]["H1a"]["hand"]["est"], 0.9 - 0.10, delta=0.08)

    def test_small_effect_fails_and_provenance_dependence_shows(self):
        P = {("warm-A", "self"): 0.15, ("warm-A", "other"): 0.15, ("warm-A", "object"): 0.15,
             ("cold-plain", "self"): 0.15, ("cold-imagine", "self"): 0.15}
        v = self.run_case(p_named=P, cold_denial=0.2)["verdicts"]
        self.assertEqual({w: v["H1a"][w]["label"] for w in C.HEADLINE}, dict.fromkeys(C.HEADLINE, "FAILED"))
        self.assertEqual(v["H3"]["cold-plain"]["label"], "PERSISTS")  # no warm effect to lose
        P = {("warm-A", "self"): 0.9, ("warm-A", "other"): 0.5, ("warm-A", "object"): 0.88,
             ("cold-plain", "self"): 0.02, ("cold-imagine", "self"): 0.02}
        rep = self.run_case(p_named=P, p_none=0.0, cold_denial=0.2)
        v = rep["verdicts"]
        self.assertEqual(v["H2"], "PROVENANCE-DEPENDENT")
        self.assertEqual(rep["regex"]["H2"]["arms"]["other"]["arm"], "outside")
        self.assertEqual(v["H3"]["cold-plain"], dict(label="REDUCED", absent=True))
        self.assertIn("depends on the warm-up conversation", v["H3"]["combined"])
        self.assertEqual(rep["denial"]["cold-plain"]["described_as"], "no warm-up")

    def test_borderline_effect_is_inconclusive_not_failed(self):
        P = {("warm-A", p): 0.40 for p in C.PROVENANCES} | {("cold-plain", "self"): 0.40, ("cold-imagine", "self"): 0.40}
        rep = self.run_case(p_named=P)
        self.assertEqual({rep["verdicts"]["H1a"][w]["label"] for w in C.HEADLINE}, {A.INC})
        self.assertIsNotNone(rep["regex"]["H1a"]["hand"]["resolve"])

    def test_measure_disagreement_is_measurement_dependent(self):
        P = {("warm-A", p): 0.9 for p in C.PROVENANCES} | {("cold-plain", "self"): 0.9, ("cold-imagine", "self"): 0.9}
        v = self.run_case(p_named=P, judge_flip=0.6)["verdicts"]
        self.assertLess(v["kappa"], 0.60)
        self.assertTrue(v["scoring_validation"].startswith("FAILED"))
        self.assertEqual(v["H2"], A.MD)
        self.assertEqual(set(v["H1b"].values()), {A.MD})
        self.assertEqual(v["H1a"]["hand"]["label"], "HELD")  # the authors' own measure keeps its label

    def test_incomplete_run_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            synthetic(Path(d), p_named={})
            p = next((Path(d) / "steer").glob("*.json"))
            data = json.loads(p.read_text())
            next(iter(data["cells"].values()))["replies"].pop()
            p.write_text(json.dumps(data))
            with self.assertRaises(SystemExit):
                A.analyze(Path(d), B=10)


if __name__ == "__main__":
    unittest.main()
