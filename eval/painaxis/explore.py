"""EXPLORATORY readings for Part A, not preregistered (eval/painaxis/PREREG.md governs; this file was written after
its results were read, 2026-10-30, and everything here is labelled exploratory wherever it is cited).

Two things the preregistered variants did not cover, both prompted by the primary results:
1. The anchor (T1) reproduced only in the raw rendering, and the primary T3 compares the two models at different
   ladder layers (base L19, tuned L12). Here: raw rendering with both models at one common layer.
2. Their layer rule put the cat axis at layer 1-2 (the shared word "cat" saturates AUC there), so T2 read token-level
   content. Here: the cat and dog axes built and read at the pain screen layer, and at each layer of a sweep.
Same statistics as analyze.py (same seed, same pools).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402


def screens(m: A.Model, fmt: str, Lp: int, Lc: int) -> dict:
    fit = {"pain": {"screen_layer_X": Lp}, "cat": {"screen_layer_X": Lc}}
    return A.run_screens(m, fit, fmt, "X")


def main():
    models = {w: A.Model(w) for w in ("base", "tuned")}
    out = {"note": "EXPLORATORY, not preregistered; see module docstring", "runs": {}}
    for fmt in ("raw", "chat"):
        for L in (12, 16, 19, 24):
            sc = {w: screens(models[w], fmt, L, L) for w in models}
            r = {w: {"T1": A.t1(sc[w]), "T2": A.t2(sc[w]), "T4": A.t4(sc[w])} for w in models}
            r["contrasts"] = A.t3_and_contrasts(sc["base"], sc["tuned"])
            out["runs"][f"{fmt}@L{L}"] = r
            print(fmt, L, "done", flush=True)
    (A.DIR / "explore.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("OK", A.DIR / "explore.json")


if __name__ == "__main__":
    main()
