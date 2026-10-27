"""TRCM Phase 1 sample selection (docs/TRCM_PHASE0.md §3-4): 500 discovery + 150 validation examples from
state/sft/v0/kept.jsonl, stratified by slice × device × lang, with the discovery/validation split taken at the
split_v3 CLUSTER level (sha1 parity), never across a cluster. 100 of the 500 are flagged for the blind audit.
The frozen 490 never enters: kept.jsonl is train-split material by invariant cat4b.select.train-split-only.

    .venv/bin/python eval/trcm/select_pilot.py
"""
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mech_common import STATE_DIR  # noqa: E402

TRCM = STATE_DIR / "eval" / "trcm"
SEED = 20261027
N_DISCOVERY, N_VALIDATION, N_AUDIT = 500, 150, 100


def main():
    kept = [json.loads(l) for l in open(STATE_DIR / "sft" / "v0" / "kept.jsonl")]
    cluster_of = {}
    with open(STATE_DIR / "split" / "split_v3.jsonl") as f:
        for l in f:
            r = json.loads(l)
            cluster_of[r["id"]] = r["cluster"]
    rows = []
    missing = 0
    for r in kept:
        cl = cluster_of.get(r["spec"]["bank_id"])
        if cl is None:
            missing += 1
            continue
        side = "discovery" if int(hashlib.sha1(cl.encode()).hexdigest(), 16) % 2 == 0 else "validation"
        rows.append({"sample_id": r["id"], "bank_id": r["spec"]["bank_id"], "cluster": cl, "trcm_split": side,
                     "slice": r["spec"]["slice"], "device": r["spec"]["device"], "lang": r["spec"]["lang"],
                     "subtype": r["spec"].get("subtype")})
    rng = random.Random(SEED)

    def stratified(pool, n):
        by = defaultdict(list)
        for r in pool:
            by[(r["slice"], r["device"], r["lang"])].append(r)
        for v in by.values():
            rng.shuffle(v)
        total = len(pool)
        take, rest = [], []
        for key, v in sorted(by.items()):
            k = max(1, round(n * len(v) / total))
            take += v[:k]
            rest += v[k:]
        rng.shuffle(rest)
        while len(take) > n:
            take.pop(rng.randrange(len(take)))
        take += rest[: n - len(take)]
        return take

    disc = stratified([r for r in rows if r["trcm_split"] == "discovery"], N_DISCOVERY)
    val = stratified([r for r in rows if r["trcm_split"] == "validation"], N_VALIDATION)
    audit = set(r["sample_id"] for r in rng.sample(disc, N_AUDIT))
    TRCM.mkdir(parents=True, exist_ok=True)
    with (TRCM / "manifest.jsonl").open("w") as f:
        for r in disc + val:
            r["audit"] = r["sample_id"] in audit
            f.write(json.dumps(r) + "\n")
    n_cl = {s: len({r['cluster'] for r in rows if r['trcm_split'] == s}) for s in ("discovery", "validation")}
    print(f"kept {len(kept)}, joined {len(rows)} (no cluster for {missing}); clusters {n_cl}")
    print(f"selected {len(disc)} discovery + {len(val)} validation, audit {len(audit)} → {TRCM}/manifest.jsonl")
    for s, pool in (("discovery", disc), ("validation", val)):
        by_slice = defaultdict(int)
        for r in pool:
            by_slice[r["slice"]] += 1
        print(f"  {s} by slice: {dict(sorted(by_slice.items()))}")


if __name__ == "__main__":
    main()
