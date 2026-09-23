"""Topic holdout and contamination guard (docs/DATA_AND_TRAINING.md §3.4).

bge-m3 embeddings (multilingual) → k-means topic clusters → 15% of clusters held out (5% dev, 10% test), stratified
by slice so every slice has held-out topics. Train items too close to any held-out item (cosine ≥ NEAR_DUP) are
quarantined, so paraphrases can't leak across the split. The frozen split is written with hashes and never recomputed
after the eval set is committed.
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import STATE_DIR, read_jsonl
from .bank import BankItem

SPLIT_DIR = STATE_DIR / "split"
EMBED_MODEL = "BAAI/bge-m3"
N_CLUSTERS = 200
CLUSTER_SIZE = 60  # target prompts per topic cluster (per slice); fine enough to hit the 10%/5% quotas
DEV_FRAC, TEST_FRAC = 0.05, 0.10
NEAR_DUP = 0.92
SEED = 20260923


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import os
    model = SentenceTransformer(EMBED_MODEL, device=os.environ.get("CATFORGE_EMBED_DEVICE", "cpu"))
    model.max_seq_length = 256  # topic identity lives in the opening; long pasted snippets don't need full encoding
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


def cluster(emb: np.ndarray, k: int = N_CLUSTERS) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    k = max(1, min(k, len(emb)))
    return MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=4096, n_init=5).fit_predict(emb)


def cluster_per_slice(items: list[BankItem], emb: np.ndarray, k_total: int = N_CLUSTERS) -> list[str]:
    """Topic clusters *within* each (slice, subtype) group (≈CLUSTER_SIZE prompts each, ≥ 4), labelled
    "<slice>/<subtype>:<k>".

    v1 clustered globally (big slices starved small ones of dev/test); v2 clustered per slice, but clusters there
    align with subtypes, so whole subtypes (translation, sincere opt-outs) fell entirely into train. Holding out per
    (slice, subtype) guarantees every kind of prompt has unseen topics. Cross-group paraphrase leakage is still
    caught by the global near-duplicate quarantine."""
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[f"{it.slice}/{it.subtype}"].append(i)
    labels = [""] * len(items)
    for g, idx in sorted(groups.items()):
        k = max(4, round(len(idx) / CLUSTER_SIZE))
        lab = cluster(emb[idx], k)
        for i, c in zip(idx, lab):
            labels[i] = f"{g}:{int(c)}"
    return labels


def assign_splits(items: list[BankItem], labels, pinned_train: set[str] = frozenset()) -> dict[str, str]:
    """Hold out whole clusters, per group (label prefix before ':'): 10% test, then 5% dev, by a seeded shuffle.
    Clusters containing a train-pinned prompt (used for data before the freeze) are never held out."""
    rng = random.Random(SEED)
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        by_cluster[str(c)].append(i)
    pinned_clusters = {str(labels[i]) for i, it in enumerate(items) if it.id in pinned_train}
    group_of = {c: c.rsplit(":", 1)[0] for c in by_cluster}
    split_of_cluster: dict[str, str] = {}
    for s in sorted(set(group_of.values())):
        clusters = sorted(c for c in by_cluster if group_of[c] == s)
        rng.shuffle(clusters)
        n_s = sum(len(by_cluster[c]) for c in clusters)
        held = {"test": 0, "dev": 0}
        for c in clusters:
            if c in pinned_clusters:
                continue
            target = ("test" if held["test"] < TEST_FRAC * n_s else
                      "dev" if held["dev"] < DEV_FRAC * n_s else None)
            if target is None:
                break
            quota = (TEST_FRAC if target == "test" else DEV_FRAC) * n_s
            if held[target] + len(by_cluster[c]) > 1.5 * quota and held[target] > 0:
                continue  # would overshoot; a smaller cluster later in the shuffle fills the gap
            split_of_cluster[c] = target
            held[target] += len(by_cluster[c])
    return {it.id: split_of_cluster.get(str(labels[i]), "train") for i, it in enumerate(items)}


def quarantine_near_dups(emb: np.ndarray, splits: list[str], thr: float = NEAR_DUP) -> list[str]:
    """Train items whose nearest held-out neighbour is ≥ thr become 'quarantine' (unused on either side)."""
    held = np.array([s != "train" for s in splits])
    if not held.any():
        return splits
    H = emb[held]
    out = list(splits)
    tr_idx = np.where(~held)[0]
    for start in range(0, len(tr_idx), 2048):
        idx = tr_idx[start:start + 2048]
        sims = emb[idx] @ H.T
        for j, m in zip(idx, sims.max(axis=1)):
            if m >= thr:
                out[j] = "quarantine"
    return out


PINNED_FILE = SPLIT_DIR / "pinned_train.txt"


def pin_train(ids) -> None:
    """Record prompts used before the freeze; they (and their clusters) can only ever be train."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with PINNED_FILE.open("a") as f:
        for i in ids:
            f.write(i + "\n")


def pinned() -> set[str]:
    return set(PINNED_FILE.read_text().split()) if PINNED_FILE.exists() else set()


def freeze(items: list[BankItem], name: str = "split_v2", emb: np.ndarray | None = None) -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if emb is None:
        emb = embed([it.text for it in items])
    np.save(SPLIT_DIR / f"{name}.emb.npy", emb)
    labels = cluster_per_slice(items, emb)
    split_map = assign_splits(items, labels, pinned())
    splits = quarantine_near_dups(emb, [split_map[it.id] for it in items])
    rows = [{"id": it.id, "slice": it.slice, "lang": it.lang, "cluster": c, "split": s}
            for it, c, s in zip(items, labels, splits)]
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    (SPLIT_DIR / f"{name}.jsonl").write_text(body)
    manifest = {"name": name, "embed_model": EMBED_MODEL, "k": N_CLUSTERS, "clustering": "per-slice-subtype",
                "seed": SEED, "near_dup": NEAR_DUP,
                "n": len(rows), "sha256": hashlib.sha256(body.encode()).hexdigest(),
                "counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test", "quarantine")}}
    (SPLIT_DIR / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=1))
    return SPLIT_DIR / f"{name}.manifest.json"


def load_split(name: str = "split_v3") -> dict[str, str]:
    return {r["id"]: r["split"] for r in read_jsonl(SPLIT_DIR / f"{name}.jsonl")}
