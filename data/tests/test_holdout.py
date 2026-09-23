import numpy as np
import pytest

pytest.importorskip("sklearn")

from catforge.bank import BankItem  # noqa: E402
from catforge import holdout  # noqa: E402


def items(n, slice_="plain"):
    return [BankItem(id=f"x{i}", source="t", license="t", slice=slice_, subtype="", lang="en", text=f"t{i}")
            for i in range(n)]


def test_whole_clusters_held_out_and_fractions():
    its = items(1000)
    labels = [f"plain:{c}" for c in np.repeat(np.arange(100), 10)]  # 100 clusters of 10
    split = holdout.assign_splits(its, labels)
    by_cluster = {}
    for it, c in zip(its, labels):
        by_cluster.setdefault(str(c), set()).add(split[it.id])
    assert all(len(s) == 1 for s in by_cluster.values())  # no cluster straddles splits
    n_test = sum(v == "test" for v in split.values())
    n_dev = sum(v == "dev" for v in split.values())
    assert 100 <= n_test <= 110 and 50 <= n_dev <= 60


def test_near_dup_quarantine():
    emb = np.eye(4, dtype=np.float32)
    emb = np.vstack([emb, emb[0] * 0.99 + emb[1] * 0.01])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    splits = ["test", "train", "train", "train", "train"]
    out = holdout.quarantine_near_dups(emb, splits, thr=0.92)
    assert out == ["test", "train", "train", "train", "quarantine"]


def test_pinned_clusters_stay_train():
    its = items(1000)
    labels = [f"plain:{c}" for c in np.repeat(np.arange(100), 10)]
    pinned = {f"x{i}" for i in range(0, 1000, 10)}  # one item in every cluster → nothing can be held out
    split = holdout.assign_splits(its, labels, pinned)
    assert set(split.values()) == {"train"}


def test_small_slices_get_their_share():
    its = items(900, "plain") + items(100, "refusal")
    for i, it in enumerate(its[900:]):
        it.id = f"r{i}"
    labels = [f"plain:{i // 10}" for i in range(900)] + [f"refusal:{i // 5}" for i in range(100)]
    split = holdout.assign_splits(its, labels)
    ref = [split[it.id] for it in its[900:]]
    assert ref.count("test") >= 10 and ref.count("dev") >= 5
