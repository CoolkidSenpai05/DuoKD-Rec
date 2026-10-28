"""Per-dataset data invariants (M11): 5-core, leave-one-out, contiguous IDs, feature rows = N+1.

Every check runs on a small synthetic dataset (always) and on each registered dataset whose
processed files exist (skipped otherwise).
"""
from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import pytest

from data.prepare import build_sequences_with_time, k_core, split_leave_one_out
from src.dataset import EvalDataset, TrainDataset, load_processed
from src.utils import load_config, resolve

DATASETS = ["beauty", "sports", "toys", "clothing"]


def _synthetic() -> dict:
    rng = random.Random(1)
    inter = []
    for u in range(60):
        for t in range(rng.randint(4, 14)):
            inter.append((f"u{u}", f"i{rng.randint(0, 40)}", 86400 * (rng.randint(0, 30) + t)))
    inter = k_core(inter, 5)
    seqs, times, _, item2id = build_sequences_with_time(inter)
    tr, va, te = split_leave_one_out(seqs)
    ttr, tva, tte = split_leave_one_out(times)
    return {"n_users": len(seqs), "n_items": len(item2id), "train_seqs": tr, "valid_items": va,
            "test_items": te, "train_ts": ttr, "valid_ts": tva, "test_ts": tte}


def _real(ds: str) -> tuple[dict, dict]:
    cfg = load_config("configs/base.yaml", [f"data.dataset={ds}"])
    d = resolve(cfg["data"]["processed_dir"])
    if not (d / "data.pkl").exists() or not (d / "timestamps.pkl").exists():
        pytest.skip(f"{ds}: processed data not found (run data/prepare.py --set data.dataset={ds})")
    return load_processed(d, with_timestamps=True), cfg


@pytest.fixture(params=["synthetic"] + DATASETS)
def dataset(request):
    if request.param == "synthetic":
        return request.param, _synthetic(), None
    data, cfg = _real(request.param)
    return request.param, data, cfg


def _full(data: dict) -> list[list[int]]:
    return [t + [v, x] for t, v, x in zip(data["train_seqs"], data["valid_items"], data["test_items"])]


def test_five_core(dataset):
    _, data, _ = dataset
    seqs = _full(data)
    assert min(len(s) for s in seqs) >= 5, "user with < 5 interactions after 5-core"
    ic = Counter(i for s in seqs for i in s)
    assert min(ic.values()) >= 5, "item with < 5 interactions after 5-core"


def test_contiguous_ids(dataset):
    _, data, _ = dataset
    items = {i for s in _full(data) for i in s}
    assert items == set(range(1, data["n_items"] + 1))
    assert len(data["train_seqs"]) == data["n_users"]
    if "item2id" in data:
        assert sorted(data["item2id"].values()) == list(range(1, data["n_items"] + 1))
        assert sorted(data["user2id"].values()) == list(range(data["n_users"]))


def test_leave_one_out_chronological(dataset):
    """valid = second to last, test = last interaction in time; inputs never contain the target."""
    _, data, _ = dataset
    for u in range(0, data["n_users"], max(1, data["n_users"] // 2000)):
        ts = data["train_ts"][u] + [data["valid_ts"][u], data["test_ts"][u]]
        assert all(a <= b for a, b in zip(ts, ts[1:])), "sequence not chronological"
    L = 50
    va, te = EvalDataset(data, "valid", L), EvalDataset(data, "test", L)
    for u in range(0, data["n_users"], max(1, data["n_users"] // 500)):
        assert va.history[u] == data["train_seqs"][u]
        assert te.history[u] == data["train_seqs"][u] + [data["valid_items"][u]]
        assert int(va[u]["target"]) == data["valid_items"][u]
        assert int(te[u]["target"]) == data["test_items"][u]
    tr = TrainDataset(data["train_seqs"], L)
    for idx in range(0, len(tr), max(1, len(tr) // 2000)):
        u, t = tr.index[idx]
        assert t < len(data["train_seqs"][u])  # label is a train interaction (never valid/test)


def test_stats_match_expected(dataset):
    name, data, cfg = dataset
    if cfg is None:
        pytest.skip("synthetic dataset has no expected statistics")
    exp = cfg["data"]["expected"]
    n_inter = sum(len(s) for s in _full(data))
    for k, v in (("n_users", data["n_users"]), ("n_items", data["n_items"]), ("n_interactions", n_inter)):
        assert abs(v - exp[k]) / exp[k] <= 0.005, f"{name} {k}: {v} vs expected {exp[k]}"


@pytest.mark.parametrize("ds", DATASETS)
def test_feature_rows(ds):
    data, cfg = _real(ds)
    base = resolve(cfg["features"]["feat_dir"])
    dirs = [d for d in [base] + sorted(base.parent.glob(base.name + "_limit*")) if (d / "feat_meta.json").exists()]
    if not dirs:
        pytest.skip(f"{ds}: features not extracted")
    for d in dirs:
        meta = json.loads((d / "feat_meta.json").read_text())
        n = meta["n_items"]
        if d == base:
            assert n == data["n_items"], "full features must cover the whole catalog"
        for name in ("text_feat", "img_feat"):
            x = np.load(d / f"{name}.npy", mmap_mode="r")
            assert x.shape[0] == n + 1 and x.dtype == np.float32 and (x[0] == 0).all()
        for name in ("has_text", "has_img"):
            h = np.load(d / f"{name}.npy")
            assert h.dtype == bool and h.shape == (n + 1,) and not h[0]
        assert int(meta["n_missing_img"]) == int(n - np.load(d / "has_img.npy")[1:].sum())


def test_no_leakage_split_check(dataset):
    """Reuse the M1 leakage check of test_data_split on every dataset (subsampled users)."""
    from test_data_split import _check

    _, data, _ = dataset
    seqs = _full(data)
    sub = list(range(0, len(seqs), max(1, len(seqs) // 400)))
    sub_data = {k: [data[k][u] for u in sub] for k in ("train_seqs", "valid_items", "test_items")}
    _check([seqs[u] for u in sub], sub_data, 50)
