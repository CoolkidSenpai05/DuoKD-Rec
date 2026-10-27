"""Leakage checks for the leave-one-out split (CLAUDE.md §4, §8)."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from data.prepare import build_sequences, k_core, split_leave_one_out
from src.dataset import EvalDataset, TrainDataset, load_processed
from src.utils import resolve

L = 5


def _synthetic():
    rng = random.Random(0)
    inter = []
    for u in range(40):
        for t in range(rng.randint(5, 15)):
            inter.append((f"u{u}", f"i{rng.randint(0, 30)}", 1000 + t * 10 + rng.randint(0, 3)))
    inter = k_core(inter, 5)
    seqs, _, _ = build_sequences(inter)
    train, valid, test = split_leave_one_out(seqs)
    data = {"train_seqs": train, "valid_items": valid, "test_items": test, "n_items": 31}
    return seqs, data


def _check(seqs, data, max_len):
    train, valid, test = data["train_seqs"], data["valid_items"], data["test_items"]
    tr = TrainDataset(train, max_len)
    # (1) every train sample comes from the user's train prefix only; labels never valid/test position
    used = [Counter() for _ in seqs]
    for idx in range(len(tr)):
        x = tr[idx]
        u = int(x["user"])
        inp = [i for i in x["seq"].tolist() if i != 0]
        tgt = int(x["target"])
        t = tr.index[idx][1]
        assert inp == seqs[u][:t][-max_len:]
        assert tgt == seqs[u][t] and t < len(seqs[u]) - 2
        used[u] |= Counter(inp + [tgt])  # max multiplicity over samples
    for u, s in enumerate(seqs):
        # multiset of items used by train samples must fit inside s[:-2]
        assert not (used[u] - Counter(s[:-2])), "valid/test interaction leaked into training"
        assert valid[u] == s[-2] and test[u] == s[-1]
    # (2) valid input = train seq ; test input = train seq + valid item
    va, te = EvalDataset(data, "valid", max_len), EvalDataset(data, "test", max_len)
    for u, s in enumerate(seqs):
        v_in = [i for i in va[u]["seq"].tolist() if i != 0]
        t_in = [i for i in te[u]["seq"].tolist() if i != 0]
        assert v_in == s[:-2][-max_len:]
        assert t_in == s[:-1][-max_len:]
        assert t_in[-1] == valid[u], "test input must end with the valid item"
        assert int(va[u]["target"]) == s[-2] and int(te[u]["target"]) == s[-1]
        assert te.history[u] == s[:-1] and va.history[u] == s[:-2]


def test_split_synthetic():
    seqs, data = _synthetic()
    for s in seqs:
        assert len(s) >= 5
    _check(seqs, data, L)


def test_left_padding():
    from src.dataset import left_pad

    assert left_pad([3, 4], 4).tolist() == [0, 0, 3, 4]
    assert left_pad([1, 2, 3, 4, 5], 3).tolist() == [3, 4, 5]


def test_split_real_data():
    path = resolve("data/processed/Beauty/data.pkl")
    if not path.exists():
        pytest.skip("processed Beauty data not found; run data/prepare.py")
    data = load_processed(path.parent)
    seqs = [t + [v, x] for t, v, x in zip(data["train_seqs"], data["valid_items"], data["test_items"])]
    sub = list(range(0, len(seqs), 50))  # subsample users for speed
    sub_data = {k: [data[k][u] for u in sub] for k in ("train_seqs", "valid_items", "test_items")}
    _check([seqs[u] for u in sub], sub_data, 50)
    items = {i for s in seqs for i in s}
    assert items == set(range(1, data["n_items"] + 1))
