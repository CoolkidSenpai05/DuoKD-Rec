"""Hand-computed Recall/NDCG/MRR vs src/metrics.py (CLAUDE.md §8)."""
from __future__ import annotations

import math

import torch

from src.metrics import mask_history, metrics_from_ranks, target_ranks


def test_metrics_by_hand():
    # 3 users, 5 items (columns = item ids 1..5)
    scores = torch.tensor([
        [0.9, 0.1, 0.5, 0.3, 0.2],   # target col 2 (0.5): one higher -> rank 1
        [0.1, 0.2, 0.3, 0.4, 0.5],   # target col 0 (0.1): rank 4
        [0.7, 0.6, 0.5, 0.4, 0.3],   # target col 0: rank 0
    ])
    target = torch.tensor([2, 0, 0])
    ranks = target_ranks(scores, target)
    assert ranks.tolist() == [1, 4, 0]
    m = metrics_from_ranks(ranks, ks=[1, 2, 5], mrr=True)
    assert m["Recall@1"].tolist() == [0, 0, 1]
    assert m["Recall@2"].tolist() == [1, 0, 1]
    assert m["Recall@5"].tolist() == [1, 1, 1]
    expect_ndcg2 = [1 / math.log2(3), 0.0, 1.0]
    for a, b in zip(m["NDCG@2"].tolist(), expect_ndcg2):
        assert abs(a - b) < 1e-6
    expect_ndcg5 = [1 / math.log2(3), 1 / math.log2(6), 1.0]
    for a, b in zip(m["NDCG@5"].tolist(), expect_ndcg5):
        assert abs(a - b) < 1e-6
    for a, b in zip(m["MRR"].tolist(), [1 / 2, 1 / 5, 1.0]):
        assert abs(a - b) < 1e-6


def test_exclude_history():
    scores = torch.tensor([[0.9, 0.1, 0.5, 0.3, 0.2]])
    hist = torch.tensor([[True, False, False, False, False]])  # item 1 seen
    ranks = target_ranks(mask_history(scores, hist), torch.tensor([2]))
    assert ranks.tolist() == [0]


def test_ties_favour_target():
    scores = torch.tensor([[0.5, 0.5, 0.5]])
    assert target_ranks(scores, torch.tensor([1])).tolist() == [0]
