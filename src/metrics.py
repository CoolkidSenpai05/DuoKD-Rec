"""Full-ranking metrics (CLAUDE.md §6)."""
from __future__ import annotations

import torch


def target_ranks(scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """0-based rank of the target among all items.

    scores: [B, N] float (excluded items already set to -inf); target_idx: [B] column index (item id - 1).
    rank = #items with strictly higher score (ties resolved in the target's favour).
    Returns [B] long.
    """
    t = scores.gather(1, target_idx.unsqueeze(1))  # [B, 1]
    return (scores > t).sum(1)


def metrics_from_ranks(ranks: torch.Tensor, ks: list[int], mrr: bool = True) -> dict[str, torch.Tensor]:
    """Per-sample metric values [B] from 0-based ranks [B]."""
    r = ranks.float()
    out: dict[str, torch.Tensor] = {}
    for k in ks:
        hit = (ranks < k).float()
        out[f"Recall@{k}"] = hit
        out[f"NDCG@{k}"] = hit / torch.log2(r + 2.0)
    if mrr:
        out["MRR"] = 1.0 / (r + 1.0)
    return out


def mask_history(scores: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    """Set scores of items already in the history to -inf. scores, history_mask: [B, N]."""
    return scores.masked_fill(history_mask, float("-inf"))
