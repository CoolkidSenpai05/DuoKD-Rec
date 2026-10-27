"""Loss functions (CLAUDE.md §3.4). Kept separate from models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Full-softmax CE. logits [B, N] over real items; target [B] item ids in 1..N."""
    return F.cross_entropy(logits, target - 1)


class SASRecLoss(nn.Module):
    """CE for the ID-only baseline. Returns (loss, logs)."""

    def forward(self, out: dict, batch: dict, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
        loss = ce_loss(out["sasrec"], batch["target"])
        return loss, {"L_sup_sasrec": loss.item()}


def moe_balance_loss(probs: torch.Tensor, gates: torch.Tensor, mask: torch.Tensor,
                     top_k: int | None) -> torch.Tensor:
    """MoE load-balancing loss (§3.4d).

    probs, gates: [M, E] router probs / applied gates; mask: [M] bool (real positions).
    Soft:  n·Σ_k P_k²          (min = 1 when uniform)
    Top-k: n·Σ_k f_k·P_k, f_k = fraction of (masked) samples assigned to expert k.
    """
    n = probs.shape[-1]
    w = mask.float().unsqueeze(-1)
    denom = w.sum().clamp_min(1.0)
    P = (probs * w).sum(0) / denom
    if top_k is None:
        return n * (P ** 2).sum()
    f = ((gates > 0).float() * w).sum(0) / denom / top_k
    return n * (f * P).sum()


def gate_stats(probs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean expert load P [E] and mean gate entropy [] over masked rows of probs [M, E]."""
    w = mask.float().unsqueeze(-1)
    denom = w.sum().clamp_min(1.0)
    P = (probs * w).sum(0) / denom
    ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
    return P, (ent * mask.float()).sum() / denom
