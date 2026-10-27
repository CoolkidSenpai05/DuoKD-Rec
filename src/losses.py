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
