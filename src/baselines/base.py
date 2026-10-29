"""Common interface of every baseline (M12, PROMPT §2.3).

Batch format = the project dataloader (src/dataset.py), left-padded:
  seq [B, L] long (0 = padding), target [B] long (1..N), user [B] long;
  timed models also get ts [B, L] float64 (unix seconds, 0 at padding) and target_ts [B].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSeqRec(nn.Module):
    """Abstract baseline.

    compute_loss(batch) -> dict[str, Tensor]   always has "loss"; other entries are logged (floats).
    score_all(batch)    -> Tensor [B, N+1]     column 0 (padding) = -inf.
    position_outputs(batch) -> Tensor [B, L, D] per-position states (only for `causal` models; used by tests).

    Class flags (declare what a faithful implementation guarantees; tests check them):
      causal:            the state at position t depends only on positions <= t.
      padding_invariant: values read at padding positions never change the scores.
      needs_timestamps:  the batch must carry `ts` / `target_ts` (src/baselines/data.py).
    """

    name: str = "base"
    causal: bool = True
    padding_invariant: bool = True
    needs_timestamps: bool = False
    modalities: tuple[str, ...] = ()

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def position_outputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not expose per-position outputs")


def with_pad_column(scores_real: torch.Tensor) -> torch.Tensor:
    """[B, N] scores of real items -> [B, N+1] with column 0 = -inf."""
    pad = scores_real.new_full((scores_real.shape[0], 1), float("-inf"))
    return torch.cat([pad, scores_real], dim=1)


def mask_pad_column(scores: torch.Tensor) -> torch.Tensor:
    """[B, N+1] scores over the padded table -> same with column 0 = -inf."""
    scores = scores.clone()
    scores[:, 0] = float("-inf")
    return scores


def ce_real(logits_all: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Full-softmax CE over the N real items. logits_all [B, N+1] (col 0 = padding) ; target [B] in 1..N.

    The padding column is dropped (project protocol: same as SASRec M3), whereas RecBole keeps it.
    """
    return F.cross_entropy(logits_all[:, 1:], target - 1)


def left_to_right(seq: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Convert left-padded [B, L, ...] tensors to RecBole's right padding.

    Returns (seq_right [B, L], lengths [B] (>= 1), [other_right, ...]). Padding positions hold 0.
    """
    B, L = seq.shape
    lengths = (seq != 0).sum(1)
    pad = L - lengths
    idx = (torch.arange(L, device=seq.device).unsqueeze(0) + pad.unsqueeze(1))  # [B, L]
    valid = idx < L
    idx = idx.clamp_max(L - 1)

    def move(x: torch.Tensor) -> torch.Tensor:
        g = idx.view(B, L, *([1] * (x.dim() - 2))).expand(B, L, *x.shape[2:])
        out = x.gather(1, g)
        return out * valid.view(B, L, *([1] * (x.dim() - 2))).to(out.dtype)

    return move(seq), lengths.clamp_min(1), [move(o) for o in others]


def right_to_left(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Inverse of left_to_right for per-position outputs [B, L, D] (padding positions -> 0)."""
    B, L = x.shape[:2]
    pad = L - lengths
    idx = torch.arange(L, device=x.device).unsqueeze(0) - pad.unsqueeze(1)  # position in right layout
    valid = idx >= 0
    idx = idx.clamp_min(0)
    out = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return out * valid.unsqueeze(-1).to(out.dtype)


def gather_last(output: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """RecBole gather_indexes: output [B, L, D] (right padded), lengths [B] -> [B, D] at lengths-1."""
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
    return output.gather(1, idx).squeeze(1)
