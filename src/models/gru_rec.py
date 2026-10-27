"""GRU student encoder (CLAUDE.md §3.3, §3.6)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def left_to_right_padded(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the real items of each left-padded row to the front.

    x: [B, L, D], mask: [B, L] bool (True = real, all True positions are a suffix).
    Returns (x_right [B, L, D], lengths [B] long, >= 1).
    """
    B, L, _ = x.shape
    lengths = mask.sum(1)
    pad = L - lengths                                              # [B]
    idx = (torch.arange(L, device=x.device).unsqueeze(0) + pad.unsqueeze(1)).clamp_max(L - 1)
    x_r = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return x_r, lengths.clamp_min(1)


class GRUEncoder(nn.Module):
    """Multi-layer GRU over left-padded sequences; padding never touches the hidden state.

    forward(h [B, L, D], mask [B, L]) -> u [B, D] (last-layer state after the last real item).
    step(x [B, D], state [n_layers, B, D] | None) -> (u [B, D], new state)  for O(1) online infer.
    """

    def __init__(self, hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.in_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, lengths = left_to_right_padded(self.in_drop(h), mask)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                                  # [n_layers, B, D]
        return h_n[-1]

    def step(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_n = self.gru(self.in_drop(x).unsqueeze(1), state)
        return h_n[-1], h_n
