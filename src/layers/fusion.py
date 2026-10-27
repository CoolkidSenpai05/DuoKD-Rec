"""Gated fusion of the two modality views (CLAUDE.md §2)."""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """g = sigmoid(W[h_t; h_v]), h_f = Proj_f(g*h_t + (1-g)*h_v).

    h_t, h_v: [..., D] -> h_f [..., D]. `Proj_f` (Linear) is the GRU branch's own projection
    so that E_f stays a learnable function of GRU-side parameters (§3.5).
    """

    def __init__(self, hidden: int, out_proj: bool = True) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * hidden, hidden)
        self.proj = nn.Linear(hidden, hidden) if out_proj else nn.Identity()

    def forward(self, h_t: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_v], dim=-1)))
        return self.proj(g * h_t + (1 - g) * h_v)
