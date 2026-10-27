"""Mixture-of-Experts layer (CLAUDE.md §2, §3.2)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """Soft / top-k MoE of 2-layer MLP experts with a linear+softmax router.

    forward(x [M, d_in]) -> (y [M, d_out], probs [M, E], gates [M, E])
      probs: router softmax (used by the balance loss)
      gates: weights actually applied (= probs for soft MoE; renormalised top-k otherwise)
    `plain_mlp=True` replaces the MoE by a single MLP (ablation); probs/gates are then [M, 1] ones.
    """

    def __init__(self, d_in: int, d_out: int, n_experts: int, d_hidden: int,
                 top_k: int | None = None, dropout: float = 0.0, plain_mlp: bool = False) -> None:
        super().__init__()
        self.n_experts = 1 if plain_mlp else n_experts
        self.top_k = None if plain_mlp else top_k
        if self.top_k is not None:
            assert 1 <= self.top_k <= n_experts
        E = self.n_experts
        self.router = None if plain_mlp else nn.Linear(d_in, E)
        # stacked expert weights: expert k = W2_k · GELU(W1_k x + b1_k) + b2_k
        self.w1 = nn.Parameter(torch.empty(E, d_in, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(E, d_hidden))
        self.w2 = nn.Parameter(torch.empty(E, d_hidden, d_out))
        self.b2 = nn.Parameter(torch.zeros(E, d_out))
        for w in (self.w1, self.w2):
            for k in range(E):
                nn.init.kaiming_uniform_(w.data[k].T, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [M, d_in] -> (probs [M, E], gates [M, E])."""
        if self.router is None:
            ones = x.new_ones(x.shape[0], 1)
            return ones, ones
        probs = F.softmax(self.router(x), dim=-1)
        if self.top_k is None:
            return probs, probs
        val, idx = probs.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, idx, val)
        return probs, gates / gates.sum(-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, gates = self.route(x)
        h = torch.einsum("md,edh->meh", x, self.w1) + self.b1  # [M, E, H]
        h = self.drop(F.gelu(h))
        y = torch.einsum("meh,eho->meo", h, self.w2) + self.b2  # [M, E, d_out]
        return (gates.unsqueeze(-1) * y).sum(1), probs, gates
