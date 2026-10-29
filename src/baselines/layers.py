"""Shared layers re-implemented from RecBole 1.2 (third_party/RecBole/recbole/model/layers.py).

Post-LN Transformer (MultiHeadAttention + FeedForward, residual + LayerNorm after each sub-layer),
additive -10000 attention masks, exactly as RecBole's SASRec / BERT4Rec / UniSRec / TedRec / HM4SR / DDMSR use.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Erf GELU as written in RecBole / BSARec."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT = {"gelu": gelu, "relu": F.relu, "swish": lambda x: x * torch.sigmoid(x), "tanh": torch.tanh,
       "sigmoid": torch.sigmoid}


def extended_attention_mask(seq: torch.Tensor, bidirectional: bool = False) -> torch.Tensor:
    """RecBole get_attention_mask: seq [B, L] -> additive mask [B, 1, L, L] (0 allowed, -10000 blocked).

    Key j is visible to query i iff seq[:, j] != 0 (and j <= i unless bidirectional).
    """
    m = (seq != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    if not bidirectional:
        m = torch.tril(m.expand(-1, -1, seq.size(-1), -1))
    return torch.where(m, 0.0, -10000.0)


class MultiHeadAttention(nn.Module):
    """RecBole MultiHeadAttention (post-LN, residual inside). x [B, L, D], mask [B, 1, L, L] -> [B, L, D]."""

    def __init__(self, n_heads: int, hidden: int, hidden_dropout: float, attn_dropout: float, eps: float) -> None:
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads, self.head = n_heads, hidden // n_heads
        self.sqrt_head = math.sqrt(self.head)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.out_dropout = nn.Dropout(hidden_dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.n_heads, self.head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self._split(self.query(x)).permute(0, 2, 1, 3)
        k = self._split(self.key(x)).permute(0, 2, 3, 1)
        v = self._split(self.value(x)).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k) / self.sqrt_head + mask
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(*ctx.shape[:-2], self.n_heads * self.head)
        return self.LayerNorm(self.out_dropout(self.dense(ctx)) + x)


class FeedForward(nn.Module):
    """RecBole FeedForward: Linear-act-Linear, dropout, residual + LayerNorm."""

    def __init__(self, hidden: int, inner: int, dropout: float, act: str, eps: float) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(hidden, inner)
        self.act = ACT[act]
        self.dense_2 = nn.Linear(inner, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.dense_2(self.act(self.dense_1(x))))
        return self.LayerNorm(h + x)


class TransformerLayer(nn.Module):
    def __init__(self, n_heads: int, hidden: int, inner: int, hidden_dropout: float, attn_dropout: float,
                 act: str, eps: float) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, hidden, hidden_dropout, attn_dropout, eps)
        self.feed_forward = FeedForward(hidden, inner, hidden_dropout, act, eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.multi_head_attention(x, mask))


class TransformerEncoder(nn.Module):
    """RecBole TransformerEncoder. forward(x [B, L, D], mask [B, 1, L, L]) -> last layer [B, L, D]."""

    def __init__(self, n_layers: int = 2, n_heads: int = 2, hidden: int = 64, inner: int = 256,
                 hidden_dropout: float = 0.5, attn_dropout: float = 0.5, act: str = "gelu",
                 eps: float = 1e-12) -> None:
        super().__init__()
        layer = TransformerLayer(n_heads, hidden, inner, hidden_dropout, attn_dropout, act, eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x, mask)
        return x


def recbole_init(module: nn.Module, std: float = 0.02) -> None:
    """RecBole SASRec._init_weights: Linear/Embedding ~ N(0, std), LayerNorm (1, 0), Linear bias 0."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class PWLayer(nn.Module):
    """UniSRec parametric whitening: lin(dropout(x) - b). `bias_shape` (TedRec DTR) defaults to [in]."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0, bias_shape: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(*(bias_shape or (d_in,))))
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.lin.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptor(nn.Module):
    """UniSRec MoE adaptor: softmax(noisy gate) weighted sum of experts. x [..., d_in] -> [..., d_out].

    Noise (softplus(x W_noise) + 1e-2) * N(0, 1) is added to the gate logits only in training.
    """

    def __init__(self, experts: list[nn.Module], d_in: int, noise: bool = True) -> None:
        super().__init__()
        self.n_exps = len(experts)
        self.noisy_gating = noise
        self.experts = nn.ModuleList(experts)
        self.w_gate = nn.Parameter(torch.zeros(d_in, self.n_exps))
        self.w_noise = nn.Parameter(torch.zeros(d_in, self.n_exps))

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        logits = x @ self.w_gate
        if self.noisy_gating and self.training:
            std = F.softplus(x @ self.w_noise) + 1e-2
            logits = logits + torch.randn_like(logits) * std
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gates(x)
        out = torch.cat([e(x).unsqueeze(-2) for e in self.experts], dim=-2)
        return (g.unsqueeze(-1) * out).sum(dim=-2)
