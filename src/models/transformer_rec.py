"""Causal Transformer sequence encoder (used by SASRec, T1, T2)."""
from __future__ import annotations

import torch
import torch.nn as nn


def causal_padding_mask(mask: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Attention mask for left-padded sequences.

    mask: [B, L] bool, True = real item.
    Returns bool [B * n_heads, L, L], True = NOT allowed to attend.
    Query i may attend key j iff j <= i and j is real. Padding queries attend only to
    themselves (avoids all-masked rows / NaN); their outputs are zeroed afterwards.
    """
    B, L = mask.shape
    causal = torch.ones(L, L, dtype=torch.bool, device=mask.device).tril()  # [L, L]
    allowed = causal.unsqueeze(0) & mask.unsqueeze(1)  # [B, L(q), L(k)]
    eye = torch.eye(L, dtype=torch.bool, device=mask.device).unsqueeze(0)
    allowed = allowed | (eye & ~mask.unsqueeze(2))
    return (~allowed).repeat_interleave(n_heads, dim=0)


class CausalTransformer(nn.Module):
    """Pre-LN causal Transformer encoder with learned positional embeddings.

    forward(h [B, L, D], mask [B, L] bool) -> [B, L, D] (padding positions are zero).
    """

    def __init__(self, hidden: int, n_layers: int, n_heads: int, max_len: int,
                 dropout: float, ffn_mult: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ffn_mult * hidden, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = h.shape
        pos = torch.arange(L, device=h.device)
        x = self.drop(h + self.pos_emb(pos).unsqueeze(0))
        x = x * mask.unsqueeze(-1)
        x = self.encoder(x, mask=causal_padding_mask(mask, self.n_heads))
        return self.norm(x) * mask.unsqueeze(-1)


class SASRec(nn.Module):
    """ID-only SASRec baseline (M3). Scores against its own item embedding table.

    forward(seq [B, L]) -> {"sasrec": logits [B, N]}.
    """

    def __init__(self, n_items: int, cfg: dict, max_len: int) -> None:
        super().__init__()
        d = cfg["hidden"]
        tcfg = cfg["transformer"]
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.encoder = CausalTransformer(d, tcfg["n_layers"], tcfg["n_heads"], max_len,
                                         cfg["dropout"], tcfg["ffn_mult"])

    def user_repr(self, seq: torch.Tensor) -> torch.Tensor:
        """seq [B, L] -> u [B, D] (state at the last position = last real item)."""
        mask = seq != 0
        return self.encoder(self.item_emb(seq), mask)[:, -1]

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        u = self.user_repr(seq)
        return {"sasrec": u @ self.item_emb.weight[1:].T}

    @torch.no_grad()
    def predict(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward(seq)
