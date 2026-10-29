"""BSARec (Shin et al., AAAI 2024), re-implemented from github.com/yehjin-shin/BSARec `src/model/bsarec.py`.

Each block: h = alpha * FrequencyLayer(x) + (1 - alpha) * SelfAttention(x), then FeedForward.
FrequencyLayer: rFFT over the sequence axis, keep the lowest c//2+1 bins (low pass), high = x - low,
out = LN(dropout(low + beta^2 * high) + x). The official code pads on the LEFT (same as this project) and
filters the whole L window including padding, so the model is neither causal nor padding invariant.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.baselines.base import BaseSeqRec, ce_real, mask_pad_column
from src.baselines.layers import FeedForward, MultiHeadAttention, extended_attention_mask, recbole_init


class FrequencyLayer(nn.Module):
    def __init__(self, hidden: int, c: int, dropout: float) -> None:
        super().__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        f = torch.fft.rfft(x, dim=1, norm="ortho")
        low = f.clone()
        low[:, self.c:, :] = 0
        low = torch.fft.irfft(low, n=L, dim=1, norm="ortho")
        high = x - low
        h = self.out_dropout(low + self.sqrt_beta ** 2 * high)
        return self.LayerNorm(h + x)


class BSARecBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, c: int, alpha: float, hidden_dropout: float,
                 attn_dropout: float, act: str) -> None:
        super().__init__()
        self.filter_layer = FrequencyLayer(hidden, c, hidden_dropout)
        self.attention_layer = MultiHeadAttention(n_heads, hidden, hidden_dropout, attn_dropout, 1e-12)
        self.alpha = alpha
        self.feed_forward = FeedForward(hidden, 4 * hidden, hidden_dropout, act, 1e-12)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.alpha * self.filter_layer(x) + (1 - self.alpha) * self.attention_layer(x, mask)
        return self.feed_forward(h)


class BSARec(BaseSeqRec):
    name = "bsarec"
    causal = False            # FFT over the whole window
    padding_invariant = False  # padding positions (pos. embedding) enter the FFT

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["bsarec"]
        d = cfg["model"]["hidden"]
        self.n_items = data["n_items"]
        L = cfg["data"]["max_len"]
        self.item_embeddings = nn.Embedding(self.n_items + 1, d, padding_idx=0)
        self.position_embeddings = nn.Embedding(L, d)
        self.LayerNorm = nn.LayerNorm(d, eps=1e-12)
        self.dropout = nn.Dropout(m["hidden_dropout_prob"])
        block = BSARecBlock(d, m["num_attention_heads"], m["c"], m["alpha"], m["hidden_dropout_prob"],
                            m["attention_probs_dropout_prob"], m["hidden_act"])
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(m["num_hidden_layers"])])
        self.apply(lambda mod: recbole_init(mod, m["initializer_range"]))

    def user_repr(self, seq: torch.Tensor) -> torch.Tensor:
        pos = self.position_embeddings(torch.arange(seq.size(1), device=seq.device)).unsqueeze(0)
        x = self.dropout(self.LayerNorm(self.item_embeddings(seq) + pos))
        mask = extended_attention_mask(seq)          # causal + padding, left padded as in the official code
        for blk in self.blocks:
            x = blk(x, mask)
        return x[:, -1]

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss = ce_real(self.user_repr(batch["seq"]) @ self.item_embeddings.weight.T, batch["target"])
        return {"loss": loss, "L_ce": loss}

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return mask_pad_column(self.user_repr(batch["seq"]) @ self.item_embeddings.weight.T)
