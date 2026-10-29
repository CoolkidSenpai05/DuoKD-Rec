"""BERT4Rec (Sun et al. 2019), re-implemented from RecBole 1.2 `bert4rec.py` + `MaskItemSequence`.

- Bidirectional post-LN Transformer, [MASK] token id = N+1 (own embedding row, never ranked).
- Training (cloze): the sample's sequence is prefix + target (a train interaction, never valid/test).
  With prob. `ft_ratio` per batch only the last item is masked (RecBole `_append_mask_last`), otherwise
  every real item is masked independently with prob. `mask_ratio`. CE over the masked positions.
- Eval: [MASK] appended after the history and the window cut back to L (RecBole reconstruct_test_data).
Right padding as in RecBole.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.base import BaseSeqRec, gather_last, left_to_right, mask_pad_column
from src.baselines.layers import TransformerEncoder, extended_attention_mask, recbole_init


class BERT4Rec(BaseSeqRec):
    name = "bert4rec"
    causal = False

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["bert4rec"]
        d = cfg["model"]["hidden"]
        self.n_items = data["n_items"]
        self.max_len = cfg["data"]["max_len"]
        self.mask_token = self.n_items + 1
        self.mask_ratio, self.ft_ratio = m["mask_ratio"], m["ft_ratio"]
        self.item_embedding = nn.Embedding(self.n_items + 2, d, padding_idx=0)   # + [MASK]
        self.position_embedding = nn.Embedding(self.max_len, d)
        self.trm_encoder = TransformerEncoder(m["n_layers"], m["n_heads"], d, m["inner_size"],
                                              m["hidden_dropout_prob"], m["attn_dropout_prob"],
                                              m["hidden_act"], m["layer_norm_eps"])
        self.LayerNorm = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.dropout = nn.Dropout(m["hidden_dropout_prob"])
        self.output_ffn = nn.Linear(d, d)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.output_bias = nn.Parameter(torch.zeros(self.n_items + 1))
        self.apply(lambda mod: recbole_init(mod, m["initializer_range"]))

    # ---- encoder (input right padded, may contain [MASK])
    def encode(self, seq_r: torch.Tensor) -> torch.Tensor:
        pos = self.position_embedding(torch.arange(seq_r.size(1), device=seq_r.device)).unsqueeze(0)
        x = self.dropout(self.LayerNorm(self.item_embedding(seq_r) + pos))
        h = self.trm_encoder(x, extended_attention_mask(seq_r, bidirectional=True))
        return self.output_ln(self.output_gelu(self.output_ffn(h)))          # [B, L, D]

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., D] -> [..., N+1] over padding + real items (the [MASK] row is excluded)."""
        return h @ self.item_embedding.weight[: self.n_items + 1].T + self.output_bias

    # ---- training input
    def make_train_input(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (masked right-padded sequence [B, L], labels [B, L]; 0 = not a prediction position)."""
        seq = torch.cat([batch["seq"][:, 1:], batch["target"].unsqueeze(1)], dim=1)  # prefix + target, last L
        seq_r, lengths, _ = left_to_right(seq)
        real = seq_r != 0
        if torch.rand(()).item() < self.ft_ratio:                                  # mask the last item only
            masked = torch.zeros_like(real)
            masked[torch.arange(seq_r.size(0), device=seq_r.device), lengths - 1] = True
        else:
            masked = (torch.rand(seq_r.shape, device=seq_r.device) < self.mask_ratio) & real
        labels = torch.where(masked, seq_r, torch.zeros_like(seq_r))
        inp = torch.where(masked, torch.full_like(seq_r, self.mask_token), seq_r)
        return inp, labels

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inp, labels = self.make_train_input(batch)
        sel = labels > 0
        h = self.encode(inp)[sel]                                                   # [M, D]
        if h.shape[0] == 0:                                                         # nothing masked in batch
            loss = (self.output_bias * 0).sum()
        else:
            loss = F.cross_entropy(self.logits(h)[:, 1:], labels[sel] - 1)
        return {"loss": loss, "L_ce": loss, "n_masked": torch.tensor(float(sel.sum()))}

    def eval_input(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """History + [MASK], cut to L, right padded. Returns (input [B, L], mask positions = lengths-1)."""
        seq = torch.cat([seq[:, 1:], torch.full_like(seq[:, :1], self.mask_token)], dim=1)
        seq_r, lengths, _ = left_to_right(seq)
        return seq_r, lengths

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq_r, lengths = self.eval_input(batch["seq"])
        return mask_pad_column(self.logits(gather_last(self.encode(seq_r), lengths)))
