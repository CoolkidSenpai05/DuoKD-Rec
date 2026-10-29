"""GRU4Rec (Hidasi et al. 2016), re-implemented from RecBole 1.2 `sequential_recommender/gru4rec.py`.

item emb (xavier normal) -> dropout -> GRU (no bias) -> Linear(hidden -> D) -> state at the last real item;
scores = state @ item_emb^T. RecBole pads on the right, so the sequence is converted to right padding and
the state is read at position len-1 (trailing padding never reaches it: same as packing in gru_rec.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_

from src.baselines.base import BaseSeqRec, ce_real, gather_last, left_to_right, mask_pad_column, right_to_left


class GRU4Rec(BaseSeqRec):
    name = "gru4rec"

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["gru4rec"]
        d = cfg["model"]["hidden"]
        self.n_items = data["n_items"]
        self.item_embedding = nn.Embedding(self.n_items + 1, d, padding_idx=0)
        self.emb_dropout = nn.Dropout(m["dropout"])
        self.gru_layers = nn.GRU(d, m["hidden_size"], m["num_layers"], bias=False, batch_first=True)
        self.dense = nn.Linear(m["hidden_size"], d)
        xavier_normal_(self.item_embedding.weight)
        xavier_uniform_(self.gru_layers.weight_hh_l0)
        xavier_uniform_(self.gru_layers.weight_ih_l0)

    def _states(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_r, lengths, _ = left_to_right(seq)
        out, _ = self.gru_layers(self.emb_dropout(self.item_embedding(seq_r)))
        return self.dense(out), lengths                                # [B, L, D] right padded

    def user_repr(self, seq: torch.Tensor) -> torch.Tensor:
        out, lengths = self._states(seq)
        return gather_last(out, lengths)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.user_repr(batch["seq"]) @ self.item_embedding.weight.T
        loss = ce_real(logits, batch["target"])
        return {"loss": loss, "L_ce": loss}

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return mask_pad_column(self.user_repr(batch["seq"]) @ self.item_embedding.weight.T)

    def position_outputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out, lengths = self._states(batch["seq"])
        return right_to_left(out, lengths)
