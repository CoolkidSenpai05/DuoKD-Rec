"""SASRec (M3) behind the BaseSeqRec interface.

Subclass of the M3 module, so parameters, initialisation order, state_dict keys (old checkpoints load)
and every floating-point operation are unchanged: training/eval is bit-identical to the M3 path.
"""
from __future__ import annotations

import torch

from src.baselines.base import BaseSeqRec, with_pad_column
from src.losses import ce_loss
from src.models.transformer_rec import SASRec


class SASRecRec(SASRec, BaseSeqRec):
    name = "sasrec"

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__(data["n_items"], cfg["model"], cfg["data"]["max_len"])

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.forward(batch["seq"])["sasrec"]
        loss = ce_loss(logits, batch["target"])
        return {"loss": loss, "L_sup_sasrec": loss}

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return with_pad_column(self.forward(batch["seq"])["sasrec"])

    def position_outputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["seq"]
        return self.encoder(self.item_emb(seq), seq != 0)
