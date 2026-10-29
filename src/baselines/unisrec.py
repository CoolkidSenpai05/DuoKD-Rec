"""UniSRec (Hou et al., KDD 2022), transductive fine-tuning variant WITHOUT pre-training, re-implemented
from github.com/RUCAIBox/UniSRec `unisrec.py` (train_stage = transductive_ft).

item_t = MoE-adaptor(PW-whitening experts)(frozen text feature);  input = item_t + id_emb + pos -> SASRec
(RecBole post-LN, causal).  Item table = MoE(text table) + id table.  logits = cos(seq, item) / temperature.
Right padding as in RecBole.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.base import BaseSeqRec, ce_real, gather_last, left_to_right, mask_pad_column, right_to_left
from src.baselines.layers import MoEAdaptor, PWLayer, TransformerEncoder, extended_attention_mask, recbole_init


class UniSRec(BaseSeqRec):
    name = "unisrec"
    modalities = ("text",)

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["unisrec"]
        d = cfg["model"]["hidden"]
        assert not m.get("pretrain", False), "pre-training stage is not part of this benchmark (pretrain: false)"
        self.n_items = data["n_items"]
        self.temperature = m["temperature"]
        # --- RecBole SASRec base
        self.item_embedding = nn.Embedding(self.n_items + 1, d, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg["data"]["max_len"], d)
        self.trm_encoder = TransformerEncoder(m["n_layers"], m["n_heads"], d, m["inner_size"],
                                              m["hidden_dropout_prob"], m["attn_dropout_prob"],
                                              m["hidden_act"], m["layer_norm_eps"])
        self.LayerNorm = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.dropout = nn.Dropout(m["hidden_dropout_prob"])
        self.apply(lambda mod: recbole_init(mod, m["initializer_range"]))
        # --- UniSRec additions (frozen text features are a non-persistent buffer: not in state_dict)
        self.register_buffer("plm_embedding", feats["text"], persistent=False)
        d_plm = feats["text"].shape[1]
        self.moe_adaptor = MoEAdaptor([PWLayer(d_plm, d, m["adaptor_dropout_prob"]) for _ in range(m["n_exps"])], d_plm)

    def _encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_r, lengths, _ = left_to_right(seq)
        item_emb = self.moe_adaptor(self.plm_embedding[seq_r])
        pos = self.position_embedding(torch.arange(seq_r.size(1), device=seq.device)).unsqueeze(0)
        x = self.dropout(self.LayerNorm(item_emb + pos + self.item_embedding(seq_r)))
        return self.trm_encoder(x, extended_attention_mask(seq_r)), lengths

    def _item_table(self) -> torch.Tensor:
        return self.moe_adaptor(self.plm_embedding) + self.item_embedding.weight      # [N+1, D]

    def _cos(self, seq: torch.Tensor) -> torch.Tensor:
        h, lengths = self._encode(seq)
        s = F.normalize(gather_last(h, lengths), dim=-1)
        return s @ F.normalize(self._item_table(), dim=-1).T                         # [B, N+1]

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss = ce_real(self._cos(batch["seq"]) / self.temperature, batch["target"])
        return {"loss": loss, "L_ce": loss}

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return mask_pad_column(self._cos(batch["seq"]))

    def position_outputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h, lengths = self._encode(batch["seq"])
        return right_to_left(h, lengths)
