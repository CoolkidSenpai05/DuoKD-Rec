"""TedRec (Xu et al., CIKM 2024), re-implemented from github.com/RUCAIBox/TedRec `tedrec.py`.

text_seq = MoE adaptor with DTR experts (position-wise whitening bias [1, L, d_plm]) on frozen text features;
contextual convolution in the frequency domain:
    item_conv   = iFFT(FFT(id_seq) * W)            (learnable complex filter W)
    fusion_conv = iFFT(FFT(text_seq) * FFT(id_seq))
    x = 2 * (item_conv * sigmoid(g_i(item_conv)) + fusion_conv * sigmoid(g_f(fusion_conv)))
then + pos -> LN -> dropout -> SASRec (RecBole, causal). Items are scored with the ID table only:
logits = cos(seq, id_emb) / temperature. Right padding as in RecBole; the FFT covers the whole window,
so the model is neither causal nor padding invariant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.base import BaseSeqRec, ce_real, gather_last, left_to_right, mask_pad_column
from src.baselines.layers import MoEAdaptor, PWLayer, TransformerEncoder, extended_attention_mask, recbole_init


class TedRec(BaseSeqRec):
    name = "tedrec"
    modalities = ("text",)
    causal = False
    padding_invariant = False

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["tedrec"]
        d = cfg["model"]["hidden"]
        L = cfg["data"]["max_len"]
        self.n_items = data["n_items"]
        self.temperature = m["temperature"]
        self.item_embedding = nn.Embedding(self.n_items + 1, d, padding_idx=0)
        self.position_embedding = nn.Embedding(L, d)
        self.trm_encoder = TransformerEncoder(m["n_layers"], m["n_heads"], d, m["inner_size"],
                                              m["hidden_dropout_prob"], m["attn_dropout_prob"],
                                              m["hidden_act"], m["layer_norm_eps"])
        self.LayerNorm = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.dropout = nn.Dropout(m["hidden_dropout_prob"])
        self.apply(lambda mod: recbole_init(mod, m["initializer_range"]))
        self.register_buffer("plm_embedding", feats["text"], persistent=False)
        d_plm = feats["text"].shape[1]
        self.item_gating = nn.Linear(d, 1)
        self.fusion_gating = nn.Linear(d, 1)
        self.moe_adaptor = MoEAdaptor(
            [PWLayer(d_plm, d, m["adaptor_dropout_prob"], bias_shape=(1, L, d_plm)) for _ in range(m["n_exps"])], d_plm)
        self.complex_weight = nn.Parameter(torch.randn(1, L // 2 + 1, d, 2) * 0.02)
        self.item_gating.weight.data.normal_(mean=0, std=0.02)
        self.fusion_gating.weight.data.normal_(mean=0, std=0.02)

    def contextual_convolution(self, item_emb: torch.Tensor, feature_emb: torch.Tensor) -> torch.Tensor:
        L = feature_emb.shape[1]
        feature_fft = torch.fft.rfft(feature_emb, dim=1, norm="ortho")
        item_fft = torch.fft.rfft(item_emb, dim=1, norm="ortho")
        w = torch.view_as_complex(self.complex_weight)
        item_conv = torch.fft.irfft(item_fft * w, n=L, dim=1, norm="ortho")
        fusion_conv = torch.fft.irfft(feature_fft * item_fft, n=L, dim=1, norm="ortho")
        return 2 * (item_conv * torch.sigmoid(self.item_gating(item_conv))
                    + fusion_conv * torch.sigmoid(self.fusion_gating(fusion_conv)))

    def user_repr(self, seq: torch.Tensor) -> torch.Tensor:
        seq_r, lengths, _ = left_to_right(seq)
        text = self.moe_adaptor(self.plm_embedding[seq_r])
        x = self.contextual_convolution(self.item_embedding(seq_r), text)
        x = x + self.position_embedding(torch.arange(seq_r.size(1), device=seq.device)).unsqueeze(0)
        h = self.trm_encoder(self.dropout(self.LayerNorm(x)), extended_attention_mask(seq_r))
        return gather_last(h, lengths)

    def _cos(self, seq: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_repr(seq), dim=-1) @ F.normalize(self.item_embedding.weight, dim=-1).T

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss = ce_real(self._cos(batch["seq"]) / self.temperature, batch["target"])
        return {"loss": loss, "L_ce": loss}

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return mask_pad_column(self._cos(batch["seq"]))
