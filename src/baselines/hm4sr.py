"""HM4SR (Zhang et al., WWW 2025), re-implemented from github.com/SStarCCat/HM4SR `recbole_model/HM4SR.py`.

Three streams (ID / text / image), each: emb + pos -> Interactive (Align) MoE residual -> Temporal MoE
(interval + absolute time embeddings route over fixed experts) -> LN -> dropout -> SASRec encoder (causal).
score = s_id @ E_id^T + s_txt @ P_t(T)^T + s_img @ P_v(V)^T.
loss  = CE + IDCL (UniSRec seq-item contrast) + CP (category BCE) + PCL (placeholder contrast).

Faithfulness notes (see docs/baselines.md):
- timestamps are the INPUT items' times (RecBole `timestamp_list`); the target's time is never used
  (`use_target_time` must stay false). They are cast to float32 like RecBole FLOAT fields.
- the official temporal experts live in a plain Python list, so they are never trained: kept here as a
  fixed buffer (saved in the checkpoint so save/load is reproducible).
- RecBole right padding makes log2(0 - t_last + 1) = NaN at the first padding slot; intervals are clamped
  at 0 there (padding positions are masked out anyway).
- NaN-free placeholders etc. follow the official code; contrastive "same target" masking keeps exp(0) = 1
  terms exactly like the official torch.where(..., 0, logits).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.base import BaseSeqRec, ce_real, gather_last, left_to_right, mask_pad_column, right_to_left
from src.baselines.layers import TransformerEncoder, extended_attention_mask, recbole_init


def max_log_interval(data: dict) -> float:
    """log2(max consecutive gap + 1) over the model INPUTS (train sequences + valid item); never test."""
    best = 0
    for tr, v in zip(data["train_ts"], data["valid_ts"]):
        ts = list(tr) + [v]
        if len(ts) > 1:
            best = max(best, int(np.max(np.diff(ts))))
    return math.log2(best + 1)


class AlignMoE(nn.Module):
    """Official Align_MoE (softmax gate per modality, shared experts on the 3D concat)."""

    def __init__(self, d: int, n_exp: int, init_weight: list[float]) -> None:
        super().__init__()
        self.d = d
        self.gate_txt = nn.Linear(d, n_exp)
        self.gate_img = nn.Linear(d, n_exp)
        self.gate_id = nn.Linear(d, n_exp)
        self.expert = nn.ModuleList([nn.Linear(3 * d, 3 * d) for _ in range(n_exp)])
        self.weight = nn.Parameter(torch.tensor(init_weight, dtype=torch.float))

    def forward(self, v: torch.Tensor) -> list[torch.Tensor]:
        d = self.d
        e = torch.cat([ex(v).unsqueeze(2) for ex in self.expert], dim=2)            # [B, L, E, 3D]
        g = [F.softmax(self.gate_id(v[..., :d]), -1), F.softmax(self.gate_txt(v[..., d:2 * d]), -1),
             F.softmax(self.gate_img(v[..., 2 * d:]), -1)]
        return [self.weight[k] * torch.sum(e[..., k * d:(k + 1) * d] * g[k].unsqueeze(3), dim=2) for k in range(3)]


class TemporalMoE(nn.Module):
    """Official Temporal_MoE_C (softmax routing over fixed, untrained experts)."""

    def __init__(self, d: int, n_exp: int, interval_scale: float, interval_num: float) -> None:
        super().__init__()
        self.d, self.scale = d, interval_scale
        self.gate = nn.Linear(2 * d, n_exp)
        self.absolute_w = nn.Linear(1, d)
        self.absolute_m = nn.Linear(d, d)  # defined (and unused) in the official code
        self.n_interval = int(interval_scale * interval_num) + 1
        self.time_embedding = nn.Embedding(self.n_interval, d)
        self.register_buffer("expert", torch.randn(n_exp, 1, 3 * d) * 0.1)          # never trained
        self.register_buffer("inv_freq", 1 / torch.pow(10000, torch.arange(0, d, 1.0) / d).view(1, -1))

    def time_embedding_of(self, ts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ts [B, L] float32 right padded -> cat[interval_emb, absolute_emb] [B, L, 2D]."""
        absolute = torch.cos(self.absolute_w(ts.unsqueeze(2)) * self.inv_freq)
        diff = (ts[:, 1:] - ts[:, :-1]).clamp_min(0)                                  # boundary NaN -> 0
        idx = torch.floor(self.scale * torch.log2(diff + 1)).long().clamp(0, self.n_interval - 1)
        pos = torch.arange(1, ts.shape[1], device=ts.device).unsqueeze(0)
        idx = torch.where(pos < lengths.unsqueeze(1), idx, torch.zeros_like(idx))
        idx = torch.cat([torch.zeros_like(idx[:, :1]), idx], dim=1)
        return torch.cat([self.time_embedding(idx), absolute], dim=-1)

    def forward(self, v: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route = F.softmax(self.gate(temb), dim=-1)                                     # [B, L, E]
        out = torch.sum(torch.stack([v * e for e in self.expert], dim=2) * route.unsqueeze(3), dim=2)
        d = self.d
        return out[..., :d], out[..., d:2 * d], out[..., 2 * d:]


class HM4SR(BaseSeqRec):
    name = "hm4sr"
    modalities = ("text", "image", "category")
    needs_timestamps = True

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["hm4sr"]
        d = cfg["model"]["hidden"]
        L = cfg["data"]["max_len"]
        if m.get("use_target_time", False):
            raise ValueError("official HM4SR never reads the target timestamp; use_target_time must be false")
        self.n_items = data["n_items"]
        self.temperature, self.beta = m["temperature"], m["beta"]
        self.phcl_temperature, self.phcl_weight = m["phcl_temperature"], m["phcl_weight"]
        self.item_embedding = nn.Embedding(self.n_items + 1, d, padding_idx=0)
        self.position_embedding = nn.Embedding(L, d)
        enc = lambda: TransformerEncoder(m["n_layers"], m["n_heads"], d, m["inner_size"],  # noqa: E731
                                         m["hidden_dropout_prob"], m["attn_dropout_prob"],
                                         m["hidden_act"], m["layer_norm_eps"])
        self.item_seq, self.txt_seq, self.img_seq = enc(), enc(), enc()
        self.item_ln = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.txt_ln = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.img_ln = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.txt_projection = nn.Linear(feats["text"].shape[1], d)
        self.img_projection = nn.Linear(feats["image"].shape[1], d)
        self.dropout = nn.Dropout(m["hidden_dropout_prob"])
        interval_num = m.get("interval_num") or max_log_interval(data)
        self.time_moe = TemporalMoE(d, m["temporal_expert_num"], m["interval_scale"], interval_num)
        self.apply(lambda mod: recbole_init(mod, m["initializer_range"]))
        self.register_buffer("txt_embedding", feats["text"], persistent=False)
        self.register_buffer("img_embedding", feats["image"], persistent=False)
        self.register_buffer("cat_embedding", feats["category"].float(), persistent=False)
        self.cat_linear = nn.Linear(3 * d, self.cat_embedding.shape[-1])
        self.start_moe = AlignMoE(d, m["start_expert_num"], m["initializer_weight"])
        self.placeholder_txt = nn.Linear(2 * d, d)
        self.placeholder_img = nn.Linear(2 * d, d)

    # ------------------------------------------------------------------ forward
    def _streams(self, seq_r: torch.Tensor, lengths: torch.Tensor, ts_r: torch.Tensor):
        item = self.item_embedding(seq_r)
        txt = self.txt_projection(self.txt_embedding[seq_r])
        img = self.img_projection(self.img_embedding[seq_r])
        pos = self.position_embedding.weight[: seq_r.shape[1]].unsqueeze(0)
        item, txt, img = item + pos, txt + pos, img + pos
        a = self.start_moe(torch.cat([item, txt, img], dim=-1))
        item, txt, img = item + a[0], txt + a[1], img + a[2]
        temb = self.time_moe.time_embedding_of(ts_r, lengths)
        item, txt, img = self.time_moe(torch.cat([item, txt, img], dim=-1), temb)
        mask = extended_attention_mask(seq_r)
        hs = [enc(self.dropout(ln(x)), mask) for enc, ln, x in
              ((self.item_seq, self.item_ln, item), (self.txt_seq, self.txt_ln, txt), (self.img_seq, self.img_ln, img))]
        return (item, txt, img), hs, temb, mask

    def _inputs(self, batch: dict[str, torch.Tensor]):
        seq_r, lengths, (ts_r,) = left_to_right(batch["seq"], batch["ts"].float())
        return seq_r, lengths, ts_r

    def _score(self, seqs: list[torch.Tensor]) -> torch.Tensor:
        return (seqs[0] @ self.item_embedding.weight.T + seqs[1] @ self.txt_projection(self.txt_embedding).T
                + seqs[2] @ self.img_projection(self.img_embedding).T)                 # [B, N+1]

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq_r, lengths, ts_r = self._inputs(batch)
        _, hs, _, _ = self._streams(seq_r, lengths, ts_r)
        return mask_pad_column(self._score([gather_last(h, lengths) for h in hs]))

    def position_outputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq_r, lengths, ts_r = self._inputs(batch)
        _, hs, _, _ = self._streams(seq_r, lengths, ts_r)
        return right_to_left(torch.cat(hs, dim=-1), lengths)

    # ------------------------------------------------------------------ losses
    @staticmethod
    def _same_target(pos_id: torch.Tensor) -> torch.Tensor:
        same = pos_id.unsqueeze(1) == pos_id.unsqueeze(0)
        return torch.logical_xor(same, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

    def _contrast(self, a: torch.Tensor, b: torch.Tensor, same: torch.Tensor, temp: float) -> torch.Tensor:
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        pos = torch.exp((a * b).sum(dim=1) / temp)
        neg = torch.where(same, torch.zeros((), device=a.device), a @ b.T / temp)
        return -torch.log(pos / torch.exp(neg).sum(dim=1)).mean()

    def _cp(self, seq_r: torch.Tensor) -> torch.Tensor:
        items = seq_r.flatten()
        nz = items != 0
        x = torch.cat([self.item_embedding(items), self.txt_projection(self.txt_embedding[items]),
                       self.img_projection(self.img_embedding[items])], dim=-1)
        return F.binary_cross_entropy_with_logits(self.cat_linear(x)[nz], self.cat_embedding[items][nz])

    def _pcl(self, streams, seqs, temb, mask, lengths, same) -> torch.Tensor:
        B, L = temb.shape[:2]
        n_mask = torch.floor(lengths.float() * self.beta).long()                     # floor(len * beta)
        r = torch.rand(B, L, device=temb.device)
        r = r.masked_fill(torch.arange(L, device=temb.device).unsqueeze(0) >= lengths.unsqueeze(1), 2.0)
        rank = r.argsort(dim=1).argsort(dim=1)                                        # random order of real slots
        ph = (rank < n_mask.unsqueeze(1)).unsqueeze(2)                                # [B, L, 1]
        txt = streams[1].masked_fill(ph, 0.0) + self.placeholder_txt(temb).masked_fill(~ph, 0.0)
        img = streams[2].masked_fill(ph, 0.0) + self.placeholder_img(temb).masked_fill(~ph, 0.0)
        t = gather_last(self.txt_seq(self.dropout(self.txt_ln(txt)), mask), lengths)
        v = gather_last(self.img_seq(self.dropout(self.img_ln(img)), mask), lengths)
        lt = self._contrast(seqs[1], t, same, self.phcl_temperature) * self.phcl_weight
        lv = self._contrast(seqs[2], v, same, self.phcl_temperature) * self.phcl_weight
        return (lt + lv) / 2

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seq_r, lengths, ts_r = self._inputs(batch)
        streams, hs, temb, mask = self._streams(seq_r, lengths, ts_r)
        seqs = [gather_last(h, lengths) for h in hs]
        ce = ce_real(self._score(seqs), batch["target"])
        same = self._same_target(batch["target"])
        idcl = self._contrast(seqs[0], self.item_embedding(batch["target"]), same, self.temperature)
        cp = self._cp(seq_r)
        pcl = self._pcl(streams, seqs, temb, mask, lengths, same)
        return {"loss": ce + idcl + cp + pcl, "L_ce": ce, "L_idcl": idcl, "L_cp": cp, "L_pcl": pcl}
