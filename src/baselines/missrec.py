"""MISSRec (Wang et al., ACM MM 2023), transductive fine-tuning WITHOUT pre-training, re-implemented from
github.com/gimpong/MM23-MISSRec (`missrec.py`, `model_utils.py`, `trainer.py`, `cluster_utils.py`).

Config of the official finetune: modal_type img_text, id_type id, seq_mm_fusion add, item_mm_fusion
dynamic_shared.
- Interest discovery before every training epoch: k-means over the adapted text + image vectors of all
  items (K = ceil(interest_ratio * 2N)); every item gets one interest id per modality.
- Encoder (non-causal) over the user's unique interest tokens (centroids, a buffer);
  decoder (self-attention NON-causal, `is_casual=False` in the official code) over
  text + image + ID + position embeddings, cross-attending the interest memory.
- logits = (cos_id + 2 * sum_m softmax(f * cos_m) * cos_m) / 3 / temperature (dynamic_shared fusion).
- loss = CE + gamma * interest orthogonality regulariser.
k-means: the official torchpq MinibatchKMeans run is reproduced: random init from the first minibatch,
one pass (the official tol check compares the updated centroids with themselves -> error 0 -> stop),
empty clusters -> 0 (torchpq NaN -> 0), euclidean distance. Right padding as in RecBole.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.base import BaseSeqRec, ce_real, gather_last, left_to_right, mask_pad_column


def _assign(x: torch.Tensor, c: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Nearest centroid (euclidean). x [n, d], c [k, d] -> labels [n]."""
    c2 = (c * c).sum(1)
    out = []
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        out.append(((xs * xs).sum(1, keepdim=True) - 2 * xs @ c.T + c2).argmin(1))
    return torch.cat(out)


def minibatch_kmeans_once(tokens: torch.Tensor, k: int, batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """cluster_utils.cluster_kmeans + torchpq MinibatchKMeans as they behave in the official run."""
    if batch_size <= k:
        batch_size = int(k * 1.1)
    centroids, counts = None, torch.ones(k, device=tokens.device)
    for ptr in range(0, len(tokens), batch_size):
        remaining = len(tokens) - ptr - batch_size
        offset = batch_size + remaining if remaining <= k else batch_size
        batch = tokens[ptr:ptr + offset]
        if centroids is None:
            centroids = batch[torch.from_numpy(np.random.choice(len(batch), size=k, replace=False)).to(tokens.device)].clone()
        labels = _assign(batch, centroids)
        sums = torch.zeros_like(centroids).index_add_(0, labels, batch)
        n = torch.bincount(labels, minlength=k).float()
        delta = torch.where(n.unsqueeze(1) > 0, sums / n.clamp_min(1).unsqueeze(1), torch.zeros_like(sums))
        w = (1 / counts).unsqueeze(1)
        centroids = centroids * (1 - w) + delta * w
        counts = counts + n
        if remaining <= k:
            break
    labels = torch.cat([_assign(tokens[p:p + batch_size], centroids) for p in range(0, len(tokens), batch_size)])
    return labels, centroids


def _init(module: nn.Module, std: float) -> None:
    """model_utils.Transformer._init_weights."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class MISSRec(BaseSeqRec):
    name = "missrec"
    modalities = ("text", "image")
    causal = False   # decoder self-attention is bidirectional in the official code

    def __init__(self, cfg: dict, data: dict, feats: dict) -> None:
        super().__init__()
        m = cfg["model"]["missrec"]
        d = cfg["model"]["hidden"]
        assert not m.get("pretrain", False), "pre-training stage is not part of this benchmark (pretrain: false)"
        assert m["n_layers"] % 2 == 0
        self.n_items = data["n_items"]
        self.temperature, self.gamma = m["temperature"], m["gamma"]
        self.item_embedding = nn.Embedding(self.n_items + 1, d, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg["data"]["max_len"], d)
        self.trm_model = nn.Transformer(d_model=d, nhead=m["n_heads"], num_encoder_layers=m["n_layers"] // 2,
                                        num_decoder_layers=m["n_layers"] // 2, dim_feedforward=m["inner_size"],
                                        dropout=m["hidden_dropout_prob"], activation=m["hidden_act"],
                                        layer_norm_eps=m["layer_norm_eps"], batch_first=True, norm_first=False)
        self.LayerNorm = nn.LayerNorm(d, eps=m["layer_norm_eps"])
        self.dropout = nn.Dropout(m["hidden_dropout_prob"])
        self.apply(lambda mod: _init(mod, m["initializer_range"]))
        self.register_buffer("plm_embedding", feats["text"], persistent=False)
        self.register_buffer("img_embedding", feats["image"], persistent=False)
        n_tok = 2 * self.n_items
        self.num_interest = max(math.ceil(n_tok * m["interest_ratio"]), 1)
        self.register_buffer("interest_embeddings", torch.zeros(self.num_interest + 1, d))
        self.register_buffer("plm_interest_lookup_table", torch.zeros(self.n_items + 1, dtype=torch.long))
        self.register_buffer("img_interest_lookup_table", torch.zeros(self.n_items + 1, dtype=torch.long))
        self.register_buffer("clustered", torch.zeros((), dtype=torch.bool))
        self.fusion_factor = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.text_adaptor = nn.Linear(feats["text"].shape[1], d)
        self.img_adaptor = nn.Linear(feats["image"].shape[1], d)

    # ---- interest discovery (official: before every training epoch)
    @torch.no_grad()
    def discover_interests(self) -> None:
        t = self.text_adaptor(self.plm_embedding[1:])
        v = self.img_adaptor(self.img_embedding[1:])
        labels, centroids = minibatch_kmeans_once(torch.cat([t, v], dim=0), self.num_interest)
        labels = labels + 1
        self.interest_embeddings[1:] = centroids
        self.plm_interest_lookup_table[1:] = labels[: self.n_items]
        self.img_interest_lookup_table[1:] = labels[self.n_items:]
        self.clustered.fill_(True)

    def on_epoch_start(self, epoch: int) -> None:
        was = self.training
        self.eval()
        self.discover_interests()
        self.train(was)

    # ---- encoder / decoder
    def _unique_interests(self, seq_r: torch.Tensor) -> torch.Tensor:
        """Per row: sorted unique interest ids of text+image (incl. 0 if padding present), 0-padded to the
        batch max count (exactly the official per-sample `unique()` + pad_sequence)."""
        ids = torch.cat([self.plm_interest_lookup_table[seq_r], self.img_interest_lookup_table[seq_r]], dim=-1)
        s, _ = ids.sort(dim=1)
        dup = torch.zeros_like(s, dtype=torch.bool)
        dup[:, 1:] = s[:, 1:] == s[:, :-1]
        big = self.num_interest + 1
        s, _ = s.masked_fill(dup, big).sort(dim=1)
        nq = int((~dup).sum(1).max())
        s = s[:, :nq]
        return s.masked_fill(s == big, 0)

    def seq_output(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.clustered):
            self.discover_interests()
        seq_r, lengths, _ = left_to_right(seq)
        interest = self._unique_interests(seq_r)
        src_pad = interest == 0
        memory = self.trm_model.encoder(src=self.interest_embeddings[interest], src_key_padding_mask=src_pad)
        w = (~src_pad).unsqueeze(-1).float().mean(1, keepdim=True)
        pooled = (memory * w).sum(1)
        reg = ((pooled * pooled).sum() / pooled.shape[1]).mean()
        pos = self.position_embedding(torch.arange(seq_r.size(1), device=seq.device))
        x = (self.text_adaptor(self.plm_embedding[seq_r]) + self.img_adaptor(self.img_embedding[seq_r])
             + pos + self.item_embedding(seq_r))
        x = self.dropout(self.LayerNorm(x))
        h = self.trm_model.decoder(x, memory, tgt_key_padding_mask=seq_r == 0, memory_key_padding_mask=src_pad)
        return F.normalize(gather_last(h, lengths), dim=1), reg

    def logits(self, s: torch.Tensor) -> torch.Tensor:
        """Official _compute_dynamic_fused_logits / temperature. s [B, D] normalised -> [B, N+1]."""
        tl = s @ F.normalize(self.text_adaptor(self.plm_embedding), dim=1).T
        il = s @ F.normalize(self.img_adaptor(self.img_embedding), dim=1).T
        ml = torch.stack([tl, il], dim=-1)
        agg = (ml * F.softmax(ml * self.fusion_factor.unsqueeze(-1), dim=-1)).sum(-1)
        idl = s @ F.normalize(self.item_embedding.weight, dim=1).T
        return (idl + agg * 2) / 3 / self.temperature

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        s, reg = self.seq_output(batch["seq"])
        ce = ce_real(self.logits(s), batch["target"])
        return {"loss": ce + self.gamma * reg, "L_ce": ce, "L_orth": reg}

    def score_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        s, _ = self.seq_output(batch["seq"])
        return mask_pad_column(self.logits(s))
