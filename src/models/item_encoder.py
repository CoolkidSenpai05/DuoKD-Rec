"""Item-level multimodal encoder (CLAUDE.md §3.2). Independent of sequence context."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.layers.fusion import GatedFusion
from src.layers.moe import MoE


class Proj(nn.Module):
    """LayerNorm + Linear from a frozen feature space to D. [*, d_in] -> [*, D]."""

    def __init__(self, d_in: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.lin = nn.Linear(d_in, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(self.norm(x))


class ItemEncoder(nn.Module):
    """Computes item tables for the whole catalog.

    Frozen features are registered buffers (never require grad):
      text_feat [N+1, d_bert], img_feat [N+1, d_vit], has_text/has_img [N+1] bool.
    forward() -> dict with
      E_id, E_t, E_v, E_f: [N+1, D]  (row 0 = padding = zeros)
      probs_t, probs_v, gates_t, gates_v: [N+1, n_experts]  (router probs / applied gates)
    `detach_shared=True` detaches h_t, h_v before Fusion (gru_detach_shared, §3.5).
    `attached_fusion=True` also returns E_f_att = Fusion(h_t, h_v) without detach (used when T2
    reads h_f, ablation §10), so the teacher's gradient reaches MoE_t / MoE_v.
    """

    def __init__(self, text_feat: np.ndarray | torch.Tensor, img_feat: np.ndarray | torch.Tensor,
                 cfg: dict, missing: str = "zeros") -> None:
        super().__init__()
        text_feat = torch.as_tensor(np.asarray(text_feat), dtype=torch.float32)
        img_feat = torch.as_tensor(np.asarray(img_feat), dtype=torch.float32)
        n1 = text_feat.shape[0]
        d = cfg["hidden"]
        mcfg = cfg["moe"]
        self.register_buffer("text_feat", text_feat, persistent=False)
        self.register_buffer("img_feat", img_feat, persistent=False)
        # vector used when a modality is dropped (same convention as missing items, §5)
        miss_t = text_feat[1:].mean(0) if missing == "mean" else torch.zeros(text_feat.shape[1])
        miss_v = img_feat[1:].mean(0) if missing == "mean" else torch.zeros(img_feat.shape[1])
        self.register_buffer("miss_t", miss_t, persistent=False)
        self.register_buffer("miss_v", miss_v, persistent=False)
        self.register_buffer("nonpad", (torch.arange(n1) > 0).float().unsqueeze(1), persistent=False)
        self.modality_dropout = cfg.get("modality_dropout", 0.0)
        # modality ablation (§10): a disabled modality contributes e_* = 0 for every item
        self.use_text = bool(cfg.get("use_text", True))
        self.use_image = bool(cfg.get("use_image", True))

        self.emb_id = nn.Embedding(n1, d, padding_idx=0)
        nn.init.normal_(self.emb_id.weight, std=0.02)
        with torch.no_grad():
            self.emb_id.weight[0].zero_()
        self.proj_t = Proj(text_feat.shape[1], d)
        self.proj_v = Proj(img_feat.shape[1], d)
        kw = dict(n_experts=mcfg["n_experts"], d_hidden=mcfg["expert_hidden"], top_k=mcfg["top_k"],
                  dropout=cfg["dropout"], plain_mlp=mcfg.get("plain_mlp", False))
        self.moe_t = MoE(2 * d, d, **kw)
        self.moe_v = MoE(2 * d, d, **kw)
        self.fusion = GatedFusion(d, out_proj=cfg.get("fusion_out_proj", True))

    @property
    def n_items(self) -> int:
        return self.text_feat.shape[0] - 1

    def _features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen features, with modality dropout per item when training. [N+1, d_*] each."""
        tf, vf = self.text_feat, self.img_feat
        p = self.modality_dropout
        if self.training and p > 0:
            drop_t = torch.rand(tf.shape[0], 1, device=tf.device) < p
            drop_v = torch.rand(vf.shape[0], 1, device=vf.device) < p
            tf = torch.where(drop_t, self.miss_t, tf)
            vf = torch.where(drop_v, self.miss_v, vf)
        return tf, vf

    def forward(self, detach_shared: bool = True, attached_fusion: bool = False) -> dict[str, torch.Tensor]:
        tf, vf = self._features()
        e_id = self.emb_id.weight                      # [N+1, D]
        e_txt = self.proj_t(tf) if self.use_text else torch.zeros_like(e_id)
        e_img = self.proj_v(vf) if self.use_image else torch.zeros_like(e_id)
        h_t, probs_t, gates_t = self.moe_t(torch.cat([e_id, e_txt], -1))
        h_v, probs_v, gates_v = self.moe_v(torch.cat([e_id, e_img], -1))
        h_t, h_v = h_t * self.nonpad, h_v * self.nonpad
        if detach_shared:
            h_f = self.fusion(h_t.detach(), h_v.detach())
        else:
            h_f = self.fusion(h_t, h_v)
        out = {"E_id": e_id, "E_t": h_t, "E_v": h_v, "E_f": h_f * self.nonpad,
               "probs_t": probs_t, "probs_v": probs_v, "gates_t": gates_t, "gates_v": gates_v}
        if attached_fusion:
            out["E_f_att"] = h_f * self.nonpad if not detach_shared else self.fusion(h_t, h_v) * self.nonpad
        return out
