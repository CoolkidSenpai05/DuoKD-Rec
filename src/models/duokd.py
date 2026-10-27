"""DuoKD-Rec: item encoder + teachers T1 (text view), T2 (image view) + GRU student (fused view)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.losses import gate_stats
from src.models.gru_rec import GRUEncoder
from src.models.item_encoder import ItemEncoder
from src.models.transformer_rec import CausalTransformer

TEACHERS = ("t1", "t2")


class DuoKD(nn.Module):
    """Multi-head sequential recommender.

    cfg keys used: model.{hidden, dropout, heads, score_against, transformer}, item_encoder, data.max_len.
    forward(seq [B, L]) -> dict:
      z_t1 / z_t2 / z_gru: logits [B, N] (only for active heads), mask [B, L], plus item-encoder tables.
    predict(seq) -> {head: logits [B, N]} (+ "ens" when both teachers are active).
    """

    def __init__(self, cfg: dict, text_feat: np.ndarray, img_feat: np.ndarray) -> None:
        super().__init__()
        mcfg = cfg["model"]
        self.heads = list(mcfg["heads"])
        assert set(self.heads) <= {"t1", "t2", "gru"}, self.heads
        self.score_against = mcfg.get("score_against", "own")
        assert self.score_against in ("own", "id")
        self.detach_shared = mcfg.get("gru_detach_shared", True)
        # which item table each head reads its input from / scores against.
        # t2_input=E_f (ablation §10): T2 reads the fused view, through a non-detached Fusion.
        t2_in = mcfg.get("t2_input", "E_v")
        assert t2_in in ("E_v", "E_f"), t2_in
        self._INPUT = {**DuoKD._INPUT, "t2": "E_v" if t2_in == "E_v" else "E_f_att"}
        d, L, tcfg = mcfg["hidden"], cfg["data"]["max_len"], mcfg["transformer"]
        ie_cfg = {"hidden": d, "dropout": mcfg["dropout"], **cfg["item_encoder"]}
        self.item_encoder = ItemEncoder(text_feat, img_feat, ie_cfg, cfg["features"]["missing"])
        self.ens_w = mcfg.get("ens_weights", [0.5, 0.5])
        # learned ensemble weights w = softmax(a); `a` is trained only by CE(z_ens) (§3.4c)
        self.learned_ens = bool(mcfg.get("learned_ens_weights", False))
        self.ens_logits = nn.Parameter(torch.zeros(2)) if self.learned_ens else None
        self.towers = nn.ModuleDict()
        for h in self.heads:
            if h in TEACHERS:
                self.towers[h] = CausalTransformer(d, tcfg["n_layers"], tcfg["n_heads"], L,
                                                   mcfg["dropout"], tcfg["ffn_mult"])
            else:
                gcfg = mcfg["gru"]
                self.towers[h] = GRUEncoder(d, gcfg["n_layers"], mcfg["dropout"])

    # default mapping head -> item table (input and scoring); instances may override t2 (see __init__)
    _INPUT = {"t1": "E_t", "t2": "E_v", "gru": "E_f"}

    def _table(self, tables: dict[str, torch.Tensor], head: str) -> torch.Tensor:
        return tables["E_id"] if self.score_against == "id" else tables[self._INPUT[head]]

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = seq != 0
        tables = self.item_encoder(detach_shared=self.detach_shared,
                                   attached_fusion=self._INPUT["t2"] == "E_f_att" and "t2" in self.heads)
        out: dict[str, torch.Tensor] = {"mask": mask, "seq": seq, **tables}
        for h in self.heads:
            x = tables[self._INPUT[h]][seq]                  # [B, L, D]
            if h == "gru":
                u = self.towers[h](x, mask)                  # [B, D] state after last real item
            else:
                u = self.towers[h](x, mask)[:, -1]           # [B, D] (left padding -> last = real)
            out[f"u_{h}"] = u
            out[f"z_{h}"] = u @ self._table(tables, h)[1:].T  # [B, N]
        if all(t in self.heads for t in TEACHERS):
            out["z_ens"] = self.ensemble(out["z_t1"], out["z_t2"])
            out["learned_ens"] = self.learned_ens
        return out

    def ens_weights(self) -> torch.Tensor:
        """[2] weights (w1, w2) of the teacher ensemble."""
        if self.learned_ens:
            return torch.softmax(self.ens_logits, dim=0)
        return torch.tensor(self.ens_w, dtype=torch.float32, device=self.item_encoder.nonpad.device)

    def ensemble(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z_ens = w1·z1.detach() + w2·z2.detach()  [B, N]. Gradient only reaches learned w."""
        w = self.ens_weights()
        return w[0] * z1.detach() + w[1] * z2.detach()

    @torch.no_grad()
    def predict(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.forward(seq)
        res = {h: out[f"z_{h}"] for h in self.heads}
        if all(t in self.heads for t in TEACHERS):
            res["ens"] = self.ensemble(res["t1"], res["t2"])
        return res

    @torch.no_grad()
    def epoch_diagnostics(self, out: dict[str, torch.Tensor], batch: dict) -> dict[str, float]:
        """Expert load / gate entropy for both MoEs (over real positions) and T1-T2 disagreement."""
        logs: dict[str, float] = {}
        m = out["mask"].reshape(-1)
        seq = out["seq"].reshape(-1)
        for v in ("t", "v"):
            P, ent = gate_stats(out[f"probs_{v}"][seq], m)
            logs[f"gate_entropy_{v}"] = float(ent)
            for k, p in enumerate(P.tolist()):
                logs[f"load_{v}{k}"] = p
        if self.learned_ens:
            logs["ens_w1"] = float(self.ens_weights()[0])
        if all(t in self.heads for t in TEACHERS):
            logs["disagree_t1_t2"] = float((out["z_t1"].argmax(1) != out["z_t2"].argmax(1)).float().mean())
        return logs
