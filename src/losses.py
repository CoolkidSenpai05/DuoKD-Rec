"""Loss functions (CLAUDE.md §3.4). Kept separate from models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.schedules import linear_warmup


def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Full-softmax CE. logits [B, N] over real items; target [B] item ids in 1..N."""
    return F.cross_entropy(logits, target - 1)


class SASRecLoss(nn.Module):
    """CE for the ID-only baseline. Returns (loss, logs)."""

    def forward(self, out: dict, batch: dict, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
        loss = ce_loss(out["sasrec"], batch["target"])
        return loss, {"L_sup_sasrec": loss.item()}


def moe_balance_loss(probs: torch.Tensor, gates: torch.Tensor, mask: torch.Tensor,
                     top_k: int | None) -> torch.Tensor:
    """MoE load-balancing loss (§3.4d).

    probs, gates: [M, E] router probs / applied gates; mask: [M] bool (real positions).
    Soft:  n·Σ_k P_k²          (min = 1 when uniform)
    Top-k: n·Σ_k f_k·P_k, f_k = fraction of (masked) samples assigned to expert k.
    """
    n = probs.shape[-1]
    w = mask.float().unsqueeze(-1)
    denom = w.sum().clamp_min(1.0)
    P = (probs * w).sum(0) / denom
    if top_k is None:
        return n * (P ** 2).sum()
    f = ((gates > 0).float() * w).sum(0) / denom / top_k
    return n * (f * P).sum()


def gate_stats(probs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean expert load P [E] and mean gate entropy [] over masked rows of probs [M, E]."""
    w = mask.float().unsqueeze(-1)
    denom = w.sum().clamp_min(1.0)
    P = (probs * w).sum(0) / denom
    ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
    return P, (ent * mask.float()).sum() / denom


def kl_teacher_student(z_teacher: torch.Tensor, z_student: torch.Tensor, tau: float) -> torch.Tensor:
    """τ²·KL(softmax(z_teacher/τ) ‖ softmax(z_student/τ)), batch mean.

    The teacher logits are detached here (detach_teacher_in_kd is mandatory, §3.5). [B, N] -> [].
    """
    p_t = F.softmax(z_teacher.detach() / tau, dim=-1)
    log_q = F.log_softmax(z_student / tau, dim=-1)
    return tau ** 2 * F.kl_div(log_q, p_t, reduction="batchmean")


def mutual_kd_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual distillation T1 <-> T2 (§3.4b). Returns (T2 teaches T1, T1 teaches T2)."""
    return kl_teacher_student(z2, z1, tau), kl_teacher_student(z1, z2, tau)


class DuoKDLoss(nn.Module):
    """L = L_sup + λ_mut(t)·L_mut + λ_bal·(L_bal_t + L_bal_v)   [KD term added in M8].

    Note: with gru_detach_shared=true the GRU gives no gradient to the MoEs; L_bal still applies.

    L_sup = Σ_head CE(z_head, y) over active heads. L_bal uses the MoE router probs of the
    items at real positions of the batch. Returns (loss, logs).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.heads = list(cfg["model"]["heads"])
        self.lambda_bal = cfg["loss"]["lambda_bal"]
        self.top_k = cfg["item_encoder"]["moe"]["top_k"]
        # a MoE is only trained when the teacher that reads its view is active (or the GRU)
        self.bal_views = [v for v, h in (("t", "t1"), ("v", "t2")) if h in self.heads or "gru" in self.heads]
        lc = cfg["loss"]
        self.tau = lc["tau"]
        self.mut_cfg = (lc["lambda_mut"], lc["mut_start_epoch"], lc["mut_warmup_epochs"])
        self.use_mut = "t1" in self.heads and "t2" in self.heads and lc["lambda_mut"] > 0

    def lambdas(self, epoch: int) -> dict[str, float]:
        return {"lambda_mut": linear_warmup(epoch, *self.mut_cfg) if self.use_mut else 0.0}

    def forward(self, out: dict, batch: dict, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
        y = batch["target"]
        logs: dict[str, float] = {}
        loss = out[f"z_{self.heads[0]}"].new_zeros(())
        for h in self.heads:
            ce = ce_loss(out[f"z_{h}"], y)
            logs[f"L_sup_{h}"] = ce.item()
            loss = loss + ce
        m = out["mask"].reshape(-1)
        seq = out["seq"].reshape(-1)
        l_bal = loss.new_zeros(())
        for v in self.bal_views:
            l_bal = l_bal + moe_balance_loss(out[f"probs_{v}"][seq], out[f"gates_{v}"][seq], m, self.top_k)
        logs["L_bal"] = l_bal.item()
        loss = loss + self.lambda_bal * l_bal

        lam = self.lambdas(epoch)
        logs.update(lam)
        if self.use_mut:
            m21, m12 = mutual_kd_loss(out["z_t1"], out["z_t2"], self.tau)
            l_mut = m21 + m12
            logs["L_mut"] = l_mut.item()
            if lam["lambda_mut"] > 0:
                loss = loss + lam["lambda_mut"] * l_mut
        return loss, logs
