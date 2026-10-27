"""Overfit one small batch: CE of T1, T2 and GRU all go close to 0 (CLAUDE.md §8)."""
from __future__ import annotations

import torch
from conftest import L, N_ITEMS

from src.build import build_loss
from src.models.duokd import DuoKD
from src.utils import set_seed


def test_overfit_batch(tiny_cfg, feats):
    set_seed(0)
    tiny_cfg["model"]["heads"] = ["t1", "t2", "gru"]
    model = DuoKD(tiny_cfg, *feats).train()
    loss_fn = build_loss(tiny_cfg)       # full loss: L_sup + L_mut + L_kd + L_bal
    seq = torch.randint(1, N_ITEMS + 1, (8, L))
    seq[0, :4] = 0
    seq[1, :6] = 0
    batch = {"seq": seq, "target": torch.randint(1, N_ITEMS + 1, (8,))}
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-2)
    first = None
    for _ in range(400):
        loss, logs = loss_fn(model(seq), batch, epoch=100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = first or logs
    model.eval()
    _, logs = loss_fn(model(seq), batch, epoch=100)
    for h in ("t1", "t2", "gru"):
        assert first[f"L_sup_{h}"] > 2.0, first
        assert logs[f"L_sup_{h}"] < 0.05, (h, logs)
