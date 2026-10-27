"""gru_detach_shared: GRU-side losses must not train Emb_ID / MoE_t / MoE_v (CLAUDE.md §3.5, §8)."""
from __future__ import annotations

import torch
from conftest import L, N_ITEMS

from src.losses import ce_loss, kl_teacher_student
from src.models.duokd import DuoKD


def _nz(module) -> bool:
    return any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in module.parameters())


def _run(tiny_cfg, feats, detach: bool):
    tiny_cfg["model"]["heads"] = ["t1", "t2", "gru"]
    tiny_cfg["model"]["gru_detach_shared"] = detach
    torch.manual_seed(0)
    model = DuoKD(tiny_cfg, *feats).train()
    seq = torch.randint(1, N_ITEMS + 1, (4, L))
    seq[1, :4] = 0
    y = torch.randint(1, N_ITEMS + 1, (4,))
    out = model(seq)
    (ce_loss(out["z_gru"], y) + kl_teacher_student(out["z_ens"], out["z_gru"], 2.0)).backward()
    return model


def test_detach_true(tiny_cfg, feats):
    model = _run(tiny_cfg, feats, True)
    ie = model.item_encoder
    for name, m in (("Emb_ID", ie.emb_id), ("MoE_t", ie.moe_t), ("MoE_v", ie.moe_v),
                    ("Proj_t", ie.proj_t), ("Proj_v", ie.proj_v)):
        assert not _nz(m), f"GRU losses reached {name}"
    assert _nz(ie.fusion), "Fusion must learn from the GRU"
    assert _nz(model.towers["gru"]), "GRU must learn"
    assert not _nz(model.towers["t1"]) and not _nz(model.towers["t2"])


def test_detach_false(tiny_cfg, feats):
    model = _run(tiny_cfg, feats, False)
    ie = model.item_encoder
    assert _nz(ie.emb_id) and _nz(ie.moe_t) and _nz(ie.moe_v), "without detach the GRU trains shared layers"
    assert not _nz(model.towers["t1"]) and not _nz(model.towers["t2"])
