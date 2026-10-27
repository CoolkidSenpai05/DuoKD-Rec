"""KD / mutual losses never send gradient to the teacher side (CLAUDE.md §3.5, §8)."""
from __future__ import annotations

import torch
from conftest import L, N_ITEMS

from src.losses import kl_teacher_student, mutual_kd_loss
from src.models.duokd import DuoKD


def _grads(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.grad for p in module.parameters() if p.requires_grad]


def _is_zero(gs) -> bool:
    return all(g is None or torch.count_nonzero(g) == 0 for g in gs)


def test_kl_detaches_teacher():
    zt = torch.randn(4, 10, requires_grad=True)
    zs = torch.randn(4, 10, requires_grad=True)
    kl_teacher_student(zt, zs, tau=2.0).backward()
    assert zt.grad is None
    assert zs.grad is not None and torch.count_nonzero(zs.grad) > 0


def test_kl_zero_when_equal():
    z = torch.randn(4, 10)
    assert abs(float(kl_teacher_student(z, z.clone(), 2.0))) < 1e-6


def _model(tiny_cfg, feats):
    torch.manual_seed(0)
    model = DuoKD(tiny_cfg, *feats).train()
    seq = torch.randint(1, N_ITEMS + 1, (4, L))
    seq[0, :3] = 0
    return model, seq


def test_mutual_each_direction_detaches_teacher(tiny_cfg, feats):
    model, seq = _model(tiny_cfg, feats)
    # direction T2 -> T1: only T1 (student side) gets gradient
    out = model(seq)
    m21, _ = mutual_kd_loss(out["z_t1"], out["z_t2"], tau=2.0)
    m21.backward()
    assert _is_zero(_grads(model.towers["t2"])) and _is_zero(_grads(model.item_encoder.moe_v))
    assert not _is_zero(_grads(model.towers["t1"]))
    model.zero_grad(set_to_none=True)
    # direction T1 -> T2: only T2 gets gradient
    out = model(seq)
    _, m12 = mutual_kd_loss(out["z_t1"], out["z_t2"], tau=2.0)
    m12.backward()
    assert _is_zero(_grads(model.towers["t1"])) and _is_zero(_grads(model.item_encoder.moe_t))
    assert not _is_zero(_grads(model.towers["t2"]))
