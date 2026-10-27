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


def _all_heads(tiny_cfg):
    tiny_cfg["model"]["heads"] = ["t1", "t2", "gru"]
    return tiny_cfg


def _teacher_side(model):
    ie = model.item_encoder
    return [model.towers["t1"], model.towers["t2"], ie.moe_t, ie.moe_v, ie.emb_id, ie.proj_t, ie.proj_v]


def test_kd_no_grad_to_teachers(tiny_cfg, feats):
    model, seq = _model(_all_heads(tiny_cfg), feats)
    out = model(seq)
    kl_teacher_student(out["z_ens"], out["z_gru"], tau=2.0).backward()
    for m in _teacher_side(model):
        assert _is_zero(_grads(m)), f"L_kd leaked gradient into {type(m).__name__}"
    assert not _is_zero(_grads(model.towers["gru"]))
    assert not _is_zero(_grads(model.item_encoder.fusion))


def test_kd_no_grad_to_teachers_without_shared_detach(tiny_cfg, feats):
    """Even with gru_detach_shared=false, L_kd must not reach the T1/T2 towers."""
    tiny_cfg["model"]["gru_detach_shared"] = False
    model, seq = _model(_all_heads(tiny_cfg), feats)
    out = model(seq)
    kl_teacher_student(out["z_ens"], out["z_gru"], tau=2.0).backward()
    assert _is_zero(_grads(model.towers["t1"])) and _is_zero(_grads(model.towers["t2"]))


def test_learned_ens_weights_only_from_ce(tiny_cfg, feats):
    from src.losses import ce_loss

    tiny_cfg["model"]["learned_ens_weights"] = True
    model, seq = _model(_all_heads(tiny_cfg), feats)
    y = torch.randint(1, N_ITEMS + 1, (4,))
    out = model(seq)
    kl_teacher_student(out["z_ens"], out["z_gru"], tau=2.0).backward()
    assert model.ens_logits.grad is None or torch.count_nonzero(model.ens_logits.grad) == 0
    model.zero_grad(set_to_none=True)
    out = model(seq)
    ce_loss(out["z_ens"], y).backward()
    assert torch.count_nonzero(model.ens_logits.grad) > 0
    for m in _teacher_side(model):
        assert _is_zero(_grads(m)), "CE(z_ens) must only train the ensemble weights"


def test_full_loss_kd_term(tiny_cfg, feats):
    from src.build import build_loss

    model, seq = _model(_all_heads(tiny_cfg), feats)
    loss_fn = build_loss(tiny_cfg)
    batch = {"seq": seq, "target": torch.randint(1, N_ITEMS + 1, (4,))}
    _, logs = loss_fn(model(seq), batch, epoch=0)
    for k in ("L_sup_t1", "L_sup_t2", "L_sup_gru", "L_mut", "L_kd", "L_bal", "lambda_mut", "lambda_kd"):
        assert k in logs, k
    assert logs["lambda_kd"] == 1.0
