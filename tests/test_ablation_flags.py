"""Ablation switches (§10): T2 reading h_f, and dropping a modality."""
from __future__ import annotations

import torch
from conftest import L, N_ITEMS

from src.losses import ce_loss, kl_teacher_student
from src.models.duokd import DuoKD


def _nz(m: torch.nn.Module) -> bool:
    return any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in m.parameters())


def _setup(cfg, feats):
    torch.manual_seed(0)
    cfg["model"]["heads"] = ["t1", "t2", "gru"]
    model = DuoKD(cfg, *feats).train()
    seq = torch.randint(1, N_ITEMS + 1, (4, L))
    seq[0, :3] = 0
    return model, seq, torch.randint(1, N_ITEMS + 1, (4,))


def test_t2_reads_fused_view(tiny_cfg, feats):
    tiny_cfg["model"]["t2_input"] = "E_f"
    model, seq, y = _setup(tiny_cfg, feats)
    out = model(seq)
    assert torch.allclose(out["E_f_att"], out["E_f"])           # same values, different graph
    ce_loss(out["z_t2"], y).backward()
    ie = model.item_encoder
    assert _nz(ie.moe_v) and _nz(ie.moe_t) and _nz(ie.fusion)    # T2 trains shared layers via h_f
    model.zero_grad(set_to_none=True)
    out = model(seq)
    (ce_loss(out["z_gru"], y) + kl_teacher_student(out["z_ens"], out["z_gru"], 2.0)).backward()
    assert not _nz(ie.moe_v) and not _nz(ie.moe_t) and not _nz(ie.emb_id)   # GRU still detached


def test_drop_text_modality(tiny_cfg, feats):
    tiny_cfg["item_encoder"]["use_text"] = False
    model, seq, y = _setup(tiny_cfg, feats)
    model.eval()
    t, v = feats
    e1 = model.item_encoder()["E_t"]
    model.item_encoder.text_feat.normal_()                        # text features must not matter
    e2 = model.item_encoder()["E_t"]
    assert torch.equal(e1, e2)
    model.train()
    out = model(seq)
    sum(ce_loss(out[f"z_{h}"], y) for h in ("t1", "t2", "gru")).backward()
    assert not _nz(model.item_encoder.proj_t) and _nz(model.item_encoder.proj_v)


def test_id_only(tiny_cfg, feats):
    tiny_cfg["item_encoder"].update(use_text=False, use_image=False)
    model, _, _ = _setup(tiny_cfg, feats)
    model.eval()
    a = model.item_encoder()
    model.item_encoder.text_feat.normal_()
    model.item_encoder.img_feat.normal_()
    b = model.item_encoder()
    assert all(torch.equal(a[k], b[k]) for k in ("E_t", "E_v", "E_f"))
