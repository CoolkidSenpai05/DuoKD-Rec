"""Frozen BERT/ViT features: no grad, unchanged by an optimiser step (CLAUDE.md §8)."""
from __future__ import annotations

import torch

from src.build import item_encoder_cfg
from src.models.item_encoder import ItemEncoder


def test_frozen_features(tiny_cfg, feats):
    enc = ItemEncoder(*feats, item_encoder_cfg(tiny_cfg))
    assert not enc.text_feat.requires_grad and not enc.img_feat.requires_grad
    param_ids = {id(p) for p in enc.parameters()}
    assert id(enc.text_feat) not in param_ids and id(enc.img_feat) not in param_ids
    assert "text_feat" not in enc.state_dict() and "img_feat" not in enc.state_dict()
    before_t, before_v = enc.text_feat.clone(), enc.img_feat.clone()
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-1)
    out = enc(detach_shared=False)
    (out["E_t"].sum() + out["E_v"].sum() + out["E_f"].sum()).backward()
    opt.step()
    torch.testing.assert_close(enc.text_feat, before_t, rtol=0, atol=0)
    torch.testing.assert_close(enc.img_feat, before_v, rtol=0, atol=0)
    assert enc.text_feat.grad is None and enc.img_feat.grad is None
