"""Shapes of every module (CLAUDE.md §3.1, §8)."""
from __future__ import annotations

import torch
from conftest import D, D_BERT, L, N_ITEMS

from src.build import item_encoder_cfg
from src.models.item_encoder import ItemEncoder

B = 3


def test_item_encoder_shapes(tiny_cfg, feats):
    enc = ItemEncoder(*feats, item_encoder_cfg(tiny_cfg))
    out = enc()
    for k in ("E_id", "E_t", "E_v", "E_f"):
        assert out[k].shape == (N_ITEMS + 1, D), k
        assert (out[k][0] == 0).all(), f"{k} padding row must be zero"
    for k in ("probs_t", "probs_v", "gates_t", "gates_v"):
        assert out[k].shape == (N_ITEMS + 1, 4)
    seq = torch.randint(0, N_ITEMS + 1, (B, L))
    for k in ("E_t", "E_v", "E_f"):
        assert out[k][seq].shape == (B, L, D)  # h_t / h_v / h_f
    assert enc.text_feat.shape == (N_ITEMS + 1, D_BERT)


def test_modality_dropout(tiny_cfg, feats):
    cfg = item_encoder_cfg(tiny_cfg)
    cfg["modality_dropout"] = 1.0
    enc = ItemEncoder(*feats, cfg)
    enc.train()
    tf, vf = enc._features()
    assert (tf == 0).all() and (vf == 0).all()
    enc.eval()
    tf, _ = enc._features()
    torch.testing.assert_close(tf, enc.text_feat)  # no dropout at eval
