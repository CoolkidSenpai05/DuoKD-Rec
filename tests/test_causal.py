"""Causality and padding invariance of the causal Transformer (CLAUDE.md §8)."""
from __future__ import annotations

import torch

from src.models.transformer_rec import CausalTransformer, SASRec

D, L = 16, 8


def _enc():
    torch.manual_seed(0)
    return CausalTransformer(D, n_layers=2, n_heads=2, max_len=L, dropout=0.0).eval()


def test_causal():
    enc = _enc()
    h = torch.randn(3, L, D)
    mask = torch.ones(3, L, dtype=torch.bool)
    out = enc(h, mask)
    h2 = h.clone()
    h2[:, 5:] = torch.randn(3, L - 5, D)  # change positions >= 5
    out2 = enc(h2, mask)
    torch.testing.assert_close(out[:, :5], out2[:, :5])
    assert not torch.allclose(out[:, 5:], out2[:, 5:])


def test_padding_invariance_and_no_nan():
    enc = _enc()
    h = torch.randn(2, L, D)
    mask = torch.ones(2, L, dtype=torch.bool)
    mask[:, :3] = False  # left padding
    out = enc(h, mask)
    assert torch.isfinite(out).all()
    h2 = h.clone()
    h2[:, :3] = 100 * torch.randn(2, 3, D)
    out2 = enc(h2, mask)
    torch.testing.assert_close(out[:, 3:], out2[:, 3:])
    assert (out[:, :3] == 0).all()


def test_sasrec_shapes():
    m = SASRec(20, {"hidden": D, "dropout": 0.0, "transformer": {"n_layers": 1, "n_heads": 2, "ffn_mult": 2}}, L)
    seq = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12]])
    assert m(seq)["sasrec"].shape == (2, 20)
