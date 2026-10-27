"""Shared tiny config / fake frozen features for model tests."""
from __future__ import annotations

import copy

import numpy as np
import pytest

N_ITEMS, D, L, D_BERT, D_VIT = 30, 16, 8, 24, 20

TINY_CFG = {
    "seed": 0,
    "data": {"max_len": L},
    "features": {"missing": "zeros"},
    "model": {"name": "duokd", "hidden": D, "dropout": 0.0, "heads": ["t1", "t2"],
              "score_against": "own", "gru_detach_shared": True, "ens_weights": [0.5, 0.5],
              "transformer": {"n_layers": 2, "n_heads": 2, "ffn_mult": 2}, "gru": {"n_layers": 2}},
    "loss": {"lambda_bal": 0.01},
    "item_encoder": {"modality_dropout": 0.0, "fusion_out_proj": True,
                     "moe": {"n_experts": 4, "expert_hidden": 32, "top_k": None, "plain_mlp": False}},
}


@pytest.fixture
def tiny_cfg():
    return copy.deepcopy(TINY_CFG)


@pytest.fixture
def feats():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(N_ITEMS + 1, D_BERT)).astype(np.float32)
    v = rng.normal(size=(N_ITEMS + 1, D_VIT)).astype(np.float32)
    t[0] = v[0] = 0
    return t, v
