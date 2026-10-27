"""Build model and loss from config."""
from __future__ import annotations

import torch.nn as nn

from src.losses import DuoKDLoss, SASRecLoss
from src.models.transformer_rec import SASRec


def build_model(cfg: dict, n_items: int) -> nn.Module:
    name = cfg["model"]["name"]
    if name == "sasrec":
        return SASRec(n_items, cfg["model"], cfg["data"]["max_len"])
    if name == "duokd":
        from src.models.duokd import DuoKD

        text_feat, img_feat = load_features(cfg)
        assert text_feat.shape[0] == n_items + 1, "features do not match the processed dataset"
        return DuoKD(cfg, text_feat, img_feat)
    raise ValueError(f"unknown model {name}")


def build_loss(cfg: dict) -> nn.Module:
    name = cfg["model"]["name"]
    if name == "sasrec":
        return SASRecLoss()
    if name == "duokd":
        return DuoKDLoss(cfg)
    raise ValueError(f"unknown model {name}")


def item_encoder_cfg(cfg: dict) -> dict:
    """Merge model.hidden / model.dropout with the item_encoder section."""
    return {"hidden": cfg["model"]["hidden"], "dropout": cfg["model"]["dropout"], **cfg["item_encoder"]}


def load_features(cfg: dict):
    """Load frozen features (text_feat, img_feat) as numpy arrays from features.feat_dir."""
    import numpy as np

    from src.utils import resolve

    d = resolve(cfg["features"]["feat_dir"])
    return np.load(d / "text_feat.npy"), np.load(d / "img_feat.npy")
