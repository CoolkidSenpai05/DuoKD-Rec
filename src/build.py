"""Build model and loss from config."""
from __future__ import annotations

import torch.nn as nn

from src.losses import DuoKDLoss


class NoLoss(nn.Module):
    """Placeholder loss for BaseSeqRec baselines: their loss lives in `model.compute_loss` (no params)."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never called by the trainer
        raise RuntimeError("baselines compute their loss with model.compute_loss(batch)")


def is_baseline(cfg: dict) -> bool:
    from src.baselines.registry import REGISTRY

    return cfg["model"]["name"] in REGISTRY


def needs_timestamps(cfg: dict) -> bool:
    """True if the configured model reads per-interaction timestamps (data loaded with timestamps.pkl)."""
    if not is_baseline(cfg):
        return False
    from src.baselines.registry import baseline_class

    return bool(baseline_class(cfg["model"]["name"]).needs_timestamps)


def build_model(cfg: dict, n_items: int, data: dict | None = None) -> nn.Module:
    """DuoKD (M4-M10) or any registered baseline (M12; `sasrec` = the M3 model behind BaseSeqRec)."""
    name = cfg["model"]["name"]
    if is_baseline(cfg):
        from src.baselines.registry import build_baseline
        from src.dataset import load_processed

        if data is None:
            data = load_processed(cfg["data"]["processed_dir"], with_timestamps=needs_timestamps(cfg))
        assert data["n_items"] == n_items
        return build_baseline(cfg, data)
    if name == "duokd":
        from src.models.duokd import DuoKD

        text_feat, img_feat = load_features(cfg)
        assert text_feat.shape[0] == n_items + 1, "features do not match the processed dataset"
        return DuoKD(cfg, text_feat, img_feat)
    raise ValueError(f"unknown model {name}")


def build_loss(cfg: dict) -> nn.Module:
    name = cfg["model"]["name"]
    if is_baseline(cfg):
        return NoLoss()
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
