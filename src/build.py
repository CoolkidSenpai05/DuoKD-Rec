"""Build model and loss from config."""
from __future__ import annotations

import torch.nn as nn

from src.losses import SASRecLoss
from src.models.transformer_rec import SASRec


def build_model(cfg: dict, n_items: int) -> nn.Module:
    name = cfg["model"]["name"]
    if name == "sasrec":
        return SASRec(n_items, cfg["model"], cfg["data"]["max_len"])
    raise ValueError(f"unknown model {name}")


def build_loss(cfg: dict) -> nn.Module:
    name = cfg["model"]["name"]
    if name == "sasrec":
        return SASRecLoss()
    raise ValueError(f"unknown model {name}")
