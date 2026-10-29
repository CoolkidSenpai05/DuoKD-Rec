"""Name -> baseline class. `build_baseline(cfg, data)` loads whatever frozen features the model needs."""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch

from src.utils import resolve

# model.name -> "module:Class" (lazy import so one broken baseline does not break the others)
REGISTRY: dict[str, str] = {
    "sasrec": "src.baselines.sasrec:SASRecRec",
    "gru4rec": "src.baselines.gru4rec:GRU4Rec",
    "bert4rec": "src.baselines.bert4rec:BERT4Rec",
    "bsarec": "src.baselines.bsarec:BSARec",
    "unisrec": "src.baselines.unisrec:UniSRec",
    "missrec": "src.baselines.missrec:MISSRec",
    "tedrec": "src.baselines.tedrec:TedRec",
    "hm4sr": "src.baselines.hm4sr:HM4SR",
    "ddmsr": "src.baselines.ddmsr:DDMSR",
}


def baseline_class(name: str) -> type:
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)


def load_frozen_features(cfg: dict, modalities: tuple[str, ...], n_items: int) -> dict[str, Any]:
    """Frozen M2 features as float32 tensors [N+1, d] (row 0 = zeros) + has_* masks, per modality."""
    out: dict[str, Any] = {}
    if not modalities:
        return out
    d = resolve(cfg["features"]["feat_dir"])
    files = {"text": "text_feat.npy", "image": "img_feat.npy", "category": "categories.npy"}
    masks = {"text": "has_text.npy", "image": "has_img.npy"}
    for m in modalities:
        x = np.load(d / files[m])
        assert x.shape[0] == n_items + 1, f"{m} features have {x.shape[0]} rows, expected N+1={n_items + 1}"
        out[m] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if m in masks and (d / masks[m]).exists():
            out[f"has_{m}"] = torch.from_numpy(np.load(d / masks[m]))
    out["feat_dir"] = d
    return out


def build_baseline(cfg: dict, data: dict) -> torch.nn.Module:
    cls = baseline_class(cfg["model"]["name"])
    feats = load_frozen_features(cfg, cls.modalities_for(cfg) if hasattr(cls, "modalities_for") else cls.modalities,
                                 data["n_items"])
    return cls(cfg, data, feats)
