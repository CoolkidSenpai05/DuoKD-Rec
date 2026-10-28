"""Shared helpers: seeding, config loading, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed random, numpy and torch (CPU + all GPUs) in one place."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


DATASET_DIR = ROOT / "configs" / "datasets"
DEFAULT_DATASET = "beauty"


def _load_chain(path: Path) -> dict[str, Any]:
    """Load one YAML file, recursively merging it over its `base:` file."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_path = Path(cfg.pop("base"))
        if not base_path.is_absolute():
            base_path = ROOT / base_path
        cfg = _deep_update(_load_chain(base_path), cfg)
    return cfg


def load_dataset_entry(name: str) -> dict[str, Any]:
    """Registry entry `configs/datasets/<name>.yaml` (name is case-insensitive)."""
    path = DATASET_DIR / f"{name.lower()}.yaml"
    if not path.exists():
        known = sorted(p.stem for p in DATASET_DIR.glob("*.yaml"))
        raise ValueError(f"unknown dataset {name!r}; registered: {known}")
    with open(path) as f:
        return yaml.safe_load(f)


def apply_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill dataset-dependent paths from the registry, keyed by `data.dataset` (default beauty).

    Only keys that are not already set are filled, so an explicit path in a config / override wins.
    Filled: data.{raw_dir, reviews_file, meta_file, processed_dir, category, expected},
    features.{feat_dir, image_dir}, train.out_dir, train.ckpt_dir, ablation.summary_dir.
    """
    data = cfg.setdefault("data", {})
    name = str(data.get("dataset") or DEFAULT_DATASET).lower()
    data["dataset"] = name
    entry = load_dataset_entry(name)
    for k, v in entry["data"].items():
        data.setdefault(k, v)
    data.setdefault("category", entry["category"])
    data.setdefault("expected", entry.get("expected"))
    feats = cfg.setdefault("features", {})
    for k, v in entry["features"].items():
        feats.setdefault(k, v)
    paths = entry["paths"]
    cfg.setdefault("train", {}).setdefault("out_dir", paths["runs_dir"])
    cfg["train"].setdefault("ckpt_dir", paths["ckpt_dir"])
    if "ablation" in cfg:
        cfg["ablation"].setdefault("summary_dir", paths["ablation_dir"])
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config. A config may set `base: <path>` to inherit from another file.

    `overrides` are strings like "train.lr=0.01" (value parsed as YAML). Dataset paths are then
    filled from the registry (`apply_dataset`), so `--set data.dataset=toys` switches dataset.
    """
    cfg = _load_chain(Path(path))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return apply_dataset(cfg)


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def get_logger(name: str = "duokd", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger
