"""Dataset registry (M11): Beauty keeps its pre-M11 paths, other datasets get per-dataset paths."""
from __future__ import annotations

import pytest

from src.utils import load_config

DATASETS = ["beauty", "sports", "toys", "clothing"]


def test_beauty_alias_keeps_old_paths():
    for c in ("configs/base.yaml", "configs/ablation_full.yaml", "configs/sasrec.yaml"):
        cfg = load_config(c)
        assert cfg["data"]["dataset"] == "beauty"
        assert cfg["data"]["raw_dir"] == "data/raw"
        assert cfg["data"]["reviews_file"] == "reviews_Beauty_5.json.gz"
        assert cfg["data"]["meta_file"] == "meta_Beauty.json.gz"
        assert cfg["data"]["processed_dir"] == "data/processed/Beauty"
        assert cfg["features"]["feat_dir"] == "data/processed/Beauty/feats"
        assert cfg["features"]["image_dir"] == "data/raw/images/Beauty"
        assert cfg["train"]["out_dir"] == "runs"
        assert cfg["ablation"]["summary_dir"] == "runs/ablation"


@pytest.mark.parametrize("ds", DATASETS[1:])
def test_other_dataset_paths(ds):
    cfg = load_config("configs/ablation_full.yaml", [f"data.dataset={ds}"])
    assert cfg["data"]["processed_dir"] == f"data/processed/{ds}"
    assert cfg["data"]["raw_dir"] == f"data/raw/{ds}"
    assert cfg["features"]["feat_dir"] == f"data/feats/{ds}"
    assert cfg["features"]["image_dir"] == f"data/raw/images/{ds}"
    assert cfg["train"]["out_dir"] == f"runs/{ds}"
    assert cfg["data"]["reviews_file"] == f"reviews_{cfg['data']['category']}_5.json.gz"
    # cold-start thresholds are dataset independent
    assert cfg["eval"]["cold_user_max_len"] == 5 and cfg["eval"]["cold_item_max_freq"] == 10


def test_explicit_path_wins_and_case_insensitive():
    cfg = load_config("configs/base.yaml", ["data.dataset=Toys", "data.processed_dir=/x"])
    assert cfg["data"]["dataset"] == "toys" and cfg["data"]["processed_dir"] == "/x"
    assert cfg["features"]["feat_dir"] == "data/feats/toys"


def test_unknown_dataset():
    with pytest.raises(ValueError):
        load_config("configs/base.yaml", ["data.dataset=nope"])
