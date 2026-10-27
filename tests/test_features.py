"""Feature-extraction helpers and saved feature files (CLAUDE.md §5)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from data.extract_feats import build_text, fill_missing
from src.utils import resolve


def test_build_text():
    r = {"title": "Lipstick", "brand": "X", "categories": [["Beauty", "Makeup"], ["Beauty", "Lips"]],
         "description": "abcdef"}
    t, has = build_text(r, desc_max_chars=3)
    assert t == "Lipstick. X. Beauty, Makeup, Lips. abc" and has
    t, has = build_text({"categories": [["Beauty"]]}, 10)
    assert t == "Beauty" and not has


@pytest.mark.parametrize("strategy", ["zeros", "mean"])
def test_fill_missing(strategy):
    f = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    has = np.array([False, True, False, True])
    fill_missing(f, has, strategy)
    assert (f[0] == 0).all()
    expect = np.zeros(3) if strategy == "zeros" else (np.array([4, 5, 6]) + np.array([10, 11, 12])) / 2
    np.testing.assert_allclose(f[2], expect)
    np.testing.assert_allclose(f[1], [4, 5, 6])


def test_saved_features():
    d = resolve("data/processed/Beauty/feats")
    if not (d / "feat_meta.json").exists():
        pytest.skip("features not extracted")
    meta = json.loads((d / "feat_meta.json").read_text())
    n = meta["n_items"]
    for name in ("text_feat", "img_feat"):
        x = np.load(d / f"{name}.npy")
        assert x.shape[0] == n + 1 and x.dtype == np.float32
        assert (x[0] == 0).all() and np.isfinite(x).all()
    for name in ("has_text", "has_img"):
        h = np.load(d / f"{name}.npy")
        assert h.dtype == bool and h.shape == (n + 1,) and not h[0]
