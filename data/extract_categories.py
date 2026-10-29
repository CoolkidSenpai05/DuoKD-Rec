"""Multi-hot item categories for HM4SR's category-prediction loss (M12).

Same rule as HM4SR `dataprocess/data_process.py::prepare_category`: the FIRST category path of each item
(meta['categories'][0]), vocabulary in order of first appearance over items 1..N.
Output: <features.feat_dir>/categories.npy  bool [N+1, C] (row 0 and items without metadata = all False),
        <features.feat_dir>/categories_vocab.json.  A new file: existing feature files are not touched.

  python data/extract_categories.py --set data.dataset=toys
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.extract_feats import load_meta  # noqa: E402
from src.dataset import load_processed  # noqa: E402
from src.utils import get_logger, load_config, resolve  # noqa: E402

log = get_logger("extract_categories")


def category_matrix(meta: dict[int, dict], n_items: int) -> tuple[np.ndarray, list[str]]:
    vocab: dict[str, int] = {}
    rows: dict[int, list[int]] = {}
    for i in range(1, n_items + 1):
        cats = (meta.get(i, {}).get("categories") or [[]])[0]
        rows[i] = [vocab.setdefault(c, len(vocab)) for c in cats]
    out = np.zeros((n_items + 1, max(len(vocab), 1)), dtype=bool)
    for i, idx in rows.items():
        out[i, idx] = True
    return out, list(vocab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    data = load_processed(cfg["data"]["processed_dir"])
    meta = load_meta(resolve(cfg["data"]["raw_dir"]) / cfg["data"]["meta_file"], data["item2id"])
    cats, vocab = category_matrix(meta, data["n_items"])
    out = resolve(cfg["features"]["feat_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "categories.npy", cats)
    (out / "categories_vocab.json").write_text(json.dumps(vocab))
    log.info("%s: %d categories, %d items without category -> %s", cfg["data"]["dataset"], len(vocab),
             int((~cats[1:].any(1)).sum()), out / "categories.npy")


if __name__ == "__main__":
    main()
