"""Prepare Amazon interactions: k-core filter, ID mapping, leave-one-out split (CLAUDE.md §4).

Output (in `data.processed_dir`):
  - `data.pkl`: dict with
      n_users, n_items,
      train_seqs: list[list[int]]  per user, chronological, without valid/test items
      valid_items: list[int]       per user (2nd to last interaction)
      test_items:  list[int]       per user (last interaction)
      item2id: {asin: id in 1..N},  user2id: {reviewerID: id in 0..U-1}
  - `stats.json`: dataset statistics.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_logger, load_config, resolve  # noqa: E402

log = get_logger("prepare")


def read_interactions(path: Path) -> list[tuple[str, str, int]]:
    """Read (user, item, timestamp) triples from an Amazon reviews json.gz file."""
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    return out


def k_core(inter: list[tuple[str, str, int]], k: int) -> list[tuple[str, str, int]]:
    """Iteratively drop users/items with fewer than k interactions until stable."""
    while True:
        uc = Counter(u for u, _, _ in inter)
        ic = Counter(i for _, i, _ in inter)
        kept = [x for x in inter if uc[x[0]] >= k and ic[x[1]] >= k]
        if len(kept) == len(inter):
            return kept
        inter = kept


def build_sequences(inter: list[tuple[str, str, int]]):
    """Map IDs and build chronological per-user sequences.

    Ties in timestamp keep the original file order (stable sort).
    Items are numbered 1..N in order of first appearance after sorting users; users 0..U-1.
    """
    by_user: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for order, (u, i, t) in enumerate(inter):
        by_user[u].append((t, order, i))
    users = sorted(by_user)  # deterministic user order
    user2id = {u: idx for idx, u in enumerate(users)}
    item2id: dict[str, int] = {}
    seqs: list[list[int]] = []
    for u in users:
        events = sorted(by_user[u])
        s = []
        for _, _, i in events:
            if i not in item2id:
                item2id[i] = len(item2id) + 1  # 0 = padding
            s.append(item2id[i])
        seqs.append(s)
    return seqs, user2id, item2id


def split_leave_one_out(seqs: list[list[int]]):
    """Last item = test, second to last = valid, the rest = train."""
    train, valid, test = [], [], []
    for s in seqs:
        assert len(s) >= 3, "every user needs >= 3 interactions for leave-one-out"
        train.append(s[:-2])
        valid.append(s[-2])
        test.append(s[-1])
    return train, valid, test


def compute_stats(seqs, train, valid, test, n_items, n_inter_raw) -> dict:
    lens = np.array([len(s) for s in seqs])
    item_cnt = Counter(i for s in seqs for i in s)
    return {
        "n_users": len(seqs),
        "n_items": n_items,
        "n_interactions": int(lens.sum()),
        "n_interactions_raw": n_inter_raw,
        "density": float(lens.sum() / (len(seqs) * n_items)),
        "seq_len_mean": float(lens.mean()),
        "seq_len_median": float(np.median(lens)),
        "seq_len_p90": float(np.percentile(lens, 90)),
        "seq_len_p95": float(np.percentile(lens, 95)),
        "seq_len_max": int(lens.max()),
        "item_freq_mean": float(np.mean(list(item_cnt.values()))),
        "n_train_samples": int(sum(len(t) - 1 for t in train)),
        # repeat consumption: target already in the input history (affects exclude_history)
        "valid_target_in_history": int(sum(v in set(t) for t, v in zip(train, valid))),
        "test_target_in_history": int(sum(x in set(t) | {v} for t, v, x in zip(train, valid, test))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["data"]

    raw = resolve(cfg["raw_dir"]) / cfg["reviews_file"]
    out_dir = resolve(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    inter = read_interactions(raw)
    n_raw = len(inter)
    log.info("read %d interactions from %s", n_raw, raw)
    inter = k_core(inter, cfg["k_core"])
    log.info("after %d-core: %d interactions", cfg["k_core"], len(inter))

    seqs, user2id, item2id = build_sequences(inter)
    train, valid, test = split_leave_one_out(seqs)
    stats = compute_stats(seqs, train, valid, test, len(item2id), n_raw)

    with open(out_dir / "data.pkl", "wb") as f:
        pickle.dump(
            {
                "n_users": len(seqs),
                "n_items": len(item2id),
                "train_seqs": train,
                "valid_items": valid,
                "test_items": test,
                "item2id": item2id,
                "user2id": user2id,
            },
            f,
        )
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info("stats: %s", json.dumps(stats))


if __name__ == "__main__":
    main()
