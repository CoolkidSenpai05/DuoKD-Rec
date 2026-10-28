"""Prepare Amazon interactions: k-core filter, ID mapping, leave-one-out split (CLAUDE.md §4).

Output (in `data.processed_dir`):
  - `data.pkl`: dict with
      n_users, n_items,
      train_seqs: list[list[int]]  per user, chronological, without valid/test items
      valid_items: list[int]       per user (2nd to last interaction)
      test_items:  list[int]       per user (last interaction)
      item2id: {asin: id in 1..N},  user2id: {reviewerID: id in 0..U-1}
  - `stats.json`: dataset statistics.
  - `timestamps.pkl` (M11, separate file so `data.pkl` of Beauty keeps its hash): dict with
      train_ts: list[list[int]], valid_ts: list[int], test_ts: list[int]  (unix seconds, aligned
      with train_seqs / valid_items / test_items).
  - `stats_check.json`: comparison with the registry's expected 5-core statistics.
An existing `data.pkl` / `stats.json` is never overwritten with different content (use --force).
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
    """Map IDs and build chronological per-user sequences (see `build_sequences_with_time`)."""
    seqs, _, user2id, item2id = build_sequences_with_time(inter)
    return seqs, user2id, item2id


def build_sequences_with_time(inter: list[tuple[str, str, int]]):
    """Map IDs and build chronological per-user sequences plus their timestamps.

    Ties in timestamp keep the original file order (stable sort).
    Items are numbered 1..N in order of first appearance after sorting users; users 0..U-1.
    Returns (seqs, times, user2id, item2id); times[u][j] is the timestamp of seqs[u][j].
    """
    by_user: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for order, (u, i, t) in enumerate(inter):
        by_user[u].append((t, order, i))
    users = sorted(by_user)  # deterministic user order
    user2id = {u: idx for idx, u in enumerate(users)}
    item2id: dict[str, int] = {}
    seqs: list[list[int]] = []
    times: list[list[int]] = []
    for u in users:
        events = sorted(by_user[u])
        s, ts = [], []
        for t, _, i in events:
            if i not in item2id:
                item2id[i] = len(item2id) + 1  # 0 = padding
            s.append(item2id[i])
            ts.append(t)
        seqs.append(s)
        times.append(ts)
    return seqs, times, user2id, item2id


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


def check_expected(stats: dict, expected: dict | None, tol: float = 0.005) -> dict:
    """Relative deviation of users/items/interactions from the expected values; ok if all <= tol."""
    out: dict = {"tolerance": tol, "fields": {}, "ok": True}
    for k, v in (expected or {}).items():
        dev = abs(stats[k] - v) / v
        out["fields"][k] = {"actual": stats[k], "expected": v, "rel_dev": dev}
        out["ok"] &= dev <= tol
    return out


def time_stats(times: list[list[int]]) -> dict:
    """Share of consecutive interactions of a user with identical timestamp (day resolution data)."""
    gaps = np.concatenate([np.diff(t) for t in times if len(t) > 1])
    return {"zero_gap_frac": float((gaps == 0).mean()), "gap_median_days": float(np.median(gaps) / 86400),
            "ts_min": int(min(t[0] for t in times)), "ts_max": int(max(t[-1] for t in times))}


def _write_if_same(path: Path, payload: bytes, force: bool) -> str:
    """Write `payload` unless the file exists with different content (then raise, unless force)."""
    if path.exists():
        if path.read_bytes() == payload:
            return "unchanged"
        if not force:
            raise FileExistsError(f"{path} exists with different content; refusing to overwrite (--force)")
    path.write_bytes(payload)
    return "written"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.dataset=toys")
    ap.add_argument("--force", action="store_true", help="overwrite data.pkl/stats.json with new content")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)["data"]

    raw = resolve(cfg["raw_dir"]) / cfg["reviews_file"]
    out_dir = resolve(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    inter = read_interactions(raw)
    n_raw = len(inter)
    log.info("dataset %s: read %d interactions from %s", cfg["dataset"], n_raw, raw)
    inter = k_core(inter, cfg["k_core"])
    log.info("after %d-core: %d interactions", cfg["k_core"], len(inter))

    seqs, times, user2id, item2id = build_sequences_with_time(inter)
    train, valid, test = split_leave_one_out(seqs)
    t_train, t_valid, t_test = split_leave_one_out(times)
    stats = compute_stats(seqs, train, valid, test, len(item2id), n_raw)

    data_bytes = pickle.dumps(
        {
            "n_users": len(seqs),
            "n_items": len(item2id),
            "train_seqs": train,
            "valid_items": valid,
            "test_items": test,
            "item2id": item2id,
            "user2id": user2id,
        }
    )
    log.info("data.pkl: %s", _write_if_same(out_dir / "data.pkl", data_bytes, args.force))
    log.info("stats.json: %s", _write_if_same(out_dir / "stats.json",
                                              json.dumps(stats, indent=2).encode(), args.force))
    ts_bytes = pickle.dumps({"train_ts": t_train, "valid_ts": t_valid, "test_ts": t_test})
    log.info("timestamps.pkl: %s", _write_if_same(out_dir / "timestamps.pkl", ts_bytes, args.force))
    check = check_expected(stats, cfg.get("expected"))
    check["time"] = time_stats(times)
    (out_dir / "stats_check.json").write_text(json.dumps(check, indent=2))
    log.info("stats: %s", json.dumps(stats))
    log.info("expected-stat check (tol %.1f%%): %s %s", 100 * check["tolerance"],
             "OK" if check["ok"] else "MISMATCH", json.dumps(check["fields"]))


if __name__ == "__main__":
    main()
