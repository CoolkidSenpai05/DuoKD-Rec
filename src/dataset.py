"""Datasets for training (sliding-window prefixes) and evaluation (leave-one-out)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import resolve


def load_processed(processed_dir: str | Path) -> dict[str, Any]:
    with open(resolve(processed_dir) / "data.pkl", "rb") as f:
        return pickle.load(f)


def left_pad(items: list[int], max_len: int) -> np.ndarray:
    """Keep the last `max_len` items and left-pad with 0. Returns int64 [max_len]."""
    items = items[-max_len:]
    out = np.zeros(max_len, dtype=np.int64)
    if items:
        out[-len(items):] = items
    return out


class TrainDataset(Dataset):
    """One sample per (user, t): input = train_seq[:t] (last L items), target = train_seq[t].

    Only the per-user train sequence is used, so valid/test items never appear.
    Item  -> dict(seq [L] int64, target [] int64, user [] int64).
    """

    def __init__(self, train_seqs: list[list[int]], max_len: int) -> None:
        self.train_seqs = train_seqs
        self.max_len = max_len
        self.index = [(u, t) for u, s in enumerate(train_seqs) for t in range(1, len(s))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u, t = self.index[idx]
        s = self.train_seqs[u]
        return {
            "seq": torch.from_numpy(left_pad(s[:t], self.max_len)),
            "target": torch.tensor(s[t], dtype=torch.long),
            "user": torch.tensor(u, dtype=torch.long),
        }


class EvalDataset(Dataset):
    """Leave-one-out evaluation, one sample per user.

    split="valid": input = train_seq,          target = valid item
    split="test":  input = train_seq + [valid], target = test item
    `history[u]` is the full (untruncated) input history, used for exclude_history.
    """

    def __init__(self, data: dict[str, Any], split: str, max_len: int) -> None:
        assert split in ("valid", "test")
        self.split = split
        self.max_len = max_len
        if split == "valid":
            self.history = [list(s) for s in data["train_seqs"]]
            self.targets = list(data["valid_items"])
        else:
            self.history = [list(s) + [v] for s, v in zip(data["train_seqs"], data["valid_items"])]
            self.targets = list(data["test_items"])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, u: int) -> dict[str, torch.Tensor]:
        return {
            "seq": torch.from_numpy(left_pad(self.history[u], self.max_len)),
            "target": torch.tensor(self.targets[u], dtype=torch.long),
            "user": torch.tensor(u, dtype=torch.long),
        }

    def history_mask(self, users: torch.Tensor, n_items: int) -> torch.Tensor:
        """Bool [B, N]: True where item (column j = item id j+1) is in the user's history."""
        mask = torch.zeros(len(users), n_items, dtype=torch.bool)
        for row, u in enumerate(users.tolist()):
            idx = torch.tensor(self.history[u], dtype=torch.long) - 1
            mask[row, idx] = True
        return mask
