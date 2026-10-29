"""Datasets that add timestamps to the standard samples (M12, HM4SR). Old formats are unchanged.

Extra keys: ts [L] float64 (unix seconds of the input items, 0 at padding, left padded like `seq`),
target_ts [] float64 (time of the target; only ever read when a model sets use_target_time).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.dataset import EvalDataset, TrainDataset


def left_pad_float(values: list[int], max_len: int) -> np.ndarray:
    values = values[-max_len:]
    out = np.zeros(max_len, dtype=np.float64)
    if values:
        out[-len(values):] = values
    return out


class TimedTrainDataset(TrainDataset):
    def __init__(self, train_seqs: list[list[int]], train_ts: list[list[int]], max_len: int) -> None:
        super().__init__(train_seqs, max_len)
        self.train_ts = train_ts

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = super().__getitem__(idx)
        u, t = self.index[idx]
        ts = self.train_ts[u]
        out["ts"] = torch.from_numpy(left_pad_float(ts[:t], self.max_len))
        out["target_ts"] = torch.tensor(float(ts[t]), dtype=torch.float64)
        return out


class TimedEvalDataset(EvalDataset):
    def __init__(self, data: dict[str, Any], split: str, max_len: int) -> None:
        super().__init__(data, split, max_len)
        if split == "valid":
            self.history_ts = [list(t) for t in data["train_ts"]]
            self.target_ts = list(data["valid_ts"])
        else:
            self.history_ts = [list(t) + [v] for t, v in zip(data["train_ts"], data["valid_ts"])]
            self.target_ts = list(data["test_ts"])

    def __getitem__(self, u: int) -> dict[str, torch.Tensor]:
        out = super().__getitem__(u)
        out["ts"] = torch.from_numpy(left_pad_float(self.history_ts[u], self.max_len))
        out["target_ts"] = torch.tensor(float(self.target_ts[u]), dtype=torch.float64)
        return out
