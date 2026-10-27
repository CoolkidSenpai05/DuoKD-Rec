"""Generic training / evaluation loop with early stopping on valid (CLAUDE.md §6)."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import EvalDataset, TrainDataset
from src.metrics import mask_history, metrics_from_ranks, target_ranks


def _seed_worker(worker_id: int) -> None:
    import random

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_train_loader(data: dict, cfg: dict, seed: int) -> DataLoader:
    ds = TrainDataset(data["train_seqs"], cfg["data"]["max_len"])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False,
                      num_workers=cfg["train"]["num_workers"], worker_init_fn=_seed_worker,
                      generator=g, pin_memory=True, persistent_workers=cfg["train"]["num_workers"] > 0)


class Evaluator:
    """Full-ranking evaluation of every head returned by `model.predict(seq)`.

    Also reports cold-start groups: users with short histories and rarely-seen target items
    (item frequency counted on train sequences only, never valid/test).
    """

    def __init__(self, data: dict, cfg: dict, split: str) -> None:
        self.ds = EvalDataset(data, split, cfg["data"]["max_len"])
        self.n_items = data["n_items"]
        self.ecfg = cfg["eval"]
        self.loader = DataLoader(self.ds, batch_size=self.ecfg["batch_size"], shuffle=False)
        freq = Counter(i for s in data["train_seqs"] for i in s)
        self.cold_user = torch.tensor([len(h) <= self.ecfg["cold_user_max_len"] for h in self.ds.history])
        self.cold_item = torch.tensor([freq[t] <= self.ecfg["cold_item_max_freq"] for t in self.ds.targets])

    @torch.no_grad()
    def run(self, model: nn.Module, device: torch.device) -> dict[str, dict[str, float]]:
        model.eval()
        per: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        users_all = []
        for batch in self.loader:
            seq = batch["seq"].to(device)
            tgt = (batch["target"] - 1).to(device)
            hist = self.ds.history_mask(batch["user"], self.n_items).to(device) \
                if self.ecfg["exclude_history"] else None
            for head, scores in model.predict(seq).items():
                if hist is not None:
                    scores = mask_history(scores, hist)
                m = metrics_from_ranks(target_ranks(scores, tgt), self.ecfg["ks"], self.ecfg["mrr"])
                for k, v in m.items():
                    per[head][k].append(v.cpu())
            users_all.append(batch["user"])
        users = torch.cat(users_all)
        groups = {"all": torch.ones(len(users), dtype=torch.bool),
                  "cold_user": self.cold_user[users], "cold_item": self.cold_item[users]}
        res: dict[str, dict[str, float]] = {}
        for head, mets in per.items():
            res[head] = {}
            for k, vs in mets.items():
                v = torch.cat(vs)
                for g, gm in groups.items():
                    name = k if g == "all" else f"{g}/{k}"
                    res[head][name] = float(v[gm].mean()) if gm.any() else float("nan")
        res["_groups"] = {g: int(gm.sum()) for g, gm in groups.items()}
        return res


class Trainer:
    """Trains `model` with `loss_fn(out, batch, epoch) -> (loss, logs)`.

    The model must implement `forward(seq) -> dict` (training outputs) and
    `predict(seq) -> dict[head, logits [B, N]]`. Optional hooks:
      model.on_epoch_start(epoch), loss_fn.on_epoch_start(epoch),
      model.epoch_diagnostics(out, batch) -> dict[str, float]  (averaged over the epoch).
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, data: dict, cfg: dict,
                 device: torch.device, out_dir: Path, logger: Any) -> None:
        self.model, self.loss_fn, self.cfg, self.device = model, loss_fn, cfg, device
        self.out_dir, self.log = out_dir, logger
        tcfg = cfg["train"]
        params = [p for p in list(model.parameters()) + list(loss_fn.parameters()) if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
        self.train_loader = make_train_loader(data, cfg, cfg["seed"])
        self.valid = Evaluator(data, cfg, "valid")
        self.test = Evaluator(data, cfg, "test")
        self.csv_path = out_dir / "log.csv"
        self.csv_fields: list[str] | None = None

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        for obj in (self.model, self.loss_fn):
            if hasattr(obj, "on_epoch_start"):
                obj.on_epoch_start(epoch)
        sums: dict[str, float] = defaultdict(float)
        n = 0
        for batch in self.train_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            out = self.model(batch["seq"])
            loss, logs = self.loss_fn(out, batch, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: {logs}")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg["train"]["grad_clip"]:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
            self.opt.step()
            logs["loss"] = loss.item()
            if hasattr(self.model, "epoch_diagnostics"):
                with torch.no_grad():
                    logs.update(self.model.epoch_diagnostics(out, batch))
            for k, v in logs.items():
                sums[k] += v
            n += 1
        return {k: v / n for k, v in sums.items()}

    def _write_csv(self, row: dict[str, float]) -> None:
        if self.csv_fields is None:
            self.csv_fields = list(row)
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, self.csv_fields).writeheader()
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, self.csv_fields, extrasaction="ignore").writerow(row)

    def save(self, path: Path, epoch: int, valid: dict) -> None:
        torch.save({"model": self.model.state_dict(), "loss_fn": self.loss_fn.state_dict(),
                    "config": self.cfg, "seed": self.cfg["seed"], "epoch": epoch, "valid": valid}, path)

    def fit(self) -> dict[str, Any]:
        tcfg = self.cfg["train"]
        head, metric = tcfg["select_head"], tcfg["select_metric"]
        best, best_epoch, bad = -1.0, -1, 0
        ckpt = self.out_dir / "best.pt"
        for epoch in range(tcfg["epochs"]):
            t0 = time.time()
            logs = self.train_epoch(epoch)
            row: dict[str, float] = {"epoch": epoch, **logs}
            if (epoch + 1) % tcfg["eval_every"] == 0:
                res = self.valid.run(self.model, self.device)
                for h, mets in res.items():
                    if h.startswith("_"):
                        continue
                    for k in self.cfg["eval"]["ks"]:
                        row[f"valid/{h}/Recall@{k}"] = mets[f"Recall@{k}"]
                        row[f"valid/{h}/NDCG@{k}"] = mets[f"NDCG@{k}"]
                score = res[head][metric]
                summary = " | ".join(f"{h}: N@20={m['NDCG@20']:.4f} R@20={m['Recall@20']:.4f}"
                                     for h, m in res.items() if not h.startswith("_"))
                self.log.info("epoch %d (%.0fs) loss=%.4f | %s", epoch, time.time() - t0,
                              logs["loss"], summary)
                extra = {k: round(v, 4) for k, v in logs.items() if k != "loss"}
                self.log.info("  train logs: %s", json.dumps(extra))
                if score > best:
                    best, best_epoch, bad = score, epoch, 0
                    self.save(ckpt, epoch, res)
                else:
                    bad += 1
            self._write_csv(row)
            if bad >= tcfg["patience"]:
                self.log.info("early stop at epoch %d (best epoch %d, valid %s/%s=%.4f)",
                              epoch, best_epoch, head, metric, best)
                break
        # final: reload best checkpoint, report valid + test once
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        valid = self.valid.run(self.model, self.device)
        test = self.test.run(self.model, self.device)
        result = {"best_epoch": best_epoch, "select": f"{head}/{metric}", "valid": valid, "test": test,
                  "seed": self.cfg["seed"]}
        with open(self.out_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        return result
