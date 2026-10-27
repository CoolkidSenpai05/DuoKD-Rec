"""Train a model from a YAML config; selects best checkpoint on valid, reports test once."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build import build_loss, build_model  # noqa: E402
from src.dataset import load_processed  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.utils import get_logger, load_config, resolve, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None, help="run name (default: config stem + seed)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=0.01")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    name = args.name or f"{Path(args.config).stem}_s{cfg['seed']}"
    out_dir = resolve(cfg["train"]["out_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("train", out_dir / "train.log")
    set_seed(cfg["seed"])
    log.info("run %s seed=%d config=%s", name, cfg["seed"], args.config)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(cfg["data"]["processed_dir"])
    model = build_model(cfg, data["n_items"]).to(device)
    loss_fn = build_loss(cfg).to(device)
    log.info("params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))
    t0 = time.time()
    res = Trainer(model, loss_fn, data, cfg, device, out_dir, log).fit()
    log.info("done in %.1f min, best epoch %d", (time.time() - t0) / 60, res["best_epoch"])
    for split in ("valid", "test"):
        for head, m in res[split].items():
            if not head.startswith("_"):
                log.info("%s %s: %s", split, head, json.dumps({k: round(v, 4) for k, v in m.items()}))


if __name__ == "__main__":
    main()
