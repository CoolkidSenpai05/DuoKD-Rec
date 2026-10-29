"""Evaluate a checkpoint on valid and test (all heads)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build import build_model, needs_timestamps  # noqa: E402
from src.dataset import load_processed  # noqa: E402
from src.trainer import Evaluator  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    args = ap.parse_args()
    log = get_logger("eval")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("config") or load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(cfg["data"]["processed_dir"], with_timestamps=needs_timestamps(cfg))
    model = build_model(cfg, data["n_items"], data).to(device)
    model.load_state_dict(state["model"])
    log.info("exclude_history=%s", cfg["eval"]["exclude_history"])
    for split in args.splits:
        res = Evaluator(data, cfg, split, needs_timestamps(cfg)).run(model, device)
        for head, m in res.items():
            log.info("%s %s: %s", split, head, json.dumps({k: round(v, 4) for k, v in m.items()}))


if __name__ == "__main__":
    main()
