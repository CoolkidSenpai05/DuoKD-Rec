"""Export the GRU student (no T1/T2) from a DuoKD checkpoint.

--verify: evaluate exported student vs. the GRU head of the full model on valid/test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build import build_model  # noqa: E402
from src.dataset import load_processed  # noqa: E402
from src.models.student import StudentRecommender, export_student  # noqa: E402
from src.trainer import Evaluator  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    log = get_logger("export")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    set_seed(cfg["seed"])
    data = load_processed(cfg["data"]["processed_dir"])
    model = build_model(cfg, data["n_items"])
    model.load_state_dict(state["model"])
    assert "gru" in model.heads, "checkpoint has no GRU student"
    st = export_student(model, args.out, cfg)
    full = sum(v.numel() for v in state["model"].values())
    stu = st["E_f"].numel() + sum(v.numel() for v in st["gru"].values())
    log.info("exported %s | full model %d params, student (E_f cache + GRU) %d values, %.1f MB on disk",
             args.out, full, stu, Path(args.out).stat().st_size / 2**20)
    if args.verify:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        student = StudentRecommender.from_checkpoint(args.out, device).to(device)
        model.to(device)
        for split in ("valid", "test"):
            ev = Evaluator(data, cfg, split)
            r_full = ev.run(model, device)["gru"]
            r_stu = ev.run(student, device)["student"]
            diff = max(abs(r_full[k] - r_stu[k]) for k in r_full)
            log.info("%s GRU head: %s", split, json.dumps({k: round(r_full[k], 4) for k in ("Recall@20", "NDCG@20")}))
            log.info("%s student : %s | max |diff| over all metrics = %.2e", split,
                     json.dumps({k: round(r_stu[k], 4) for k in ("Recall@20", "NDCG@20")}), diff)


if __name__ == "__main__":
    main()
