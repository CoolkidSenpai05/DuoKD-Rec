"""Latency / memory of the inference paths: T1, T2 (full sequence) vs GRU (full sequence and one step).

All paths use cached item tables (E_t / E_v / E_f computed once), so the timing covers the sequence
model + scoring against the full catalog, i.e. what runs per request. Settings in `bench` (base.yaml).

  python scripts/benchmark.py --ckpt runs/ablation_full_s42/best.pt
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.build import build_model  # noqa: E402
from src.dataset import load_processed  # noqa: E402
from src.utils import get_logger, load_config, resolve, set_seed  # noqa: E402


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def time_fn(fn, dev: torch.device, warmup: int, iters: int) -> dict[str, float]:
    """Median / p90 latency in ms and peak extra memory (MiB, CUDA only) of fn()."""
    for _ in range(warmup):
        fn()
    _sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    res = {"ms_median": statistics.median(ts), "ms_p90": ts[int(0.9 * (len(ts) - 1))]}
    if dev.type == "cuda":
        res["peak_mem_MiB"] = (torch.cuda.max_memory_allocated(dev) - base) / 2**20
    return res


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="DuoKD best.pt with all three heads")
    ap.add_argument("--student", default=None, help="optional exported student file (size is reported)")
    ap.add_argument("--base", default="configs/base.yaml", help="config holding the `bench` section")
    args = ap.parse_args()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    bcfg = load_config(ROOT / args.base)["bench"]
    out_dir = resolve(load_config(ROOT / args.base)["ablation"]["summary_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("bench", out_dir / "benchmark.log")
    set_seed(cfg["seed"])
    data = load_processed(cfg["data"]["processed_dir"])
    model = build_model(cfg, data["n_items"])
    model.load_state_dict(state["model"])
    model.eval()
    L = cfg["data"]["max_len"]

    report: dict = {"ckpt": args.ckpt, "max_len": L, "n_items": data["n_items"],
                    "params": {h: n_params(model.towers[h]) for h in model.heads},
                    "params_item_encoder": n_params(model.item_encoder), "latency": {}}
    report["params_full_model"] = n_params(model)
    if args.student:
        report["student_file_MiB"] = Path(args.student).stat().st_size / 2**20
    report["full_ckpt_MiB"] = Path(args.ckpt).stat().st_size / 2**20

    # realistic inputs: the last L items of real test histories
    from src.dataset import EvalDataset

    ds = EvalDataset(data, "test", L)
    for dev_name in bcfg["devices"]:
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        dev = torch.device(dev_name)
        model.to(dev)
        with torch.no_grad():
            tables = model.item_encoder(detach_shared=True)
        tabs = {"t1": tables["E_t"], "t2": tables["E_v"], "gru": tables["E_f"]}
        for B in bcfg["batch_sizes"]:
            seq = torch.stack([ds[i]["seq"] for i in range(B)]).to(dev)
            mask = seq != 0
            res = {}
            for h in ("t1", "t2"):
                if h in model.heads:
                    E = tabs[h]
                    res[h] = time_fn(lambda E=E, h=h: model.towers[h](E[seq], mask)[:, -1] @ E[1:].T,
                                     dev, bcfg["warmup_iters"], bcfg["iters"])
            E = tabs["gru"]
            gru = model.towers["gru"]
            res["gru_full_seq"] = time_fn(lambda: gru(E[seq], mask) @ E[1:].T,
                                          dev, bcfg["warmup_iters"], bcfg["iters"])
            with torch.no_grad():
                _, state_h = gru.step(E[seq[:, -2]], None)     # some previous state
            res["gru_step"] = time_fn(lambda: gru.step(E[seq[:, -1]], state_h)[0] @ E[1:].T,
                                      dev, bcfg["warmup_iters"], bcfg["iters"])
            report["latency"][f"{dev_name}_B{B}"] = res
            for k, v in res.items():
                log.info("%s B=%d %-13s %s", dev_name, B, k, json.dumps({a: round(b, 3) for a, b in v.items()}))
        model.cpu()
    report["user_state_bytes"] = {
        "gru": cfg["model"]["gru"]["n_layers"] * cfg["model"]["hidden"] * 4,     # hidden state per user
        "transformer": L * 8,                                                    # last L item ids (int64)
    }
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2))
    log.info("params %s, item encoder %d; written %s", report["params"], report["params_item_encoder"],
             out_dir / "benchmark.json")


if __name__ == "__main__":
    main()
