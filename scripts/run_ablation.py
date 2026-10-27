"""Run ablation configs over several seeds and summarise mean ± std (CLAUDE.md §10).

Each (config, seed) is trained by scripts/train.py in its own process (run dir runs/<stem>_s<seed>);
finished runs (result.json present) are skipped, so the script can be re-run / resumed.
Summary: <ablation.summary_dir>/summary.{md,csv}, test metrics of every head, mean ± std over seeds.

  python scripts/run_ablation.py --configs configs/ablation_*.yaml
  python scripts/run_ablation.py --configs configs/ablation_*.yaml --summary-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import get_logger, load_config, resolve  # noqa: E402

HEAD_ORDER = ["sasrec", "t1", "t2", "ens", "gru"]


def run_dir(cfg: dict, cfg_path: Path, seed: int) -> Path:
    return resolve(cfg["train"]["out_dir"]) / f"{cfg_path.stem}_s{seed}"


def launch(jobs: list[tuple[Path, int]], parallel: int, log) -> None:
    """Run `scripts/train.py` for each (config, seed), at most `parallel` at a time."""
    pending, running = list(jobs), []
    # numpy/mkl-service sets MKL_THREADING_LAYER=INTEL in this process; inherited by a child that
    # imports torch first it aborts ("incompatible with libgomp"), so drop it for the children.
    env = {k: v for k, v in os.environ.items() if k != "MKL_THREADING_LAYER"}
    while pending or running:
        while pending and len(running) < parallel:
            path, seed = pending.pop(0)
            cfg = load_config(path)
            out = resolve(cfg["train"]["out_dir"]) / f"{path.stem}_s{seed}.out"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(path), "--seed", str(seed)]
            log.info("start %s seed %d -> %s", path.name, seed, out)
            fh = open(out, "w")
            running.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env), fh, path, seed))
        time.sleep(5)
        still = []
        for proc, fh, path, seed in running:
            if proc.poll() is None:
                still.append((proc, fh, path, seed))
                continue
            fh.close()
            log.info("%s %s seed %d (exit %d)", "done" if proc.returncode == 0 else "FAILED",
                     path.name, seed, proc.returncode)
        running = still


def summarise(paths: list[Path], seeds: list[int], metrics: list[str]) -> list[dict]:
    """One row per (config, head): mean/std over the seeds that finished."""
    rows = []
    for path in paths:
        cfg = load_config(path)
        results = []
        for s in seeds:
            f = run_dir(cfg, path, s) / "result.json"
            if f.exists():
                results.append(json.loads(f.read_text()))
        if not results:
            continue
        heads = [h for h in HEAD_ORDER if h in results[0]["test"]]
        for h in heads:
            row = {"config": path.stem.removeprefix("ablation_"), "head": h, "n_seeds": len(results),
                   "select": results[0]["select"]}
            for m in metrics:
                vals = np.array([r["test"][h][m] for r in results])
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            sel_head, sel_metric = results[0]["select"].split("/", 1)
            row["valid_select_mean"] = float(np.mean([r["valid"][sel_head][sel_metric] for r in results]))
            row["epoch_mean"] = float(np.mean([r["best_epoch"] for r in results]))
            rows.append(row)
    return rows


def write_summary(rows: list[dict], metrics: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = ["| config | head | seeds | best ep. | " + " | ".join(metrics) + " |",
             "|---|---|---|---|" + "---|" * len(metrics)]
    for r in rows:
        cells = [f"{r[m + '_mean']:.4f} ± {r[m + '_std']:.4f}" for m in metrics]
        lines.append(f"| {r['config']} | {r['head']} | {r['n_seeds']} | {r['epoch_mean']:.0f} | "
                     + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text(
        "Test metrics, mean ± std (ddof=1) over seeds; checkpoint chosen on valid NDCG@20 of the "
        "student head (gru; sasrec for the baseline); full ranking, history excluded.\n\n" + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--base", default="configs/base.yaml", help="config holding the `ablation` section")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override ablation.seeds")
    ap.add_argument("--parallel", type=int, default=None, help="override ablation.parallel")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    acfg = load_config(ROOT / args.base)["ablation"]
    seeds = args.seeds or acfg["seeds"]
    parallel = args.parallel or acfg["parallel"]
    out_dir = resolve(acfg["summary_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("ablation", out_dir / "run_ablation.log")
    paths = [Path(p).resolve() for p in args.configs]

    if not args.summary_only:
        jobs = []
        for p in paths:
            cfg = load_config(p)
            for s in seeds:
                if (run_dir(cfg, p, s) / "result.json").exists():
                    log.info("skip %s seed %d (result.json exists)", p.name, s)
                else:
                    jobs.append((p, s))
        log.info("%d jobs, %d in parallel, seeds %s", len(jobs), parallel, seeds)
        launch(jobs, parallel, log)

    rows = summarise(paths, seeds, acfg["metrics"])
    write_summary(rows, acfg["metrics"], out_dir)
    log.info("summary -> %s (%d rows)", out_dir / "summary.md", len(rows))


if __name__ == "__main__":
    main()
