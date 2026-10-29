"""Benchmark runner (M12 §3): datasets x models x seeds through scripts/train.py.

  python scripts/run_benchmark.py --datasets beauty toys --models gru4rec sasrec duokd_full --seeds 42 43 44
  python scripts/run_benchmark.py ... --dry-run          # print the commands only
  python scripts/run_benchmark.py ... --smoke            # small user subset, <= 2 epochs, runs/smoke/ (not results)
  python scripts/run_benchmark.py ... --estimate         # ~50 train batches on the full data -> GPU-hours
  python scripts/run_benchmark.py ... --summary-only

Run dirs: <dataset runs_dir>/benchmark/<model>_s<seed>   (beauty: runs/benchmark/..., toys: runs/toys/benchmark/...)
          runs/smoke/<dataset>/<model>_s<seed>            with --smoke
A run with result.json is skipped (resume). Summary per dataset: <...>/benchmark/summary.{md,csv}, same
format as the M10 ablation table (test metrics, mean ± std over seeds); smoke: runs/smoke/summary.md.
`duokd_full` = configs/ablation_full.yaml (M10 reference row), runnable on every dataset.
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
from src.utils import get_logger, load_config, load_dataset_entry, resolve  # noqa: E402

ALL_DATASETS = ["beauty", "sports", "toys", "clothing"]
ALL_MODELS = ["gru4rec", "sasrec", "bert4rec", "bsarec", "unisrec", "missrec", "tedrec", "hm4sr", "ddmsr", "duokd_full"]
METRICS = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MRR", "cold_user/NDCG@20", "cold_item/NDCG@20"]
# epochs actually trained (best epoch + patience 10) observed in M3/M10 on Beauty; used by --estimate
EST_EPOCHS = {"duokd_full": 49, "sasrec": 43}
EST_EPOCHS_DEFAULT = 45


def config_path(model: str) -> Path:
    return ROOT / ("configs/ablation_full.yaml" if model == "duokd_full" else f"configs/baselines/{model}.yaml")


def run_root(ds: str, smoke: bool) -> Path:
    if smoke:
        return resolve("runs/smoke") / ds
    return resolve(load_dataset_entry(ds)["paths"]["runs_dir"]) / "benchmark"


def overrides(ds: str, args: argparse.Namespace) -> list[str]:
    ov = [f"data.dataset={ds}", f"train.out_dir={run_root(ds, args.smoke)}"]
    if args.smoke:
        ov += [f"data.limit_users={args.smoke_users}", f"train.epochs={args.smoke_epochs}", "train.patience=100"]
    return ov


def command(ds: str, model: str, seed: int, args: argparse.Namespace) -> list[str]:
    return [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config_path(model)), "--seed", str(seed),
            "--name", f"{model}_s{seed}", "--set", *overrides(ds, args)]


def head_of(model: str) -> list[str]:
    return ["t1", "t2", "ens", "gru"] if model == "duokd_full" else [model]


def launch(jobs: list[tuple[str, str, int]], args: argparse.Namespace, log) -> None:
    """Run jobs with at most args.parallel processes; stdout -> <run dir>.out."""
    env = {k: v for k, v in os.environ.items() if k != "MKL_THREADING_LAYER"}  # see run_ablation.py
    pending, running = list(jobs), []
    while pending or running:
        while pending and len(running) < args.parallel:
            ds, model, seed = pending.pop(0)
            out = run_root(ds, args.smoke) / f"{model}_s{seed}.out"
            out.parent.mkdir(parents=True, exist_ok=True)
            log.info("start %s %s seed %d -> %s", ds, model, seed, out)
            fh = open(out, "w")
            p = subprocess.Popen(command(ds, model, seed, args), stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            running.append((p, fh, ds, model, seed, time.time()))
        time.sleep(3)
        still = []
        for p, fh, ds, model, seed, t0 in running:
            if p.poll() is None:
                still.append((p, fh, ds, model, seed, t0))
                continue
            fh.close()
            log.info("%s %s %s seed %d (exit %d, %.1f min)", "done" if p.returncode == 0 else "FAILED",
                     ds, model, seed, p.returncode, (time.time() - t0) / 60)
        running = still


# ---------------------------------------------------------------------------------------- summaries
def summarise(ds: str, models: list[str], seeds: list[int], smoke: bool) -> list[dict]:
    rows = []
    for model in models:
        results = [json.loads(f.read_text()) for s in seeds
                   if (f := run_root(ds, smoke) / f"{model}_s{s}" / "result.json").exists()]
        if not results:
            continue
        for h in [h for h in head_of(model) if h in results[0]["test"]]:
            row = {"dataset": ds, "model": model, "head": h, "n_seeds": len(results), "select": results[0]["select"]}
            for m in METRICS:
                vals = np.array([r["test"][h][m] for r in results])
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row["epoch_mean"] = float(np.mean([r["best_epoch"] for r in results]))
            rows.append(row)
    return rows


def write_summary(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| model | head | seeds | best ep. | " + " | ".join(METRICS) + " |", "|---|---|---|---|" + "---|" * len(METRICS)]
    for r in rows:
        cells = [f"{r[m + '_mean']:.4f} ± {r[m + '_std']:.4f}" for m in METRICS]
        lines.append(f"| {r['model']} | {r['head']} | {r['n_seeds']} | {r['epoch_mean']:.0f} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text(
        f"Dataset `{rows[0]['dataset']}`. Test metrics, mean ± std (ddof=1) over seeds; checkpoint chosen on valid "
        "NDCG@20 of each model (DuoKD: GRU head); full ranking, history excluded.\n\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------- smoke checks
RESULT_KEYS = ["best_epoch", "select", "valid", "test", "seed"]
GROUP_KEYS = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "MRR"]


def check_smoke_run(ds: str, model: str, seed: int) -> dict:
    """Pass/fail checks of one smoke run (PROMPT §4)."""
    import math

    d = run_root(ds, True) / f"{model}_s{seed}"
    out: dict = {"dataset": ds, "model": model, "seed": seed, "checks": {}, "notes": ""}
    c = out["checks"]
    rj = d / "result.json"
    if not rj.exists():
        tail = (d.parent / f"{model}_s{seed}.out").read_text().splitlines()[-3:] if (d.parent / f"{model}_s{seed}.out").exists() else []
        out["notes"] = " / ".join(tail)[-300:]
        c["finished"] = False
        return out
    c["finished"] = True
    res = json.loads(rj.read_text())
    ok = all(k in res for k in RESULT_KEYS)
    for split in ("valid", "test"):
        for h in head_of(model):
            m = res[split].get(h, {})
            ok &= all(k in m and f"cold_user/{k}" in m and f"cold_item/{k}" in m for k in GROUP_KEYS)
        ok &= set(res[split].get("_groups", {})) == {"all", "cold_user", "cold_item"}
    c["result_schema"] = bool(ok)
    with open(d / "log.csv") as f:
        losses = [float(r["loss"]) for r in csv.DictReader(f)]
    c["loss_finite"] = all(math.isfinite(x) for x in losses)
    c["loss_decreases"] = len(losses) >= 2 and losses[-1] < losses[0]
    h = head_of(model)[-1]
    out["test_N@20"] = res["test"][h]["NDCG@20"]
    out["test_R@20"] = res["test"][h]["Recall@20"]
    c["metric_nonzero"] = out["test_R@20"] > 0
    info = json.loads((d / "run_info.json").read_text()) if (d / "run_info.json").exists() else {}
    out.update({k: info.get(k) for k in ("n_params", "minutes", "peak_mem_mb")})
    c["checkpoint_reload"] = _reload_ok(d)
    return out


def _reload_ok(d: Path) -> bool:
    """best.pt -> fresh model of the saved config -> load_state_dict(strict=True)."""
    import torch

    from src.build import build_model, needs_timestamps
    from src.dataset import limit_users, load_processed

    try:
        state = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
        cfg = state["config"]
        data = limit_users(load_processed(cfg["data"]["processed_dir"], with_timestamps=needs_timestamps(cfg)),
                           cfg["data"].get("limit_users"))
        model = build_model(cfg, data["n_items"], data)
        model.load_state_dict(state["model"], strict=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"reload failed for {d}: {e}")
        return False


def write_smoke_summary(rows: list[dict], resume_ok: bool | None, args: argparse.Namespace) -> Path:
    names = ["finished", "loss_finite", "loss_decreases", "result_schema", "checkpoint_reload", "metric_nonzero"]
    lines = ["# Smoke matrix (M12) — NOT RESULTS", "",
             f"Subset: first {args.smoke_users} users per dataset (full catalog), {args.smoke_epochs} epochs, seed(s) "
             f"{args.seeds}. Metrics below are only a sanity check (non-zero / plausible); they are **not** results.",
             f"Resume check (second invocation skips every finished run): **{'PASS' if resume_ok else 'not run' if resume_ok is None else 'FAIL'}**.",
             "", "| dataset | model | " + " | ".join(names) + " | PASS | minutes | peak MB | params | test R@20 (smoke) | notes |",
             "|---|---|" + "---|" * (len(names) + 6)]
    for r in rows:
        c = r["checks"]
        cells = ["✓" if c.get(n) else "✗" for n in names]
        passed = all(c.get(n) for n in names)
        fmt = lambda v, f: (f.format(v) if v is not None else "-")  # noqa: E731
        lines.append(f"| {r['dataset']} | {r['model']} | " + " | ".join(cells) + f" | {'PASS' if passed else '**FAIL**'} | "
                     f"{fmt(r.get('minutes'), '{:.1f}')} | {fmt(r.get('peak_mem_mb'), '{:.0f}')} | "
                     f"{fmt(r.get('n_params'), '{:,}')} | {fmt(r.get('test_R@20'), '{:.4f}')} | {r['notes']} |")
    out = resolve("runs/smoke/summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.parent / "summary.json").write_text(json.dumps(rows, indent=2))
    return out


# ---------------------------------------------------------------------------------------- estimate
def estimate_one(ds: str, model: str, n_batches: int) -> dict:
    """Time n_batches training steps + a few eval batches on the FULL data (one process, GPU)."""
    import torch

    from src.build import build_loss, build_model, needs_timestamps
    from src.dataset import load_processed
    from src.trainer import Evaluator, Trainer
    from src.utils import set_seed

    cfg = load_config(config_path(model), [f"data.dataset={ds}", "train.num_workers=2"])
    set_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(cfg["data"]["processed_dir"], with_timestamps=needs_timestamps(cfg))
    model_ = build_model(cfg, data["n_items"], data).to(dev)
    tr = Trainer(model_, build_loss(cfg).to(dev), data, cfg, dev, Path("/nonexistent"), get_logger("est"))
    if hasattr(model_, "on_epoch_start"):
        t0 = time.time()
        model_.on_epoch_start(0)
        epoch_hook = time.time() - t0
    else:
        epoch_hook = 0.0
    tr.model.train()
    it = iter(tr.train_loader)
    times = []
    for i in range(min(n_batches + 5, len(tr.train_loader))):
        batch = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        if hasattr(model_, "compute_loss"):
            loss = model_.compute_loss(batch)["loss"]
        else:
            loss, _ = tr.loss_fn(model_(batch["seq"]), batch, 10)  # epoch 10: every DuoKD loss term active
        tr.opt.zero_grad()
        loss.backward()
        tr.opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= 5:
            times.append(time.time() - t0)
    ev = Evaluator(data, cfg, "valid", bool(getattr(model_, "needs_timestamps", False)))
    t0 = time.time()
    n_eval = 0
    model_.eval()
    with torch.no_grad():
        for j, b in enumerate(ev.loader):
            b = {k: v.to(dev) for k, v in b.items()}
            if hasattr(model_, "score_all"):
                model_.score_all(b)
            else:
                model_.predict(b["seq"])
            n_eval += 1
            if j >= 4:
                break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    eval_batch = (time.time() - t0) / max(n_eval, 1)
    per_batch = float(np.median(times))
    epoch_s = per_batch * len(tr.train_loader) + epoch_hook + 2 * eval_batch * len(ev.loader)  # train + valid (+test once)
    return {"dataset": ds, "model": model, "batch_size": cfg["train"]["batch_size"], "train_batches": len(tr.train_loader),
            "sec_per_batch": per_batch, "eval_sec_per_batch": eval_batch, "epoch_minutes": epoch_s / 60,
            "est_epochs": EST_EPOCHS.get(model, EST_EPOCHS_DEFAULT),
            "peak_mem_mb": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None}


def run_estimates(datasets: list[str], models: list[str], seeds: list[int], n_batches: int, log) -> None:
    env = {k: v for k, v in os.environ.items() if k != "MKL_THREADING_LAYER"}
    rows = []
    for ds in datasets:
        for model in models:
            cmd = [sys.executable, str(Path(__file__)), "--_estimate-one", ds, model, "--estimate-batches", str(n_batches)]
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
            line = [x for x in p.stdout.splitlines() if x.startswith("ESTIMATE ")]
            if p.returncode != 0 or not line:
                log.info("estimate FAILED %s %s: %s", ds, model, p.stderr[-500:])
                rows.append({"dataset": ds, "model": model, "error": p.stderr[-300:]})
                continue
            r = json.loads(line[-1][len("ESTIMATE "):])
            r["gpu_hours_all_seeds"] = r["epoch_minutes"] * r["est_epochs"] * len(seeds) / 60
            log.info("estimate %s %s: %.3f s/batch, %.1f min/epoch, %.2f GPU-h (%d seeds)", ds, model,
                     r["sec_per_batch"], r["epoch_minutes"], r["gpu_hours_all_seeds"], len(seeds))
            rows.append(r)
    out = resolve("runs/estimate")
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimate.json").write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if "error" not in r]
    lines = ["| dataset | model | batch | s/batch | min/epoch | est. epochs | GPU-h (all seeds) | peak MB |",
             "|---|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['batch_size']} | {r['sec_per_batch']:.3f} | "
                     f"{r['epoch_minutes']:.1f} | {r['est_epochs']} | {r['gpu_hours_all_seeds']:.2f} | "
                     f"{r['peak_mem_mb'] or 0:.0f} |")
    total = sum(r["gpu_hours_all_seeds"] for r in ok)
    lines.append(f"\n**Total: {total:.1f} GPU-hours** for {len(ok)} (dataset, model) pairs x {len(seeds)} seeds "
                 "(sequential, one job at a time; measured while the GPU/CPU were shared with other users).")
    for r in rows:
        if "error" in r:
            lines.append(f"- FAILED {r['dataset']} {r['model']}: {r['error'][-200:]}")
    (out / "estimate.md").write_text("\n".join(lines) + "\n")
    log.info("estimate -> %s (total %.1f GPU-h)", out / "estimate.md", total)


# ---------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-users", type=int, default=500)
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--estimate-batches", type=int, default=50)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--_estimate-one", nargs=2, metavar=("DATASET", "MODEL"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._estimate_one:
        print("ESTIMATE " + json.dumps(estimate_one(*args._estimate_one, args.estimate_batches)))
        return
    for m in args.models:
        assert m in ALL_MODELS, f"unknown model {m}"
    log_dir = resolve("runs/smoke" if args.smoke else "runs/estimate" if args.estimate else "runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("benchmark", None if args.dry_run else log_dir / "run_benchmark.log")
    if args.estimate:
        run_estimates(args.datasets, args.models, args.seeds, args.estimate_batches, log)
        return

    jobs, skipped = [], 0
    for ds in args.datasets:
        for model in args.models:
            for s in args.seeds:
                if (run_root(ds, args.smoke) / f"{model}_s{s}" / "result.json").exists():
                    skipped += 1
                    if not args.dry_run:
                        log.info("skip %s %s seed %d (result.json exists)", ds, model, s)
                else:
                    jobs.append((ds, model, s))
    if args.dry_run:
        for ds, model, s in jobs:
            print(" ".join(command(ds, model, s, args)))
        print(f"# {len(jobs)} runs to launch, {skipped} already finished (skipped)")
        return
    if not args.summary_only:
        log.info("%d jobs (%d skipped), %d in parallel", len(jobs), skipped, args.parallel)
        launch(jobs, args, log)
    if args.smoke:
        rows = [check_smoke_run(ds, m, s) for ds in args.datasets for m in args.models for s in args.seeds]
        # resume check = a second invocation of the same matrix launches nothing (every run skipped)
        resume_ok = (len(jobs) == 0 and skipped == len(rows)) if (skipped and not args.summary_only) else None
        out = write_smoke_summary(rows, resume_ok, args)
        log.info("smoke summary -> %s", out)
        return
    for ds in args.datasets:
        rows = summarise(ds, args.models, args.seeds, False)
        write_summary(rows, run_root(ds, False))
        log.info("%s: summary -> %s (%d rows)", ds, run_root(ds, False) / "summary.md", len(rows))


if __name__ == "__main__":
    main()
