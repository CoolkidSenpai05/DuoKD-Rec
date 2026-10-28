# Regression log (M11/M12)

Reference = commit d505772 (pre-M11), run from a git worktree on the same GPU, Beauty, seed 42:
- `ablation_full`, 1 epoch;  `sasrec`, 1 epoch;
- `fullkd`: `ablation_full` 2 epochs with `mut/kd_start_epoch=0`, warm-up 1 (exercises L_mut and L_kd).
Compared with `scripts/compare_runs.py` (every log.csv value + result.json valid/test metrics).

Beauty processed-file hashes before refactor: `beauty_sha256_before.txt`. pytest before: 46 passed.

| step | pytest | Beauty sha256 (data.pkl, stats.json, feats) | smoke ablation_full / sasrec / fullkd |
|---|---|---|---|
| M11.1 registry + refactor | 52 passed | identical (re-running prepare.py into a temp dir also reproduces both hashes) | identical / identical / identical |
