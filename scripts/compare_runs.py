"""Regression check: are two run directories bit-identical (log.csv rows + result.json metrics)?

  python scripts/compare_runs.py <run_dir_a> <run_dir_b> [...more pairs]
Exit code 1 on any difference. Timing fields are ignored.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

IGNORE = {"time", "seconds"}


def load(d: Path) -> tuple[list[dict], dict]:
    with open(d / "log.csv") as f:
        rows = [{k: v for k, v in r.items() if k not in IGNORE} for r in csv.DictReader(f)]
    return rows, json.loads((d / "result.json").read_text())


def compare(a: Path, b: Path) -> list[str]:
    (ra, ja), (rb, jb) = load(a), load(b)
    diffs = []
    if len(ra) != len(rb):
        diffs.append(f"log.csv rows {len(ra)} vs {len(rb)}")
    for i, (x, y) in enumerate(zip(ra, rb)):
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k):
                diffs.append(f"log.csv row {i} {k}: {x.get(k)} vs {y.get(k)}")
    for split in ("valid", "test"):
        if ja[split] != jb[split]:
            for h in sorted(set(ja[split]) | set(jb[split])):
                if ja[split].get(h) != jb[split].get(h):
                    diffs.append(f"result {split}/{h} differs")
    if ja["best_epoch"] != jb["best_epoch"]:
        diffs.append("best_epoch differs")
    return diffs


def main() -> None:
    args = [Path(p) for p in sys.argv[1:]]
    assert args and len(args) % 2 == 0, "give pairs of run dirs"
    bad = False
    for a, b in zip(args[::2], args[1::2]):
        d = compare(a, b)
        print(f"{'IDENTICAL' if not d else 'DIFFERENT'}: {a.name} vs {b}")
        for line in d[:20]:
            print("   ", line)
        bad |= bool(d)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
