"""Warm-up schedules for the loss weights (CLAUDE.md §3.4)."""
from __future__ import annotations


def linear_warmup(epoch: int, max_value: float, start_epoch: int, warmup_epochs: int) -> float:
    """0 before `start_epoch`, then linear from 0 to `max_value` over `warmup_epochs` epochs.

    epoch = start_epoch -> 0;  epoch >= start_epoch + warmup_epochs -> max_value.
    """
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(max_value)
    return float(max_value) * min(1.0, (epoch - start_epoch) / warmup_epochs)
