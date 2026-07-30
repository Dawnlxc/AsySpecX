"""Chronological and purged split utilities for SafeRoute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


class PurgedTimeSeriesSplit:
    """Expanding-window CV with contiguous validation and an origin purge gap."""

    def __init__(self, n_splits: int = 4, purge_steps: int = 0, pred_len: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)
        self.purge_steps = int(purge_steps)
        self.pred_len = int(pred_len)

    @property
    def effective_purge(self) -> int:
        requested = self.purge_steps if self.purge_steps > 0 else self.pred_len
        return max(requested, self.pred_len)

    def split(self, X=None, origins: Optional[Sequence[int]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if origins is None:
            if X is None:
                raise ValueError("X or origins is required")
            origins_arr = np.arange(len(X), dtype=np.int64)
        else:
            origins_arr = np.asarray(origins, dtype=np.int64)
        if origins_arr.ndim != 1 or origins_arr.size < self.n_splits + 1:
            raise ValueError("not enough one-dimensional origins for requested folds")
        order = np.argsort(origins_arr, kind="stable")
        sorted_origins = origins_arr[order]
        boundaries = np.linspace(0, len(order), self.n_splits + 2, dtype=int)
        yielded = 0
        for fold in range(self.n_splits):
            val_start_pos = boundaries[fold + 1]
            val_end_pos = boundaries[fold + 2]
            if val_end_pos <= val_start_pos:
                continue
            val_idx = order[val_start_pos:val_end_pos]
            first_val_origin = sorted_origins[val_start_pos]
            train_limit = first_val_origin - self.effective_purge
            train_mask = sorted_origins[:val_start_pos] < train_limit
            train_idx = order[:val_start_pos][train_mask]
            if train_idx.size == 0 or val_idx.size == 0:
                continue
            yielded += 1
            yield train_idx, val_idx
        if yielded == 0:
            raise ValueError(
                "purge removed every training fold; reduce folds/purge or provide more samples"
            )


def rolling_oof_windows(
    n_samples: int,
    purge_steps: int,
    train_fractions: Sequence[float] = (0.6, 0.8),
    validation_fraction: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the locked 60/20 and 80/20 chronological OOF windows."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0,1)")
    windows = []
    width = max(1, int(round(n_samples * validation_fraction)))
    for fraction in train_fractions:
        if not (0.0 < fraction < 1.0):
            raise ValueError("train fractions must be in (0,1)")
        val_start = int(round(n_samples * fraction))
        val_end = min(n_samples, val_start + width)
        train_end = max(0, val_start - int(purge_steps))
        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(val_start, val_end, dtype=np.int64)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("rolling OOF window is empty after purge")
        if int(train_idx[-1]) + int(purge_steps) >= int(val_idx[0]):
            raise AssertionError("rolling OOF chronology/purge invariant failed")
        windows.append((train_idx, val_idx))
    return windows
