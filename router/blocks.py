"""Horizon-block definitions and loss calculations."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def horizon_blocks(horizon: int, num_blocks: int = 4) -> List[Tuple[int, int]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > horizon:
        num_blocks = horizon
    base, extra = divmod(horizon, num_blocks)
    blocks: List[Tuple[int, int]] = []
    start = 0
    for block in range(num_blocks):
        width = base + (1 if block < extra else 0)
        blocks.append((start, start + width))
        start += width
    if blocks[0][0] != 0 or blocks[-1][1] != horizon:
        raise AssertionError("internal horizon partition error")
    return blocks


def block_losses(
    predictions: np.ndarray,
    target: np.ndarray,
    blocks: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-sample, per-expert, per-block MSE and MAE.

    ``predictions`` is ``[N, K, H, C]`` and ``target`` is ``[N, H, C]``.
    Arrays are batch-local; callers must not materialize a full dataset tensor.
    """

    pred = np.asarray(predictions)
    true = np.asarray(target)
    if pred.ndim != 4 or true.ndim != 3:
        raise ValueError("predictions must be [N,K,H,C] and target [N,H,C]")
    if pred.shape[0] != true.shape[0] or pred.shape[2:] != true.shape[1:]:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    mse = np.empty((pred.shape[0], pred.shape[1], len(blocks)), dtype=np.float64)
    mae = np.empty_like(mse)
    for block_index, (start, end) in enumerate(blocks):
        if not (0 <= start < end <= pred.shape[2]):
            raise ValueError(f"invalid horizon block {(start, end)} for H={pred.shape[2]}")
        error = pred[:, :, start:end, :] - true[:, None, start:end, :]
        mse[:, :, block_index] = np.mean(np.square(error, dtype=np.float64), axis=(2, 3))
        mae[:, :, block_index] = np.mean(np.abs(error, dtype=np.float64), axis=(2, 3))
    return mse, mae


def advantage_targets(
    mse: np.ndarray,
    anchor_index: int = 0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return advantage and log-relative-regret for non-anchor experts."""

    values = np.asarray(mse, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("mse must include an expert dimension")
    if not (0 <= anchor_index < values.shape[1]):
        raise ValueError("anchor_index out of range")
    alternatives = [index for index in range(values.shape[1]) if index != anchor_index]
    anchor = values[:, anchor_index : anchor_index + 1, ...]
    alt = values[:, alternatives, ...]
    advantage = anchor - alt
    relative_regret = np.log(alt + eps) - np.log(anchor + eps)
    return advantage, relative_regret
