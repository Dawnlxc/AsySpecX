"""Batch-local feature and label assembly for compact router metadata."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .blocks import advantage_targets, block_losses
from .features import context_features, forecast_features


def compact_meta_batch(
    context: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    seed_variances: Mapping[str, np.ndarray],
    sample_ids: np.ndarray,
    blocks: Sequence[Tuple[int, int]],
    anchor_name: str,
    periods: Sequence[int],
    dataset: str,
    router_scope: str,
    target: np.ndarray | None = None,
    snippet_bins: int = 0,
    max_channels: int = 64,
    channel_groups: np.ndarray | None = None,
) -> Tuple[Dict[str, np.ndarray], Sequence[str]]:
    """Build sample-block rows without materialising dataset-wide predictions."""

    names = list(predictions)
    if anchor_name not in names:
        raise ValueError("anchor prediction missing from batch")
    if target is not None and not np.isfinite(np.asarray(target)).all():
        raise FloatingPointError("router metadata target contains NaN or Inf")
    for name in names:
        if not np.isfinite(np.asarray(predictions[name])).all():
            raise FloatingPointError(f"expert prediction {name!r} contains NaN or Inf")
        if not np.isfinite(np.asarray(seed_variances[name])).all():
            raise FloatingPointError(f"expert seed variance {name!r} contains NaN or Inf")
    anchor_index = names.index(anchor_name)
    ctx, ctx_names = context_features(
        context,
        pred_len=np.asarray(predictions[anchor_name]).shape[1],
        periods=periods,
        dataset=dataset,
        router_scope=router_scope,
        max_channels=max_channels,
    )
    forecast, forecast_names = forecast_features(
        context,
        predictions,
        seed_variances,
        blocks,
        anchor_name,
        snippet_bins=snippet_bins,
    )
    batch, n_blocks, _ = forecast.shape
    features = np.concatenate(
        (
            np.repeat(ctx[:, None, :], n_blocks, axis=1),
            forecast,
        ),
        axis=2,
    ).reshape(batch * n_blocks, -1)
    block_index = np.tile(np.arange(n_blocks, dtype=np.int16), batch)
    starts = np.tile(np.asarray([item[0] for item in blocks], dtype=np.int32), batch)
    ends = np.tile(np.asarray([item[1] for item in blocks], dtype=np.int32), batch)
    arrays: Dict[str, np.ndarray] = {
        "features": features.astype(np.float32, copy=False),
        "sample_id": np.repeat(np.asarray(sample_ids, dtype=np.int64), n_blocks),
        "origin": np.repeat(np.asarray(sample_ids, dtype=np.int64), n_blocks),
        "block": block_index,
        "block_start": starts,
        "block_end": ends,
    }

    seed_summary = np.empty((batch, n_blocks, len(names)), dtype=np.float32)
    for expert_index, name in enumerate(names):
        variance = np.asarray(seed_variances[name], dtype=np.float64)
        for block, (start, end) in enumerate(blocks):
            seed_summary[:, block, expert_index] = variance[:, start:end, :].mean(axis=(1, 2))
    arrays["seed_variance"] = seed_summary.reshape(batch * n_blocks, len(names))

    if target is not None:
        prediction_stack = np.stack([predictions[name] for name in names], axis=1)
        mse, mae = block_losses(prediction_stack, target, blocks)
        mse_rows = np.transpose(mse, (0, 2, 1)).reshape(batch * n_blocks, len(names))
        mae_rows = np.transpose(mae, (0, 2, 1)).reshape(batch * n_blocks, len(names))
        advantage, relative_regret = advantage_targets(mse, anchor_index=anchor_index)
        arrays.update(
            {
                "loss_mse": mse_rows.astype(np.float32),
                "loss_mae": mae_rows.astype(np.float32),
                "advantage": np.transpose(advantage, (0, 2, 1)).reshape(batch * n_blocks, -1).astype(np.float32),
                "relative_regret": np.transpose(relative_regret, (0, 2, 1)).reshape(batch * n_blocks, -1).astype(np.float32),
            }
        )
        if channel_groups is not None:
            assignments = np.asarray(channel_groups, dtype=np.int64)
            if assignments.shape != (target.shape[2],):
                raise ValueError("channel group assignment does not match target channels")
            unique_groups = np.unique(assignments)
            if not np.array_equal(unique_groups, np.arange(len(unique_groups))):
                raise ValueError("channel group labels must be contiguous from zero")
            group_loss = np.empty(
                (batch, n_blocks, len(names), len(unique_groups)), dtype=np.float32
            )
            for block_index, (start, end) in enumerate(blocks):
                for group in unique_groups:
                    channel_mask = assignments == group
                    error = (
                        prediction_stack[:, :, start:end, channel_mask]
                        - target[:, None, start:end, channel_mask]
                    )
                    group_loss[:, block_index, :, group] = np.mean(
                        error.astype(np.float64) ** 2, axis=(2, 3)
                    )
            arrays["channel_group_loss_mse"] = group_loss.reshape(
                batch * n_blocks, len(names), len(unique_groups)
            )
    return arrays, list(ctx_names) + list(forecast_names)
