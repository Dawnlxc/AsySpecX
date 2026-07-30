"""Deterministic, finite, fixed-dimensional SafeRoute features."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-8
AGGREGATIONS = ("mean", "std", "q10", "q50", "q90", "max")
DATASETS = ("ETTh1", "ETTm1", "weather", "electricity", "traffic", "PEMS04", "PEMS08")
FAMILIES = ("ETT", "Weather", "LargeC", "PEMS", "Other")


def dataset_family(dataset: str) -> str:
    if dataset.startswith("ETT"):
        return "ETT"
    if dataset == "weather":
        return "Weather"
    if dataset in {"electricity", "traffic"}:
        return "LargeC"
    if dataset.startswith("PEMS"):
        return "PEMS"
    return "Other"


def deterministic_channel_indices(channels: int, max_channels: int = 64) -> np.ndarray:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if max_channels <= 0 or channels <= max_channels:
        return np.arange(channels, dtype=np.int64)
    return np.linspace(0, channels - 1, max_channels, dtype=np.int64)


def train_only_channel_groups(
    train_series: np.ndarray,
    n_groups: int,
    random_state: int = 2024,
) -> Tuple[np.ndarray, List[str]]:
    """Cluster channels from train-only structural descriptors.

    This helper is used only for the optional channel-group oracle audit. It
    never consumes validation/test labels or forecasts.
    """

    values = np.asarray(train_series, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
        raise ValueError("train_series must be [time, channels] with at least four steps")
    channels = values.shape[1]
    if n_groups < 1 or n_groups > channels:
        raise ValueError("n_groups must be between 1 and the channel count")
    if n_groups == 1:
        return np.zeros(channels, dtype=np.int32), ["variance", "slope", "dominant_period", "spectral_entropy", "acf_strength"]

    # Bound descriptor construction cost without introducing random sampling.
    if values.shape[0] > 20000:
        values = values[np.linspace(0, values.shape[0] - 1, 20000, dtype=np.int64)]
    values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    steps = values.shape[0]
    centered = values - values.mean(axis=0, keepdims=True)
    variance = np.mean(centered * centered, axis=0)
    time = np.linspace(-1.0, 1.0, steps)
    slope = np.einsum("t,tc->c", time, centered) / max(float(np.sum(time * time)), EPS)
    power = np.abs(np.fft.rfft(centered, axis=0))[1:] ** 2
    if power.shape[0] == 0:
        power = np.zeros((1, channels), dtype=np.float64)
    total = np.maximum(power.sum(axis=0), EPS)
    probability = power / total[None, :]
    entropy = -np.sum(probability * np.log(np.maximum(probability, EPS)), axis=0)
    entropy /= math.log(max(2, power.shape[0]))
    dominant_period = steps / np.maximum(np.argmax(power, axis=0) + 1, 1)
    acfs = []
    for lag in (1, 2, 4, 8, 12, 24):
        if lag >= steps:
            continue
        left, right = centered[:-lag], centered[lag:]
        numerator = np.mean(left * right, axis=0)
        denominator = np.sqrt(np.mean(left * left, axis=0) * np.mean(right * right, axis=0))
        acfs.append(np.abs(numerator / np.maximum(denominator, EPS)))
    acf_strength = np.max(np.stack(acfs), axis=0) if acfs else np.zeros(channels)
    descriptors = np.column_stack((variance, slope, dominant_period, entropy, acf_strength))
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=1e6, neginf=-1e6)
    scale = descriptors.std(axis=0)
    descriptors = (descriptors - descriptors.mean(axis=0)) / np.where(scale > EPS, scale, 1.0)

    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=n_groups, random_state=random_state, n_init=10).fit_predict(descriptors)
    if len(np.unique(labels)) != n_groups:
        raise ValueError("channel descriptors did not support the requested number of non-empty groups")
    # Relabel by the first channel in each cluster so persisted assignments are stable.
    order = sorted(range(n_groups), key=lambda label: int(np.flatnonzero(labels == label)[0]))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=np.int32)
    return labels, ["variance", "slope", "dominant_period", "spectral_entropy", "acf_strength"]


def _aggregate(values: np.ndarray) -> List[np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return [
        np.mean(values, axis=1),
        np.std(values, axis=1),
        np.quantile(values, 0.10, axis=1),
        np.quantile(values, 0.50, axis=1),
        np.quantile(values, 0.90, axis=1),
        np.max(values, axis=1),
    ]


def _append_channel_feature(
    columns: List[np.ndarray],
    names: List[str],
    base_name: str,
    values: np.ndarray,
) -> None:
    for aggregation, column in zip(AGGREGATIONS, _aggregate(values)):
        names.append(f"ctx__{base_name}__{aggregation}")
        columns.append(column)


def _safe_acf(x: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0 or lag >= x.shape[1]:
        return np.zeros((x.shape[0], x.shape[2]), dtype=np.float64)
    left = x[:, :-lag, :]
    right = x[:, lag:, :]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.mean(left * right, axis=1)
    denominator = np.sqrt(np.mean(left * left, axis=1) * np.mean(right * right, axis=1))
    return numerator / np.maximum(denominator, EPS)


def _cross_channel_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, _, channels = x.shape
    corr_mean = np.zeros(batch, dtype=np.float64)
    corr_top = np.zeros(batch, dtype=np.float64)
    coherence = np.zeros(batch, dtype=np.float64)
    if channels < 2:
        return corr_mean, corr_top, coherence
    upper = np.triu_indices(channels, 1)
    for sample in range(batch):
        values = x[sample]
        centered = values - values.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.sum(centered * centered, axis=0, keepdims=True))
        normalised = centered / np.maximum(scale, EPS)
        corr = np.abs(normalised.T @ normalised)[upper]
        corr_mean[sample] = float(corr.mean()) if corr.size else 0.0
        if corr.size:
            top_count = min(max(1, channels), corr.size)
            corr_top[sample] = float(np.partition(corr, -top_count)[-top_count:].mean())

        spectrum = np.fft.rfft(centered, axis=0)[1:]
        if spectrum.shape[0] == 0:
            continue
        spec_scale = np.sqrt(np.sum(np.abs(spectrum) ** 2, axis=0, keepdims=True))
        spec_norm = spectrum / np.maximum(spec_scale, EPS)
        coh = np.abs(spec_norm.conj().T @ spec_norm)[upper]
        coherence[sample] = float(coh.mean()) if coh.size else 0.0
    return corr_mean, corr_top, coherence


def context_features(
    context: np.ndarray,
    pred_len: int,
    periods: Sequence[int] = (),
    dataset: str = "",
    router_scope: str = "cell",
    max_channels: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """Extract structural context features from ``[B,T,C]`` without labels."""

    raw = np.asarray(context, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1] < 2 or raw.shape[2] < 1:
        raise ValueError("context must be [B,T,C] with T>=2 and C>=1")
    batch, steps, original_channels = raw.shape
    indices = deterministic_channel_indices(original_channels, max_channels)
    x = raw[:, :, indices]
    columns: List[np.ndarray] = []
    names: List[str] = []

    channel_mean = x.mean(axis=1)
    channel_std = x.std(axis=1)
    _append_channel_feature(columns, names, "level_mean", channel_mean)
    _append_channel_feature(columns, names, "level_std", channel_std)
    _append_channel_feature(columns, names, "last", x[:, -1, :])
    _append_channel_feature(columns, names, "range", x.max(axis=1) - x.min(axis=1))

    time = np.linspace(-1.0, 1.0, steps, dtype=np.float64)
    time_var = np.sum(time * time)
    centered = x - channel_mean[:, None, :]
    slope = np.einsum("t,btc->bc", time, centered) / max(time_var, EPS)
    fitted = slope[:, None, :] * time[None, :, None]
    residual = centered - fitted
    r2 = 1.0 - np.sum(residual * residual, axis=1) / np.maximum(
        np.sum(centered * centered, axis=1), EPS
    )
    midpoint = max(1, steps // 2)
    first, second = x[:, :midpoint, :], x[:, midpoint:, :]
    if second.shape[1] == 0:
        second = first
    _append_channel_feature(columns, names, "trend_slope", slope)
    _append_channel_feature(columns, names, "trend_r2", r2)
    _append_channel_feature(columns, names, "half_mean_shift", second.mean(1) - first.mean(1))
    _append_channel_feature(
        columns,
        names,
        "half_std_ratio",
        second.std(1) / np.maximum(first.std(1), EPS),
    )

    diff1 = np.diff(x, axis=1)
    diff2 = np.diff(x, n=2, axis=1) if steps >= 3 else np.zeros_like(diff1[:, :1])
    _append_channel_feature(columns, names, "diff1_rms", np.sqrt(np.mean(diff1 * diff1, axis=1)))
    _append_channel_feature(columns, names, "diff2_rms", np.sqrt(np.mean(diff2 * diff2, axis=1)))
    _append_channel_feature(columns, names, "total_variation", np.mean(np.abs(diff1), axis=1))
    outlier = np.mean(np.abs(centered) > (3.0 * channel_std[:, None, :] + EPS), axis=1)
    _append_channel_feature(columns, names, "outlier_ratio", outlier)

    spectrum = np.fft.rfft(centered, axis=1)
    power = np.abs(spectrum) ** 2
    non_dc = power[:, 1:, :]
    if non_dc.shape[1] == 0:
        non_dc = np.zeros((batch, 1, x.shape[2]), dtype=np.float64)
    total_power = np.maximum(non_dc.sum(axis=1), EPS)
    low_bins = max(1, non_dc.shape[1] // 4)
    low_ratio = non_dc[:, :low_bins, :].sum(axis=1) / total_power
    probabilities = non_dc / total_power[:, None, :]
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, EPS)), axis=1)
    entropy /= math.log(max(2, non_dc.shape[1]))
    dominant_index = np.argmax(non_dc, axis=1) + 1
    dominant_strength = np.max(non_dc, axis=1) / total_power
    dominant_period = steps / np.maximum(dominant_index, 1)
    top_count = min(3, non_dc.shape[1])
    top3 = np.partition(non_dc, -top_count, axis=1)[:, -top_count:, :].sum(axis=1) / total_power
    _append_channel_feature(columns, names, "low_frequency_energy_ratio", low_ratio)
    _append_channel_feature(columns, names, "spectral_entropy", entropy)
    _append_channel_feature(columns, names, "dominant_frequency_strength", dominant_strength)
    _append_channel_feature(columns, names, "dominant_period", dominant_period)
    _append_channel_feature(columns, names, "top3_frequency_energy_ratio", top3)

    fixed_lags = [1, 2, 4, 8, 12, 24]
    valid_periods = sorted({int(period) for period in periods if 0 < int(period) < steps})
    for lag in fixed_lags:
        _append_channel_feature(columns, names, f"acf_lag{lag}", _safe_acf(x, lag))
    if valid_periods:
        period_acf = np.stack([_safe_acf(x, lag) for lag in valid_periods], axis=0).mean(axis=0)
        primary_period = valid_periods[0]
        seasonal_diff = x[:, primary_period:, :] - x[:, :-primary_period, :]
        seasonal_rms = np.sqrt(np.mean(seasonal_diff * seasonal_diff, axis=1))
        strength = 1.0 - np.var(seasonal_diff, axis=1) / np.maximum(np.var(x, axis=1), EPS)
    else:
        period_acf = np.zeros_like(channel_mean)
        seasonal_rms = np.zeros_like(channel_mean)
        strength = np.zeros_like(channel_mean)
    _append_channel_feature(columns, names, "acf_period_mean", period_acf)
    _append_channel_feature(columns, names, "seasonal_difference_rms", seasonal_rms)
    _append_channel_feature(columns, names, "seasonality_strength", strength)

    _append_channel_feature(columns, names, "near_zero_ratio", np.mean(np.abs(x) <= 1e-6, axis=1))
    _append_channel_feature(columns, names, "repeated_value_ratio", np.mean(np.abs(diff1) <= 1e-6, axis=1))

    corr_mean, corr_top, coherence = _cross_channel_features(x)
    for name, values in (
        ("cross_abs_correlation_mean", corr_mean),
        ("cross_topk_correlation_mean", corr_top),
        ("cross_spectral_coherence", coherence),
    ):
        names.append(f"ctx__{name}")
        columns.append(values)

    metadata = {
        "meta__log_channels": np.full(batch, math.log(max(original_channels, 1))),
        "meta__log_context": np.full(batch, math.log(max(steps, 1))),
        "meta__log_horizon": np.full(batch, math.log(max(pred_len, 1))),
        "meta__horizon_context_ratio": np.full(batch, pred_len / float(steps)),
    }
    for name, values in metadata.items():
        names.append(name)
        columns.append(values)

    family = dataset_family(dataset)
    if router_scope == "global":
        for item in DATASETS:
            names.append(f"meta__dataset__{item}")
            columns.append(np.full(batch, float(dataset == item)))
    if router_scope in {"global", "family"}:
        for item in FAMILIES:
            names.append(f"meta__family__{item}")
            columns.append(np.full(batch, float(family == item)))

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    return matrix, names


def _forecast_slope(values: np.ndarray) -> np.ndarray:
    # values: [B,H,C]
    horizon = values.shape[1]
    if horizon < 2:
        return np.zeros(values.shape[0], dtype=np.float64)
    time = np.linspace(-1.0, 1.0, horizon)
    channel_mean = values.mean(axis=2)
    channel_mean -= channel_mean.mean(axis=1, keepdims=True)
    return np.einsum("h,bh->b", time, channel_mean) / max(np.sum(time * time), EPS)


def forecast_features(
    context: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    seed_variances: Mapping[str, np.ndarray],
    blocks: Sequence[Tuple[int, int]],
    anchor_name: str,
    snippet_bins: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """Return per-sample/per-block compact expert forecast features."""

    if anchor_name not in predictions:
        raise ValueError(f"anchor prediction {anchor_name!r} is missing")
    expert_names = list(predictions)
    shape = np.asarray(predictions[anchor_name]).shape
    if len(shape) != 3:
        raise ValueError("expert forecasts must be [B,H,C]")
    for name in expert_names:
        if np.asarray(predictions[name]).shape != shape:
            raise ValueError(f"forecast shape mismatch for expert {name!r}")
        if np.asarray(seed_variances[name]).shape != shape:
            raise ValueError(f"seed variance shape mismatch for expert {name!r}")
    if snippet_bins < 0 or snippet_bins > 16:
        raise ValueError("snippet_bins must be between 0 and 16")
    effective_snippet_bins = (
        min(snippet_bins, min(end - start for start, end in blocks)) if snippet_bins else 0
    )

    context_scale = np.std(np.asarray(context, dtype=np.float64), axis=(1, 2))
    block_matrices: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    for block_index, (start, end) in enumerate(blocks):
        columns: List[np.ndarray] = []
        names: List[str] = []
        block_predictions = {name: np.asarray(predictions[name], dtype=np.float64)[:, start:end, :] for name in expert_names}
        anchor = block_predictions[anchor_name]
        for name in expert_names:
            values = block_predictions[name]
            variance = np.asarray(seed_variances[name], dtype=np.float64)[:, start:end, :]
            prefix = f"fc__{name}"
            entries = (
                ("mean", values.mean(axis=(1, 2))),
                ("std", values.std(axis=(1, 2))),
                ("slope", _forecast_slope(values)),
                ("diff_rms", np.sqrt(np.mean(np.diff(values, axis=1) ** 2, axis=(1, 2))) if values.shape[1] > 1 else np.zeros(values.shape[0])),
                ("amplitude_ratio", values.std(axis=(1, 2)) / np.maximum(context_scale, EPS)),
                ("anchor_disagreement_rms", np.sqrt(np.mean((values - anchor) ** 2, axis=(1, 2)))),
                ("anchor_disagreement_max", np.max(np.abs(values - anchor), axis=(1, 2))),
                ("seed_variance_mean", variance.mean(axis=(1, 2))),
                ("seed_variance_max", variance.max(axis=(1, 2))),
            )
            for suffix, column in entries:
                names.append(f"{prefix}__{suffix}")
                columns.append(column)
            if effective_snippet_bins:
                for bin_index, indices in enumerate(
                    np.array_split(np.arange(values.shape[1]), effective_snippet_bins)
                ):
                    names.append(f"{prefix}__snippet{bin_index}")
                    columns.append(values[:, indices, :].mean(axis=(1, 2)))

        pairwise = []
        for left in range(len(expert_names)):
            for right in range(left + 1, len(expert_names)):
                delta = block_predictions[expert_names[left]] - block_predictions[expert_names[right]]
                pairwise.append(np.sqrt(np.mean(delta * delta, axis=(1, 2))))
        if pairwise:
            pair_values = np.stack(pairwise, axis=1)
            sorted_pair = np.sort(pair_values, axis=1)
            top_two = sorted_pair[:, -min(2, sorted_pair.shape[1]) :].mean(axis=1)
            pool_mean = pair_values.mean(axis=1)
            pool_max = pair_values.max(axis=1)
        else:
            top_two = pool_mean = pool_max = np.zeros(shape[0], dtype=np.float64)
        anchor_deltas = []
        for name in expert_names:
            if name == anchor_name:
                continue
            delta = block_predictions[name] - anchor
            anchor_deltas.append(np.sqrt(np.mean(delta * delta, axis=(1, 2))))
        anchor_pool = np.stack(anchor_deltas, axis=1).mean(axis=1) if anchor_deltas else np.zeros(shape[0])
        for suffix, column in (
            ("pairwise_diversity_mean", pool_mean),
            ("pairwise_diversity_max", pool_max),
            ("top2_disagreement", top_two),
            ("anchor_pool_disagreement", anchor_pool),
        ):
            names.append(f"pool__{suffix}")
            columns.append(column)
        names.extend(("meta__block_index", "meta__block_start_ratio", "meta__block_end_ratio"))
        columns.extend(
            (
                np.full(shape[0], block_index, dtype=np.float64),
                np.full(shape[0], start / float(shape[1]), dtype=np.float64),
                np.full(shape[0], end / float(shape[1]), dtype=np.float64),
            )
        )
        matrix = np.column_stack(columns)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        block_matrices.append(matrix.astype(np.float32, copy=False))
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise AssertionError("forecast feature names changed across blocks")
    return np.stack(block_matrices, axis=1), feature_names or []
