#!/usr/bin/env python3
"""Build a train-only ridge/SVD initializer for the Phase 11 kernel.

The fitted affine map is channel-separable: every train window/channel pair is
treated as one independent record, and a single H x T temporal map is shared by
all channels.  No cross-channel statistic is accumulated anywhere in this
script.

Example
-------
python scripts/build_phase11_forecastability_init.py \
  --dataset Weather --data custom --root_path ./dataset/weather/ \
  --data_path weather.csv --features M --target OT \
  --seq_len 96 --pred_len 720 --enc_in 21 --rank 16 \
  --max_windows 2048 --ridge 1e-2 \
  --output artifacts/phase11/weather_sl96_pl720_r16.pt
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MomentAccumulator:
    """Float64 sufficient statistics for independent temporal records."""

    def __init__(self, seq_len: int, pred_len: int):
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.records = 0
        self.sum_x = torch.zeros(self.seq_len, dtype=torch.float64)
        self.sum_y = torch.zeros(self.pred_len, dtype=torch.float64)
        self.xtx = torch.zeros(self.seq_len, self.seq_len, dtype=torch.float64)
        self.ytx = torch.zeros(self.pred_len, self.seq_len, dtype=torch.float64)
        self.yty = torch.zeros(self.pred_len, self.pred_len, dtype=torch.float64)

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 2 or tuple(x.shape[1:]) != (self.seq_len,):
            raise ValueError(f"x must be [N,{self.seq_len}], got {tuple(x.shape)}")
        if y.ndim != 2 or tuple(y.shape[1:]) != (self.pred_len,):
            raise ValueError(f"y must be [N,{self.pred_len}], got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y record counts differ")
        x = x.to(dtype=torch.float64, device="cpu")
        y = y.to(dtype=torch.float64, device="cpu")
        self.records += int(x.shape[0])
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.xtx += x.transpose(0, 1) @ x
        self.ytx += y.transpose(0, 1) @ x
        self.yty += y.transpose(0, 1) @ y

    def centered_covariances(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.records < 2:
            raise ValueError("at least two temporal records are required")
        n = float(self.records)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        cxx = self.xtx / n - torch.outer(mean_x, mean_x)
        cyx = self.ytx / n - torch.outer(mean_y, mean_x)
        cyy = self.yty / n - torch.outer(mean_y, mean_y)
        return cxx, cyx, cyy


def normalize_windows(
    x: torch.Tensor,
    y: torch.Tensor,
    norm_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match the input-derived normalization used by ``AsySpecX.Model``."""
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    if norm_mode == "rin_noaffine":
        loc = x.mean(dim=1, keepdim=True)
        centered = x - loc
        # torch.var in AsySpecX uses the default correction=1.
        scale = torch.sqrt(torch.var(centered, dim=1, keepdim=True, correction=1) + 1e-5)
        return centered / scale, (y - loc) / scale
    if norm_mode == "revin_affine":
        loc = x.mean(dim=1, keepdim=True)
        centered = x - loc
        scale = torch.sqrt(torch.var(centered, dim=1, keepdim=True, correction=1) + 1e-5)
        # Model starts with gamma=1 and beta=0.  Its inverse divides by
        # gamma+1e-5, so this is the exact normalized target at initialization.
        return centered / scale, (1.0 + 1e-5) * (y - loc) / scale
    if norm_mode == "subtract_last":
        loc = x[:, -1:, :]
        return x - loc, y - loc
    if norm_mode == "none":
        return x, y
    raise ValueError(
        f"unsupported norm_mode={norm_mode!r}; initializer supports rin_noaffine, "
        "revin_affine, subtract_last, or none"
    )


def fit_ridge_svd(
    moments: MomentAccumulator,
    rank: int,
    ridge: float,
) -> Dict[str, object]:
    """Solve the centered ridge map and return a balanced rank-R factorization."""
    rank = int(rank)
    ridge = float(ridge)
    if rank < 1 or rank > min(moments.seq_len, moments.pred_len):
        raise ValueError(
            f"rank must be in [1,{min(moments.seq_len, moments.pred_len)}], got {rank}"
        )
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    cxx, cyx, cyy = moments.centered_covariances()
    eye = torch.eye(moments.seq_len, dtype=torch.float64)
    regularized = cxx + ridge * eye
    # W = Cyx (Cxx + lambda I)^-1, avoiding an explicit matrix inverse.
    weight = torch.linalg.solve(regularized, cyx.transpose(0, 1)).transpose(0, 1)
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    root = torch.sqrt(singular[:rank].clamp_min(0.0))
    future_basis = u[:, :rank] * root.view(1, rank)
    past_basis = root.view(rank, 1) * vh[:rank]
    weight_rank = future_basis @ past_basis

    n = float(moments.records)
    mean_x = moments.sum_x / n
    mean_y = moments.sum_y / n
    horizon_bias = mean_y - weight_rank @ mean_x
    total_energy = torch.sum(singular ** 2).clamp_min(torch.finfo(torch.float64).tiny)
    retained_energy = torch.sum(singular[:rank] ** 2) / total_energy

    # Expected centered training MSE of the truncated affine map, computed from
    # sufficient statistics without revisiting or materializing all records.
    residual_trace = (
        torch.trace(cyy)
        - 2.0 * torch.sum(weight_rank * cyx)
        + torch.trace(weight_rank @ cxx @ weight_rank.transpose(0, 1))
    )
    centered_train_mse = residual_trace.clamp_min(0.0) / float(moments.pred_len)
    return {
        "past_basis": past_basis.to(torch.float32),
        "future_basis": future_basis.to(torch.float32),
        "horizon_bias": horizon_bias.to(torch.float32),
        "singular_values": singular.to(torch.float32),
        "retained_energy": float(retained_energy),
        "centered_train_mse": float(centered_train_mse),
    }


def evenly_spaced_indices(total: int, maximum: int) -> np.ndarray:
    if total < 1:
        raise ValueError("the train split has no complete input/forecast windows")
    if maximum <= 0 or maximum >= total:
        return np.arange(total, dtype=np.int64)
    # unique guards against integer rounding when maximum is close to total.
    return np.unique(np.linspace(0, total - 1, num=maximum, dtype=np.int64))


def iter_window_batches(
    series: np.ndarray,
    indices: Iterable[int],
    seq_len: int,
    pred_len: int,
    batch_windows: int,
):
    indices = np.asarray(list(indices), dtype=np.int64)
    for start in range(0, len(indices), batch_windows):
        batch_idx = indices[start : start + batch_windows]
        x = np.stack([series[i : i + seq_len] for i in batch_idx], axis=0)
        y = np.stack(
            [series[i + seq_len : i + seq_len + pred_len] for i in batch_idx],
            axis=0,
        )
        yield torch.from_numpy(x), torch.from_numpy(y)


def load_train_series(args) -> np.ndarray:
    """Load exactly the repository's train split; no split argument is exposed."""
    from data_provider.data_factory import data_dict

    if args.data not in data_dict:
        raise ValueError(f"unknown data type {args.data!r}; choose from {sorted(data_dict)}")
    dataset = data_dict[args.data](
        root_path=args.root_path,
        data_path=args.data_path,
        flag="train",
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=1 if args.embed == "timeF" else 0,
        freq=args.freq,
        cycle=args.cycle,
    )
    series = np.asarray(dataset.data_x)
    if series.ndim != 2:
        raise ValueError(f"train series must be [time,channel], got {series.shape}")
    if args.enc_in > 0 and series.shape[1] != args.enc_in:
        raise ValueError(
            f"enc_in mismatch: requested {args.enc_in}, train data has {series.shape[1]} channels"
        )
    return series


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="auditable dataset label, e.g. Weather")
    p.add_argument("--data", default="custom", choices=["custom", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Solar", "PEMS"])
    p.add_argument("--root_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--features", default="M")
    p.add_argument("--target", default="OT")
    p.add_argument("--embed", default="timeF")
    p.add_argument("--freq", default="h")
    p.add_argument("--cycle", type=int, default=24)
    p.add_argument("--enc_in", type=int, default=0)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--norm_mode", default="rin_noaffine", choices=["rin_noaffine", "revin_affine", "subtract_last", "none"])
    p.add_argument("--max_windows", type=int, default=2048, help="0 uses every complete train window")
    p.add_argument("--batch_windows", type=int, default=64)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.seq_len < 2 and args.norm_mode in {"rin_noaffine", "revin_affine"}:
        raise ValueError(f"{args.norm_mode} requires seq_len >= 2")
    if args.batch_windows < 1:
        raise ValueError("batch_windows must be positive")

    series = load_train_series(args)
    total_windows = len(series) - args.seq_len - args.pred_len + 1
    indices = evenly_spaced_indices(total_windows, args.max_windows)
    moments = MomentAccumulator(args.seq_len, args.pred_len)
    for x, y in iter_window_batches(
        series, indices, args.seq_len, args.pred_len, args.batch_windows
    ):
        x_norm, y_norm = normalize_windows(x, y, args.norm_mode)
        # [B,T,C] -> [B*C,T].  The channel dimension only becomes an
        # independent record dimension; it is never a feature dimension.
        x_records = x_norm.permute(0, 2, 1).reshape(-1, args.seq_len)
        y_records = y_norm.permute(0, 2, 1).reshape(-1, args.pred_len)
        moments.update(x_records, y_records)

    fitted = fit_ridge_svd(moments, rank=args.rank, ridge=args.ridge)
    source_path = os.path.abspath(os.path.join(args.root_path, args.data_path))
    source_stat = os.stat(source_path)
    meta = {
        "format": "asyspecx_phase11_forecastability_v1",
        "dataset": args.dataset,
        "data": args.data,
        "root_path": os.path.abspath(args.root_path),
        "data_path": args.data_path,
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "split": "train",
        "train_only": True,
        "features": args.features,
        "target": args.target,
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "channels": int(series.shape[1]),
        "rank": int(args.rank),
        "ridge": float(args.ridge),
        "norm_mode": args.norm_mode,
        "available_train_windows": int(total_windows),
        "sampled_train_windows": int(len(indices)),
        "temporal_records": int(moments.records),
        "sampling": "all" if len(indices) == total_windows else "evenly_spaced",
        "retained_frobenius_energy": fitted["retained_energy"],
        "centered_train_mse": fitted["centered_train_mse"],
    }
    payload = {
        "past_basis": fitted["past_basis"],
        "future_basis": fitted["future_basis"],
        "horizon_bias": fitted["horizon_bias"],
        "singular_values": fitted["singular_values"],
        "meta": meta,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"output": output, **meta}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
