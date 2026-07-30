"""Locked Phase 9 expert configurations and checkpoint naming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DATASETS = {
    "ETTh1": dict(data="ETTh1", data_path="ETTh1.csv", subdir="ETT-small", enc_in=7, cycle=24, batch_size=256, learning_rate=0.0005, pred_lens=(96, 192, 336, 720)),
    "ETTm1": dict(data="ETTm1", data_path="ETTm1.csv", subdir="ETT-small", enc_in=7, cycle=96, batch_size=256, learning_rate=0.0005, pred_lens=(96, 192, 336, 720)),
    "weather": dict(data="custom", data_path="weather.csv", subdir="weather", enc_in=21, cycle=144, batch_size=64, learning_rate=0.0005, pred_lens=(96, 192, 336, 720)),
    "electricity": dict(data="custom", data_path="electricity.csv", subdir="electricity", enc_in=321, cycle=168, batch_size=32, learning_rate=0.0005, pred_lens=(96, 192, 336, 720)),
    "traffic": dict(data="custom", data_path="traffic.csv", subdir="traffic", enc_in=862, cycle=168, batch_size=16, learning_rate=0.0005, pred_lens=(96, 192, 336, 720)),
    "PEMS04": dict(data="PEMS", data_path="PEMS04.npz", subdir="PEMS", enc_in=307, cycle=288, batch_size=4, learning_rate=0.001, pred_lens=(12, 24, 48, 96)),
    "PEMS08": dict(data="PEMS", data_path="PEMS08.npz", subdir="PEMS", enc_in=170, cycle=288, batch_size=4, learning_rate=0.001, pred_lens=(12, 24, 48, 96)),
}

MANUAL_PERIODS = {
    "ETTh1": (24, 168),
    "ETTm1": (96, 672),
    "weather": (144,),
    "electricity": (24, 168),
    "traffic": (24, 168),
    "PEMS04": (24,),
    "PEMS08": (24,),
}

LOCKED_EXPERTS = {
    "anchor": "phase7_period_multi_auto_acf_patchlinear",
    "dlinear": "phase8_auto_acf_patchlinear_dlinear",
    "split_clip": "phase8_auto_acf_patchlinear_split_clip05",
    "individual_revin": "phase6_asx_individual_revin",
    "individual_period": "phase6_asx_individual_period",
}

# Phase 6/7/8 frozen experts were all trained with the legacy CLI setting
# ``--cycle 24``.  The dataset-specific cycle in DATASETS is the Phase 9
# runtime value and must not be substituted into the on-disk checkpoint name.
FROZEN_CHECKPOINT_SETTING_CYCLE = 24


def _as_period_text(periods: Sequence[int]) -> str:
    return "+".join(str(int(period)) for period in periods)


def discover_cached_periods(repo_root: Path, dataset: str, seq_len: int) -> List[int]:
    candidates = (
        repo_root / "phase8_results" / "hydra" / "auto_periods" / f"{dataset}_sl{seq_len}_auto_acf.json",
        repo_root / "phase7_results" / "breakthrough" / "auto_periods" / f"{dataset}_sl{seq_len}_auto_acf.json",
    )
    for path in candidates:
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                values = json.load(handle).get("periods", [])
            periods = [int(value) for value in values if int(value) > 0]
            if periods:
                return periods
    raise FileNotFoundError(
        f"missing train-only auto-ACF period cache for {dataset}/sl{seq_len}; "
        "run scripts/discover_periods.py before building the Phase 9 manifest"
    )


def _base_runtime(dataset: str, seq_len: int, pred_len: int, data_root: Path) -> Dict[str, object]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported Phase 9 dataset {dataset!r}")
    spec = DATASETS[dataset]
    if pred_len not in spec["pred_lens"]:
        raise ValueError(f"pred_len={pred_len} is not configured for {dataset}")
    rank = 2 if dataset in {"electricity", "traffic", "PEMS04", "PEMS08"} else 8
    if dataset.startswith("PEMS"):
        cut_freq = seq_len // 2 + 1
        num_bands = 16
    else:
        base_period = {"ETTh1": 24, "ETTm1": 96, "weather": 144, "electricity": 24, "traffic": 24}[dataset]
        cut_freq = max(2, seq_len // base_period * 6 + 1)
        num_bands = 8
    return {
        "model": "AsySpecX",
        "dataset": dataset,
        "data": spec["data"],
        "root_path": str((data_root / spec["subdir"]).resolve()) + "/",
        "data_path": spec["data_path"],
        "features": "M",
        "target": "OT",
        "freq": "h",
        "embed": "timeF",
        "seq_len": int(seq_len),
        "label_len": 0,
        "pred_len": int(pred_len),
        "cycle": int(spec["cycle"]),
        "enc_in": int(spec["enc_in"]),
        "batch_size": int(spec["batch_size"]),
        "num_workers": 4,
        "learning_rate": float(spec["learning_rate"]),
        "train_epochs": 30,
        "rank": rank,
        "num_bands": num_bands,
        "cut_freq": cut_freq,
        "individual": 0,
    }


def _cross_period_config(periods: Sequence[int]) -> Dict[str, object]:
    return {
        "spectral_lift": "fits_linear",
        "lift_sharing": "shared",
        "cross_mode": "asym_lowrank",
        "residual_part": "split",
        "gate_type": "hier_channel_band",
        "gate_init_logit": -6.0,
        "gate_max": 1.0,
        "gate_lr_mult": 5.0,
        "residual_clip_eta": -1.0,
        "norm_mode": "rin_noaffine",
        "temporal_adapter": "sparse_period",
        "period": int(periods[0]),
        "periods": _as_period_text(periods),
        "periodic_init": "seasonal_naive",
        "period_fusion": "sum_gated",
        "period_gate_type": "period",
        "period_gate_init_logit": 0.0,
        "temporal_fusion": "convex",
        "temporal_gate_type": "horizon",
        "temporal_gate_init_logit": -4.0,
    }


def expert_config(
    name: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    data_root: Path,
    auto_periods: Sequence[int],
) -> Dict[str, object]:
    base = _base_runtime(dataset, seq_len, pred_len, data_root)
    if name in {"anchor", "dlinear", "split_clip"}:
        base.update(_cross_period_config(auto_periods))
        base.update(
            {
                "period_mode": "auto_acf",
                "patch_adapter": "patch_linear",
                "patch_len": 16,
                "patch_stride": 8,
                "patch_fusion": "convex",
                "patch_gate_type": "horizon",
                "patch_gate_init_logit": -6.0,
            }
        )
        if name == "dlinear":
            base.update(
                {
                    "linear_adapter": "dlinear_decomp",
                    "linear_fusion": "convex",
                    "linear_gate_type": "horizon",
                    "linear_gate_init_logit": -6.0,
                }
            )
        elif name == "split_clip":
            base["residual_clip_eta"] = 0.5
    elif name == "individual_revin":
        base.update(
            {
                "spectral_lift": "fits_linear",
                "lift_sharing": "individual",
                "cross_mode": "none",
                "norm_mode": "revin_affine",
                "temporal_adapter": "none",
            }
        )
    elif name == "individual_period":
        manual = MANUAL_PERIODS[dataset]
        base.update(_cross_period_config(manual))
        base.update({"lift_sharing": "individual", "cross_mode": "none"})
    else:
        raise ValueError(f"unknown locked expert {name!r}")
    return base


def checkpoint_path(
    checkpoint_root: Path,
    arm: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    seed: int,
) -> Path:
    spec = DATASETS[dataset]
    run_id = f"{arm}_{dataset}_sl{seq_len}_pl{pred_len}_sd{seed}"
    setting = (
        f"{run_id}_AsySpecX_{spec['data']}_ftM_sl{seq_len}_pl{pred_len}_"
        f"cycle{FROZEN_CHECKPOINT_SETTING_CYCLE}_seed{seed}"
    )
    return checkpoint_root / setting / "checkpoint.pth"
