"""Ordered data loading and strict frozen-expert inference."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_factory import data_dict
from models.AsySpecX import Model as AsySpecXModel

from .manifest import ExpertManifest


RUNTIME_DEFAULTS = {
    "model": "AsySpecX",
    "data": "custom",
    "root_path": "./dataset/weather/",
    "data_path": "weather.csv",
    "dataset": "weather",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "embed": "timeF",
    "seq_len": 96,
    "label_len": 0,
    "pred_len": 96,
    "cycle": 24,
    "enc_in": 1,
    "batch_size": 16,
    "num_workers": 0,
    "individual": 0,
    "cut_freq": 0,
    "spectral_lift": "complex_mlp",
    "lift_sharing": "shared",
    "norm_mode": "rin_noaffine",
    "rank": 8,
    "num_bands": 8,
    "gate_init": 0.0,
    "gate_init_logit": None,
    "gate_max": 1.0,
    "gate_type": "global",
    "residual_part": None,
    "mask_self_transfer": 0,
    "residual_clip_eta": -1.0,
    "force_cross_off": 0,
    "skip_dc_cross": 1,
    "log_asyspecx_diagnostics": 0,
    "eval_residual_part": "default",
    "gate_lr_mult": 1.0,
    "self_gain_init_std": 1e-3,
    "temporal_adapter": "none",
    "period": 24,
    "periods": "",
    "periodic_init": "seasonal_naive",
    "periodic_sharing": "shared",
    "temporal_fusion": "convex",
    "temporal_gate_type": "global",
    "temporal_gate_init_logit": -4.0,
    "period_fusion": "sum_gated",
    "period_gate_type": "period",
    "period_gate_init_logit": 0.0,
    "periodic_l1_weight": 0.0,
    "periodic_l2_weight": 0.0,
    "temporal_gate_l1_weight": 0.0,
    "energy_control": "none",
    "learned_clip_scope": "component_channel_band",
    "learned_clip_eta_init": 1.0,
    "learned_clip_eta_max": 2.0,
    "patch_adapter": "none",
    "patch_len": 16,
    "patch_stride": 8,
    "patch_basis_dim": 0,
    "patch_fusion": "convex",
    "patch_gate_type": "horizon",
    "patch_gate_init_logit": -6.0,
    "patch_l1_weight": 0.0,
    "patch_l2_weight": 0.0,
    "max_periods": 5,
    "linear_adapter": "none",
    "linear_sharing": "shared",
    "individual_linear_max_channels": 64,
    "linear_init": "zeros",
    "moving_avg_kernel": 25,
    "multiscale_factors": "1,2,4",
    "multiscale_fusion": "softmax",
    "multiscale_gate_type": "scale",
    "linear_fusion": "convex",
    "linear_gate_type": "horizon",
    "linear_gate_init_logit": -6.0,
    "linear_l1_weight": 0.0,
    "linear_l2_weight": 0.0,
    "branch_fusion": "sequential",
    "branch_fusion_scope": "horizon",
    "branch_init_main_logit": 4.0,
    "branch_init_aux_logit": -4.0,
}


def namespace_from_config(config: Mapping[str, object], **overrides) -> Namespace:
    values = dict(RUNTIME_DEFAULTS)
    values.update(config)
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Namespace(**values)


def make_ordered_loader(
    config: Mapping[str, object],
    split: str,
    batch_size: int = 0,
    num_workers: Optional[int] = None,
):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    args = namespace_from_config(config)
    if args.data not in data_dict:
        raise ValueError(f"unsupported dataset loader {args.data!r}")
    data_class = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    dataset = data_class(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=split,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        cycle=args.cycle,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers if num_workers is None else num_workers),
    )
    return dataset, loader


def _normalise_state_dict(state):
    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a state dict")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


class FrozenExpertPool:
    """Strictly load frozen checkpoints and average seed predictions per batch."""

    def __init__(
        self,
        manifest: ExpertManifest,
        seeds: Sequence[int | str],
        device: str = "auto",
        device_policy: str = "one_at_a_time",
    ):
        if device_policy not in {"one_at_a_time", "resident"}:
            raise ValueError("device_policy must be one_at_a_time or resident")
        self.manifest = manifest
        self.seeds = [str(seed) for seed in seeds]
        if not self.seeds:
            raise ValueError("at least one expert seed is required")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.device_policy = device_policy
        self.models: Dict[str, Dict[str, torch.nn.Module]] = {}
        self._load_all()

    @property
    def expert_names(self):
        return self.manifest.names

    def _load_all(self) -> None:
        for expert in self.manifest.experts:
            self.models[expert.name] = {}
            for seed in self.seeds:
                checkpoint = expert.checkpoint_for(seed)
                args = namespace_from_config(expert.config)
                model = AsySpecXModel(args).float()
                state = torch.load(checkpoint, map_location="cpu")
                try:
                    model.load_state_dict(_normalise_state_dict(state), strict=True)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"checkpoint/config incompatibility for {expert.name}[seed={seed}] at {checkpoint}: {exc}"
                    ) from exc
                model.eval()
                if self.device_policy == "resident":
                    model.to(self.device)
                self.models[expert.name][seed] = model

    def predict(self, batch_x: torch.Tensor) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        x = batch_x.float().to(self.device, non_blocking=True)
        means: Dict[str, np.ndarray] = {}
        variances: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for expert in self.manifest.experts:
                seed_predictions = []
                for seed in self.seeds:
                    model = self.models[expert.name][seed]
                    if self.device_policy == "one_at_a_time":
                        model.to(self.device)
                    output = model(
                        x,
                        eval_residual_part=str(expert.config.get("eval_residual_part", "default")),
                    )
                    if isinstance(output, Mapping):
                        output = output["pred"]
                    horizon = int(expert.config["pred_len"])
                    output = output[:, -horizon:, :].detach().float()
                    if not torch.isfinite(output).all():
                        raise FloatingPointError(
                            f"non-finite prediction from {expert.name}[seed={seed}]"
                        )
                    prediction = output.cpu().numpy()
                    seed_predictions.append(prediction)
                    if self.device_policy == "one_at_a_time":
                        model.to("cpu")
                stack = np.stack(seed_predictions, axis=0).astype(np.float32, copy=False)
                means[expert.name] = stack.mean(axis=0, dtype=np.float64).astype(np.float32)
                variances[expert.name] = stack.var(axis=0, dtype=np.float64).astype(np.float32)
                del stack
                if self.device.type == "cuda" and self.device_policy == "one_at_a_time":
                    torch.cuda.empty_cache()
        return means, variances


class FullPredictionMemmaps:
    """Optional chunked full export, one memmap per expert rather than [N,K,H,C]."""

    def __init__(self, output_dir: str, expert_names: Sequence[str], shape: Tuple[int, int, int]):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.predictions = {
            name: np.lib.format.open_memmap(
                self.root / f"{name}_mean.npy", mode="w+", dtype=np.float32, shape=shape
            )
            for name in expert_names
        }
        self.variances = {
            name: np.lib.format.open_memmap(
                self.root / f"{name}_seed_variance.npy", mode="w+", dtype=np.float32, shape=shape
            )
            for name in expert_names
        }
        self.target = np.lib.format.open_memmap(
            self.root / "target.npy", mode="w+", dtype=np.float32, shape=shape
        )

    def write(
        self,
        start: int,
        predictions: Mapping[str, np.ndarray],
        variances: Mapping[str, np.ndarray],
        target: np.ndarray,
    ) -> None:
        end = start + len(target)
        self.target[start:end] = target
        for name in self.predictions:
            self.predictions[name][start:end] = predictions[name]
            self.variances[name][start:end] = variances[name]

    def flush(self) -> None:
        self.target.flush()
        for array in list(self.predictions.values()) + list(self.variances.values()):
            array.flush()
