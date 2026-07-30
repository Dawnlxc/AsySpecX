#!/usr/bin/env python3
"""Same-GPU fixed-work audit for validation-selected Phase 11 checkpoints.

This script uses synthetic tensors only.  It never constructs a dataset and
therefore cannot inspect validation or test labels.  The purpose is to remove
node/scheduler noise from train-step, inference-step, and peak-memory ratios.
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.AsySpecX import Model


def make_config(arm, args):
    values = dict(
        data=args.data_name,
        data_path=args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.channels,
        cycle=args.cycle,
        cut_freq=args.cut_freq,
        rank=args.cross_rank,
        num_bands=args.num_bands,
        spectral_lift="fits_linear",
        lift_sharing="individual",
        individual=0,
        norm_mode="revin_affine",
        cross_mode="none",
        temporal_adapter="none",
        patch_adapter="none",
        linear_adapter="none",
        forecast_kernel="none",
        forecast_kernel_extension_shrink="none",
        branch_fusion="sequential",
        cycle_residual=1,
        cycle_residual_rank=0,
        cycle_residual_init_std=0.02,
    )
    if args.base_profile == "compact_period_cycle_full":
        values.update(
            lift_sharing="shared",
            norm_mode="rin_noaffine",
            cross_mode="asym_lowrank",
            residual_part="split",
            gate_type="hier_channel_band",
            gate_init_logit=-6.0,
            gate_max=1.0,
            gate_lr_mult=5.0,
            residual_clip_eta=-1.0,
            temporal_adapter="compact_period",
            periods=args.periods,
            periodic_init="seasonal_naive",
            period_fusion="sum_gated",
            period_gate_type="period",
            period_gate_init_logit=0.0,
            temporal_fusion="convex",
            temporal_gate_type="horizon",
            temporal_gate_init_logit=-4.0,
        )
    elif args.base_profile != "ind_cycle_full":
        raise ValueError(f"unsupported base profile {args.base_profile!r}")

    if arm == "anchor":
        pass
    elif arm == "dense_direct":
        values.update(
            linear_adapter="direct_linear",
            linear_sharing="shared",
            linear_init="zeros",
            linear_fusion="additive",
            linear_gate_type="horizon",
            linear_gate_init_logit=4.0,
        )
    elif arm in {
        "fk_r4", "fk_r8", "fk_r8_cs", "fk_sm2_shared", "fk_sm2_mode",
        "fk_sm2_tail2", "fk_sm4_mode", "fk_sm4_frozen", "fk_sm4_ph2_q", "fk_sm4_ph4_q",
        "fk_sm4_ph4_h",
    }:
        values.update(
            forecast_kernel="lowrank_time",
            forecast_kernel_rank=4 if arm == "fk_r4" else 8,
            forecast_kernel_init="small_random",
            forecast_kernel_channel_scale=1 if arm in {
                "fk_r8_cs", "fk_sm2_shared", "fk_sm2_mode", "fk_sm2_tail2",
                "fk_sm4_mode", "fk_sm4_frozen", "fk_sm4_ph2_q",
                "fk_sm4_ph4_q", "fk_sm4_ph4_h",
            } else 0,
            forecast_kernel_fusion="convex",
            forecast_kernel_gate_type="horizon",
            forecast_kernel_gate_init_logit=-4.0,
        )
        if arm.startswith("fk_sm"):
            values.update(
                forecast_kernel_spectral_mixtures=4 if "sm4" in arm else 2,
                forecast_kernel_sm_sharing="shared" if arm.endswith("shared") else "mode",
                forecast_kernel_sm_base_trainable=0 if arm.endswith("frozen") else 1,
            )
            phase_map = {
                "fk_sm4_ph2_q": (2, 0.7853981633974483),
                "fk_sm4_ph4_q": (4, 0.7853981633974483),
                "fk_sm4_ph4_h": (4, 1.5707963267948966),
            }
            if arm in phase_map:
                basis_dim, phase_max = phase_map[arm]
                values.update(
                    forecast_kernel_phase_basis_dim=basis_dim,
                    forecast_kernel_phase_max=phase_max,
                )
            if arm == "fk_sm2_tail2":
                values.update(forecast_kernel_extension_shrink="tail2_linear")
    else:
        raise ValueError(f"unsupported benchmark arm {arm!r}")
    return SimpleNamespace(**values)


def checkpoint_path(root, run_tag, arm, args):
    run_id = (
        f"{run_tag}_{arm}_{args.dataset}_sl{args.seq_len}_pl{args.pred_len}_"
        f"cf{args.cut_freq}_sd{args.seed}"
    )
    setting = (
        f"{run_id}_AsySpecX_{args.data_name}_ftM_sl{args.seq_len}_pl{args.pred_len}_"
        f"cycle{args.cycle}_seed{args.seed}"
    )
    return os.path.join(root, setting, "checkpoint.pth")


def elapsed_per_step(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / repeats


def benchmark_arm(arm, args, x, phase):
    model = Model(make_config(arm, args)).to(args.device)
    checkpoint = checkpoint_path(args.checkpoint_root, args.run_tag, arm, args)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state, strict=True)

    model.eval()
    with torch.no_grad():
        def inference_step():
            model(x, cycle_index=phase)

        inference_ms = elapsed_per_step(inference_step, args.warmup, args.inference_repeats)

    model.train()
    target = torch.zeros(args.batch_size, args.pred_len, args.channels, device=args.device)

    def train_step():
        model.zero_grad(set_to_none=True)
        prediction = model(x, cycle_index=phase)
        torch.mean((prediction - target) ** 2).backward()

    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.train_repeats):
        train_step()
    torch.cuda.synchronize()
    train_ms = 1000.0 * (time.perf_counter() - start) / args.train_repeats
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    result = {
        "arm": arm,
        "n_param": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_ms_per_batch": inference_ms,
        "train_forward_backward_ms_per_batch": train_ms,
        "fixed_work_peak_cuda_mb": peak_mb,
        "checkpoint": checkpoint,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="dense_direct,fk_r8_cs,fk_r8,fk_r4")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--run_tag", default="phase11_screen_0715v1")
    p.add_argument("--base_profile", default="ind_cycle_full")
    p.add_argument("--dataset", default="weather")
    p.add_argument("--data_name", default="custom")
    p.add_argument("--data_path", default="weather.csv")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=720)
    p.add_argument("--channels", type=int, default=21)
    p.add_argument("--cycle", type=int, default=144)
    p.add_argument("--cut_freq", type=int, default=13)
    p.add_argument("--cross_rank", type=int, default=8)
    p.add_argument("--num_bands", type=int, default=8)
    p.add_argument("--periods", default="144")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--inference_repeats", type=int, default=50)
    p.add_argument("--train_repeats", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-work audit requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    x = torch.randn(args.batch_size, args.seq_len, args.channels, device=args.device)
    phase = torch.randint(0, args.cycle, (args.batch_size,), device=args.device)
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    rows = [benchmark_arm(arm, args, x, phase) for arm in arms]
    dense = next((row for row in rows if row["arm"] == "dense_direct"), None)
    stage_a = next((row for row in rows if row["arm"] == "fk_r8_cs"), None)
    for row in rows:
        if dense is not None:
            row["inference_ratio_vs_dense"] = (
                row["inference_ms_per_batch"] / dense["inference_ms_per_batch"]
            )
            row["train_ratio_vs_dense"] = (
                row["train_forward_backward_ms_per_batch"]
                / dense["train_forward_backward_ms_per_batch"]
            )
            row["peak_memory_ratio_vs_dense"] = (
                row["fixed_work_peak_cuda_mb"] / dense["fixed_work_peak_cuda_mb"]
            )
        if stage_a is not None:
            row["inference_ratio_vs_stage_a"] = (
                row["inference_ms_per_batch"] / stage_a["inference_ms_per_batch"]
            )
            row["train_ratio_vs_stage_a"] = (
                row["train_forward_backward_ms_per_batch"]
                / stage_a["train_forward_backward_ms_per_batch"]
            )
            row["peak_memory_ratio_vs_stage_a"] = (
                row["fixed_work_peak_cuda_mb"]
                / stage_a["fixed_work_peak_cuda_mb"]
            )

    payload = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "base_profile": args.base_profile,
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "channels": args.channels,
        "cycle": args.cycle,
        "cut_freq": args.cut_freq,
        "batch_size": args.batch_size,
        "synthetic_only_no_dataset_read": True,
        "warmup": args.warmup,
        "inference_repeats": args.inference_repeats,
        "train_repeats": args.train_repeats,
        "rows": rows,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
