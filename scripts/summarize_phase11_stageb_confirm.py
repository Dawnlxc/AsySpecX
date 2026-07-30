#!/usr/bin/env python3
"""Aggregate a frozen three-seed Stage-B candidate against Stage A."""

import argparse
import csv
import json
import statistics
from pathlib import Path


REQUIRED_SEEDS = (2024, 2025, 2026)


def mean(values):
    return statistics.fmean(float(value) for value in values)


def pstdev(values):
    values = [float(value) for value in values]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def load_rows(root):
    rows = []
    for path in sorted(Path(root).glob("**/run_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_file"] = str(path)
        rows.append(row)
    return rows


def summarize(rows, candidate):
    required_arms = ("fk_r8_cs", str(candidate))
    by_key = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("test_deferred") is not False:
            continue
        if row.get("mse") is None or row.get("val_mse") is None:
            continue
        by_key[(str(row["arm"]), int(row["seed"]))] = row
    missing = [
        (arm, seed)
        for arm in required_arms
        for seed in REQUIRED_SEEDS
        if (arm, seed) not in by_key
    ]
    if missing:
        raise ValueError(f"Stage-B confirmation incomplete: {missing}")

    reference = {seed: by_key[("fk_r8_cs", seed)] for seed in REQUIRED_SEEDS}
    paired = []
    for arm in required_arms:
        for seed in REQUIRED_SEEDS:
            row = by_key[(arm, seed)]
            stage_a = reference[seed]
            paired.append(
                {
                    **row,
                    "stage_a_mse_same_seed": float(stage_a["mse"]),
                    "stage_a_val_same_seed": float(stage_a["val_mse"]),
                    "test_delta_vs_stage_a": float(row["mse"]) - float(stage_a["mse"]),
                    "val_delta_vs_stage_a": float(row["val_mse"])
                    - float(stage_a["val_mse"]),
                    "test_win_vs_stage_a": int(float(row["mse"]) < float(stage_a["mse"])),
                }
            )

    groups = {
        arm: sorted(
            (row for row in paired if row["arm"] == arm),
            key=lambda row: int(row["seed"]),
        )
        for arm in required_arms
    }
    stage_a_val = mean(row["val_mse"] for row in groups["fk_r8_cs"])
    stage_a_test = mean(row["mse"] for row in groups["fk_r8_cs"])
    summaries = []
    for arm in required_arms:
        group = groups[arm]
        val_mean = mean(row["val_mse"] for row in group)
        test_mean = mean(row["mse"] for row in group)
        wins = sum(int(row["test_win_vs_stage_a"]) for row in group)
        stable = arm == candidate and wins >= 2 and val_mean < stage_a_val
        sm_gates = [
            row["forecast_kernel_sm_gate_abs_max"]
            for row in group
            if row.get("forecast_kernel_sm_gate_abs_max") is not None
        ]
        summaries.append(
            {
                "arm": arm,
                "seeds": len(group),
                "val_mse_mean": val_mean,
                "val_mse_std": pstdev(row["val_mse"] for row in group),
                "test_mse_mean": test_mean,
                "test_mse_std": pstdev(row["mse"] for row in group),
                "test_mae_mean": mean(row["mae"] for row in group),
                "test_mae_std": pstdev(row["mae"] for row in group),
                "test_wins_vs_stage_a": wins,
                "mean_test_delta_vs_stage_a": test_mean - stage_a_test,
                "mean_test_delta_vs_stage_a_pct": 100.0
                * (test_mean / stage_a_test - 1.0),
                "mean_val_delta_vs_stage_a": val_mean - stage_a_val,
                "n_param": max(int(row["n_param"]) for row in group),
                "peak_cuda_mb": max(float(row["peak_cuda_mb"]) for row in group),
                "t_train_mean": mean(row["t_train"] for row in group),
                "t_inf_mean": mean(row["t_inf"] for row in group),
                "kernel_gate_mean": mean(row["forecast_kernel_gate_mean"] for row in group),
                "sm_gate_abs_max_mean": mean(sm_gates) if sm_gates else None,
                "stable_win": int(stable),
            }
        )
    return paired, summaries


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fixed_work_json", default="")
    args = parser.parse_args()
    paired, summaries = summarize(load_rows(args.root), args.candidate)

    if args.fixed_work_json:
        audit = json.loads(Path(args.fixed_work_json).read_text(encoding="utf-8"))
        by_arm = {row["arm"]: row for row in audit.get("rows", [])}
        for row in summaries:
            measured = by_arm.get(row["arm"], {})
            row["fixed_train_ratio_vs_stage_a"] = measured.get("train_ratio_vs_stage_a")
            row["fixed_inference_ratio_vs_stage_a"] = measured.get(
                "inference_ratio_vs_stage_a"
            )
            row["fixed_peak_memory_ratio_vs_stage_a"] = measured.get(
                "peak_memory_ratio_vs_stage_a"
            )

    output = Path(args.output_dir)
    write_csv(
        output / "paired_seed_results.csv",
        paired,
        [
            "arm", "seed", "val_mse", "mse", "mae", "stage_a_val_same_seed",
            "stage_a_mse_same_seed", "val_delta_vs_stage_a", "test_delta_vs_stage_a",
            "test_win_vs_stage_a", "n_param", "t_train", "t_inf", "peak_cuda_mb",
            "forecast_kernel_gate_mean", "forecast_kernel_sm_gate_abs_max", "job_id",
            "summary_file",
        ],
    )
    write_csv(
        output / "three_seed_summary.csv",
        summaries,
        [
            "arm", "seeds", "val_mse_mean", "val_mse_std", "test_mse_mean",
            "test_mse_std", "test_mae_mean", "test_mae_std", "test_wins_vs_stage_a",
            "mean_test_delta_vs_stage_a", "mean_test_delta_vs_stage_a_pct",
            "mean_val_delta_vs_stage_a", "n_param", "peak_cuda_mb", "t_train_mean",
            "t_inf_mean", "kernel_gate_mean", "sm_gate_abs_max_mean",
            "fixed_train_ratio_vs_stage_a", "fixed_inference_ratio_vs_stage_a",
            "fixed_peak_memory_ratio_vs_stage_a", "stable_win",
        ],
    )
    lines = [
        "# Phase 11 Stage B real-SM confirmation",
        "",
        "Stable win = at least 2/3 paired test wins over Stage A and lower mean validation MSE.",
        "",
        "| arm | val mean | test MSE mean +/- std | wins | MAE | params | fixed train | fixed infer | stable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        ft = row.get("fixed_train_ratio_vs_stage_a")
        fi = row.get("fixed_inference_ratio_vs_stage_a")
        lines.append(
            f"| {row['arm']} | {row['val_mse_mean']:.7f} | "
            f"{row['test_mse_mean']:.7f} +/- {row['test_mse_std']:.7f} | "
            f"{row['test_wins_vs_stage_a']}/3 | {row['test_mae_mean']:.7f} | "
            f"{row['n_param']} | {'' if ft is None else f'{ft:.3f}x'} | "
            f"{'' if fi is None else f'{fi:.3f}x'} | {row['stable_win']} |"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "PHASE11_STAGEB_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "summaries": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
