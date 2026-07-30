#!/usr/bin/env python3
"""Aggregate the frozen Phase 11 three-seed confirmation."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED_ARMS = ("dense_direct", "fk_r8_cs", "fk_r8")
REQUIRED_SEEDS = (2024, 2025, 2026)
PUBLISHED_WEATHER_H720 = 0.3387


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


def summarize(rows):
    by_key = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("test_deferred") is not False:
            continue
        if row.get("mse") is None or row.get("val_mse") is None:
            continue
        by_key[(str(row["arm"]), int(row["seed"]))] = row
    missing = [
        (arm, seed) for arm in REQUIRED_ARMS for seed in REQUIRED_SEEDS
        if (arm, seed) not in by_key
    ]
    if missing:
        raise ValueError(f"confirmation is incomplete: {missing}")

    dense = {seed: by_key[("dense_direct", seed)] for seed in REQUIRED_SEEDS}
    paired = []
    for arm in REQUIRED_ARMS:
        for seed in REQUIRED_SEEDS:
            row = by_key[(arm, seed)]
            reference = dense[seed]
            paired.append({
                **row,
                "dense_mse_same_seed": float(reference["mse"]),
                "dense_val_same_seed": float(reference["val_mse"]),
                "test_delta_vs_dense": float(row["mse"]) - float(reference["mse"]),
                "val_delta_vs_dense": float(row["val_mse"]) - float(reference["val_mse"]),
                "test_win_vs_dense": int(float(row["mse"]) < float(reference["mse"])),
                "test_win_vs_published": int(float(row["mse"]) < PUBLISHED_WEATHER_H720),
            })

    grouped = defaultdict(list)
    for row in paired:
        grouped[row["arm"]].append(row)
    arm_rows = []
    dense_val_mean = mean(row["val_mse"] for row in grouped["dense_direct"])
    dense_test_mean = mean(row["mse"] for row in grouped["dense_direct"])
    for arm in REQUIRED_ARMS:
        group = sorted(grouped[arm], key=lambda row: int(row["seed"]))
        val_mean = mean(row["val_mse"] for row in group)
        test_mean = mean(row["mse"] for row in group)
        test_wins = sum(int(row["test_win_vs_dense"]) for row in group)
        val_mean_improves = val_mean < dense_val_mean
        stable_win = arm != "dense_direct" and test_wins >= 2 and val_mean_improves
        arm_rows.append({
            "arm": arm,
            "seeds": 3,
            "val_mse_mean": val_mean,
            "val_mse_std": pstdev(row["val_mse"] for row in group),
            "test_mse_mean": test_mean,
            "test_mse_std": pstdev(row["mse"] for row in group),
            "test_mae_mean": mean(row["mae"] for row in group),
            "test_mae_std": pstdev(row["mae"] for row in group),
            "test_wins_vs_dense": test_wins,
            "test_wins_vs_published": sum(int(row["test_win_vs_published"]) for row in group),
            "mean_test_delta_vs_dense": test_mean - dense_test_mean,
            "mean_test_delta_vs_dense_pct": 100.0 * (test_mean / dense_test_mean - 1.0),
            "mean_val_delta_vs_dense": val_mean - dense_val_mean,
            "n_param": max(int(row["n_param"]) for row in group),
            "peak_cuda_mb": max(float(row["peak_cuda_mb"]) for row in group),
            "t_train_mean": mean(row["t_train"] for row in group),
            "t_inf_mean": mean(row["t_inf"] for row in group),
            "gate_mean": mean(
                row["forecast_kernel_gate_mean"] for row in group
                if row.get("forecast_kernel_gate_mean") is not None
            ) if arm != "dense_direct" else None,
            "stable_win": int(stable_win),
        })
    return paired, arm_rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--fixed_work_json", default="")
    args = p.parse_args()
    paired, arms = summarize(load_rows(args.root))

    if args.fixed_work_json:
        audit = json.loads(Path(args.fixed_work_json).read_text(encoding="utf-8"))
        audit_by_arm = {row["arm"]: row for row in audit.get("rows", [])}
        for row in arms:
            measured = audit_by_arm.get(row["arm"], {})
            row["fixed_train_ratio"] = measured.get("train_ratio_vs_dense")
            row["fixed_inference_ratio"] = measured.get("inference_ratio_vs_dense")
            row["fixed_peak_memory_ratio"] = measured.get("peak_memory_ratio_vs_dense")

    output = Path(args.output_dir)
    write_csv(output / "paired_seed_results.csv", paired, [
        "arm", "seed", "val_mse", "mse", "mae", "dense_val_same_seed",
        "dense_mse_same_seed", "val_delta_vs_dense", "test_delta_vs_dense",
        "test_win_vs_dense", "test_win_vs_published", "n_param", "t_train",
        "t_inf", "peak_cuda_mb", "forecast_kernel_gate_mean", "job_id", "summary_file",
    ])
    write_csv(output / "three_seed_summary.csv", arms, [
        "arm", "seeds", "val_mse_mean", "val_mse_std", "test_mse_mean",
        "test_mse_std", "test_mae_mean", "test_mae_std", "test_wins_vs_dense",
        "test_wins_vs_published", "mean_test_delta_vs_dense",
        "mean_test_delta_vs_dense_pct", "mean_val_delta_vs_dense", "n_param",
        "peak_cuda_mb", "t_train_mean", "t_inf_mean", "gate_mean",
        "fixed_train_ratio", "fixed_inference_ratio", "fixed_peak_memory_ratio",
        "stable_win",
    ])

    lines = [
        "# AsySpecX Phase 11 final confirmation",
        "",
        "A stable win requires at least two seed-matched test wins over dense and lower mean validation MSE.",
        "",
        "| arm | val mean | test MSE mean ± std | wins vs dense | MAE mean | params | fixed train | fixed infer | stable win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in arms:
        ft = row.get("fixed_train_ratio")
        fi = row.get("fixed_inference_ratio")
        lines.append(
            f"| {row['arm']} | {row['val_mse_mean']:.7f} | "
            f"{row['test_mse_mean']:.7f} ± {row['test_mse_std']:.7f} | "
            f"{row['test_wins_vs_dense']}/3 | {row['test_mae_mean']:.7f} | "
            f"{row['n_param']} | {'' if ft is None else f'{ft:.3f}x'} | "
            f"{'' if fi is None else f'{fi:.3f}x'} | {row['stable_win']} |"
        )
    lines += ["", f"Published Weather H720 reference: {PUBLISHED_WEATHER_H720:.4f}.", ""]
    output.mkdir(parents=True, exist_ok=True)
    (output / "PHASE11_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(paired), "arms": arms, "output_dir": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
