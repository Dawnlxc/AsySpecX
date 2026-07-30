#!/usr/bin/env python3
"""Three-seed confirmation of one complex arm against the real-SM anchor."""

import argparse
import csv
import json
import statistics
from pathlib import Path


REFERENCE_ARM = "fk_sm4_mode"
SEEDS = (2024, 2025, 2026)


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
    arms = (REFERENCE_ARM, str(candidate))
    by_key = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("test_deferred") is not False:
            continue
        if row.get("val_mse") is None or row.get("mse") is None:
            continue
        by_key[(str(row["arm"]), int(row["seed"]))] = row
    missing = [
        (arm, seed) for arm in arms for seed in SEEDS if (arm, seed) not in by_key
    ]
    if missing:
        raise ValueError(f"complex confirmation incomplete: {missing}")
    reference = {seed: by_key[(REFERENCE_ARM, seed)] for seed in SEEDS}
    paired = []
    for arm in arms:
        for seed in SEEDS:
            row = by_key[(arm, seed)]
            base = reference[seed]
            paired.append(
                {
                    **row,
                    "real_sm_mse_same_seed": float(base["mse"]),
                    "real_sm_val_same_seed": float(base["val_mse"]),
                    "test_delta_vs_real_sm": float(row["mse"]) - float(base["mse"]),
                    "val_delta_vs_real_sm": float(row["val_mse"])
                    - float(base["val_mse"]),
                    "test_win_vs_real_sm": int(float(row["mse"]) < float(base["mse"])),
                }
            )
    groups = {
        arm: sorted(
            (row for row in paired if row["arm"] == arm),
            key=lambda row: int(row["seed"]),
        )
        for arm in arms
    }
    ref_val = mean(row["val_mse"] for row in groups[REFERENCE_ARM])
    ref_test = mean(row["mse"] for row in groups[REFERENCE_ARM])
    summaries = []
    for arm in arms:
        group = groups[arm]
        val_mean = mean(row["val_mse"] for row in group)
        test_mean = mean(row["mse"] for row in group)
        wins = sum(int(row["test_win_vs_real_sm"]) for row in group)
        stable = arm == candidate and wins >= 2 and val_mean < ref_val
        phases = [
            row["forecast_kernel_phase_abs_max"]
            for row in group
            if row.get("forecast_kernel_phase_abs_max") is not None
        ]
        summaries.append(
            {
                "arm": arm,
                "seeds": 3,
                "val_mse_mean": val_mean,
                "val_mse_std": pstdev(row["val_mse"] for row in group),
                "test_mse_mean": test_mean,
                "test_mse_std": pstdev(row["mse"] for row in group),
                "test_mae_mean": mean(row["mae"] for row in group),
                "test_mae_std": pstdev(row["mae"] for row in group),
                "test_wins_vs_real_sm": wins,
                "mean_test_delta_vs_real_sm": test_mean - ref_test,
                "mean_test_delta_vs_real_sm_pct": 100.0 * (test_mean / ref_test - 1.0),
                "mean_val_delta_vs_real_sm": val_mean - ref_val,
                "n_param": max(int(row["n_param"]) for row in group),
                "peak_cuda_mb": max(float(row["peak_cuda_mb"]) for row in group),
                "t_train_mean": mean(row["t_train"] for row in group),
                "t_inf_mean": mean(row["t_inf"] for row in group),
                "phase_abs_max_mean": mean(phases) if phases else None,
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
        ref = by_arm[REFERENCE_ARM]
        for row in summaries:
            measured = by_arm[row["arm"]]
            row["fixed_train_ratio_vs_real_sm"] = (
                measured["train_forward_backward_ms_per_batch"]
                / ref["train_forward_backward_ms_per_batch"]
            )
            row["fixed_inference_ratio_vs_real_sm"] = (
                measured["inference_ms_per_batch"] / ref["inference_ms_per_batch"]
            )
            row["fixed_peak_memory_ratio_vs_real_sm"] = (
                measured["fixed_work_peak_cuda_mb"] / ref["fixed_work_peak_cuda_mb"]
            )
    output = Path(args.output_dir)
    write_csv(
        output / "paired_seed_results.csv",
        paired,
        [
            "arm", "seed", "val_mse", "mse", "mae", "real_sm_val_same_seed",
            "real_sm_mse_same_seed", "val_delta_vs_real_sm", "test_delta_vs_real_sm",
            "test_win_vs_real_sm", "n_param", "t_train", "t_inf", "peak_cuda_mb",
            "forecast_kernel_phase_abs_max", "job_id", "summary_file",
        ],
    )
    write_csv(
        output / "three_seed_summary.csv",
        summaries,
        [
            "arm", "seeds", "val_mse_mean", "val_mse_std", "test_mse_mean",
            "test_mse_std", "test_mae_mean", "test_mae_std", "test_wins_vs_real_sm",
            "mean_test_delta_vs_real_sm", "mean_test_delta_vs_real_sm_pct",
            "mean_val_delta_vs_real_sm", "n_param", "peak_cuda_mb", "t_train_mean",
            "t_inf_mean", "phase_abs_max_mean", "fixed_train_ratio_vs_real_sm",
            "fixed_inference_ratio_vs_real_sm", "fixed_peak_memory_ratio_vs_real_sm",
            "stable_win",
        ],
    )
    lines = [
        "# Complex phase three-seed confirmation",
        "",
        "Stable win = at least 2/3 paired test wins and lower mean validation MSE than real SM.",
        "",
        "| arm | val mean | test MSE mean +/- std | wins | MAE | params | train | infer | stable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        ft = row.get("fixed_train_ratio_vs_real_sm")
        fi = row.get("fixed_inference_ratio_vs_real_sm")
        lines.append(
            f"| {row['arm']} | {row['val_mse_mean']:.7f} | "
            f"{row['test_mse_mean']:.7f} +/- {row['test_mse_std']:.7f} | "
            f"{row['test_wins_vs_real_sm']}/3 | {row['test_mae_mean']:.7f} | "
            f"{row['n_param']} | {'' if ft is None else f'{ft:.3f}x'} | "
            f"{'' if fi is None else f'{fi:.3f}x'} | {row['stable_win']} |"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "PHASE11_COMPLEX_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "summaries": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
