#!/usr/bin/env python3
"""Leakage-safe selector for the Phase 11 Stage-B real-SM screen."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_rows(root):
    rows = []
    for path in sorted(Path(root).glob("**/run_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_file"] = str(path)
        log_path = Path(str(row.get("log_file", "")))
        if log_path.is_file():
            matches = re.findall(
                r"Epoch:\s*(\d+),\s*Steps:",
                log_path.read_text(encoding="utf-8", errors="replace"),
            )
            row["epochs_ran"] = max(map(int, matches)) if matches else None
        rows.append(row)
    return rows


def attach_fixed_work(rows, path):
    if not path:
        return
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("synthetic_only_no_dataset_read") is not True:
        raise ValueError("fixed-work audit must be synthetic and dataset-free")
    by_arm = {row["arm"]: row for row in payload.get("rows", [])}
    for row in rows:
        audit = by_arm.get(row.get("arm"))
        if audit is None:
            continue
        row["fixed_train_ratio_vs_stage_a"] = audit.get("train_ratio_vs_stage_a")
        row["fixed_inference_ratio_vs_stage_a"] = audit.get(
            "inference_ratio_vs_stage_a"
        )
        row["fixed_peak_memory_ratio_vs_stage_a"] = audit.get(
            "peak_memory_ratio_vs_stage_a"
        )


def build_ranking(rows, resource_tolerance=0.10, min_abs_val_gain=0.0):
    ok = [row for row in rows if row.get("status") == "ok" and finite(row.get("val_mse"))]
    stage_a_rows = [row for row in ok if row.get("arm") == "fk_r8_cs"]
    dense_rows = [row for row in ok if row.get("arm") == "dense_direct"]
    if len(stage_a_rows) != 1 or len(dense_rows) != 1:
        raise ValueError("selection requires exactly one fk_r8_cs and dense_direct row")
    stage_a = stage_a_rows[0]
    dense = dense_rows[0]
    stage_a_val = float(stage_a["val_mse"])
    dense_params = int(dense.get("n_param") or 0)
    stage_a_time = float(stage_a.get("t_train") or 0.0)
    stage_a_epochs = int(stage_a.get("epochs_ran") or 0)
    stage_a_per_epoch = stage_a_time / stage_a_epochs if stage_a_epochs else stage_a_time

    ranking = []
    for row in ok:
        arm = str(row.get("arm", ""))
        val = float(row["val_mse"])
        params = int(row.get("n_param") or 0)
        is_sm = arm.startswith("fk_sm")
        strict_val_gain = val < stage_a_val - float(min_abs_val_gain)
        parameter_ok = params > 0 and params < dense_params
        fixed_train = row.get("fixed_train_ratio_vs_stage_a")
        fixed_infer = row.get("fixed_inference_ratio_vs_stage_a")
        if finite(fixed_train) and finite(fixed_infer):
            resource_ok = (
                float(fixed_train) <= 1.0 + resource_tolerance
                and float(fixed_infer) <= 1.0 + resource_tolerance
            )
        else:
            run_time = float(row.get("t_train") or 0.0)
            epochs = int(row.get("epochs_ran") or 0)
            per_epoch = run_time / epochs if epochs else run_time
            resource_ok = (
                stage_a_per_epoch <= 0.0
                or per_epoch <= stage_a_per_epoch * (1.0 + resource_tolerance)
            )
        kernel_gate = row.get("forecast_kernel_gate_mean")
        kernel_gate_ok = finite(kernel_gate) and 0.002 <= float(kernel_gate) <= 0.998
        sm_gate = row.get("forecast_kernel_sm_gate_abs_max")
        factor_min = row.get("forecast_kernel_sm_factor_min")
        factor_max = row.get("forecast_kernel_sm_factor_max")
        sm_active = (
            finite(sm_gate)
            and float(sm_gate) >= 1e-3
            and finite(factor_min)
            and finite(factor_max)
            and 0.0 < float(factor_min) <= float(factor_max) < 20.0
        )
        eligible = (
            is_sm
            and strict_val_gain
            and parameter_ok
            and resource_ok
            and kernel_gate_ok
            and sm_active
        )
        ranking.append(
            {
                **row,
                "stage_a_val_mse": stage_a_val,
                "delta_val_vs_stage_a": val - stage_a_val,
                "delta_val_vs_stage_a_pct": 100.0 * (val / stage_a_val - 1.0),
                "param_ratio_vs_dense": params / dense_params if dense_params else None,
                "strict_val_gain": int(strict_val_gain),
                "parameter_ok": int(parameter_ok),
                "resource_ok": int(resource_ok),
                "kernel_gate_ok": int(kernel_gate_ok),
                "sm_active": int(sm_active),
                "eligible": int(eligible),
            }
        )
    ranking.sort(key=lambda row: (float(row["val_mse"]), int(row.get("n_param") or 0)))
    promoted = [row for row in ranking if row["eligible"]][:2]
    return ranking, promoted


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fixed_work_json", default="")
    parser.add_argument("--resource_tolerance", type=float, default=0.10)
    parser.add_argument("--min_abs_val_gain", type=float, default=0.0)
    parser.add_argument(
        "--require_arms",
        default=(
            "dense_direct,fk_r8_cs,fk_sm2_shared,fk_sm2_mode,"
            "fk_sm4_mode,fk_sm4_frozen"
        ),
    )
    args = parser.parse_args()

    rows = load_rows(args.root)
    required = {item.strip() for item in args.require_arms.split(",") if item.strip()}
    present = {str(row.get("arm", "")) for row in rows}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"Stage-B screen incomplete; missing arms: {missing}")
    leaked = [row.get("arm") for row in rows if row.get("test_deferred") is not True]
    if leaked:
        raise ValueError(f"selector refuses rows that opened test: {leaked}")
    attach_fixed_work(rows, args.fixed_work_json)
    ranking, promoted = build_ranking(
        rows,
        resource_tolerance=args.resource_tolerance,
        min_abs_val_gain=args.min_abs_val_gain,
    )

    output = Path(args.output_dir)
    fields = [
        "arm", "dataset", "seq_len", "pred_len", "seed", "cut_freq", "status",
        "n_param", "val_mse", "stage_a_val_mse", "delta_val_vs_stage_a",
        "delta_val_vs_stage_a_pct", "param_ratio_vs_dense", "t_train",
        "peak_cuda_mb", "forecast_sm_components", "forecast_sm_sharing",
        "forecast_sm_base_trainable", "forecast_kernel_gate_mean",
        "forecast_kernel_sm_gate_mean", "forecast_kernel_sm_gate_abs_max",
        "forecast_kernel_sm_factor_min", "forecast_kernel_sm_factor_max",
        "fixed_train_ratio_vs_stage_a", "fixed_inference_ratio_vs_stage_a",
        "fixed_peak_memory_ratio_vs_stage_a", "strict_val_gain", "parameter_ok",
        "resource_ok", "kernel_gate_ok", "sm_active", "eligible", "job_id",
        "summary_file",
    ]
    write_csv(output / "validation_ranking.csv", ranking, fields)
    write_csv(output / "promoted.csv", promoted, fields)
    output.mkdir(parents=True, exist_ok=True)
    for filename, selected in (
        ("promoted.tsv", promoted),
        (
            "eval_selected.tsv",
            [row for row in ranking if row["arm"] == "fk_r8_cs"] + promoted,
        ),
    ):
        with (output / filename).open("w", encoding="utf-8") as handle:
            handle.write("# arm\tdataset\tseq_len\tpred_len\tseed\tcut_freq\n")
            for row in selected:
                handle.write(
                    f"{row['arm']}\t{row['dataset']}\t{row['seq_len']}\t"
                    f"{row['pred_len']}\t{row['seed']}\t{row['cut_freq']}\n"
                )

    lines = [
        "# Phase 11 Stage B validation-only selection",
        "",
        "No test metric was read or used.",
        "",
        "| arm | val MSE | vs Stage A | params | fixed train | fixed infer | SM gate | envelope | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in ranking:
        train = row.get("fixed_train_ratio_vs_stage_a")
        infer = row.get("fixed_inference_ratio_vs_stage_a")
        sm_gate = row.get("forecast_kernel_sm_gate_abs_max")
        envelope = ""
        if finite(row.get("forecast_kernel_sm_factor_min")) and finite(
            row.get("forecast_kernel_sm_factor_max")
        ):
            envelope = (
                f"{float(row['forecast_kernel_sm_factor_min']):.3f}.."
                f"{float(row['forecast_kernel_sm_factor_max']):.3f}"
            )
        lines.append(
            f"| {row['arm']} | {float(row['val_mse']):.7f} | "
            f"{float(row['delta_val_vs_stage_a_pct']):+.4f}% | {row['n_param']} | "
            f"{float(train):.3f}x | {float(infer):.3f}x | "
            f"{float(sm_gate):.4f} | {envelope} | {row['eligible']} |"
            if finite(train) and finite(infer) and finite(sm_gate)
            else f"| {row['arm']} | {float(row['val_mse']):.7f} | "
            f"{float(row['delta_val_vs_stage_a_pct']):+.4f}% | {row['n_param']} |  |  |  | {envelope} | {row['eligible']} |"
        )
    lines.extend(["", "Promoted: " + (", ".join(row["arm"] for row in promoted) or "none"), ""])
    (output / "summary_stageb.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "promoted": [row["arm"] for row in promoted]}, sort_keys=True))


if __name__ == "__main__":
    main()
