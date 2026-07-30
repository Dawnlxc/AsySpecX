#!/usr/bin/env python3
"""Validation-only Phase 11 selector and promotion report."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def load_rows(root):
    rows = []
    for path in sorted(Path(root).glob("**/run_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_file"] = str(path)
        log_path = Path(str(row.get("log_file", "")))
        if log_path.is_file():
            epochs = [int(value) for value in re.findall(
                r"Epoch:\s*(\d+),\s*Steps:", log_path.read_text(encoding="utf-8", errors="replace")
            )]
            row["epochs_ran"] = max(epochs) if epochs else None
        rows.append(row)
    return rows


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_ranking(rows, val_tolerance=0.005, resource_tolerance=0.10):
    ok = [r for r in rows if r.get("status") == "ok" and finite(r.get("val_mse"))]
    anchors = [r for r in ok if r.get("arm") in {"anchor", "dense_direct"}]
    if len(anchors) < 2:
        raise ValueError("both anchor and dense_direct must finish before selection")
    dense = min((r for r in anchors if r.get("arm") == "dense_direct"), key=lambda r: float(r["val_mse"]))
    reference = min(anchors, key=lambda r: float(r["val_mse"]))
    dense_val = float(dense["val_mse"])
    ref_val = float(reference["val_mse"])
    dense_params = int(dense.get("n_param") or 0)
    dense_time = float(dense.get("t_train") or 0.0)
    dense_epochs = int(dense.get("epochs_ran") or 0)
    dense_time_per_epoch = dense_time / dense_epochs if dense_epochs else dense_time

    ranking = []
    for row in ok:
        val = float(row["val_mse"])
        params = int(row.get("n_param") or 0)
        train_time = float(row.get("t_train") or 0.0)
        epochs_ran = int(row.get("epochs_ran") or 0)
        train_time_per_epoch = train_time / epochs_ran if epochs_ran else train_time
        gate = row.get("forecast_kernel_gate_mean")
        fixed_train_ratio = row.get("fixed_work_train_ratio_vs_dense")
        fixed_inference_ratio = row.get("fixed_work_inference_ratio_vs_dense")
        is_kernel = str(row.get("arm", "")).startswith("fk_")
        within_val_band = val <= ref_val * (1.0 + val_tolerance)
        parameter_ok = params > 0 and params < dense_params
        if finite(fixed_train_ratio) and finite(fixed_inference_ratio):
            resource_ok = (
                float(fixed_train_ratio) <= 1.0 + resource_tolerance
                and float(fixed_inference_ratio) <= 1.0 + resource_tolerance
            )
        else:
            resource_ok = (
                dense_time_per_epoch <= 0.0
                or train_time_per_epoch <= dense_time_per_epoch * (1.0 + resource_tolerance)
            )
        gate_open = finite(gate) and 0.002 <= float(gate) <= 0.998
        gate_ok = (not is_kernel) or gate_open or val < ref_val
        eligible = is_kernel and within_val_band and parameter_ok and resource_ok and gate_ok
        ranking.append({
            **row,
            "reference_arm": reference["arm"],
            "reference_val_mse": ref_val,
            "delta_val_vs_reference_pct": 100.0 * (val / ref_val - 1.0),
            "delta_val_vs_dense_pct": 100.0 * (val / dense_val - 1.0),
            "param_ratio_vs_dense": params / dense_params if dense_params else None,
            "train_time_ratio_vs_dense": train_time / dense_time if dense_time else None,
            "train_seconds_per_epoch": train_time_per_epoch,
            "train_per_epoch_ratio_vs_dense": (
                train_time_per_epoch / dense_time_per_epoch if dense_time_per_epoch else None
            ),
            "within_val_band": int(within_val_band),
            "parameter_ok": int(parameter_ok),
            "resource_ok": int(resource_ok),
            "gate_ok": int(gate_ok),
            "eligible": int(eligible),
        })
    ranking.sort(key=lambda r: (float(r["val_mse"]), int(r.get("n_param") or 0)))
    promoted = [row for row in ranking if row["eligible"]][:2]
    return ranking, promoted


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
    p.add_argument("--val_tolerance", type=float, default=0.005)
    p.add_argument("--resource_tolerance", type=float, default=0.10)
    p.add_argument(
        "--fixed_work_json",
        default="",
        help="same-GPU synthetic resource audit; overrides cross-node runtime ratios",
    )
    p.add_argument(
        "--require_arms",
        default="anchor,dense_direct,fk_r4,fk_r8,fk_r8_cs,fk_r8_svd,fk_r16_svd",
        help="comma-separated completeness gate",
    )
    args = p.parse_args()

    rows = load_rows(args.root)
    required = {value.strip() for value in args.require_arms.split(",") if value.strip()}
    present = {str(row.get("arm", "")) for row in rows}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"screen is incomplete; missing arms: {missing}")
    leaked = [row.get("arm") for row in rows if row.get("test_deferred") is not True]
    if leaked:
        raise ValueError(f"validation-only selector refuses rows that opened test: {leaked}")
    if args.fixed_work_json:
        audit = json.loads(Path(args.fixed_work_json).read_text(encoding="utf-8"))
        if audit.get("synthetic_only_no_dataset_read") is not True:
            raise ValueError("fixed-work audit must declare synthetic_only_no_dataset_read=true")
        audit_by_arm = {row["arm"]: row for row in audit.get("rows", [])}
        for row in rows:
            measured = audit_by_arm.get(row.get("arm"))
            if measured is None:
                continue
            row["fixed_work_train_ratio_vs_dense"] = measured.get("train_ratio_vs_dense")
            row["fixed_work_inference_ratio_vs_dense"] = measured.get("inference_ratio_vs_dense")
            row["fixed_work_peak_memory_ratio_vs_dense"] = measured.get("peak_memory_ratio_vs_dense")
    ranking, promoted = build_ranking(rows, args.val_tolerance, args.resource_tolerance)
    output = Path(args.output_dir)
    fields = [
        "arm", "dataset", "seq_len", "pred_len", "seed", "cut_freq", "status",
        "n_param", "val_mse", "val_mse_seg0", "val_mse_seg1", "val_mse_seg2",
        "val_mse_seg3", "t_train", "peak_cuda_mb", "forecast_rank",
        "forecast_init", "forecast_channel_scale", "forecast_kernel_gate_mean",
        "forecast_kernel_effective_rank", "forecast_kernel_raw_rms",
        "forecast_kernel_delta_rms", "reference_arm", "reference_val_mse",
        "delta_val_vs_reference_pct", "delta_val_vs_dense_pct",
        "param_ratio_vs_dense", "epochs_ran", "train_seconds_per_epoch",
        "train_time_ratio_vs_dense", "train_per_epoch_ratio_vs_dense", "within_val_band",
        "fixed_work_train_ratio_vs_dense", "fixed_work_inference_ratio_vs_dense",
        "fixed_work_peak_memory_ratio_vs_dense",
        "parameter_ok", "resource_ok", "gate_ok", "eligible", "job_id", "summary_file",
    ]
    write_csv(output / "validation_ranking.csv", ranking, fields)
    write_csv(output / "promoted.csv", promoted, fields)

    with (output / "promoted.tsv").open("w", encoding="utf-8") as handle:
        handle.write("# arm\tdataset\tseq_len\tpred_len\tseed\tcut_freq\n")
        for row in promoted:
            handle.write(
                f"{row['arm']}\t{row['dataset']}\t{row['seq_len']}\t{row['pred_len']}\t"
                f"{row['seed']}\t{row['cut_freq']}\n"
            )
    eval_rows = [row for row in ranking if row["arm"] == "dense_direct"] + promoted
    with (output / "eval_selected.tsv").open("w", encoding="utf-8") as handle:
        handle.write("# arm\tdataset\tseq_len\tpred_len\tseed\tcut_freq\n")
        for row in eval_rows:
            handle.write(
                f"{row['arm']}\t{row['dataset']}\t{row['seq_len']}\t{row['pred_len']}\t"
                f"{row['seed']}\t{row['cut_freq']}\n"
            )

    lines = [
        "# AsySpecX Phase 11 validation-only screen",
        "",
        "No test metric was read or used by this selector.",
        "",
        "| arm | val MSE | vs reference | params | dense ratio | fixed train | fixed infer | peak MiB | gate | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranking:
        gate = row.get("forecast_kernel_gate_mean")
        gate_text = "" if not finite(gate) else f"{float(gate):.4f}"
        fixed_train = row.get("fixed_work_train_ratio_vs_dense")
        fixed_infer = row.get("fixed_work_inference_ratio_vs_dense")
        fixed_train_text = "" if not finite(fixed_train) else f"{float(fixed_train):.3f}x"
        fixed_infer_text = "" if not finite(fixed_infer) else f"{float(fixed_infer):.3f}x"
        lines.append(
            f"| {row['arm']} | {float(row['val_mse']):.7f} | "
            f"{float(row['delta_val_vs_reference_pct']):+.3f}% | {row['n_param']} | "
            f"{float(row['param_ratio_vs_dense']):.3f} | {fixed_train_text} | {fixed_infer_text} | "
            f"{float(row.get('peak_cuda_mb') or 0.0):.1f} | {gate_text} | {row['eligible']} |"
        )
    lines += ["", "Promoted: " + (", ".join(r["arm"] for r in promoted) or "none"), ""]
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary_phase11.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "ok": sum(r.get("status") == "ok" for r in rows),
        "promoted": [r["arm"] for r in promoted],
        "output_dir": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
