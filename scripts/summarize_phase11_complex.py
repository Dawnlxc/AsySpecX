#!/usr/bin/env python3
"""Validation-only selector for bounded within-variable complex phase."""

import argparse
import csv
import json
import math
from pathlib import Path


REFERENCE_ARM = "fk_sm4_mode"


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
        rows.append(row)
    return rows


def attach_fixed_work(rows, path):
    if not path:
        return
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("synthetic_only_no_dataset_read") is not True:
        raise ValueError("fixed-work audit must be synthetic and dataset-free")
    by_arm = {row["arm"]: row for row in payload.get("rows", [])}
    reference = by_arm.get(REFERENCE_ARM)
    if reference is None:
        raise ValueError(f"fixed-work audit is missing {REFERENCE_ARM}")
    for row in rows:
        measured = by_arm.get(row.get("arm"))
        if measured is None:
            continue
        row["fixed_train_ratio_vs_real_sm"] = (
            float(measured["train_forward_backward_ms_per_batch"])
            / float(reference["train_forward_backward_ms_per_batch"])
        )
        row["fixed_inference_ratio_vs_real_sm"] = (
            float(measured["inference_ms_per_batch"])
            / float(reference["inference_ms_per_batch"])
        )
        row["fixed_peak_memory_ratio_vs_real_sm"] = (
            float(measured["fixed_work_peak_cuda_mb"])
            / float(reference["fixed_work_peak_cuda_mb"])
        )


def build_ranking(rows, resource_tolerance=0.10):
    ok = [row for row in rows if row.get("status") == "ok" and finite(row.get("val_mse"))]
    references = [row for row in ok if row.get("arm") == REFERENCE_ARM]
    dense_rows = [row for row in ok if row.get("arm") == "dense_direct"]
    if len(references) != 1 or len(dense_rows) != 1:
        raise ValueError("complex selection requires one real-SM anchor and one dense row")
    reference = references[0]
    dense = dense_rows[0]
    reference_val = float(reference["val_mse"])
    dense_params = int(dense.get("n_param") or 0)
    ranking = []
    for row in ok:
        arm = str(row.get("arm", ""))
        is_phase = arm.startswith("fk_sm4_ph")
        val = float(row["val_mse"])
        params = int(row.get("n_param") or 0)
        strict_val_gain = val < reference_val
        parameter_ok = 0 < params < dense_params
        train_ratio = row.get("fixed_train_ratio_vs_real_sm")
        inference_ratio = row.get("fixed_inference_ratio_vs_real_sm")
        resource_ok = (
            finite(train_ratio)
            and finite(inference_ratio)
            and float(train_ratio) <= 1.0 + resource_tolerance
            and float(inference_ratio) <= 1.0 + resource_tolerance
        )
        kernel_gate = row.get("forecast_kernel_gate_mean")
        kernel_gate_ok = finite(kernel_gate) and 0.002 <= float(kernel_gate) <= 0.998
        phase_abs = row.get("forecast_kernel_phase_abs_max")
        configured_max = row.get("forecast_phase_max")
        phase_active = (
            finite(phase_abs)
            and float(phase_abs) >= 1e-4
            and finite(configured_max)
            and 0.0 < float(phase_abs) <= float(configured_max) + 1e-6
        )
        eligible = (
            is_phase
            and strict_val_gain
            and parameter_ok
            and resource_ok
            and kernel_gate_ok
            and phase_active
        )
        ranking.append(
            {
                **row,
                "real_sm_val_mse": reference_val,
                "delta_val_vs_real_sm": val - reference_val,
                "delta_val_vs_real_sm_pct": 100.0 * (val / reference_val - 1.0),
                "param_ratio_vs_dense": params / dense_params if dense_params else None,
                "strict_val_gain": int(strict_val_gain),
                "parameter_ok": int(parameter_ok),
                "resource_ok": int(resource_ok),
                "kernel_gate_ok": int(kernel_gate_ok),
                "phase_active": int(phase_active),
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
    parser.add_argument("--fixed_work_json", required=True)
    parser.add_argument("--resource_tolerance", type=float, default=0.10)
    parser.add_argument(
        "--require_arms",
        default="dense_direct,fk_sm4_mode,fk_sm4_ph2_q,fk_sm4_ph4_q,fk_sm4_ph4_h",
    )
    args = parser.parse_args()
    rows = load_rows(args.root)
    required = {item.strip() for item in args.require_arms.split(",") if item.strip()}
    present = {str(row.get("arm", "")) for row in rows}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"complex screen incomplete; missing arms: {missing}")
    leaked = [row.get("arm") for row in rows if row.get("test_deferred") is not True]
    if leaked:
        raise ValueError(f"complex selector refuses rows that opened test: {leaked}")
    attach_fixed_work(rows, args.fixed_work_json)
    ranking, promoted = build_ranking(rows, args.resource_tolerance)
    output = Path(args.output_dir)
    fields = [
        "arm", "dataset", "seq_len", "pred_len", "seed", "cut_freq", "status",
        "n_param", "val_mse", "real_sm_val_mse", "delta_val_vs_real_sm",
        "delta_val_vs_real_sm_pct", "param_ratio_vs_dense", "forecast_phase_basis_dim",
        "forecast_phase_max", "forecast_kernel_gate_mean", "forecast_kernel_sm_gate_abs_max",
        "forecast_kernel_phase_abs_mean", "forecast_kernel_phase_abs_max",
        "forecast_kernel_phase_rms", "fixed_train_ratio_vs_real_sm",
        "fixed_inference_ratio_vs_real_sm", "fixed_peak_memory_ratio_vs_real_sm",
        "strict_val_gain", "parameter_ok", "resource_ok", "kernel_gate_ok",
        "phase_active", "eligible", "job_id", "summary_file",
    ]
    write_csv(output / "validation_ranking.csv", ranking, fields)
    write_csv(output / "promoted.csv", promoted, fields)
    output.mkdir(parents=True, exist_ok=True)
    for filename, selected in (
        ("promoted.tsv", promoted),
        ("eval_selected.tsv", [r for r in ranking if r["arm"] == REFERENCE_ARM] + promoted),
    ):
        with (output / filename).open("w", encoding="utf-8") as handle:
            handle.write("# arm\tdataset\tseq_len\tpred_len\tseed\tcut_freq\n")
            for row in selected:
                handle.write(
                    f"{row['arm']}\t{row['dataset']}\t{row['seq_len']}\t"
                    f"{row['pred_len']}\t{row['seed']}\t{row['cut_freq']}\n"
                )
    lines = [
        "# Complex-phase validation-only selection",
        "",
        "No test metric was read or used.",
        "",
        "| arm | val MSE | vs real SM | params | train | infer | phase max | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranking:
        train = row.get("fixed_train_ratio_vs_real_sm")
        infer = row.get("fixed_inference_ratio_vs_real_sm")
        phase = row.get("forecast_kernel_phase_abs_max")
        lines.append(
            f"| {row['arm']} | {float(row['val_mse']):.7f} | "
            f"{float(row['delta_val_vs_real_sm_pct']):+.4f}% | {row['n_param']} | "
            f"{'' if not finite(train) else f'{float(train):.3f}x'} | "
            f"{'' if not finite(infer) else f'{float(infer):.3f}x'} | "
            f"{'' if not finite(phase) else f'{float(phase):.4f}'} | {row['eligible']} |"
        )
    lines.extend(["", "Promoted: " + (", ".join(r["arm"] for r in promoted) or "none"), ""])
    (output / "summary_complex.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "promoted": [r["arm"] for r in promoted]}, sort_keys=True))


if __name__ == "__main__":
    main()
