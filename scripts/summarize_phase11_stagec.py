#!/usr/bin/env python3
"""Leakage-safe aggregate selector for Phase 11 Stage C Wave 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ARMS = ("anchor", "fk_r8_cs", "fk_sm2_mode", "fk_sm4_ph4_h")
CANDIDATES = ("fk_sm2_mode", "fk_sm4_ph4_h")
STAGE_A = "fk_r8_cs"


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_manifest(path):
    specs = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            profile, arm, dataset, seq_len, pred_len, seed, cut_freq = line.split("\t")
            specs.append(
                {
                    "base_profile": profile,
                    "arm": arm,
                    "dataset": dataset,
                    "seq_len": int(seq_len),
                    "pred_len": int(pred_len),
                    "seed": int(seed),
                    "cut_freq": int(cut_freq),
                }
            )
    return specs


def load_rows(root):
    rows = []
    for path in sorted(Path(root).glob("**/run_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["summary_file"] = str(path)
        rows.append(payload)
    return rows


def load_resources(paths):
    resources = {}
    profiles = set()
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("synthetic_only_no_dataset_read") is not True:
            raise ValueError("fixed-work audit must be synthetic and dataset-free")
        profile = str(payload.get("base_profile", ""))
        if not profile:
            raise ValueError(f"fixed-work audit lacks base_profile: {path}")
        profiles.add(profile)
        for row in payload.get("rows", []):
            resources[(profile, str(row["arm"]))] = row
    return resources, profiles


def cell_key(row):
    return (
        str(row["base_profile"]),
        str(row["dataset"]),
        int(row["seq_len"]),
        int(row["pred_len"]),
        int(row["seed"]),
        int(row["cut_freq"]),
    )


def diagnostic_ok(row):
    gate = row.get("forecast_kernel_gate_mean")
    kernel_ok = finite(gate) and 0.002 <= float(gate) <= 0.998
    sm_gate = row.get("forecast_kernel_sm_gate_abs_max")
    factor_min = row.get("forecast_kernel_sm_factor_min")
    factor_max = row.get("forecast_kernel_sm_factor_max")
    sm_ok = (
        finite(sm_gate)
        and float(sm_gate) >= 1e-3
        and finite(factor_min)
        and finite(factor_max)
        and 0.0 < float(factor_min) <= float(factor_max) < 20.0
    )
    if row.get("arm") == "fk_sm4_ph4_h":
        phase = row.get("forecast_kernel_phase_abs_max")
        phase_max = row.get("forecast_phase_max")
        phase_ok = (
            finite(phase)
            and finite(phase_max)
            and 1e-4 <= float(phase) <= float(phase_max) + 1e-6
        )
    else:
        phase_ok = True
    return kernel_ok and sm_ok and phase_ok


def summarize(
    rows,
    expected_specs,
    resources,
    resource_profiles,
    min_wins=4,
    max_regression_pct=0.05,
    train_limit=1.15,
    inference_limit=1.10,
    memory_limit=1.02,
    parameter_limit=1.15,
):
    expected = {
        (
            spec["base_profile"], spec["arm"], spec["dataset"],
            spec["seq_len"], spec["pred_len"], spec["seed"], spec["cut_freq"],
        )
        for spec in expected_specs
    }
    by_spec = {}
    for row in rows:
        key = (
            str(row.get("base_profile", "")), str(row.get("arm", "")),
            str(row.get("dataset", "")), int(row.get("seq_len", -1)),
            int(row.get("pred_len", -1)), int(row.get("seed", -1)),
            int(row.get("cut_freq", -1)),
        )
        if key in by_spec:
            raise ValueError(f"duplicate Stage-C row: {key}")
        by_spec[key] = row
    missing = sorted(expected - set(by_spec))
    extra = sorted(set(by_spec) - expected)
    if missing or extra:
        raise ValueError(f"Stage-C manifest mismatch: missing={missing}, extra={extra}")
    ordered = [by_spec[key] for key in sorted(expected)]
    failed = [row for row in ordered if row.get("status") != "ok" or not finite(row.get("val_mse"))]
    if failed:
        raise ValueError(f"Stage-C rows failed or lack validation: {[row.get('arm') for row in failed]}")
    leaked = [
        row for row in ordered
        if row.get("test_deferred") is not True
        or row.get("mse") is not None
        or row.get("mae") is not None
    ]
    if leaked:
        raise ValueError(f"Stage-C selector refuses test-open rows: {[row.get('arm') for row in leaked]}")

    grouped = defaultdict(dict)
    for row in ordered:
        grouped[cell_key(row)][str(row["arm"])] = row
    for key, group in grouped.items():
        if set(group) != set(ARMS):
            raise ValueError(f"cell {key} does not contain exactly {ARMS}")

    cell_rows = []
    candidate_cells = defaultdict(list)
    for key in sorted(grouped):
        group = grouped[key]
        anchor_val = float(group["anchor"]["val_mse"])
        stage_a_val = float(group[STAGE_A]["val_mse"])
        sm2_val = float(group["fk_sm2_mode"]["val_mse"])
        for arm in ARMS:
            row = group[arm]
            val = float(row["val_mse"])
            stage_delta_pct = 100.0 * (val / stage_a_val - 1.0)
            anchor_delta_pct = 100.0 * (val / anchor_val - 1.0)
            record = {
                **row,
                "anchor_val_mse": anchor_val,
                "stage_a_val_mse": stage_a_val,
                "delta_val_vs_stage_a": val - stage_a_val,
                "delta_val_vs_stage_a_pct": stage_delta_pct,
                "delta_val_vs_anchor": val - anchor_val,
                "delta_val_vs_anchor_pct": anchor_delta_pct,
                "win_vs_stage_a": int(val < stage_a_val),
                "win_vs_anchor": int(val < anchor_val),
                "win_vs_sm2": int(val < sm2_val),
                "diagnostic_ok": int(diagnostic_ok(row)) if arm in CANDIDATES else 1,
            }
            cell_rows.append(record)
            if arm in CANDIDATES:
                candidate_cells[arm].append(record)

    aggregates = []
    for arm in CANDIDATES:
        cells = candidate_cells[arm]
        profiles = sorted({str(row["base_profile"]) for row in cells})
        missing_resources = sorted(set(profiles) - set(resource_profiles))
        train_ratios = []
        inference_ratios = []
        memory_ratios = []
        parameter_ratios = []
        for profile in profiles:
            stage = resources.get((profile, STAGE_A))
            candidate = resources.get((profile, arm))
            if stage is None or candidate is None:
                continue
            train_ratios.append(
                float(candidate["train_forward_backward_ms_per_batch"])
                / float(stage["train_forward_backward_ms_per_batch"])
            )
            inference_ratios.append(
                float(candidate["inference_ms_per_batch"])
                / float(stage["inference_ms_per_batch"])
            )
            memory_ratios.append(
                float(candidate["fixed_work_peak_cuda_mb"])
                / float(stage["fixed_work_peak_cuda_mb"])
            )
            parameter_ratios.append(
                float(candidate["n_param"]) / float(stage["n_param"])
            )
        resource_complete = not missing_resources and len(train_ratios) == len(profiles)
        max_train = max(train_ratios) if train_ratios else None
        max_inference = max(inference_ratios) if inference_ratios else None
        max_memory = max(memory_ratios) if memory_ratios else None
        max_parameter = max(parameter_ratios) if parameter_ratios else None
        resource_ok = (
            resource_complete
            and max_train <= train_limit
            and max_inference <= inference_limit
            and max_memory <= memory_limit
            and max_parameter <= parameter_limit
        )
        deltas = [float(row["delta_val_vs_stage_a_pct"]) for row in cells]
        wins_stage = sum(int(row["win_vs_stage_a"]) for row in cells)
        wins_anchor = sum(int(row["win_vs_anchor"]) for row in cells)
        aggregate = {
            "arm": arm,
            "cells": len(cells),
            "wins_vs_stage_a": wins_stage,
            "wins_vs_anchor": wins_anchor,
            "wins_vs_sm2": sum(int(row["win_vs_sm2"]) for row in cells),
            "median_delta_vs_stage_a_pct": statistics.median(deltas),
            "mean_delta_vs_stage_a_pct": statistics.fmean(deltas),
            "worst_delta_vs_stage_a_pct": max(deltas),
            "diagnostics_ok": int(all(row["diagnostic_ok"] for row in cells)),
            "resource_complete": int(resource_complete),
            "max_train_ratio_vs_stage_a": max_train,
            "max_inference_ratio_vs_stage_a": max_inference,
            "max_memory_ratio_vs_stage_a": max_memory,
            "max_parameter_ratio_vs_stage_a": max_parameter,
        }
        aggregate["eligible"] = int(
            wins_stage >= min_wins
            and wins_anchor >= min_wins
            and aggregate["median_delta_vs_stage_a_pct"] < 0.0
            and aggregate["worst_delta_vs_stage_a_pct"] <= max_regression_pct
            and aggregate["diagnostics_ok"]
            and resource_ok
        )
        aggregates.append(aggregate)

    by_arm = {row["arm"]: row for row in aggregates}
    sm2 = by_arm["fk_sm2_mode"]
    phase = by_arm["fk_sm4_ph4_h"]
    complex_head_to_head_ok = (
        phase["wins_vs_sm2"] >= 3
        and phase["median_delta_vs_stage_a_pct"]
        < sm2["median_delta_vs_stage_a_pct"]
    )
    selected = None
    reason = "neither Stage-B candidate passed the frozen Wave-1 gate"
    if phase["eligible"] and not sm2["eligible"]:
        selected = phase["arm"]
        reason = "complex passed while real SM did not"
    elif phase["eligible"] and sm2["eligible"] and complex_head_to_head_ok:
        selected = phase["arm"]
        reason = "both passed; complex won the frozen validation aggregate tie-break"
    elif sm2["eligible"]:
        selected = sm2["arm"]
        reason = "real SM passed; complex failed eligibility or aggregate tie-break"
    decision = {
        "selected_arm": selected,
        "advance_to_wave2": int(selected is not None),
        "reason": reason,
        "complex_head_to_head_ok": int(complex_head_to_head_ok),
        "cells": len(grouped),
        "test_metrics_used": False,
    }
    return cell_rows, aggregates, decision


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return "" if value is None else f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fixed_work_json", action="append", default=[])
    parser.add_argument("--min_wins", type=int, default=4)
    parser.add_argument("--max_regression_pct", type=float, default=0.05)
    parser.add_argument("--train_limit", type=float, default=1.15)
    parser.add_argument("--inference_limit", type=float, default=1.10)
    parser.add_argument("--memory_limit", type=float, default=1.02)
    parser.add_argument("--parameter_limit", type=float, default=1.15)
    args = parser.parse_args()

    resources, profiles = load_resources(args.fixed_work_json)
    cell_rows, aggregates, decision = summarize(
        load_rows(args.root), read_manifest(args.manifest), resources, profiles,
        min_wins=args.min_wins,
        max_regression_pct=args.max_regression_pct,
        train_limit=args.train_limit,
        inference_limit=args.inference_limit,
        memory_limit=args.memory_limit,
        parameter_limit=args.parameter_limit,
    )
    output = Path(args.output_dir)
    write_csv(output / "cell_validation.csv", cell_rows, [
        "base_profile", "arm", "dataset", "seq_len", "pred_len", "seed",
        "cut_freq", "status", "n_param", "val_mse", "anchor_val_mse",
        "stage_a_val_mse", "delta_val_vs_stage_a",
        "delta_val_vs_stage_a_pct", "delta_val_vs_anchor",
        "delta_val_vs_anchor_pct", "win_vs_stage_a", "win_vs_anchor",
        "win_vs_sm2", "diagnostic_ok", "forecast_kernel_gate_mean",
        "forecast_kernel_sm_gate_abs_max", "forecast_kernel_sm_factor_min",
        "forecast_kernel_sm_factor_max", "forecast_kernel_phase_abs_max",
        "job_id", "summary_file",
    ])
    write_csv(output / "aggregate_validation.csv", aggregates, [
        "arm", "cells", "wins_vs_stage_a", "wins_vs_anchor", "wins_vs_sm2",
        "median_delta_vs_stage_a_pct", "mean_delta_vs_stage_a_pct",
        "worst_delta_vs_stage_a_pct", "diagnostics_ok", "resource_complete",
        "max_train_ratio_vs_stage_a", "max_inference_ratio_vs_stage_a",
        "max_memory_ratio_vs_stage_a", "max_parameter_ratio_vs_stage_a",
        "eligible",
    ])
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase 11 Stage C Wave-1 validation-only decision", "",
        "No test metric was opened or used.", "",
        "| arm | wins vs Stage A | wins vs anchor | wins vs sm2 | median delta | worst delta | train max | infer max | memory max | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['arm']} | {row['wins_vs_stage_a']}/{row['cells']} | "
            f"{row['wins_vs_anchor']}/{row['cells']} | {row['wins_vs_sm2']}/{row['cells']} | "
            f"{row['median_delta_vs_stage_a_pct']:+.4f}% | "
            f"{row['worst_delta_vs_stage_a_pct']:+.4f}% | "
            f"{fmt(row['max_train_ratio_vs_stage_a'], 3)}x | "
            f"{fmt(row['max_inference_ratio_vs_stage_a'], 3)}x | "
            f"{fmt(row['max_memory_ratio_vs_stage_a'], 3)}x | {row['eligible']} |"
        )
    lines.extend([
        "", f"Selected: {decision['selected_arm'] or 'none'}", "",
        f"Advance to Wave 2: {decision['advance_to_wave2']}", "",
        f"Reason: {decision['reason']}", "",
    ])
    (output / "summary_stagec_wave1.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()

