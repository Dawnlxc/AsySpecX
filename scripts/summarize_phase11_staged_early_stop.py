#!/usr/bin/env python3
"""Audit a mathematically decisive partial Stage-D screen after early stopping."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.summarize_phase11_staged import (
    ARMS,
    STAGE_A,
    TAIL2,
    cell_key,
    expected_scale,
    load_resources,
    load_rows,
    read_manifest,
    spec_key,
)


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_partial(rows, expected_specs, resources):
    expected = {spec_key(spec): spec for spec in expected_specs}
    if len(expected) != len(expected_specs):
        raise ValueError("duplicate Stage-D manifest rows")
    observed = {}
    for row in rows:
        item_key = spec_key(row)
        if item_key not in expected:
            raise ValueError(f"partial screen contains an unexpected row: {item_key}")
        if item_key in observed:
            raise ValueError(f"partial screen contains a duplicate row: {item_key}")
        if (
            row.get("status") != "ok"
            or row.get("test_deferred") is not True
            or row.get("mse") is not None
            or row.get("mae") is not None
            or not finite(row.get("val_mse"))
        ):
            raise ValueError(f"partial screen contains failed or test-open row: {item_key}")
        observed[item_key] = row

    grouped = {}
    for spec in expected_specs:
        item_key = spec_key(spec)
        if item_key in observed:
            grouped.setdefault(cell_key(spec), {})[spec["arm"]] = observed[item_key]
    complete = {key: group for key, group in grouped.items() if set(group) == set(ARMS)}
    partial_groups = {key: sorted(group) for key, group in grouped.items() if set(group) != set(ARMS)}

    cells = []
    identity_ok = True
    active_deltas = []
    active_wins = 0
    for item_key in sorted(complete):
        role, profile, dataset, seq_len, pred_len, seed, cut_freq = item_key
        group = complete[item_key]
        stage = group[STAGE_A]
        tail = group[TAIL2]
        scale = expected_scale(seq_len, pred_len)
        delta = float(tail["val_mse"]) - float(stage["val_mse"])
        delta_pct = 100.0 * (float(tail["val_mse"]) / float(stage["val_mse"]) - 1.0)
        cells.append({
            "role": role,
            "base_profile": profile,
            "dataset": dataset,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "seed": seed,
            "cut_freq": cut_freq,
            "stage_a_val_mse": float(stage["val_mse"]),
            "tail2_val_mse": float(tail["val_mse"]),
            "delta_val": delta,
            "delta_val_pct": delta_pct,
            "tail2_win": int(delta < 0.0),
            "expected_extension_scale": scale,
            "reported_extension_scale": tail.get("forecast_kernel_extension_scale"),
        })
        if not finite(tail.get("forecast_kernel_extension_scale")) or abs(
            float(tail["forecast_kernel_extension_scale"]) - scale
        ) > 1e-9:
            raise ValueError(f"extension scale mismatch in completed cell: {item_key}")
        if role == "identity":
            identity_ok = identity_ok and (
                scale == 0.0
                and abs(delta) <= 1e-7
                and finite(tail.get("forecast_kernel_sm_factor_min"))
                and finite(tail.get("forecast_kernel_sm_factor_max"))
                and abs(float(tail["forecast_kernel_sm_factor_min"]) - 1.0) <= 1e-7
                and abs(float(tail["forecast_kernel_sm_factor_max"]) - 1.0) <= 1e-7
                and finite(tail.get("forecast_kernel_sm_gate_abs_max"))
                and abs(float(tail["forecast_kernel_sm_gate_abs_max"])) <= 1e-8
            )
        if role == "active":
            active_deltas.append(delta_pct)
            active_wins += int(delta < 0.0)

    all_cells = {cell_key(spec) for spec in expected_specs}
    total_active = sum(key[0] == "active" for key in all_cells)
    completed_active = sum(key[0] == "active" for key in complete)
    remaining_active = total_active - completed_active
    maximum_possible_wins = active_wins + remaining_active
    win_gate_impossible = maximum_possible_wins < 3
    regression_failures = [value for value in active_deltas if value > 0.05]

    resource_rows = []
    resource_failure = False
    for key, group in sorted(resources.items()):
        if STAGE_A not in group or TAIL2 not in group:
            raise ValueError(f"resource audit lacks required arms: {key}")
        stage = group[STAGE_A]
        tail = group[TAIL2]
        ratios = {
            "dataset": key[0],
            "seq_len": key[1],
            "pred_len": key[2],
            "train_ratio_vs_stage_a": float(tail["train_forward_backward_ms_per_batch"])
            / float(stage["train_forward_backward_ms_per_batch"]),
            "inference_ratio_vs_stage_a": float(tail["inference_ms_per_batch"])
            / float(stage["inference_ms_per_batch"]),
            "memory_ratio_vs_stage_a": float(tail["fixed_work_peak_cuda_mb"])
            / float(stage["fixed_work_peak_cuda_mb"]),
            "parameter_ratio_vs_stage_a": float(tail["n_param"]) / float(stage["n_param"]),
        }
        ratios["passes_active_limits"] = int(
            ratios["train_ratio_vs_stage_a"] <= 1.10
            and ratios["inference_ratio_vs_stage_a"] <= 1.05
            and ratios["memory_ratio_vs_stage_a"] <= 1.02
            and ratios["parameter_ratio_vs_stage_a"] <= 1.01
        )
        resource_failure = resource_failure or not bool(ratios["passes_active_limits"])
        resource_rows.append(ratios)

    decisive = bool(win_gate_impossible or regression_failures or resource_failure)
    aggregate = {
        "manifest_rows": len(expected_specs),
        "completed_rows": len(observed),
        "complete_cells": len(complete),
        "partial_cells": len(partial_groups),
        "completed_active_cells": completed_active,
        "total_active_cells": total_active,
        "active_wins_so_far": active_wins,
        "maximum_possible_active_wins": maximum_possible_wins,
        "minimum_required_active_wins": 3,
        "win_gate_impossible": int(win_gate_impossible),
        "active_regressions_over_0_05_pct": len(regression_failures),
        "worst_completed_active_regression_pct": max(active_deltas) if active_deltas else None,
        "completed_identity_ok": int(identity_ok),
        "observed_resource_failure": int(resource_failure),
        "decisive_early_stop": int(decisive),
    }
    decision = {
        "selected_arm": None,
        "advance_to_confirmation": 0,
        "open_test": 0,
        "test_metrics_used": False,
        "decisive_early_stop": int(decisive),
        "reason": (
            "tail2 cannot satisfy the frozen Stage-D accuracy/resource gates"
            if decisive
            else "partial screen is not yet mathematically decisive"
        ),
    }
    return cells, resource_rows, aggregate, decision


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fixed_work_json", action="append", default=[])
    args = parser.parse_args()
    cells, resources, aggregate, decision = audit_partial(
        load_rows(args.root),
        read_manifest(args.manifest),
        load_resources(args.fixed_work_json),
    )
    output = Path(args.output_dir)
    write_csv(output / "completed_cell_validation.csv", cells, [
        "role", "base_profile", "dataset", "seq_len", "pred_len", "seed",
        "cut_freq", "stage_a_val_mse", "tail2_val_mse", "delta_val",
        "delta_val_pct", "tail2_win", "expected_extension_scale",
        "reported_extension_scale",
    ])
    write_csv(output / "observed_resource_ratios.csv", resources, [
        "dataset", "seq_len", "pred_len", "train_ratio_vs_stage_a",
        "inference_ratio_vs_stage_a", "memory_ratio_vs_stage_a",
        "parameter_ratio_vs_stage_a", "passes_active_limits",
    ])
    output.mkdir(parents=True, exist_ok=True)
    (output / "early_stop_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase 11 Stage D decisive early stop",
        "",
        "No test metric was opened or used.",
        "",
        "The run stopped once the frozen gate became mathematically impossible; pending jobs were cancelled.",
        "",
        "| completed active | wins | maximum possible wins | regressions > 0.05% | worst regression | resource failure | advance |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {aggregate['completed_active_cells']}/{aggregate['total_active_cells']} | "
            f"{aggregate['active_wins_so_far']} | "
            f"{aggregate['maximum_possible_active_wins']}/{aggregate['total_active_cells']} | "
            f"{aggregate['active_regressions_over_0_05_pct']} | "
            f"{aggregate['worst_completed_active_regression_pct']:+.4f}% | "
            f"{aggregate['observed_resource_failure']} | 0 |"
        ),
    ]
    (output / "PHASE11_STAGED_EARLY_STOP_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "decision": decision}, sort_keys=True))
    if not decision["decisive_early_stop"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
