#!/usr/bin/env python3
"""Leakage-safe selector for Phase 11 Stage D horizon-safe SM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ARMS = ("fk_r8_cs", "fk_sm2_mode", "fk_sm2_tail2")
STAGE_A = "fk_r8_cs"
UNSHRUNK = "fk_sm2_mode"
TAIL2 = "fk_sm2_tail2"
ROLES = {"bridge", "identity", "active"}


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def expected_scale(seq_len, pred_len):
    return max(0.0, 1.0 - 2.0 * float(seq_len) / float(pred_len))


def read_manifest(path):
    specs = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 8:
                raise ValueError(f"Stage-D manifest row needs 8 fields: {line}")
            role, profile, arm, dataset, seq_len, pred_len, seed, cut_freq = fields
            if role not in ROLES:
                raise ValueError(f"unsupported Stage-D role {role!r}")
            specs.append(
                {
                    "role": role,
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
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("synthetic_only_no_dataset_read") is not True:
            raise ValueError("fixed-work audit must be synthetic and dataset-free")
        key = (
            str(payload.get("dataset", "")),
            int(payload.get("seq_len", -1)),
            int(payload.get("pred_len", -1)),
        )
        if not key[0] or key[1] <= 0 or key[2] <= 0:
            raise ValueError(f"fixed-work audit lacks dataset dimensions: {path}")
        if key in resources:
            raise ValueError(f"duplicate fixed-work audit for {key}")
        resources[key] = {str(row["arm"]): row for row in payload.get("rows", [])}
    return resources


def spec_key(item):
    return (
        str(item.get("base_profile", "")),
        str(item.get("arm", "")),
        str(item.get("dataset", "")),
        int(item.get("seq_len", -1)),
        int(item.get("pred_len", -1)),
        int(item.get("seed", -1)),
        int(item.get("cut_freq", -1)),
    )


def cell_key(spec):
    return (
        str(spec["role"]),
        str(spec["base_profile"]),
        str(spec["dataset"]),
        int(spec["seq_len"]),
        int(spec["pred_len"]),
        int(spec["seed"]),
        int(spec["cut_freq"]),
    )


def ratio(candidate, reference, key):
    return float(candidate[key]) / float(reference[key])


def summarize(
    rows,
    expected_specs,
    resources,
    identity_val_tolerance=1e-7,
    identity_factor_tolerance=1e-7,
    identity_gate_tolerance=1e-8,
    min_active_wins=3,
    max_active_regression_pct=0.05,
    max_bridge_regression_pct=0.02,
    active_train_limit=1.10,
    active_inference_limit=1.05,
    active_memory_limit=1.02,
    identity_resource_limit=1.02,
    parameter_limit=1.01,
):
    expected = {spec_key(spec): spec for spec in expected_specs}
    if len(expected) != len(expected_specs):
        raise ValueError("duplicate Stage-D manifest rows")
    by_spec = {}
    for row in rows:
        key = spec_key(row)
        if key in by_spec:
            raise ValueError(f"duplicate Stage-D result row: {key}")
        by_spec[key] = row
    missing = sorted(set(expected) - set(by_spec))
    extra = sorted(set(by_spec) - set(expected))
    if missing or extra:
        raise ValueError(f"Stage-D manifest mismatch: missing={missing}, extra={extra}")

    ordered = []
    for key, spec in expected.items():
        row = dict(by_spec[key])
        row["role"] = spec["role"]
        ordered.append(row)
    failed = [
        row for row in ordered
        if row.get("status") != "ok" or not finite(row.get("val_mse"))
    ]
    if failed:
        raise ValueError(
            f"Stage-D rows failed or lack validation: {[spec_key(row) for row in failed]}"
        )
    leaked = [
        row for row in ordered
        if row.get("test_deferred") is not True
        or row.get("mse") is not None
        or row.get("mae") is not None
    ]
    if leaked:
        raise ValueError(
            f"Stage-D selector refuses test-open rows: {[spec_key(row) for row in leaked]}"
        )

    grouped = {}
    for spec in expected_specs:
        key = cell_key(spec)
        grouped.setdefault(key, {})[spec["arm"]] = by_spec[spec_key(spec)]
    for key, group in grouped.items():
        if set(group) != set(ARMS):
            raise ValueError(f"Stage-D cell {key} does not contain exactly {ARMS}")

    cell_rows = []
    tail_records = []
    identity_checks = []
    active_diagnostic_checks = []
    parameter_ratios = []
    for key in sorted(grouped):
        role, profile, dataset, seq_len, pred_len, seed, cut_freq = key
        group = grouped[key]
        stage_val = float(group[STAGE_A]["val_mse"])
        unshrunk_val = float(group[UNSHRUNK]["val_mse"])
        scale = expected_scale(seq_len, pred_len)
        for arm in ARMS:
            row = group[arm]
            val = float(row["val_mse"])
            record = {
                **row,
                "role": role,
                "stage_a_val_mse": stage_val,
                "unshrunk_val_mse": unshrunk_val,
                "delta_val_vs_stage_a": val - stage_val,
                "delta_val_vs_stage_a_pct": 100.0 * (val / stage_val - 1.0),
                "delta_val_vs_unshrunk": val - unshrunk_val,
                "delta_val_vs_unshrunk_pct": 100.0 * (val / unshrunk_val - 1.0),
                "win_vs_stage_a": int(val < stage_val),
                "win_vs_unshrunk": int(val < unshrunk_val),
                "expected_extension_scale": scale,
            }
            cell_rows.append(record)
            if arm == TAIL2:
                tail_records.append(record)

        tail = group[TAIL2]
        if tail.get("forecast_extension_shrink") != "tail2_linear":
            raise ValueError(f"tail2 row lacks frozen schedule at {key}")
        for arm in (STAGE_A, UNSHRUNK):
            if group[arm].get("forecast_extension_shrink", "none") != "none":
                raise ValueError(f"control arm changed shrink schedule at {key}: {arm}")
        reported_scale = tail.get("forecast_kernel_extension_scale")
        if not finite(reported_scale) or abs(float(reported_scale) - scale) > 1e-9:
            raise ValueError(
                f"tail2 scale mismatch at {key}: expected={scale}, got={reported_scale}"
            )
        parameter_ratios.append(float(tail["n_param"]) / float(group[STAGE_A]["n_param"]))

        if role == "identity":
            identity_ok = (
                scale == 0.0
                and abs(float(tail["val_mse"]) - stage_val) <= identity_val_tolerance
                and finite(tail.get("forecast_kernel_extension_identity"))
                and float(tail["forecast_kernel_extension_identity"]) == 1.0
                and finite(tail.get("forecast_kernel_sm_factor_min"))
                and finite(tail.get("forecast_kernel_sm_factor_max"))
                and abs(float(tail["forecast_kernel_sm_factor_min"]) - 1.0)
                <= identity_factor_tolerance
                and abs(float(tail["forecast_kernel_sm_factor_max"]) - 1.0)
                <= identity_factor_tolerance
                and finite(tail.get("forecast_kernel_sm_gate_abs_max"))
                and abs(float(tail["forecast_kernel_sm_gate_abs_max"]))
                <= identity_gate_tolerance
            )
            identity_checks.append(identity_ok)
        elif role in {"active", "bridge"}:
            factor_min = tail.get("forecast_kernel_sm_factor_min")
            factor_max = tail.get("forecast_kernel_sm_factor_max")
            effective_gate = tail.get("forecast_kernel_sm_effective_gate_abs_max")
            active_ok = (
                0.0 < scale < 1.0
                and finite(tail.get("forecast_kernel_extension_identity"))
                and float(tail["forecast_kernel_extension_identity"]) == 0.0
                and finite(effective_gate)
                and float(effective_gate) >= 1e-4
                and finite(factor_min)
                and finite(factor_max)
                and 0.0 < float(factor_min) <= float(factor_max) < 20.0
                and max(
                    abs(float(factor_min) - 1.0),
                    abs(float(factor_max) - 1.0),
                ) >= 1e-4
            )
            active_diagnostic_checks.append(active_ok)

    identity = [row for row in tail_records if row["role"] == "identity"]
    active = [row for row in tail_records if row["role"] == "active"]
    bridge = [row for row in tail_records if row["role"] == "bridge"]
    if len(identity) != 4 or len(active) != 4 or len(bridge) != 1:
        raise ValueError(
            f"Stage-D role sizes must be identity=4 active=4 bridge=1, got "
            f"{len(identity)}/{len(active)}/{len(bridge)}"
        )

    active_deltas = [float(row["delta_val_vs_stage_a_pct"]) for row in active]
    wins_stage = sum(int(row["win_vs_stage_a"]) for row in active)
    wins_unshrunk = sum(int(row["win_vs_unshrunk"]) for row in active)
    identity_max_abs_delta = max(
        abs(float(row["delta_val_vs_stage_a"])) for row in identity
    )
    bridge_delta = float(bridge[0]["delta_val_vs_stage_a_pct"])

    required_resources = {
        ("traffic", 96, 192): "identity",
        ("ETTm1", 96, 720): "active",
        ("traffic", 96, 720): "active",
    }
    resource_rows = []
    resource_complete = set(resources) == set(required_resources)
    for key, role in required_resources.items():
        group = resources.get(key, {})
        if not set((STAGE_A, UNSHRUNK, TAIL2)).issubset(group):
            resource_complete = False
            continue
        stage = group[STAGE_A]
        tail = group[TAIL2]
        resource_rows.append(
            {
                "role": role,
                "dataset": key[0],
                "seq_len": key[1],
                "pred_len": key[2],
                "train_ratio_vs_stage_a": ratio(
                    tail, stage, "train_forward_backward_ms_per_batch"
                ),
                "inference_ratio_vs_stage_a": ratio(
                    tail, stage, "inference_ms_per_batch"
                ),
                "memory_ratio_vs_stage_a": ratio(
                    tail, stage, "fixed_work_peak_cuda_mb"
                ),
                "parameter_ratio_vs_stage_a": ratio(tail, stage, "n_param"),
            }
        )
    active_resources = [row for row in resource_rows if row["role"] == "active"]
    identity_resources = [row for row in resource_rows if row["role"] == "identity"]
    resource_complete = (
        resource_complete
        and len(active_resources) == 2
        and len(identity_resources) == 1
    )

    def maximum(rows_, key_):
        return max((float(row[key_]) for row in rows_), default=None)

    active_train_max = maximum(active_resources, "train_ratio_vs_stage_a")
    active_inference_max = maximum(active_resources, "inference_ratio_vs_stage_a")
    active_memory_max = maximum(active_resources, "memory_ratio_vs_stage_a")
    active_parameter_max = maximum(active_resources, "parameter_ratio_vs_stage_a")
    identity_train_max = maximum(identity_resources, "train_ratio_vs_stage_a")
    identity_inference_max = maximum(identity_resources, "inference_ratio_vs_stage_a")
    identity_memory_max = maximum(identity_resources, "memory_ratio_vs_stage_a")
    parameter_max = max(parameter_ratios + [
        value for value in (active_parameter_max,) if value is not None
    ])

    active_resource_ok = (
        resource_complete
        and active_train_max <= active_train_limit
        and active_inference_max <= active_inference_limit
        and active_memory_max <= active_memory_limit
        and parameter_max <= parameter_limit
    )
    identity_resource_ok = (
        resource_complete
        and identity_train_max <= identity_resource_limit
        and identity_inference_max <= identity_resource_limit
        and identity_memory_max <= identity_resource_limit
    )
    identity_ok = all(identity_checks)
    diagnostics_ok = all(active_diagnostic_checks)
    eligible = (
        identity_ok
        and diagnostics_ok
        and wins_stage >= min_active_wins
        and wins_unshrunk >= min_active_wins
        and statistics.median(active_deltas) < 0.0
        and statistics.fmean(active_deltas) < 0.0
        and max(active_deltas) <= max_active_regression_pct
        and bridge_delta <= max_bridge_regression_pct
        and active_resource_ok
        and identity_resource_ok
    )
    aggregate = {
        "active_cells": len(active),
        "identity_cells": len(identity),
        "wins_vs_stage_a": wins_stage,
        "wins_vs_unshrunk": wins_unshrunk,
        "median_delta_vs_stage_a_pct": statistics.median(active_deltas),
        "mean_delta_vs_stage_a_pct": statistics.fmean(active_deltas),
        "worst_delta_vs_stage_a_pct": max(active_deltas),
        "bridge_delta_vs_stage_a_pct": bridge_delta,
        "identity_max_abs_val_delta": identity_max_abs_delta,
        "identity_ok": int(identity_ok),
        "active_diagnostics_ok": int(diagnostics_ok),
        "resource_complete": int(resource_complete),
        "active_train_max": active_train_max,
        "active_inference_max": active_inference_max,
        "active_memory_max": active_memory_max,
        "identity_train_max": identity_train_max,
        "identity_inference_max": identity_inference_max,
        "identity_memory_max": identity_memory_max,
        "parameter_max": parameter_max,
        "eligible": int(eligible),
    }
    decision = {
        "selected_arm": TAIL2 if eligible else None,
        "advance_to_confirmation": int(eligible),
        "reason": (
            "tail2 passed every frozen Stage-D gate"
            if eligible
            else "tail2 failed at least one frozen Stage-D gate"
        ),
        "test_metrics_used": False,
        "cells": len(grouped),
    }
    return cell_rows, resource_rows, aggregate, decision


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
    args = parser.parse_args()

    cell_rows, resource_rows, aggregate, decision = summarize(
        load_rows(args.root),
        read_manifest(args.manifest),
        load_resources(args.fixed_work_json),
    )
    output = Path(args.output_dir)
    write_csv(output / "cell_validation.csv", cell_rows, [
        "role", "base_profile", "arm", "dataset", "seq_len", "pred_len",
        "seed", "cut_freq", "status", "n_param", "val_mse",
        "stage_a_val_mse", "unshrunk_val_mse", "delta_val_vs_stage_a",
        "delta_val_vs_stage_a_pct", "delta_val_vs_unshrunk",
        "delta_val_vs_unshrunk_pct", "win_vs_stage_a",
        "win_vs_unshrunk", "expected_extension_scale",
        "forecast_kernel_extension_scale", "forecast_kernel_extension_identity",
        "forecast_kernel_sm_gate_abs_max",
        "forecast_kernel_sm_effective_gate_abs_max",
        "forecast_kernel_sm_factor_min", "forecast_kernel_sm_factor_max",
        "job_id", "summary_file",
    ])
    write_csv(output / "resource_ratios.csv", resource_rows, [
        "role", "dataset", "seq_len", "pred_len",
        "train_ratio_vs_stage_a", "inference_ratio_vs_stage_a",
        "memory_ratio_vs_stage_a", "parameter_ratio_vs_stage_a",
    ])
    write_csv(output / "aggregate_validation.csv", [aggregate], list(aggregate))
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase 11 Stage D validation-only decision",
        "",
        "No test metric was opened or used.",
        "",
        "| active wins vs A | wins vs unshrunk | median delta | mean delta | worst delta | identity max abs | bridge delta | train max | infer max | memory max | eligible |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {aggregate['wins_vs_stage_a']}/{aggregate['active_cells']} | "
            f"{aggregate['wins_vs_unshrunk']}/{aggregate['active_cells']} | "
            f"{aggregate['median_delta_vs_stage_a_pct']:+.4f}% | "
            f"{aggregate['mean_delta_vs_stage_a_pct']:+.4f}% | "
            f"{aggregate['worst_delta_vs_stage_a_pct']:+.4f}% | "
            f"{aggregate['identity_max_abs_val_delta']:.2e} | "
            f"{aggregate['bridge_delta_vs_stage_a_pct']:+.4f}% | "
            f"{fmt(aggregate['active_train_max'], 3)}x | "
            f"{fmt(aggregate['active_inference_max'], 3)}x | "
            f"{fmt(aggregate['active_memory_max'], 3)}x | "
            f"{aggregate['eligible']} |"
        ),
        "",
        f"Selected: {decision['selected_arm'] or 'none'}",
        "",
        f"Advance to confirmation: {decision['advance_to_confirmation']}",
        "",
        f"Reason: {decision['reason']}",
        "",
    ]
    (output / "summary_staged.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
