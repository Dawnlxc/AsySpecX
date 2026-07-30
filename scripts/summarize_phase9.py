#!/usr/bin/env python3
"""Aggregate SafeRoute headroom and routed cell results."""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_csvs(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source_path"] = path
                rows.append(row)
    return rows


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarise_router(rows):
    if not rows:
        return None
    anchor = mean(f(row, "anchor_mse") for row in rows)
    routed = mean(f(row, "routed_mse") for row in rows)
    gains = np.asarray(
        [f(row, "anchor_mse") - f(row, "routed_mse") for row in rows], dtype=np.float64
    )
    wins = int(np.sum(gains > 1e-12))
    losses = int(np.sum(gains < -1e-12))
    non_ties = wins + losses
    sign_p = 1.0
    if non_ties:
        tail = sum(math.comb(non_ties, index) for index in range(min(wins, losses) + 1))
        sign_p = min(1.0, 2.0 * tail / (2.0 ** non_ties))
    return {
        "cells": len(rows),
        "anchor_mse": anchor,
        "anchor_mae": mean(f(row, "anchor_mae") for row in rows),
        "routed_mse": routed,
        "routed_mae": mean(f(row, "routed_mae") for row in rows),
        "gain_vs_anchor": anchor - routed,
        "paired_gain_median": float(np.median(gains)),
        "paired_gain_std": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
        "paired_gain_standard_error": float(np.std(gains, ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else 0.0,
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": int(len(gains) - non_ties),
        "paired_sign_test_p": float(sign_p),
        "router_coverage": mean(f(row, "router_coverage") for row in rows),
        "fallback_fraction": mean(f(row, "fallback_fraction") for row in rows),
        "mean_alpha": mean(f(row, "mean_alpha") for row in rows),
        "false_activation_rate": mean(f(row, "false_activation_rate") for row in rows),
        "catastrophic_activation_rate": mean(f(row, "catastrophic_activation_rate") for row in rows),
    }


def grouped_summaries(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {value: summarise_router(group) for value, group in sorted(groups.items())}


def activation_counts(rows):
    counts = Counter()
    seen = set()
    for row in rows:
        path = str(Path(row["_source_path"]).with_name("routing_diagnostics.csv"))
        if path in seen or not Path(path).is_file():
            continue
        seen.add(path)
        with open(path, newline="", encoding="utf-8") as handle:
            for diagnostic in csv.DictReader(handle):
                counts[diagnostic["expert"]] += int(float(diagnostic.get("activations", 0)))
    return dict(sorted(counts.items()))


def add_router_breakdowns(payload, prefix, rows):
    if not rows:
        return
    payload[f"{prefix}_per_dataset"] = grouped_summaries(rows, "dataset")
    payload[f"{prefix}_per_seq_len"] = grouped_summaries(rows, "seq_len")
    payload[f"{prefix}_per_pred_len"] = grouped_summaries(rows, "pred_len")
    payload[f"{prefix}_activation_counts"] = activation_counts(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headroom_dir", default="phase9_results/headroom")
    parser.add_argument("--quick_glob", default="phase9_results/quick/*/routed_results.csv")
    parser.add_argument("--oof_glob", default="phase9_results/oof/*/routed_results.csv")
    parser.add_argument("--output_dir", default="phase9_results/summary")
    args = parser.parse_args()

    headroom_path = Path(args.headroom_dir) / "headroom_summary.json"
    headroom = json.loads(headroom_path.read_text()) if headroom_path.is_file() else None
    quick_rows, oof_rows = read_csvs(args.quick_glob), read_csvs(args.oof_glob)
    quick, oof = summarise_router(quick_rows), summarise_router(oof_rows)
    payload = {"headroom": headroom, "quick_router": quick, "oof_router": oof}
    add_router_breakdowns(payload, "quick", quick_rows)
    add_router_breakdowns(payload, "oof", oof_rows)

    if headroom:
        payload["headroom_go"] = float(headroom["sample_block_gain_vs_best_fixed"]) >= 0.004
    if quick:
        payload["quick_go_oof"] = quick["gain_vs_anchor"] >= 0.002
        payload["quick_any_dataset_regression_gt_0_003"] = any(
            summary["routed_mse"] - summary["anchor_mse"] > 0.003
            for summary in payload["quick_per_dataset"].values()
        )
    if oof:
        payload["oof_go_distillation"] = (
            oof["gain_vs_anchor"] >= 0.003 and oof["catastrophic_activation_rate"] <= 0.01
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase9_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Phase 9 SafeRoute Summary", ""]
    if headroom:
        lines += [
            "## Headroom Audit", "",
            f"**{headroom['warning']}**", "",
            f"- best_fixed_expert: {headroom['best_fixed_expert']}",
            f"- best_fixed_mse: {headroom['best_fixed_mse']:.6f}",
            f"- cell_oracle_mse: {headroom['cell_oracle_mse']:.6f}",
            f"- sample_oracle_mse: {headroom['sample_oracle_mse']:.6f}",
            f"- horizon_block_oracle_mse: {headroom['horizon_block_oracle_mse']:.6f}",
            f"- sample_block_oracle_mse: {headroom['sample_block_oracle_mse']:.6f}",
            f"- sample_block_gain_vs_best_fixed: {headroom['sample_block_gain_vs_best_fixed']:.6f}",
            f"- decision: {headroom['opportunity']}", "",
        ]
    for title, summary in (("Quick Validation-Adapted Router", quick), ("Rolling-OOF Router", oof)):
        if not summary:
            continue
        lines += [f"## {title}", ""] + [f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: {value}" for key, value in summary.items()] + [""]
    if quick_rows:
        lines += ["## Quick Router Per Dataset", "", "| dataset | anchor_mse | routed_mse | gain |", "| --- | ---: | ---: | ---: |"]
        for dataset, summary in payload["quick_per_dataset"].items():
            lines.append(f"| {dataset} | {summary['anchor_mse']:.6f} | {summary['routed_mse']:.6f} | {summary['gain_vs_anchor']:.6f} |")
        lines.append("")
    if oof_rows:
        lines += ["## Rolling-OOF Router Per Dataset", "", "| dataset | anchor_mse | routed_mse | gain |", "| --- | ---: | ---: | ---: |"]
        for dataset, summary in payload["oof_per_dataset"].items():
            lines.append(f"| {dataset} | {summary['anchor_mse']:.6f} | {summary['routed_mse']:.6f} | {summary['gain_vs_anchor']:.6f} |")
        lines.append("")
    for prefix, title in (("quick", "Quick"), ("oof", "Rolling-OOF")):
        if not payload.get(f"{prefix}_per_seq_len"):
            continue
        for dimension, label in (("seq_len", "Sequence Length"), ("pred_len", "Prediction Length")):
            lines += [f"## {title} By {label}", "", f"| {dimension} | cells | anchor_mse | routed_mse | gain |", "| ---: | ---: | ---: | ---: | ---: |"]
            for value, summary in payload[f"{prefix}_per_{dimension}"].items():
                lines.append(f"| {value} | {summary['cells']} | {summary['anchor_mse']:.6f} | {summary['routed_mse']:.6f} | {summary['gain_vs_anchor']:.6f} |")
            lines.append("")
        counts = payload.get(f"{prefix}_activation_counts", {})
        if counts:
            lines += [f"## {title} Expert Activation Counts", "", "| expert | activations |", "| --- | ---: |"]
            lines.extend(f"| {expert} | {count} |" for expert, count in counts.items())
            lines.append("")
    lines += [
        "## Locked Stop/Go Rules", "",
        "1. sample-block oracle gain < 0.004: stop internal routing.",
        "2. quick router gain < 0.002: do not run rolling OOF.",
        "3. any dataset regression > 0.003: increase confidence or use conservative dataset calibration.",
        "4. rolling OOF gain >= 0.003 with low catastrophic activation: only then consider distillation.",
        "5. Oracles are analysis only and are never valid model results.", "",
    ]
    (output / "summary_phase9_saferoute.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
