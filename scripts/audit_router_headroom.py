#!/usr/bin/env python3
"""Fine-grained frozen-expert oracle audit. Test oracles are analysis only."""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.io import CompactMetaDataset, expand_meta_paths


WARNING = "ANALYSIS ONLY -- test labels used -- not a valid model result"


def weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.maximum(np.sum(weights), 1e-12))


def analyse_cell(dataset):
    parts = list(dataset.iter_parts())
    if not parts or "loss_mse" not in parts[0]:
        raise ValueError(f"metadata has no labels: {dataset.root}")
    loss = np.concatenate([part["loss_mse"] for part in parts], axis=0).astype(np.float64)
    mae = np.concatenate([part["loss_mae"] for part in parts], axis=0).astype(np.float64)
    sample = np.concatenate([part["sample_id"] for part in parts], axis=0)
    block = np.concatenate([part["block"] for part in parts], axis=0)
    block_start = np.concatenate([part["block_start"] for part in parts], axis=0)
    block_end = np.concatenate([part["block_end"] for part in parts], axis=0)
    width = np.concatenate([part["block_end"] - part["block_start"] for part in parts], axis=0).astype(np.float64)
    names = dataset.expert_names
    expert_mse = np.array([weighted_mean(loss[:, k], width) for k in range(loss.shape[1])])
    expert_mae = np.array([weighted_mean(mae[:, k], width) for k in range(mae.shape[1])])
    cell_best_index = int(np.argmin(expert_mse))

    sample_values = []
    sample_choices = Counter()
    for sample_id in np.unique(sample):
        mask = sample == sample_id
        scores = np.array([weighted_mean(loss[mask, k], width[mask]) for k in range(loss.shape[1])])
        best = int(np.argmin(scores))
        sample_choices[names[best]] += 1
        sample_values.append(scores[best])
    sample_oracle = float(np.mean(sample_values))

    block_values = []
    block_choices = Counter()
    for block_id in np.unique(block):
        mask = block == block_id
        scores = np.array([weighted_mean(loss[mask, k], width[mask]) for k in range(loss.shape[1])])
        best = int(np.argmin(scores))
        block_choices[names[best]] += 1
        block_values.append(
            {
                "block": int(block_id),
                "block_start": int(block_start[mask][0]),
                "block_end": int(block_end[mask][0]),
                "oracle_mse": float(scores[best]),
                "oracle_expert": names[best],
                "expert_mse": scores,
                "width": float(np.mean(width[mask])),
            }
        )
    horizon_block_oracle = weighted_mean(
        np.array([item["oracle_mse"] for item in block_values]),
        np.array([item["width"] for item in block_values]),
    )
    row_best = np.argmin(loss, axis=1)
    sample_block_oracle = weighted_mean(loss[np.arange(len(loss)), row_best], width)
    sample_block_choices = Counter(names[index] for index in row_best)

    channel_group_oracle = None
    channel_group_choices = Counter()
    if "channel_group_loss_mse" in parts[0]:
        group_loss = np.concatenate(
            [part["channel_group_loss_mse"] for part in parts], axis=0
        ).astype(np.float64)
        if group_loss.ndim != 3 or group_loss.shape[:2] != loss.shape:
            raise ValueError("channel-group loss must have shape [rows, experts, groups]")
        group_sizes = np.asarray(
            dataset.manifest.get("channel_group_sizes", [1] * group_loss.shape[2]),
            dtype=np.float64,
        )
        if group_sizes.shape != (group_loss.shape[2],):
            raise ValueError("channel_group_sizes do not match channel-group loss")
        group_best = np.argmin(group_loss, axis=1)
        group_best_loss = np.take_along_axis(
            group_loss, group_best[:, None, :], axis=1
        )[:, 0, :]
        channel_group_oracle = weighted_mean(
            group_best_loss.reshape(-1),
            (width[:, None] * group_sizes[None, :]).reshape(-1),
        )
        channel_group_choices.update(names[index] for index in group_best.reshape(-1))

    return {
        "dataset": dataset.manifest["dataset"],
        "seq_len": int(dataset.manifest["seq_len"]),
        "pred_len": int(dataset.manifest["pred_len"]),
        "samples": int(len(np.unique(sample))),
        "expert_names": names,
        "expert_mse": expert_mse,
        "expert_mae": expert_mae,
        "cell_best_expert": names[cell_best_index],
        "cell_oracle_mse": float(expert_mse[cell_best_index]),
        "sample_oracle_mse": sample_oracle,
        "horizon_block_oracle_mse": horizon_block_oracle,
        "sample_block_oracle_mse": sample_block_oracle,
        "channel_group_oracle_mse": channel_group_oracle,
        "sample_choices": sample_choices,
        "block_choices": block_choices,
        "sample_block_choices": sample_block_choices,
        "channel_group_choices": channel_group_choices,
        "block_values": block_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="comma paths/globs of compact metadata dirs")
    parser.add_argument("--current_validation_selected", type=float, default=0.333805)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    datasets = [CompactMetaDataset(path) for path in expand_meta_paths(args.meta)]
    for dataset in datasets:
        if str(dataset.manifest.get("split")) != "test":
            raise SystemExit("headroom audit requires test-labelled metadata and is analysis only")
    cells = [analyse_cell(dataset) for dataset in datasets]
    names = cells[0]["expert_names"]
    if any(cell["expert_names"] != names for cell in cells):
        raise SystemExit("all headroom cells must use the same ordered expert pool")

    fixed = {name: float(np.mean([cell["expert_mse"][i] for cell in cells])) for i, name in enumerate(names)}
    best_fixed = min(fixed, key=fixed.get)
    best_fixed_mse = fixed[best_fixed]
    metrics = {
        "best_fixed_mse": best_fixed_mse,
        "cell_oracle_mse": float(np.mean([cell["cell_oracle_mse"] for cell in cells])),
        "sample_oracle_mse": float(np.mean([cell["sample_oracle_mse"] for cell in cells])),
        "horizon_block_oracle_mse": float(np.mean([cell["horizon_block_oracle_mse"] for cell in cells])),
        "sample_block_oracle_mse": float(np.mean([cell["sample_block_oracle_mse"] for cell in cells])),
    }
    channel = [cell["channel_group_oracle_mse"] for cell in cells if cell["channel_group_oracle_mse"] is not None]
    if channel:
        metrics["channel_group_oracle_mse"] = float(np.mean(channel))
        metrics["channel_group_gain_vs_sample_block"] = (
            metrics["sample_block_oracle_mse"] - metrics["channel_group_oracle_mse"]
        )
    gain = best_fixed_mse - metrics["sample_block_oracle_mse"]
    if gain >= 0.008:
        opportunity = "strong routing opportunity"
    elif gain >= 0.004:
        opportunity = "moderate routing opportunity"
    else:
        opportunity = "limited routing headroom; do not run expensive router phase"

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cell_fields = [
        "dataset", "seq_len", "pred_len", "samples", "best_fixed_expert", "best_fixed_mse",
        "cell_best_expert", "cell_oracle_mse", "sample_oracle_mse",
        "horizon_block_oracle_mse", "sample_block_oracle_mse",
        "sample_block_gain_vs_best_fixed",
    ]
    with (output / "headroom_by_cell.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields)
        writer.writeheader()
        for cell in cells:
            best_index = names.index(best_fixed)
            cell_fixed = float(cell["expert_mse"][best_index])
            writer.writerow(
                {
                    **{key: cell[key] for key in ("dataset", "seq_len", "pred_len", "samples")},
                    "best_fixed_expert": best_fixed,
                    "best_fixed_mse": cell_fixed,
                    "cell_best_expert": cell["cell_best_expert"],
                    "cell_oracle_mse": cell["cell_oracle_mse"],
                    "sample_oracle_mse": cell["sample_oracle_mse"],
                    "horizon_block_oracle_mse": cell["horizon_block_oracle_mse"],
                    "sample_block_oracle_mse": cell["sample_block_oracle_mse"],
                    "sample_block_gain_vs_best_fixed": cell_fixed - cell["sample_block_oracle_mse"],
                }
            )

    by_dataset = defaultdict(list)
    for cell in cells:
        by_dataset[cell["dataset"]].append(cell)
    with (output / "headroom_by_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "cells", "best_fixed_mse", "cell_oracle_mse", "sample_oracle_mse", "horizon_block_oracle_mse", "sample_block_oracle_mse", "sample_block_gain"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, group in sorted(by_dataset.items()):
            best_index = names.index(best_fixed)
            fixed_value = float(np.mean([cell["expert_mse"][best_index] for cell in group]))
            sample_block = float(np.mean([cell["sample_block_oracle_mse"] for cell in group]))
            writer.writerow(
                {
                    "dataset": dataset,
                    "cells": len(group),
                    "best_fixed_mse": fixed_value,
                    "cell_oracle_mse": np.mean([cell["cell_oracle_mse"] for cell in group]),
                    "sample_oracle_mse": np.mean([cell["sample_oracle_mse"] for cell in group]),
                    "horizon_block_oracle_mse": np.mean([cell["horizon_block_oracle_mse"] for cell in group]),
                    "sample_block_oracle_mse": sample_block,
                    "sample_block_gain": fixed_value - sample_block,
                }
            )

    with (output / "headroom_by_horizon.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "seq_len", "pred_len", "block", "block_start", "block_end",
            "best_fixed_expert", "best_fixed_mse", "oracle_expert", "oracle_mse",
            "oracle_gain_vs_best_fixed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        best_index = names.index(best_fixed)
        for cell in cells:
            for item in cell["block_values"]:
                fixed_mse = float(item["expert_mse"][best_index])
                writer.writerow(
                    {
                        "dataset": cell["dataset"],
                        "seq_len": cell["seq_len"],
                        "pred_len": cell["pred_len"],
                        "block": item["block"],
                        "block_start": item["block_start"],
                        "block_end": item["block_end"],
                        "best_fixed_expert": best_fixed,
                        "best_fixed_mse": fixed_mse,
                        "oracle_expert": item["oracle_expert"],
                        "oracle_mse": item["oracle_mse"],
                        "oracle_gain_vs_best_fixed": fixed_mse - item["oracle_mse"],
                    }
                )

    choice_counts = {
        "cell": Counter(),
        "sample": Counter(),
        "horizon_block": Counter(),
        "sample_block": Counter(),
        "channel_group": Counter(),
    }
    for cell in cells:
        choice_counts["cell"][cell["cell_best_expert"]] += 1
        choice_counts["sample"].update(cell["sample_choices"])
        choice_counts["horizon_block"].update(cell["block_choices"])
        choice_counts["sample_block"].update(cell["sample_block_choices"])
        choice_counts["channel_group"].update(cell["channel_group_choices"])
    oracle_gains = {
        key: best_fixed_mse - metrics[f"{key}_mse"]
        for key in ("cell_oracle", "sample_oracle", "horizon_block_oracle", "sample_block_oracle")
    }
    oracle_gains_vs_selected = {
        key: args.current_validation_selected - metrics[f"{key}_mse"]
        for key in ("cell_oracle", "sample_oracle", "horizon_block_oracle", "sample_block_oracle")
    }
    payload = {
        "warning": WARNING,
        "cells": len(cells),
        "best_fixed_expert": best_fixed,
        "fixed_expert_mse": fixed,
        **metrics,
        "current_validation_selected_mse": args.current_validation_selected,
        "sample_block_gain_vs_best_fixed": gain,
        "sample_block_gain_vs_current_validation_selected": args.current_validation_selected - metrics["sample_block_oracle_mse"],
        "oracle_gains_vs_best_fixed": oracle_gains,
        "oracle_gains_vs_current_validation_selected": oracle_gains_vs_selected,
        "opportunity": opportunity,
        "oracle_expert_counts": {
            key: dict(sorted(counts.items())) for key, counts in choice_counts.items() if counts
        },
        "sample_block_oracle_expert_counts": dict(sorted(choice_counts["sample_block"].items())),
    }
    with (output / "headroom_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# Phase 9 SafeRoute Headroom Audit",
        "",
        f"**{WARNING}**",
        "",
        f"- cells: {len(cells)}",
        f"- best_fixed_expert: {best_fixed}",
        f"- best_fixed_mse: {best_fixed_mse:.6f}",
        f"- cell_oracle_mse: {metrics['cell_oracle_mse']:.6f}",
        f"- sample_oracle_mse: {metrics['sample_oracle_mse']:.6f}",
        f"- horizon_block_oracle_mse: {metrics['horizon_block_oracle_mse']:.6f}",
        f"- sample_block_oracle_mse: {metrics['sample_block_oracle_mse']:.6f}",
        f"- sample_block_gain_vs_best_fixed: {gain:.6f}",
        f"- sample_block_gain_vs_current_validation_selected: {args.current_validation_selected - metrics['sample_block_oracle_mse']:.6f}",
        f"- current_validation_selected_mse: {args.current_validation_selected:.6f}",
        f"- decision: {opportunity}",
        "",
        "## Fixed Experts",
        "",
        "| expert | mse |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {fixed[name]:.6f} |" for name in names)
    lines += ["", "## Oracle Gains", "", "| oracle | vs best fixed | vs validation-selected |", "| --- | ---: | ---: |"]
    lines.extend(
        f"| {key} | {oracle_gains[key]:.6f} | {oracle_gains_vs_selected[key]:.6f} |"
        for key in oracle_gains
    )
    if "channel_group_oracle_mse" in metrics:
        lines += [
            f"| channel_group_oracle | {best_fixed_mse - metrics['channel_group_oracle_mse']:.6f} | {args.current_validation_selected - metrics['channel_group_oracle_mse']:.6f} |",
            "",
            f"- channel_group_gain_vs_sample_block: {metrics['channel_group_gain_vs_sample_block']:.6f}",
            f"- channel_group_routing_allowed: {metrics['channel_group_gain_vs_sample_block'] >= 0.002}",
        ]
    for oracle_name, counts in choice_counts.items():
        if not counts:
            continue
        lines += ["", f"## {oracle_name.replace('_', ' ').title()} Oracle Choice Counts", "", "| expert | selections |", "| --- | ---: |"]
        lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
    lines += ["", "Never report any oracle above as a valid selected model.", ""]
    (output / "headroom_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
