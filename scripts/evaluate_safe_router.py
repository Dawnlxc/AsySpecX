#!/usr/bin/env python3
"""Stream routed test evaluation without writing full prediction arrays."""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.blocks import horizon_blocks
from router.manifest import load_expert_manifest
from router.pipeline import compact_meta_batch
from router.runtime import FrozenExpertPool, make_ordered_loader
from router.safe import safe_route
from router.training import RouterBundle


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_periods(value):
    return [int(item) for item in str(value).replace("+", ",").split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_manifest", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor_expert", default="anchor")
    parser.add_argument("--expert_seeds", default="2024,2025,2026")
    parser.add_argument("--experts", default="")
    parser.add_argument("--router_num_horizon_blocks", type=int, default=4)
    parser.add_argument("--router_decision", choices=["hard_top1", "safe_top1_blend", "safe_multi_mix"], default="safe_top1_blend")
    parser.add_argument("--router_min_gain", type=float, default=0.0)
    parser.add_argument("--router_full_gain", type=float, default=0.02)
    parser.add_argument("--router_uncertainty_beta", type=float, default=0.1)
    parser.add_argument("--router_temperature", type=float, default=0.1)
    parser.add_argument("--catastrophic_threshold", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expert_device_policy", choices=["one_at_a_time", "resident"], default="one_at_a_time")
    parser.add_argument("--router_forecast_snippet_bins", type=int, default=0)
    parser.add_argument("--max_feature_channels", type=int, default=64)
    args = parser.parse_args()

    seeds = parse_csv(args.expert_seeds)
    selected = parse_csv(args.experts) or None
    manifest = load_expert_manifest(
        args.expert_manifest,
        args.anchor_expert,
        selected,
        seeds,
        require_checkpoints=True,
    )
    bundle = RouterBundle.load(args.router)
    if list(bundle.expert_names) != manifest.names:
        raise SystemExit("router and expert manifest use different expert ordering")
    dataset, loader = make_ordered_loader(
        manifest.anchor.config,
        "test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pool = FrozenExpertPool(manifest, seeds, args.device, args.expert_device_policy)
    blocks = horizon_blocks(int(manifest.cell["pred_len"]), args.router_num_horizon_blocks)
    periods = parse_periods(manifest.anchor.config.get("periods", manifest.anchor.config.get("period", "")))
    names = manifest.names
    anchor_index = names.index(args.anchor_expert)
    alternative_indices = [index for index, name in enumerate(names) if name != args.anchor_expert]
    alternative_names = [names[index] for index in alternative_indices]

    routed_se = routed_ae = anchor_se = anchor_ae = 0.0
    elements = samples = 0
    activation = Counter()
    fallback = alpha_sum = active_rows = false_rows = catastrophic_rows = 0
    predicted_advantage_sum = actual_advantage_sum = 0.0
    diagnostics_rows = defaultdict(
        lambda: {
            "rows": 0,
            "activations": 0,
            "alpha_sum": 0.0,
            "false": 0,
            "catastrophic": 0,
            "seed_variance_sum": 0.0,
            "predicted_advantage_sum": 0.0,
            "actual_advantage_sum": 0.0,
            "routed_gain_sum": 0.0,
        }
    )
    sample_offset = 0

    for batch in loader:
        batch_x, batch_y = batch[0], batch[1]
        predictions, variances = pool.predict(batch_x)
        target = batch_y[:, -int(manifest.cell["pred_len"]) :, :].float().cpu().numpy()
        sample_ids = np.arange(sample_offset, sample_offset + len(batch_x), dtype=np.int64)
        arrays, feature_names = compact_meta_batch(
            batch_x.float().cpu().numpy(), predictions, variances, sample_ids, blocks,
            args.anchor_expert, periods, str(manifest.cell["dataset"]), bundle.router_scope,
            target=target, snippet_bins=args.router_forecast_snippet_bins,
            max_channels=args.max_feature_channels,
        )
        if list(feature_names) != list(bundle.feature_names):
            raise SystemExit("evaluation features do not match trained router features")
        batch_size = len(batch_x)
        feature_cube = arrays["features"].reshape(batch_size, len(blocks), -1)
        loss_cube = arrays["loss_mse"].reshape(batch_size, len(blocks), len(names))
        seed_cube = arrays["seed_variance"].reshape(batch_size, len(blocks), len(names))
        anchor_prediction = predictions[args.anchor_expert]
        routed = anchor_prediction.copy()

        for block_index, (start, end) in enumerate(blocks):
            predicted, quantiles = bundle.predict(
                feature_cube[:, block_index, :],
                str(manifest.cell["dataset"]),
                int(manifest.cell["seq_len"]),
                int(manifest.cell["pred_len"]),
                block_index,
            )
            alternatives = np.stack(
                [predictions[name][:, start:end, :] for name in alternative_names], axis=1
            )
            block_routed, diag = safe_route(
                anchor_prediction[:, start:end, :],
                alternatives,
                predicted,
                quantiles,
                seed_cube[:, block_index, alternative_indices],
                decision=args.router_decision,
                min_gain=args.router_min_gain,
                full_gain=args.router_full_gain,
                uncertainty_beta=args.router_uncertainty_beta,
                temperature=args.router_temperature,
            )
            routed[:, start:end, :] = block_routed
            anchor_loss = loss_cube[:, block_index, anchor_index]
            alternative_loss = loss_cube[:, block_index, alternative_indices]
            top = diag["top_index"]
            active = diag["active"]
            chosen_loss = alternative_loss[np.arange(batch_size), top]
            regret = chosen_loss - anchor_loss
            routed_block_loss = np.mean(
                (
                    block_routed.astype(np.float64)
                    - target[:, start:end, :].astype(np.float64)
                )
                ** 2,
                axis=(1, 2),
            )
            routed_gain = anchor_loss - routed_block_loss
            fallback += int((~active).sum())
            active_rows += int(active.sum())
            false_rows += int(((regret > 0) & active).sum())
            catastrophic_rows += int(((regret > args.catastrophic_threshold) & active).sum())
            alpha_sum += float(diag["alpha"].sum())
            predicted_advantage_sum += float(np.where(active, predicted[np.arange(batch_size), top], 0.0).sum())
            actual_advantage_sum += float(np.where(active, -regret, 0.0).sum())
            for expert in names:
                diagnostics_rows[(block_index, expert)]["rows"] += batch_size
            for row in range(batch_size):
                expert = alternative_names[int(top[row])] if active[row] else args.anchor_expert
                activation[expert] += 1
                key = (block_index, expert)
                rec = diagnostics_rows[key]
                rec["activations"] += 1
                rec["alpha_sum"] += float(diag["alpha"][row])
                rec["false"] += int(active[row] and regret[row] > 0)
                rec["catastrophic"] += int(active[row] and regret[row] > args.catastrophic_threshold)
                rec["routed_gain_sum"] += float(routed_gain[row])
                if active[row]:
                    rec["seed_variance_sum"] += float(seed_cube[row, block_index, alternative_indices[int(top[row])]])
                    rec["predicted_advantage_sum"] += float(predicted[row, top[row]])
                    rec["actual_advantage_sum"] += float(-regret[row])

        error = routed.astype(np.float64) - target.astype(np.float64)
        anchor_error = anchor_prediction.astype(np.float64) - target.astype(np.float64)
        routed_se += float(np.square(error).sum())
        routed_ae += float(np.abs(error).sum())
        anchor_se += float(np.square(anchor_error).sum())
        anchor_ae += float(np.abs(anchor_error).sum())
        elements += int(error.size)
        samples += batch_size
        sample_offset += batch_size

    routed_mse, routed_mae = routed_se / elements, routed_ae / elements
    anchor_mse, anchor_mae = anchor_se / elements, anchor_ae / elements
    total_rows = samples * len(blocks)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": manifest.cell["dataset"],
        "seq_len": int(manifest.cell["seq_len"]),
        "pred_len": int(manifest.cell["pred_len"]),
        "samples": samples,
        "anchor_expert": args.anchor_expert,
        "anchor_mse": anchor_mse,
        "anchor_mae": anchor_mae,
        "routed_mse": routed_mse,
        "routed_mae": routed_mae,
        "delta_mse": routed_mse - anchor_mse,
        "gain_vs_anchor": anchor_mse - routed_mse,
        "router_coverage": active_rows / max(total_rows, 1),
        "fallback_fraction": fallback / max(total_rows, 1),
        "mean_alpha": alpha_sum / max(total_rows, 1),
        "false_activation_rate": false_rows / max(active_rows, 1),
        "catastrophic_activation_rate": catastrophic_rows / max(active_rows, 1),
        "mean_predicted_advantage_activated": predicted_advantage_sum / max(active_rows, 1),
        "mean_actual_advantage_activated_analysis_only": actual_advantage_sum / max(active_rows, 1),
        "test_labels_used_for_decision": False,
    }
    with (output / "routed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader(); writer.writerow(result)
    with (output / "routing_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "seq_len", "pred_len", "block", "block_start", "block_end",
            "expert", "rows", "activations", "activation_fraction",
            "mean_alpha", "mean_alpha_when_selected", "false_activation_rate",
            "catastrophic_activation_rate", "mean_seed_variance_activated",
            "mean_predicted_advantage_activated",
            "mean_actual_advantage_activated_analysis_only",
            "mean_routed_gain_all_rows_analysis_only",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for (block, expert), rec in sorted(diagnostics_rows.items()):
            writer.writerow(
                {
                    "dataset": manifest.cell["dataset"],
                    "seq_len": int(manifest.cell["seq_len"]),
                    "pred_len": int(manifest.cell["pred_len"]),
                    "block": block,
                    "block_start": blocks[block][0],
                    "block_end": blocks[block][1],
                    "expert": expert,
                    "rows": rec["rows"],
                    "activations": rec["activations"],
                    "activation_fraction": rec["activations"] / max(rec["rows"], 1),
                    "mean_alpha": rec["alpha_sum"] / max(rec["rows"], 1),
                    "mean_alpha_when_selected": rec["alpha_sum"] / max(rec["activations"], 1),
                    "false_activation_rate": rec["false"] / max(rec["activations"], 1),
                    "catastrophic_activation_rate": rec["catastrophic"] / max(rec["activations"], 1),
                    "mean_seed_variance_activated": rec["seed_variance_sum"] / max(rec["activations"], 1),
                    "mean_predicted_advantage_activated": rec["predicted_advantage_sum"] / max(rec["activations"], 1),
                    "mean_actual_advantage_activated_analysis_only": rec["actual_advantage_sum"] / max(rec["activations"], 1),
                    "mean_routed_gain_all_rows_analysis_only": rec["routed_gain_sum"] / max(rec["rows"], 1),
                }
            )
    lines = [
        "# Phase 9 SafeRoute Cell Evaluation", "",
        "Router and confidence calibration use no test labels. Actual advantage diagnostics below are analysis only.", "",
    ] + [f"- {key}: {value}" for key, value in result.items()] + ["", "## Activation Counts", ""]
    lines += [f"- {name}: {count}" for name, count in sorted(activation.items())]
    (output / "summary_phase9_saferoute.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
