#!/usr/bin/env python3
"""Leakage-safe Stage-D confirmation gate and final test aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


STAGE_A = "fk_r8_cs"
TAIL2 = "fk_sm2_tail2"
ARMS = (STAGE_A, TAIL2)
DATASETS = ("ETTm1", "traffic")
HORIZONS = (336, 720)
SEEDS = (2024, 2025, 2026)
CUT_FREQ = {"ETTm1": 7, "traffic": 25}


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_json_rows(root, filename):
    rows = []
    for path in sorted(Path(root).glob(f"**/{filename}")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_file"] = str(path)
        rows.append(row)
    return rows


def key(row):
    return (
        str(row.get("arm", "")),
        str(row.get("dataset", "")),
        int(row.get("seq_len", -1)),
        int(row.get("pred_len", -1)),
        int(row.get("seed", -1)),
        int(row.get("cut_freq", -1)),
    )


def expected_key(arm, dataset, pred_len, seed):
    return (arm, dataset, 96, pred_len, seed, CUT_FREQ[dataset])


def index_unique(rows, label):
    indexed = {}
    for row in rows:
        item_key = key(row)
        if item_key in indexed:
            raise ValueError(f"duplicate {label} row: {item_key}")
        indexed[item_key] = row
    return indexed


def summarize_validation(screen_rows, confirm_rows, screen_decision):
    if (
        int(screen_decision.get("advance_to_confirmation", 0)) != 1
        or screen_decision.get("selected_arm") != TAIL2
        or screen_decision.get("test_metrics_used") is not False
    ):
        raise ValueError("Stage-D screen did not authorize confirmation")

    screen = index_unique(screen_rows, "screen")
    confirm = index_unique(confirm_rows, "confirmation")
    expected_confirm = {
        expected_key(arm, dataset, pred_len, seed)
        for arm in ARMS
        for dataset in DATASETS
        for pred_len in HORIZONS
        for seed in (2024, 2025)
    }
    if set(confirm) != expected_confirm:
        missing = sorted(expected_confirm - set(confirm))
        extra = sorted(set(confirm) - expected_confirm)
        raise ValueError(f"confirmation manifest mismatch: missing={missing}, extra={extra}")

    selected = {}
    for arm in ARMS:
        for dataset in DATASETS:
            for pred_len in HORIZONS:
                for seed in SEEDS:
                    item_key = expected_key(arm, dataset, pred_len, seed)
                    source = screen if seed == 2026 else confirm
                    if item_key not in source:
                        raise ValueError(f"missing validation row: {item_key}")
                    row = source[item_key]
                    if (
                        row.get("status") != "ok"
                        or row.get("test_deferred") is not True
                        or row.get("mse") is not None
                        or row.get("mae") is not None
                        or not finite(row.get("val_mse"))
                    ):
                        raise ValueError(f"non-finite, failed, or test-open validation row: {item_key}")
                    selected[item_key] = row

    pairs = []
    stage_values = []
    tail_values = []
    for dataset in DATASETS:
        for pred_len in HORIZONS:
            for seed in SEEDS:
                stage = selected[expected_key(STAGE_A, dataset, pred_len, seed)]
                tail = selected[expected_key(TAIL2, dataset, pred_len, seed)]
                stage_val = float(stage["val_mse"])
                tail_val = float(tail["val_mse"])
                stage_values.append(stage_val)
                tail_values.append(tail_val)
                pairs.append(
                    {
                        "dataset": dataset,
                        "seq_len": 96,
                        "pred_len": pred_len,
                        "seed": seed,
                        "cut_freq": CUT_FREQ[dataset],
                        "stage_a_val_mse": stage_val,
                        "tail2_val_mse": tail_val,
                        "delta_val": tail_val - stage_val,
                        "delta_val_pct": 100.0 * (tail_val / stage_val - 1.0),
                        "tail2_win": int(tail_val < stage_val),
                        "stage_a_summary": stage["summary_file"],
                        "tail2_summary": tail["summary_file"],
                    }
                )

    stage_mean = statistics.fmean(stage_values)
    tail_mean = statistics.fmean(tail_values)
    open_test = tail_mean < stage_mean
    aggregate = {
        "pairs": len(pairs),
        "stage_a_val_mse_macro_mean": stage_mean,
        "tail2_val_mse_macro_mean": tail_mean,
        "tail2_val_delta": tail_mean - stage_mean,
        "tail2_val_delta_pct": 100.0 * (tail_mean / stage_mean - 1.0),
        "tail2_validation_wins": sum(row["tail2_win"] for row in pairs),
        "open_test": int(open_test),
    }
    decision = {
        "selected_arm": TAIL2,
        "open_test": int(open_test),
        "reason": (
            "three-seed macro validation mean remained below Stage A"
            if open_test
            else "three-seed macro validation mean did not remain below Stage A"
        ),
        "test_metrics_used": False,
        "pairs": len(pairs),
    }
    return pairs, aggregate, decision


def summarize_final(test_rows, validation_decision):
    if (
        int(validation_decision.get("open_test", 0)) != 1
        or validation_decision.get("selected_arm") != TAIL2
        or validation_decision.get("test_metrics_used") is not False
    ):
        raise ValueError("validation gate did not authorize opening test")

    indexed = index_unique(test_rows, "test")
    expected = {
        expected_key(arm, dataset, pred_len, seed)
        for arm in ARMS
        for dataset in DATASETS
        for pred_len in HORIZONS
        for seed in SEEDS
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"test manifest mismatch: missing={missing}, extra={extra}")
    for item_key, row in indexed.items():
        if (
            row.get("status") != "ok"
            or row.get("test_opened_after_validation_selection") is not True
            or not finite(row.get("mse"))
            or not finite(row.get("mae"))
        ):
            raise ValueError(f"invalid authorized test row: {item_key}")

    pairs = []
    by_dataset = {dataset: [] for dataset in DATASETS}
    for dataset in DATASETS:
        for pred_len in HORIZONS:
            for seed in SEEDS:
                stage = indexed[expected_key(STAGE_A, dataset, pred_len, seed)]
                tail = indexed[expected_key(TAIL2, dataset, pred_len, seed)]
                stage_mse = float(stage["mse"])
                tail_mse = float(tail["mse"])
                delta_pct = 100.0 * (tail_mse / stage_mse - 1.0)
                by_dataset[dataset].append(delta_pct)
                pairs.append(
                    {
                        "dataset": dataset,
                        "seq_len": 96,
                        "pred_len": pred_len,
                        "seed": seed,
                        "stage_a_test_mse": stage_mse,
                        "tail2_test_mse": tail_mse,
                        "delta_test_mse": tail_mse - stage_mse,
                        "delta_test_mse_pct": delta_pct,
                        "tail2_win": int(tail_mse < stage_mse),
                        "stage_a_test_mae": float(stage["mae"]),
                        "tail2_test_mae": float(tail["mae"]),
                        "stage_a_summary": stage["summary_file"],
                        "tail2_summary": tail["summary_file"],
                    }
                )

    wins = sum(row["tail2_win"] for row in pairs)
    macro_delta = statistics.fmean(row["delta_test_mse_pct"] for row in pairs)
    dataset_delta = {
        dataset: statistics.fmean(values) for dataset, values in by_dataset.items()
    }
    stable = wins >= 8 and macro_delta < 0.0 and all(
        value < 0.0 for value in dataset_delta.values()
    )
    aggregate = {
        "pairs": len(pairs),
        "tail2_test_wins": wins,
        "macro_mean_relative_test_mse_delta_pct": macro_delta,
        "ETTm1_mean_relative_test_mse_delta_pct": dataset_delta["ETTm1"],
        "traffic_mean_relative_test_mse_delta_pct": dataset_delta["traffic"],
        "stable_generalization": int(stable),
    }
    decision = {
        "selected_arm": TAIL2 if stable else STAGE_A,
        "stable_generalization": int(stable),
        "reason": (
            "tail2 passed the frozen 8/12, macro, and per-dataset test gates"
            if stable
            else "tail2 failed at least one frozen final generalization gate"
        ),
        "test_opened_after_validation_selection": True,
        "test_metrics_used_for_tuning": False,
    }
    return pairs, aggregate, decision


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_eval_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# arm\tdataset\tseq_len\tpred_len\tseed\tcut_freq\n")
        for row in rows:
            handle.write(
                f"{row['arm']}\t{row['dataset']}\t96\t{row['pred_len']}\t"
                f"{row['seed']}\t{CUT_FREQ[row['dataset']]}\n"
            )


def validation_command(args):
    screen_decision = json.loads(Path(args.screen_decision).read_text(encoding="utf-8"))
    pairs, aggregate, decision = summarize_validation(
        load_json_rows(args.screen_root, "run_summary.json"),
        load_json_rows(args.confirm_root, "run_summary.json"),
        screen_decision,
    )
    output = Path(args.output_dir)
    write_csv(output / "validation_pairs.csv", pairs, [
        "dataset", "seq_len", "pred_len", "seed", "cut_freq",
        "stage_a_val_mse", "tail2_val_mse", "delta_val", "delta_val_pct",
        "tail2_win", "stage_a_summary", "tail2_summary",
    ])
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "validation_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if decision["open_test"]:
        screen_eval = []
        confirm_eval = []
        for arm in ARMS:
            for dataset in DATASETS:
                for pred_len in HORIZONS:
                    for seed in SEEDS:
                        target = screen_eval if seed == 2026 else confirm_eval
                        target.append({
                            "arm": arm, "dataset": dataset,
                            "pred_len": pred_len, "seed": seed,
                        })
        write_eval_manifest(output / "eval_screen_seed2026.tsv", screen_eval)
        write_eval_manifest(output / "eval_confirm_seeds2024_2025.tsv", confirm_eval)
    print(json.dumps({"aggregate": aggregate, "decision": decision}, sort_keys=True))


def final_command(args):
    validation_decision = json.loads(
        Path(args.validation_decision).read_text(encoding="utf-8")
    )
    rows = []
    for root in args.eval_root:
        rows.extend(load_json_rows(root, "test_summary.json"))
    pairs, aggregate, decision = summarize_final(rows, validation_decision)
    output = Path(args.output_dir)
    write_csv(output / "paired_test_results.csv", pairs, [
        "dataset", "seq_len", "pred_len", "seed", "stage_a_test_mse",
        "tail2_test_mse", "delta_test_mse", "delta_test_mse_pct", "tail2_win",
        "stage_a_test_mae", "tail2_test_mae", "stage_a_summary", "tail2_summary",
    ])
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "final_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase 11 Stage D final decision",
        "",
        "Test was opened only after the frozen three-seed validation gate passed.",
        "",
        "| wins | macro relative delta | ETTm1 delta | Traffic delta | stable |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {aggregate['tail2_test_wins']}/12 | "
            f"{aggregate['macro_mean_relative_test_mse_delta_pct']:+.4f}% | "
            f"{aggregate['ETTm1_mean_relative_test_mse_delta_pct']:+.4f}% | "
            f"{aggregate['traffic_mean_relative_test_mse_delta_pct']:+.4f}% | "
            f"{aggregate['stable_generalization']} |"
        ),
    ]
    (output / "PHASE11_STAGED_FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "decision": decision}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validation")
    validation.add_argument("--screen_root", required=True)
    validation.add_argument("--confirm_root", required=True)
    validation.add_argument("--screen_decision", required=True)
    validation.add_argument("--output_dir", required=True)
    validation.set_defaults(func=validation_command)
    final = subparsers.add_parser("final")
    final.add_argument("--eval_root", action="append", required=True)
    final.add_argument("--validation_decision", required=True)
    final.add_argument("--output_dir", required=True)
    final.set_defaults(func=final_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
