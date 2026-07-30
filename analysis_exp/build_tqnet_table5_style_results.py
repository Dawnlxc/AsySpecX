#!/usr/bin/env python3
"""Render the locked seed-2026 main results in TQNet Table 5 style.

The normalized CSV is the single source of truth. The generated Markdown uses
the same structural layout as TQNet Table 5: models across columns (separate
MSE/MAE subcolumns), datasets down rows, four horizons plus a per-dataset Avg.

This script intentionally preserves the project's locked evaluation protocol:
  * non-PEMS datasets: L=720, H in {96, 192, 336, 720}
  * PEMS datasets: L=96, H in {12, 24, 48, 96}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


OURS = "Ours (Compact + Echo)"
MODELS = (
    OURS,
    "TQNet",
    "CycleNet",
    "FITS",
    "SparseTSF",
    "FreTS",
    "FilterNet",
    "iTransformer",
    "PatchTST",
    "DLinear",
    "MixLinear",
    "PhaseFormer",
    "FreqCycle",
)
DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "electricity",
    "traffic",
    "weather",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
)
DISPLAY_DATASET = {
    "electricity": "Electricity",
    "traffic": "Traffic",
    "weather": "Weather",
}
HORIZONS = {
    dataset: ((12, 24, 48, 96) if dataset.startswith("PEMS") else (96, 192, 336, 720))
    for dataset in DATASETS
}
SEQ_LEN = {dataset: (96 if dataset.startswith("PEMS") else 720) for dataset in DATASETS}
EXPECTED_ROWS = len(MODELS) * sum(len(HORIZONS[dataset]) for dataset in DATASETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_and_validate(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != EXPECTED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if Counter(row["model"] for row in rows) != Counter({model: 44 for model in MODELS}):
        raise AssertionError("Unexpected model coverage")
    if {row["dataset"] for row in rows} != set(DATASETS):
        raise AssertionError(f"Unexpected dataset coverage: {sorted({row['dataset'] for row in rows})}")
    if {row["seed"] for row in rows} != {"2026"}:
        raise AssertionError(f"Expected seed=2026 only, found {Counter(row['seed'] for row in rows)}")
    if {row["status"] for row in rows} != {"ok"}:
        raise AssertionError(f"Incomplete rows: {Counter(row['status'] for row in rows)}")

    seen: set[tuple[str, str, int, int]] = set()
    for row in rows:
        dataset = row["dataset"]
        key = (row["model"], dataset, int(row["seq_len"]), int(row["pred_len"]))
        if key in seen:
            raise AssertionError(f"Duplicate row: {key}")
        seen.add(key)
        if int(row["seq_len"]) != SEQ_LEN[dataset]:
            raise AssertionError(f"Wrong look-back policy: {key}")
        if int(row["pred_len"]) not in HORIZONS[dataset]:
            raise AssertionError(f"Wrong horizon policy: {key}")
        for metric in ("mse", "mae", "param_count"):
            if not row[metric] or not math.isfinite(float(row[metric])):
                raise AssertionError(f"Invalid {metric}: {key}")
    return rows


def metric_markup(value: float, competitors: list[float]) -> str:
    unique = sorted(set(competitors))
    rendered = f"{value:.3f}"
    if math.isclose(value, unique[0], rel_tol=0, abs_tol=1e-12):
        return f"**{rendered}**"
    if len(unique) > 1 and math.isclose(value, unique[1], rel_tol=0, abs_tol=1e-12):
        return f"_{rendered}_"
    return rendered


def format_params(values: list[int]) -> str:
    low, high = min(values), max(values)
    return f"{low:,}" if low == high else f"{low:,} - {high:,}"


def render(rows: list[dict[str, str]], input_csv: Path, output_md: Path) -> str:
    index = {
        (row["model"], row["dataset"], int(row["pred_len"])): row
        for row in rows
    }
    lines = [
        "# Main forecasting results - TQNet Table 5 style",
        "",
        f"> Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        "## Locked evaluation scope",
        "",
        "- Format reference: [TQNet, Table 5](https://arxiv.org/abs/2505.12917).",
        "- Non-PEMS: `L=720`, `H={96,192,336,720}`.",
        "- PEMS03/04/07/08: `L=96`, `H={12,24,48,96}`.",
        "- Models: 13; datasets: 11; evaluated cells: 44; normalized rows: 572/572.",
        "- All entries are held-out **test** MSE/MAE with `seed=2026`; validation metrics are not substituted.",
        "- Cell format: `MSE / MAE`; lower is better.",
        "- Best is **bold** and second-best is _italic_, ranked from unrounded values. Italic replaces Table 5's underline because Markdown has no native underline.",
        "- `Avg` is the arithmetic mean over the four horizons for that dataset.",
        "",
        "> This file adopts the **layout** of TQNet Table 5, not its input protocol. The original paper fixes `L=96` for every dataset and uses `H={96,192,336,720}`; this project keeps the user-locked mixed-L/PEMS protocol above.",
        "",
        "> **HPO disclosure:** Ours uses profiles selected by seed-2025 test MSE and frozen-replayed at seed=2026. These are test-selected stability results, not an untouched confirmatory test set.",
        "",
        "## Full results",
        "",
    ]
    for dataset in DATASETS:
        horizons = HORIZONS[dataset]
        display = DISPLAY_DATASET.get(dataset, dataset)
        model_labels = [f"**{model}**" if model == OURS else model for model in MODELS]
        lines.extend(
            [
                f"### {display} - L={SEQ_LEN[dataset]}",
                "",
                "| H | " + " | ".join(model_labels) + " |",
                "|---:|" + "---:|" * len(MODELS),
            ]
        )
        for horizon in horizons:
            mse_values = [float(index[(model, dataset, horizon)]["mse"]) for model in MODELS]
            mae_values = [float(index[(model, dataset, horizon)]["mae"]) for model in MODELS]
            cells: list[str] = []
            for model in MODELS:
                row = index[(model, dataset, horizon)]
                cells.append(
                    f'{metric_markup(float(row["mse"]), mse_values)} / '
                    f'{metric_markup(float(row["mae"]), mae_values)}'
                )
            lines.append(f"| {horizon} | " + " | ".join(cells) + " |")

        avg_mse = {
            model: statistics.mean(float(index[(model, dataset, horizon)]["mse"]) for horizon in horizons)
            for model in MODELS
        }
        avg_mae = {
            model: statistics.mean(float(index[(model, dataset, horizon)]["mae"]) for horizon in horizons)
            for model in MODELS
        }
        avg_cells: list[str] = []
        for model in MODELS:
            avg_cells.append(
                f'{metric_markup(avg_mse[model], list(avg_mse.values()))} / '
                f'{metric_markup(avg_mae[model], list(avg_mae.values()))}'
            )
        lines.extend([f"| **Avg** | " + " | ".join(avg_cells) + " |", ""])

    lines.extend(
        [
            "",
            "## Parameter audit",
            "",
            "Parameter counts are kept outside the Table 5-style performance grid because the original Table 5 reports only MSE/MAE.",
            "",
            "| Model | Registered trainable parameters (min - max) | Median | Distinct counts |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        values = [int(float(row["param_count"])) for row in rows if row["model"] == model]
        median = statistics.median(values)
        median_text = f"{int(median):,}" if float(median).is_integer() else f"{median:,.1f}"
        label = f"**{model}**" if model == OURS else model
        lines.append(f"| {label} | {format_params(values)} | {median_text} | {len(set(values))} |")

    lines.extend(
        [
            "",
            "## Maintenance contract",
            "",
            "- Single source of truth: the normalized CSV below. Do not hand-edit numeric table cells.",
            "- Re-running the generator validates full coverage, seed, status, look-back lengths, horizons, and finite metrics before replacing this file.",
            "- Solar-Energy is not part of the current locked 11-dataset result bundle, so no Solar values are fabricated here.",
            "",
            "```bash",
            "python analysis_exp/build_tqnet_table5_style_results.py \\",
            f"  --input-csv {input_csv.name} \\",
            f"  --output-md {output_md.name}",
            "```",
            "",
            f"- Input CSV: `{input_csv.name}`",
            f"- Input CSV SHA256: `{sha256(input_csv)}`",
            "- Ours protocol: frozen seed-2026 replay of a seed-2025 test-selected profile.",
            "- PhaseFormer protocol: validation-selected checkpoint, followed by one deferred held-out test.",
            "- Other baselines: locked local seed-2026 reproduction/completion results.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    rows = read_and_validate(args.input_csv)
    report = render(rows, args.input_csv, args.output_md)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"wrote={args.output_md} rows={len(rows)}")
    print(f"sha256={sha256(args.output_md)}")


if __name__ == "__main__":
    main()
