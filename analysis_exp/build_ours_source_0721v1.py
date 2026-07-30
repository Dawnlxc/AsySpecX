#!/usr/bin/env python3
"""Build the 0721v1 88-cell Compact + Echo source for the all-baselines table.

Composition rule (recorded in the output's `source` column, never relabelled):

1. weather / electricity (16 cells): unchanged 0717v2 production-route rows
   (validation-locked routing; arms asy1_echo / compact_echo / fallback).
2. the nine former fallback datasets (72 cells): the compact_true9_0719v1
   three-seed rerun with an actually enabled AsyCycleCompact
   (arm=true_compact_echo, fixed research profile, seeds 2024/2025/2026).

The script refuses to write a partial source: it requires exactly 16 + 72
unique (dataset, seq_len, pred_len) keys with no overlap.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROD_KEEP_DATASETS = {"weather", "electricity"}
TRUE9_DATASETS = {
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "traffic",
    "PEMS03", "PEMS04", "PEMS07", "PEMS08",
}
OUT_FIELDS = [
    "dataset", "seq_len", "pred_len", "arm", "seeds", "n_seeds",
    "test_mse_mean", "test_mse_std", "test_mae_mean", "test_mae_std",
    "mape_mean", "mape_std", "param_count", "train_time_mean_s",
    "test_peak_gpu_mem_mean_mb", "our_compact_profile", "source",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production-csv",
        type=Path,
        default=here / "source_snapshot_0717v2" / "compact_echo_production_88cells.csv",
    )
    parser.add_argument(
        "--true9-csv",
        type=Path,
        default=Path(
            "/scratch3/lin250/bldgFM/DUBABA/asy/outputs/compact_true9/"
            "compact_true9_0719v1_20260719T023045/COMPACT_TRUE9_0719V1_SUMMARY_3SEED.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=here / "source_snapshot_0721v1" / "compact_echo_true9_88cells.csv",
    )
    args = parser.parse_args()

    out_rows: dict[tuple[str, int, int], dict[str, str]] = {}

    for row in read_rows(args.production_csv):
        if row["dataset"] not in PROD_KEEP_DATASETS:
            continue
        key = (row["dataset"], int(row["seq_len"]), int(row["pred_len"]))
        if key in out_rows:
            raise AssertionError(f"duplicate production row {key}")
        out_rows[key] = {field: row.get(field, "") for field in OUT_FIELDS}
    if len(out_rows) != 16:
        raise AssertionError(f"expected 16 weather/electricity production cells, got {len(out_rows)}")

    true9_count = 0
    for row in read_rows(args.true9_csv):
        dataset = row["dataset"]
        if dataset not in TRUE9_DATASETS:
            raise AssertionError(f"unexpected true9 dataset {dataset}")
        if row["arm"] != "true_compact_echo":
            raise AssertionError(f"unexpected true9 arm {row['arm']} for {dataset}")
        if int(row["n_seeds"]) != 3:
            raise AssertionError(f"true9 cell {dataset} is not a full three-seed mean")
        key = (dataset, int(row["seq_len"]), int(row["pred_len"]))
        if key in out_rows:
            raise AssertionError(f"true9 row overlaps existing cell {key}")
        out = {field: row.get(field, "") for field in OUT_FIELDS}
        out["our_compact_profile"] = row.get("profile", "")
        out["source"] = row.get("source", "compact_true9_0719v1")
        out_rows[key] = out
        true9_count += 1
    if true9_count != 72:
        raise AssertionError(f"expected 72 true9 cells, got {true9_count}")
    if len(out_rows) != 88:
        raise AssertionError(f"expected 88 total cells, got {len(out_rows)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for key in sorted(out_rows):
            writer.writerow(out_rows[key])
    print(f"wrote {args.output_csv} (88 rows: 16 production + 72 true_compact_echo)")


if __name__ == "__main__":
    main()
