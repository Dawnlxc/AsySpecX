#!/usr/bin/env python3
"""Aggregate AsySpecX phase-1 run_summary.json files."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "arm",
    "dataset",
    "seq_len",
    "pred_len",
    "seed",
    "status",
    "val_mse",
    "mse",
    "mae",
    "cut_freq",
    "periods",
    "exit_code",
    "job_id",
    "log_file",
]


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def load_rows(root: Path):
    rows = []
    for path in sorted(root.rglob("run_summary.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception as exc:
            row = {
                "arm": "",
                "dataset": "",
                "seq_len": "",
                "pred_len": "",
                "seed": "",
                "status": "failed",
                "mse": None,
                "mae": None,
                "exit_code": "",
                "job_id": "",
                "log_file": "",
                "error": f"cannot read {path}: {exc}",
            }
        row["_summary_path"] = str(path)
        rows.append(row)
    return rows


def write_csv(root: Path, rows):
    csv_path = root / "results.csv"
    # Dynamic extra columns (e.g. Phase 5 val_mse_seg*/val_mae_seg*, diagnostics)
    # are appended after the fixed FIELDS so old summaries keep working.
    extra = []
    seen = set(FIELDS) | {"_summary_path", "error"}
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                extra.append(k)
    extra.sort()
    fieldnames = FIELDS + extra
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return csv_path


def write_summary(root: Path, rows):
    summary_path = root / "summary.md"
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]

    by_arm = defaultdict(list)
    by_cell = defaultdict(list)
    for row in ok:
        by_arm[row.get("arm", "")].append(row)
        key = (row.get("dataset", ""), row.get("seq_len", ""), row.get("pred_len", ""))
        by_cell[key].append(row)

    root_name = str(root)
    if "phase3" in root_name:
        title = "AsySpecX Phase 3-GapClose Summary"
    elif "phase2" in root_name:
        title = "AsySpecX Phase 2 Summary"
    else:
        title = "AsySpecX Phase 1 Summary"
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- total_runs: {len(rows)}")
    lines.append(f"- ok_runs: {len(ok)}")
    lines.append(f"- failed_runs: {len(failed)}")
    lines.append(f"- results_csv: {root / 'results.csv'}")
    lines.append("")

    lines.append("## Arm Means")
    lines.append("")
    lines.append("| arm | n | val_mse_mean | mse_mean | mae_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for arm in sorted(by_arm):
        group = by_arm[arm]
        lines.append(
            f"| {arm} | {len(group)} | {_fmt(_mean([r.get('val_mse') for r in group]))} | "
            f"{_fmt(_mean([r.get('mse') for r in group]))} | "
            f"{_fmt(_mean([r.get('mae') for r in group]))} |"
        )
    lines.append("")

    lines.append("## Best Arm Per Dataset/Length")
    lines.append("")
    lines.append("| dataset | seq_len | pred_len | best_arm | mse | mae |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for key in sorted(by_cell):
        group = by_cell[key]
        arm_scores = defaultdict(lambda: {"mse": [], "mae": []})
        for row in group:
            arm_scores[row.get("arm", "")]["mse"].append(row.get("mse"))
            arm_scores[row.get("arm", "")]["mae"].append(row.get("mae"))
        best_arm = None
        best_mse = None
        best_mae = None
        for arm, scores in arm_scores.items():
            mse = _mean(scores["mse"])
            mae = _mean(scores["mae"])
            if mse is not None and (best_mse is None or mse < best_mse):
                best_arm = arm
                best_mse = mse
                best_mae = mae
        dataset, seq_len, pred_len = key
        lines.append(f"| {dataset} | {seq_len} | {pred_len} | {best_arm} | {_fmt(best_mse)} | {_fmt(best_mae)} |")
    lines.append("")

    if failed:
        lines.append("## Failed Runs")
        lines.append("")
        lines.append("| arm | dataset | seq_len | pred_len | seed | exit_code | log_file |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for row in failed[:200]:
            lines.append(
                f"| {row.get('arm', '')} | {row.get('dataset', '')} | {row.get('seq_len', '')} | "
                f"{row.get('pred_len', '')} | {row.get('seed', '')} | {row.get('exit_code', '')} | "
                f"{row.get('log_file', '')} |"
            )
        if len(failed) > 200:
            lines.append(f"| ... | ... | ... | ... | ... | ... | {len(failed) - 200} more |")
        lines.append("")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return summary_path


def _fmt(value):
    if value is None:
        return ""
    return f"{float(value):.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="phase1_results/main")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    csv_path = write_csv(root, rows)
    summary_path = write_summary(root, rows)
    failed = sum(1 for row in rows if row.get("status") != "ok")
    print(f"runs={len(rows)} failed={failed} csv={csv_path} summary={summary_path}")


if __name__ == "__main__":
    main()
