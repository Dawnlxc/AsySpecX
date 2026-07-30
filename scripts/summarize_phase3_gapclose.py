#!/usr/bin/env python3
"""Summarize AsySpecX Phase 3-GapClose results."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return "" if v is None else f"{v:.6g}"


def key(row):
    return (row.get("dataset", ""), row.get("seq_len", ""), row.get("pred_len", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="phase3_gapclose_results/main")
    parser.add_argument("--csv", default="")
    parser.add_argument("--baseline_csv", default="")
    args = parser.parse_args()
    root = Path(args.root)
    csv_path = Path(args.csv) if args.csv else root / "results.csv"
    rows = [r for r in read_csv(csv_path) if r.get("status", "ok") == "ok"]
    root.mkdir(parents=True, exist_ok=True)

    by_arm = defaultdict(list)
    by_cell = defaultdict(list)
    for row in rows:
        by_arm[row.get("arm", "")].append(row)
        by_cell[key(row)].append(row)

    best_rows = []
    for cell, group in sorted(by_cell.items()):
        candidates = [r for r in group if r.get("mse", "") != ""]
        if candidates:
            best_rows.append(min(candidates, key=lambda r: float(r["mse"])))

    best_count = Counter(r.get("arm", "") for r in best_rows)
    anchor = "phase3_anchor_hier_split"
    anchor_by_cell = {}
    for row in rows:
        if row.get("arm") == anchor and row.get("mse", "") != "":
            anchor_by_cell.setdefault(key(row), []).append(row)

    lines = ["# Phase 3-GapClose Summary", ""]
    lines.append(f"- total_ok_runs: {len(rows)}")
    lines.append(f"- csv: {csv_path}")
    lines.append("")
    lines.append("## Arm Means")
    lines.append("")
    lines.append("| arm | n | mse_mean | mae_mean | val_mse_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for arm in sorted(by_arm):
        group = by_arm[arm]
        lines.append(
            f"| {arm} | {len(group)} | {fmt(mean(r.get('mse') for r in group))} | "
            f"{fmt(mean(r.get('mae') for r in group))} | {fmt(mean(r.get('val_mse') for r in group))} |"
        )
    lines.append("")
    lines.append("## Best Arm Per Dataset/Length")
    lines.append("")
    lines.append("| dataset | seq_len | pred_len | best_arm | mse | mae |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for row in best_rows:
        lines.append(
            f"| {row.get('dataset','')} | {row.get('seq_len','')} | {row.get('pred_len','')} | "
            f"{row.get('arm','')} | {row.get('mse','')} | {row.get('mae','')} |"
        )
    lines.append("")
    lines.append("## Best Cell Count")
    lines.append("")
    lines.append("| arm | n |")
    lines.append("| --- | ---: |")
    for arm, n in sorted(best_count.items()):
        lines.append(f"| {arm} | {n} |")
    lines.append("")
    lines.append("## Delta Versus Anchor")
    lines.append("")
    lines.append("| arm | cells | delta_mse_mean | delta_mae_mean |")
    lines.append("| --- | ---: | ---: | ---: |")
    for arm in sorted(by_arm):
        if arm == anchor:
            continue
        dm, da = [], []
        for row in by_arm[arm]:
            anchors = anchor_by_cell.get(key(row), [])
            if not anchors or row.get("mse", "") == "":
                continue
            a_mse = mean(a.get("mse") for a in anchors)
            a_mae = mean(a.get("mae") for a in anchors)
            if a_mse is not None:
                dm.append(float(row["mse"]) - a_mse)
            if a_mae is not None and row.get("mae", "") != "":
                da.append(float(row["mae"]) - a_mae)
        lines.append(f"| {arm} | {len(dm)} | {fmt(mean(dm))} | {fmt(mean(da))} |")

    if args.baseline_csv:
        baselines = read_csv(args.baseline_csv)
        baseline_by_key = defaultdict(list)
        for row in baselines:
            baseline_by_key[(row.get("dataset", ""), row.get("pred_len", ""), row.get("model", ""))].append(row)
        lines.append("")
        lines.append("## External Baseline Gap")
        lines.append("")
        lines.append("| dataset | pred_len | baseline | best_mse | baseline_mse | gap_abs | gap_pct |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
        for best in best_rows:
            for model in ("FITS", "PatchTST", "SparseTSF"):
                matches = baseline_by_key.get((best.get("dataset", ""), best.get("pred_len", ""), model), [])
                if not matches or best.get("mse", "") == "":
                    continue
                b = mean(r.get("mse") for r in matches)
                bm = float(best["mse"])
                gap = bm - b
                lines.append(
                    f"| {best.get('dataset','')} | {best.get('pred_len','')} | {model} | "
                    f"{bm:.6g} | {b:.6g} | {gap:.6g} | {100*gap/b:.3g}% |"
                )

    lines.append("")
    lines.append("## Dataset Notes")
    lines.append("")
    for dataset in ("weather", "electricity"):
        group = [r for r in rows if r.get("dataset") == dataset]
        if not group:
            continue
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(f"- runs: {len(group)}")
        lines.append(f"- mse_mean: {fmt(mean(r.get('mse') for r in group))}")
        lines.append(f"- mae_mean: {fmt(mean(r.get('mae') for r in group))}")
        lines.append("")

    out = root / "summary_phase3_gapclose.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"summary={out}")


if __name__ == "__main__":
    main()
