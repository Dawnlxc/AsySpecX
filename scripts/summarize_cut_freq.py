#!/usr/bin/env python3
"""Cut_freq diagnostics summary for AsySpecX Phase 4.

Per dataset/pred_len/arm/cut_freq: val_mse / mse / mae means (+ low_freq_energy
_ratio if present). Reports best cut_freq by validation (for selection) and by
test (analysis only -- clearly marked NOT for selection).
"""

import argparse
import csv
from collections import defaultdict


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return "" if v is None else f"{v:.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = [r for r in read_rows(args.csv) if r.get("status", "ok") == "ok"]
    header = rows[0].keys() if rows else []
    has_val = "val_mse" in header
    has_lfe = "low_freq_energy_ratio" in header
    if "cut_freq" not in header:
        raise SystemExit("csv has no cut_freq column")

    cells = defaultdict(list)  # (dataset,pred_len,arm,cut_freq) -> rows
    for r in rows:
        key = (r.get("dataset", ""), r.get("pred_len", ""), r.get("arm", ""), r.get("cut_freq", ""))
        cells[key].append(r)

    lines = ["# Phase 4-Finalize Cut_freq Diagnostics", ""]
    lines.append("Best cut_freq **by validation** is the selection-safe choice. "
                 "Best-by-test is shown for analysis only and must NOT drive selection.")
    lines.append("")
    cols = "| dataset | pred_len | arm | cut_freq | n | val_mse | mse | mae |"
    sep = "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"
    if has_lfe:
        cols = cols[:-1] + " low_freq_energy_ratio |"
        sep = sep[:-1] + " ---: |"
    lines.append("## Per cut_freq")
    lines.append("")
    lines.append(cols)
    lines.append(sep)
    for key in sorted(cells):
        group = cells[key]
        ds, pl, arm, cf = key
        row = (f"| {ds} | {pl} | {arm} | {cf} | {len(group)} | "
               f"{fmt(mean(r.get('val_mse') for r in group)) if has_val else ''} | "
               f"{fmt(mean(r.get('mse') for r in group))} | {fmt(mean(r.get('mae') for r in group))} |")
        if has_lfe:
            row = row[:-1] + f" {fmt(mean(r.get('low_freq_energy_ratio') for r in group))} |"
        lines.append(row)
    lines.append("")

    # best cut_freq per dataset/pred_len/arm
    by_group = defaultdict(dict)  # (ds,pl,arm) -> {cf: {val,mse,mae}}
    for key, group in cells.items():
        ds, pl, arm, cf = key
        by_group[(ds, pl, arm)][cf] = {
            "val": mean(r.get("val_mse") for r in group) if has_val else None,
            "mse": mean(r.get("mse") for r in group),
        }

    lines.append("## Best cut_freq by validation (selection-safe)")
    lines.append("")
    lines.append("| dataset | pred_len | arm | best_cut_freq_val | val_mse |")
    lines.append("| --- | ---: | --- | ---: | ---: |")
    for (ds, pl, arm), cfs in sorted(by_group.items()):
        if not has_val:
            continue
        valid = {cf: d["val"] for cf, d in cfs.items() if d["val"] is not None}
        if not valid:
            continue
        best = min(valid, key=lambda c: valid[c])
        lines.append(f"| {ds} | {pl} | {arm} | {best} | {fmt(valid[best])} |")
    if not has_val:
        lines.append("| (val_mse absent -- validation-best unavailable) | | | | |")
    lines.append("")

    lines.append("## Best cut_freq by test (ANALYSIS ONLY -- not for selection)")
    lines.append("")
    lines.append("| dataset | pred_len | arm | best_cut_freq_test | test_mse |")
    lines.append("| --- | ---: | --- | ---: | ---: |")
    for (ds, pl, arm), cfs in sorted(by_group.items()):
        valid = {cf: d["mse"] for cf, d in cfs.items() if d["mse"] is not None}
        if not valid:
            continue
        best = min(valid, key=lambda c: valid[c])
        lines.append(f"| {ds} | {pl} | {arm} | {best} | {fmt(valid[best])} |")
    lines.append("")

    out = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"summary={args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
