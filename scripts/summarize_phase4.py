#!/usr/bin/env python3
"""AsySpecX Phase 4-Finalize summary.

Combines the raw results.csv with the validation-selected results.csv and (opt)
external baselines. Selection is fair: test metrics are only reported after arms
were chosen by mean val_mse over replicate seeds.
"""

import argparse
import csv
from collections import Counter, defaultdict

CANDIDATES = [
    "phase4_asx_cross",
    "phase4_asx_individual",
    "phase4_asx_period_single",
    "phase4_asx_period_multi",
    "phase4_asx_individual_period",
]
BASELINES = ["FITS", "PatchTST", "SparseTSF"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return "" if v is None else f"{v:.6g}"


def cell_key(row):
    return (row.get("dataset", ""), row.get("seq_len", ""), row.get("pred_len", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--selected_csv", default="")
    parser.add_argument("--baseline_csv", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = [r for r in read_rows(args.csv) if r.get("status", "ok") == "ok"]
    by_arm = defaultdict(list)
    by_cell = defaultdict(list)
    for r in rows:
        by_arm[r.get("arm", "")].append(r)
        by_cell[cell_key(r)].append(r)

    lines = ["# Phase 4-Finalize Summary", ""]
    lines.append("Validation selection is performed using val_mse averaged over "
                 "replicate seeds for each dataset/seq_len/pred_len group. Test "
                 "metrics are used only after selection.")
    lines.append("")
    lines.append(f"- total_ok_runs: {len(rows)}")
    lines.append(f"- csv: {args.csv}")
    lines.append("")

    # 1. arm means
    lines.append("## Arm Means")
    lines.append("")
    lines.append("| arm | n | mse_mean | mae_mean | val_mse_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for arm in sorted(by_arm):
        g = by_arm[arm]
        lines.append(f"| {arm} | {len(g)} | {fmt(mean(r.get('mse') for r in g))} | "
                     f"{fmt(mean(r.get('mae') for r in g))} | {fmt(mean(r.get('val_mse') for r in g))} |")
    lines.append("")

    # 2. best arm per cell by test (analysis only)
    lines.append("## Best Arm Per Dataset/Seq_len/Pred_len BY TEST (analysis only -- not for selection)")
    lines.append("")
    lines.append("| dataset | seq_len | pred_len | best_arm | mse_mean | mae_mean |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for key in sorted(by_cell):
        arm_scores = defaultdict(list)
        for r in by_cell[key]:
            arm_scores[r.get("arm", "")].append(r)
        best_arm, best_mse, best_mae = None, None, None
        for arm, arm_rows in arm_scores.items():
            m = mean(r.get("mse") for r in arm_rows)
            if m is not None and (best_mse is None or m < best_mse):
                best_arm, best_mse, best_mae = arm, m, mean(r.get("mae") for r in arm_rows)
        ds, sl, pl = key
        lines.append(f"| {ds} | {sl} | {pl} | {best_arm} | {fmt(best_mse)} | {fmt(best_mae)} |")
    lines.append("")

    # 3. validation-selected summary
    selected = read_rows(args.selected_csv) if args.selected_csv else []
    sel_by_cell = defaultdict(list)
    for r in selected:
        sel_by_cell[cell_key(r)].append(r)
    if selected:
        lines.append("## Validation-Selected Summary")
        lines.append("")
        lines.append(f"- selected_test_mse_mean: {fmt(mean(r.get('test_mse') for r in selected))}")
        lines.append(f"- selected_test_mae_mean: {fmt(mean(r.get('test_mae') for r in selected))}")
        lines.append("")
        counts = Counter(r.get("selected_arm", "") for r in sel_by_cell_first(sel_by_cell))
        lines.append("### Selected Arm Counts (per group)")
        lines.append("")
        lines.append("| arm | groups |")
        lines.append("| --- | ---: |")
        for arm, n in sorted(counts.items()):
            lines.append(f"| {arm} | {n} |")
        lines.append("")
        lines.append("### Selected Arm Per Dataset/Pred_len")
        lines.append("")
        lines.append("| dataset | pred_len | selected_arm | test_mse_mean | test_mae_mean |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        by_dp = defaultdict(list)
        for r in selected:
            by_dp[(r.get("dataset", ""), r.get("pred_len", ""))].append(r)
        for (ds, pl), g in sorted(by_dp.items()):
            arm = g[0].get("selected_arm", "")
            lines.append(f"| {ds} | {pl} | {arm} | {fmt(mean(r.get('test_mse') for r in g))} | "
                         f"{fmt(mean(r.get('test_mae') for r in g))} |")
        lines.append("")
        lines.append("### Per Dataset Selected")
        lines.append("")
        lines.append("| dataset | test_mse_mean | test_mae_mean |")
        lines.append("| --- | ---: | ---: |")
        by_ds = defaultdict(list)
        for r in selected:
            by_ds[r.get("dataset", "")].append(r)
        for ds, g in sorted(by_ds.items()):
            lines.append(f"| {ds} | {fmt(mean(r.get('test_mse') for r in g))} | "
                         f"{fmt(mean(r.get('test_mae') for r in g))} |")
        lines.append("")

    # 4. external baseline comparison against selected
    if args.baseline_csv and selected:
        baselines = read_rows(args.baseline_csv)
        bkey = defaultdict(list)
        for r in baselines:
            bkey[(r.get("dataset", ""), r.get("pred_len", ""), r.get("model", ""))].append(r)
        # selected mse mean per (dataset,pred_len)
        sel_mse = defaultdict(list)
        for r in selected:
            sel_mse[(r.get("dataset", ""), r.get("pred_len", ""))].append(r)
        for model in BASELINES:
            lines.append(f"## Selected AsySpecX vs {model}")
            lines.append("")
            lines.append("| dataset | pred_len | asx_mse | base_mse | gap_abs | gap_pct | verdict |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
            wins = losses = 0
            for (ds, pl), g in sorted(sel_mse.items()):
                matches = bkey.get((ds, pl, model), [])
                asx = mean(r.get("test_mse") for r in g)
                base = mean(r.get("mse") for r in matches)
                if asx is None or base is None:
                    continue
                gap = asx - base
                verdict = "win" if gap < 0 else ("tie" if gap == 0 else "loss")
                wins += gap < 0
                losses += gap > 0
                lines.append(f"| {ds} | {pl} | {asx:.6g} | {base:.6g} | {gap:.6g} | "
                             f"{100*gap/base:.3g}% | {verdict} |")
            lines.append("")
            lines.append(f"- vs {model}: wins={wins} losses={losses}")
            lines.append("")

    # 5. single-arm candidate summary
    lines.append("## Single-Arm Candidate Summary")
    lines.append("")
    lines.append("| arm | n | mse_mean | mae_mean | val_mse_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for arm in CANDIDATES:
        g = by_arm.get(arm, [])
        if not g:
            lines.append(f"| {arm} | 0 | | | |")
            continue
        lines.append(f"| {arm} | {len(g)} | {fmt(mean(r.get('mse') for r in g))} | "
                     f"{fmt(mean(r.get('mae') for r in g))} | {fmt(mean(r.get('val_mse') for r in g))} |")
    lines.append("")

    # 6. fairness note
    lines.append("## Fairness Note")
    lines.append("")
    lines.append("Validation selection is performed using val_mse averaged over "
                 "replicate seeds for each dataset/seq_len/pred_len group. Test "
                 "metrics are used only after selection.")
    lines.append("")

    out = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"summary={args.output}")
    else:
        print(out)


def sel_by_cell_first(sel_by_cell):
    """One representative selected row per group (they share selected_arm)."""
    return [g[0] for g in sel_by_cell.values() if g]


if __name__ == "__main__":
    main()
