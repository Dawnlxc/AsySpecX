#!/usr/bin/env python3
"""Phase 6-Protocol selector audit.

Compares several validation-selected CSVs against each other, against the best
fixed single-arm, and against a TEST oracle upper bound (analysis only). The
oracle must NEVER be reported as a valid selected model.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return "" if v is None else f"{v:.6g}"


def cell(r):
    return (r.get("dataset", ""), r.get("seq_len", ""), r.get("pred_len", ""))


def resolve(path, output_dir):
    if os.path.isfile(path):
        return path
    alt = os.path.join(output_dir, os.path.basename(path.strip()))
    return alt if os.path.isfile(alt) else None


def selector_name(path):
    base = os.path.basename(path)
    base = base[:-4] if base.endswith(".csv") else base
    return base[len("selected_"):] if base.startswith("selected_") else base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--selected_files", required=True,
                        help="comma-separated selected csv paths or basenames")
    parser.add_argument("--output_dir", default="phase5_results/main")
    args = parser.parse_args()

    raw = [r for r in read_rows(args.csv) if r.get("status", "ok") == "ok"]
    by_arm = defaultdict(list)
    by_cell = defaultdict(list)
    for r in raw:
        by_arm[r.get("arm", "")].append(r)
        by_cell[cell(r)].append(r)

    # best fixed single-arm (by mean test mse over all runs)
    arm_mse = {a: mean(to_float(r.get("mse")) for r in g) for a, g in by_arm.items()}
    arm_mse = {a: v for a, v in arm_mse.items() if v is not None}
    best_single_arm = min(arm_mse, key=arm_mse.get) if arm_mse else None
    best_single_mse = arm_mse.get(best_single_arm)
    best_single_mae = mean(to_float(r.get("mae")) for r in by_arm.get(best_single_arm, []))

    # TEST oracle per group (ANALYSIS ONLY)
    oracle = {}
    for key, g in by_cell.items():
        scores = defaultdict(list)
        for r in g:
            scores[r.get("arm", "")].append(r)
        best_a, best_m, best_ma = None, None, None
        for a, rs in scores.items():
            m = mean(to_float(r.get("mse")) for r in rs)
            if m is not None and (best_m is None or m < best_m):
                best_a, best_m, best_ma = a, m, mean(to_float(r.get("mae")) for r in rs)
        oracle[key] = (best_a, best_m, best_ma)
    oracle_mse = mean(v[1] for v in oracle.values())
    oracle_mae = mean(v[2] for v in oracle.values())

    # load selectors
    selectors = []
    for raw_path in args.selected_files.split(","):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        p = resolve(raw_path, args.output_dir)
        if p is None:
            print(f"[warn] selected file not found, skipping: {raw_path}")
            continue
        rows = read_rows(p)
        selectors.append((selector_name(p), p, rows))

    fixed_best_by_cell = {k: best_single_arm for k in by_cell}  # fixed arm same everywhere

    # per-selector aggregates + group details
    group_detail_rows = []
    selector_summ = []
    for name, path, rows in selectors:
        mse_mean = mean(to_float(r.get("test_mse")) for r in rows)
        mae_mean = mean(to_float(r.get("test_mae")) for r in rows)
        # one representative per group for arm counts
        by_g = defaultdict(list)
        for r in rows:
            by_g[cell(r)].append(r)
        counts = Counter(g[0].get("selected_arm", "") for g in by_g.values())
        selector_summ.append({
            "selector": name,
            "mse_mean": mse_mean,
            "mae_mean": mae_mean,
            "delta_vs_best_single_mse": (mse_mean - best_single_mse) if (mse_mean is not None and best_single_mse is not None) else None,
            "delta_vs_oracle_mse": (mse_mean - oracle_mse) if (mse_mean is not None and oracle_mse is not None) else None,
            "arm_counts": dict(sorted(counts.items())),
        })
        for key, g in sorted(by_g.items()):
            ds, sl, pl = key
            orc = oracle.get(key, (None, None, None))
            group_detail_rows.append({
                "dataset": ds, "seq_len": sl, "pred_len": pl, "selector": name,
                "selected_arm": g[0].get("selected_arm", ""),
                "test_mse": fmt(mean(to_float(r.get("test_mse")) for r in g)),
                "test_mae": fmt(mean(to_float(r.get("test_mae")) for r in g)),
                "oracle_arm": orc[0], "oracle_mse": fmt(orc[1]),
                "fixed_best_arm": best_single_arm,
            })

    os.makedirs(args.output_dir, exist_ok=True)

    # selector_audit.csv
    with open(os.path.join(args.output_dir, "selector_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["selector", "mse_mean", "mae_mean", "delta_vs_best_single_mse", "delta_vs_oracle_mse", "arm_counts"])
        for s in selector_summ:
            w.writerow([s["selector"], fmt(s["mse_mean"]), fmt(s["mae_mean"]),
                        fmt(s["delta_vs_best_single_mse"]), fmt(s["delta_vs_oracle_mse"]),
                        ";".join(f"{k}:{v}" for k, v in s["arm_counts"].items())])

    # selector_group_details.csv
    gd_fields = ["dataset", "seq_len", "pred_len", "selector", "selected_arm",
                 "test_mse", "test_mae", "oracle_arm", "oracle_mse", "fixed_best_arm"]
    with open(os.path.join(args.output_dir, "selector_group_details.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gd_fields)
        w.writeheader()
        w.writerows(group_detail_rows)

    # selector_audit.md
    L = ["# Phase 6-Protocol Selector Audit", ""]
    L.append("Oracle is analysis only and must not be reported as a valid selected model.")
    L.append("")
    L.append(f"- best_fixed_single_arm: {best_single_arm} (mse_mean={fmt(best_single_mse)}, mae_mean={fmt(best_single_mae)})")
    L.append(f"- test_oracle_mse_mean (ANALYSIS ONLY): {fmt(oracle_mse)}")
    L.append(f"- test_oracle_mae_mean (ANALYSIS ONLY): {fmt(oracle_mae)}")
    L.append("")
    L.append("## Selector Comparison")
    L.append("")
    L.append("| selector | mse_mean | mae_mean | delta_vs_best_single_mse | delta_vs_oracle_mse | selected_arm_counts |")
    L.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for s in selector_summ:
        L.append(f"| {s['selector']} | {fmt(s['mse_mean'])} | {fmt(s['mae_mean'])} | "
                 f"{fmt(s['delta_vs_best_single_mse'])} | {fmt(s['delta_vs_oracle_mse'])} | "
                 f"{'; '.join(f'{k}:{v}' for k, v in s['arm_counts'].items())} |")
    L.append("")
    L.append("## Weather / Electricity Detail")
    L.append("")
    L.append("| dataset | pred_len | selector | selected_arm | test_mse | test_mae | oracle_arm | oracle_mse | fixed_best_arm |")
    L.append("| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |")
    for r in group_detail_rows:
        if r["dataset"] not in ("weather", "electricity"):
            continue
        L.append(f"| {r['dataset']} | {r['pred_len']} | {r['selector']} | {r['selected_arm']} | "
                 f"{r['test_mse']} | {r['test_mae']} | {r['oracle_arm']} | {r['oracle_mse']} | {r['fixed_best_arm']} |")
    L.append("")
    L.append("## Fairness Note")
    L.append("")
    L.append("Selectors use validation metrics aggregated over seeds. Test metrics "
             "(and the oracle) are shown only after selection, for analysis.")

    with open(os.path.join(args.output_dir, "selector_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"selectors={len(selectors)} best_single={best_single_arm} "
          f"oracle_mse={fmt(oracle_mse)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
