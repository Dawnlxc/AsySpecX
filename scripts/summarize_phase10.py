#!/usr/bin/env python3
"""Aggregate Phase 10 run summaries without selecting on test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PUBLISHED_BASELINES_SL96 = {
    "weather": {96: 0.1583, 192: 0.2060, 336: 0.2634, 720: 0.3387},
    "electricity": {96: 0.1375, 192: 0.1555, 336: 0.1723, 720: 0.2089},
}
PHASE8_CURRENT = {
    ("weather", 96): {96: 0.177397, 192: 0.230566, 336: 0.284441, 720: 0.356700},
    ("electricity", 96): {96: 0.192454, 192: 0.194564, 336: 0.207282, 720: 0.250780},
    ("electricity", 720): {96: 0.136534, 192: 0.151307, 336: 0.166620, 720: 0.200907},
}


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(values) if values else None


def pop_std(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def fmt(value, digits=6):
    return "" if value is None else f"{value:.{digits}f}"


def load_rows(roots):
    rows = {}
    for root in roots:
        for path in sorted(Path(root).glob("**/run_summary.json")):
            payload = json.loads(path.read_text())
            payload["summary_file"] = str(path)
            key = tuple(payload.get(k) for k in (
                "arm", "dataset", "seq_len", "pred_len", "seed", "cut_freq"
            ))
            rows[key] = payload
    return list(rows.values())


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    rows = load_rows(args.roots)
    rows.sort(key=lambda r: (
        r.get("dataset", ""), int(r.get("seq_len", 0)), int(r.get("pred_len", 0)),
        r.get("arm", ""), int(r.get("cut_freq", 0)), int(r.get("seed", 0)),
    ))
    fields = [
        "run_tag", "arm", "dataset", "seq_len", "pred_len", "seed", "cut_freq",
        "cycle", "periods", "status", "exit_code", "job_id", "n_param", "val_mse",
        "mse", "mae", "t_train", "t_inf", "peak_cuda_mb", "log_file", "summary_file",
        "val_mse_seg0", "val_mse_seg1", "val_mse_seg2", "val_mse_seg3",
    ]
    write_csv(output / "results.csv", rows, fields)

    ok = [r for r in rows if r.get("status") == "ok" and r.get("val_mse") is not None]
    grouped = defaultdict(list)
    for row in ok:
        grouped[(row["dataset"], int(row["seq_len"]), row["arm"], int(row["cut_freq"]))].append(row)

    ranking = []
    for (dataset, seq_len, arm, cut_freq), group in grouped.items():
        horizons = sorted({int(r["pred_len"]) for r in group})
        seeds = sorted({int(r["seed"]) for r in group})
        screen_group = [r for r in group if int(r["pred_len"]) in {96, 336}]
        screen_horizons = sorted({int(r["pred_len"]) for r in screen_group})
        ranking.append({
            "dataset": dataset,
            "seq_len": seq_len,
            "arm": arm,
            "cut_freq": cut_freq,
            "runs": len(group),
            "horizons": "+".join(map(str, horizons)),
            "seeds": "+".join(map(str, seeds)),
            "screen_complete": int(screen_horizons == [96, 336]),
            "screen_runs": len(screen_group),
            "screen_val_mse_mean": mean(r.get("val_mse") for r in screen_group),
            "screen_test_mse_mean": mean(r.get("mse") for r in screen_group),
            "all_val_mse_mean": mean(r.get("val_mse") for r in group),
            "all_test_mse_mean": mean(r.get("mse") for r in group),
            "n_param_max": max(int(r.get("n_param") or 0) for r in group),
            "peak_cuda_mb_max": max(float(r.get("peak_cuda_mb") or 0.0) for r in group),
            "t_train_mean": mean(r.get("t_train") for r in group),
            "t_inf_mean": mean(r.get("t_inf") for r in group),
        })
    ranking.sort(key=lambda r: (
        r["dataset"], r["seq_len"], -r["screen_complete"],
        float("inf") if r["screen_val_mse_mean"] is None else r["screen_val_mse_mean"],
    ))
    write_csv(output / "config_ranking.csv", ranking, [
        "dataset", "seq_len", "arm", "cut_freq", "runs", "horizons", "seeds",
        "screen_complete", "screen_runs", "screen_val_mse_mean", "screen_test_mse_mean",
        "all_val_mse_mean", "all_test_mse_mean", "n_param_max", "peak_cuda_mb_max",
        "t_train_mean", "t_inf_mean",
    ])

    cell_groups = defaultdict(list)
    for row in ok:
        cell_groups[(row["dataset"], int(row["seq_len"]), int(row["pred_len"]), row["arm"], int(row["cut_freq"]))].append(row)
    selected = []
    by_cell = defaultdict(list)
    for (dataset, seq_len, pred_len, arm, cut_freq), group in cell_groups.items():
        baseline = PUBLISHED_BASELINES_SL96.get(dataset, {}).get(pred_len)
        test_values = [r.get("mse") for r in group if r.get("mse") is not None]
        by_cell[(dataset, seq_len, pred_len)].append({
            "dataset": dataset,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "arm": arm,
            "cut_freq": cut_freq,
            "seeds": len({int(r["seed"]) for r in group}),
            "val_mse": mean(r.get("val_mse") for r in group),
            "mse": mean(r.get("mse") for r in group),
            "mse_std": pop_std(test_values),
            "wins_vs_baseline": (
                None if baseline is None
                else sum(float(value) < baseline for value in test_values)
            ),
            "mae": mean(r.get("mae") for r in group),
            "n_param": max(int(r.get("n_param") or 0) for r in group),
            "peak_cuda_mb": max(float(r.get("peak_cuda_mb") or 0.0) for r in group),
        })
    for cell, candidates in by_cell.items():
        # Do not let a one-seed exploratory cell displace a replicated
        # confirmation merely through seed noise.  Prefer the largest available
        # seed count, then select by mean validation MSE only.
        max_seeds = max(candidate["seeds"] for candidate in candidates)
        confirmed = [
            candidate for candidate in candidates
            if candidate["seeds"] == max_seeds
        ]
        winner = min(confirmed, key=lambda r: r["val_mse"])
        # Published reference numbers all use seq_len=96.  We still display
        # them for longer-lookback experiments, but label that comparison
        # explicitly so it is not mistaken for an apples-to-apples protocol.
        baseline = PUBLISHED_BASELINES_SL96.get(winner["dataset"], {}).get(winner["pred_len"])
        old = PHASE8_CURRENT.get((winner["dataset"], winner["seq_len"]), {}).get(winner["pred_len"])
        winner["baseline_mse"] = baseline
        winner["delta_vs_baseline"] = None if baseline is None else winner["mse"] - baseline
        winner["phase8_mse"] = old
        winner["gain_vs_phase8"] = None if old is None else old - winner["mse"]
        selected.append(winner)
    selected.sort(key=lambda r: (r["dataset"], r["seq_len"], r["pred_len"]))
    write_csv(output / "selected_by_validation.csv", selected, [
        "dataset", "seq_len", "pred_len", "arm", "cut_freq", "seeds", "val_mse",
        "mse", "mse_std", "wins_vs_baseline", "mae", "n_param", "peak_cuda_mb",
        "baseline_mse", "delta_vs_baseline",
        "phase8_mse", "gain_vs_phase8",
    ])

    lines = [
        "# AsySpecX Phase 10 Summary", "",
        f"- discovered summaries: {len(rows)}", f"- ok: {len(ok)}",
        f"- failed_or_incomplete: {len(rows) - len(ok)}", "",
        "Selection first prefers the largest replicated seed count in each cell, then uses mean validation MSE only. Test MSE is read after selection.", "",
        "## Validation-selected cells", "",
        "| dataset | sl | pl | arm | cf | seeds | wins | val | test mean | test std | vs Phase8 | vs published sl96 baseline | params | peak MiB |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row['dataset']} | {row['seq_len']} | {row['pred_len']} | {row['arm']} | "
            f"{row['cut_freq']} | {row['seeds']} | {row['wins_vs_baseline']} | "
            f"{fmt(row['val_mse'])} | {fmt(row['mse'])} | {fmt(row['mse_std'])} | "
            f"{fmt(row['gain_vs_phase8'])} | {fmt(row['delta_vs_baseline'])} | "
            f"{row['n_param']} | {row['peak_cuda_mb']:.1f} |"
        )
    lines += ["", "## Fixed-config ranking (validation metric)", "",
              "Only the common screening horizons 96 and 336 are used for ranking; all-horizon means are descriptive.", "",
              "| dataset | sl | arm | cf | screen complete | screen runs | seeds | screen val | screen test | all horizons | all test | params max | peak MiB |",
              "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in ranking:
        lines.append(
            f"| {row['dataset']} | {row['seq_len']} | {row['arm']} | {row['cut_freq']} | "
            f"{row['screen_complete']} | {row['screen_runs']} | {row['seeds']} | "
            f"{fmt(row['screen_val_mse_mean'])} | {fmt(row['screen_test_mse_mean'])} | "
            f"{row['horizons']} | {fmt(row['all_test_mse_mean'])} | {row['n_param_max']} | "
            f"{row['peak_cuda_mb_max']:.1f} |"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary_phase10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(rows), "ok": len(ok), "failed_or_incomplete": len(rows) - len(ok),
        "output_dir": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
