#!/usr/bin/env python3
"""Fair, robust validation selection for AsySpecX (Phase 5-Lockdown).

For each selection group (default: dataset,seq_len,pred_len) an arm is chosen by
an aggregate VALIDATION score over replicate seeds, then ALL seed rows of the
selected arm are emitted with their held-out test metrics.

Robustness knobs (Phase 5):
- --metric_mode {mean, mean_plus_std, last_segment}
- --std_weight, --selection_margin_abs, --selection_margin_pct
- --prefer_arm_order  (tie/near-best resolver, e.g. prefer simpler/safer arms)
- --arm_allowlist_json  (per-dataset candidate pools)

Hard rules:
- Test mse/mae NEVER drive selection (unless --allow_test_selection is passed).
- select_metric must exist; last_segment requires the seg columns.
- Selection is aggregated over seeds, not per-seed.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value in ("", None):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def pstd(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def fmt(v):
    return "" if v is None else f"{v:.6g}"


def _seg_cols_sorted(header, base):
    """All seg columns for a base metric, sorted by index: val_mse_seg0.., seg1.."""
    pat = re.compile(rf"^{re.escape(base)}_seg(\d+)$")
    idxs = sorted((int(m.group(1)), c) for c in header for m in [pat.match(c)] if m)
    return [c for _, c in idxs]


def last_segment_col(header, base):
    """Highest-index seg column for a base metric, e.g. val_mse -> val_mse_seg3."""
    cols = _seg_cols_sorted(header, base)
    return cols[-1] if cols else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--selection_keys", default="dataset,seq_len,pred_len",
                        help="Recommended: dataset,seq_len,pred_len (aggregate over seeds).")
    parser.add_argument("--replicate_key", default="seed")
    parser.add_argument("--arm_key", default="arm")
    parser.add_argument("--group_keys", default="", help="Legacy alias for --selection_keys.")
    parser.add_argument("--select_metric", default="val_mse")
    parser.add_argument("--metric_mode", default="mean",
                        choices=["mean", "mean_plus_std", "last_segment", "segment_mean_plus_std"])
    parser.add_argument("--std_weight", type=float, default=0.0)
    parser.add_argument("--selection_margin_abs", type=float, default=0.0)
    parser.add_argument("--selection_margin_pct", type=float, default=0.0)
    parser.add_argument("--prefer_arm_order", default="")
    parser.add_argument("--arm_allowlist_json", default="")
    parser.add_argument("--test_metrics", default="mse,mae")
    parser.add_argument("--allow_test_selection", action="store_true",
                        help="Escape hatch to select by test metric; OFF by default.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("input csv has no rows")

    selection_keys = parse_list(args.group_keys) if args.group_keys else parse_list(args.selection_keys)
    replicate_key, arm_key = args.replicate_key, args.arm_key
    test_metrics = parse_list(args.test_metrics)
    header = list(rows[0].keys())

    if args.select_metric in {"mse", "mae", "test_mse", "test_mae"} and not args.allow_test_selection:
        raise SystemExit(f"refusing to select by test metric {args.select_metric!r}; "
                         f"pass --allow_test_selection to override")
    if args.select_metric not in header:
        raise SystemExit(f"select_metric {args.select_metric!r} missing from csv")
    for m in test_metrics:
        if m not in header:
            raise SystemExit(f"test metric {m!r} missing from csv")
    if arm_key not in header:
        raise SystemExit(f"arm key {arm_key!r} missing from csv")

    # Resolve the effective scoring column(s).
    seg_cols = None
    if args.metric_mode == "last_segment":
        metric_col = last_segment_col(header, args.select_metric)
        if metric_col is None:
            raise SystemExit(f"metric_mode=last_segment needs {args.select_metric}_seg* columns "
                             f"(run with --val_num_segments>1); none found")
    elif args.metric_mode == "segment_mean_plus_std":
        seg_cols = _seg_cols_sorted(header, args.select_metric)
        if not seg_cols:
            raise SystemExit(f"metric_mode=segment_mean_plus_std needs {args.select_metric}_seg* "
                             f"columns (run with --val_num_segments>1); none found")
        metric_col = args.select_metric  # display only
    else:
        metric_col = args.select_metric

    allowlist = {}
    if args.arm_allowlist_json:
        with open(args.arm_allowlist_json, encoding="utf-8") as f:
            allowlist = json.load(f)
    prefer_order = parse_list(args.prefer_arm_order)

    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    groups = defaultdict(lambda: defaultdict(list))
    for row in ok_rows:
        gkey = tuple(row.get(k, "") for k in selection_keys)
        groups[gkey][row.get(arm_key, "")].append(row)

    def allowed_arms_for(gkey):
        if not allowlist:
            return None  # no restriction
        dataset = dict(zip(selection_keys, gkey)).get("dataset", "")
        pool = allowlist.get(dataset, allowlist.get("default"))
        return set(pool) if pool is not None else None

    selected_rows = []
    group_records = []
    for gkey, arms in sorted(groups.items()):
        allowed = allowed_arms_for(gkey)
        scores = {}
        for arm, arm_rows in arms.items():
            if allowed is not None and arm not in allowed:
                continue
            if args.metric_mode == "segment_mean_plus_std":
                # Pool every segment metric across every seed, then penalize the
                # spread -> weather's high segment mismatch is punished.
                pool = [to_float(r.get(c, "")) for r in arm_rows for c in seg_cols]
                m = mean(pool)
                if m is None:
                    continue
                m = m + args.std_weight * pstd(pool)
            else:
                vals = [to_float(r.get(metric_col, "")) for r in arm_rows]
                m = mean(vals)
                if m is None:
                    continue
                if args.metric_mode == "mean_plus_std":
                    m = m + args.std_weight * pstd(vals)
            scores[arm] = m
        if not scores:
            raise SystemExit(f"group {gkey} has no usable arm for metric {metric_col!r} "
                             f"(allowlist={allowed})")

        best_arm = min(scores, key=lambda a: scores[a])
        best_score = scores[best_arm]
        # Near-best tolerance = the looser of the abs / pct margins so that
        # either margin alone admits arms (pct-only is the policy_family case).
        tol = best_score + max(args.selection_margin_abs, best_score * args.selection_margin_pct)
        near_best = [a for a, s in scores.items() if s <= tol]
        chosen_arm = best_arm
        if prefer_order:
            for pa in prefer_order:
                if pa in near_best:
                    chosen_arm = pa
                    break

        chosen = arms[chosen_arm]
        for r in chosen:
            out = {k: r.get(k, "") for k in selection_keys}
            out[replicate_key] = r.get(replicate_key, "")
            out["selected_arm"] = chosen_arm
            out["val_score"] = scores[chosen_arm]
            out[args.select_metric] = r.get(args.select_metric, "")
            for m in test_metrics:
                out[f"test_{m}"] = r.get(m, "")
            selected_rows.append(out)

        rec = dict(zip(selection_keys, gkey))
        rec["selected_arm"] = chosen_arm
        rec["mean_val_score"] = scores[chosen_arm]
        rec["raw_best_arm"] = best_arm
        rec["raw_best_score"] = best_score
        rec["near_best_arms"] = sorted(near_best, key=lambda a: scores[a])
        for m in test_metrics:
            rec[f"mean_test_{m}"] = mean(to_float(r.get(m, "")) for r in chosen)
        group_records.append(rec)

    out_fields = (list(selection_keys) + [replicate_key, "selected_arm", "val_score",
                  args.select_metric] + [f"test_{m}" for m in test_metrics])
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    # ---- summary ----
    arm_counts = Counter(rec["selected_arm"] for rec in group_records)
    sel_means = {m: mean(rec.get(f"mean_test_{m}") for rec in group_records) for m in test_metrics}
    by_dataset = defaultdict(list)
    by_predlen = defaultdict(list)
    for rec in group_records:
        by_dataset[rec.get("dataset", "")].append(rec)
        by_predlen[rec.get("pred_len", "")].append(rec)

    lines = ["# Phase 5-Lockdown Validation Selection Summary", ""]
    lines.append("Selection uses validation metrics aggregated over seeds. Test "
                 "metrics are reported only after arm selection.")
    lines.append("")
    lines.append(f"- selection_keys: {','.join(selection_keys)}")
    lines.append(f"- replicate_key: {replicate_key}")
    lines.append(f"- select_metric: {args.select_metric} (metric_mode={args.metric_mode}, col={metric_col})")
    lines.append(f"- std_weight: {args.std_weight}  margin_abs: {args.selection_margin_abs}  margin_pct: {args.selection_margin_pct}")
    lines.append(f"- prefer_arm_order: {args.prefer_arm_order or '(none)'}")
    lines.append(f"- arm_allowlist_json: {args.arm_allowlist_json or '(none)'}")
    lines.append(f"- selection_groups: {len(group_records)}")
    for m in test_metrics:
        lines.append(f"- selected_test_{m}_mean: {fmt(sel_means[m])}")
    lines.append("")

    lines.append("## Selected Arm Counts")
    lines.append("")
    lines.append("| arm | groups |")
    lines.append("| --- | ---: |")
    for arm, n in sorted(arm_counts.items()):
        lines.append(f"| {arm} | {n} |")
    lines.append("")

    lines.append("## Selected Per Dataset")
    lines.append("")
    lines.append("| dataset | groups | mse_mean | mae_mean |")
    lines.append("| --- | ---: | ---: | ---: |")
    for ds, recs in sorted(by_dataset.items()):
        lines.append(f"| {ds} | {len(recs)} | {fmt(mean(r.get('mean_test_mse') for r in recs))} | "
                     f"{fmt(mean(r.get('mean_test_mae') for r in recs))} |")
    lines.append("")

    lines.append("## Selected Per Pred_len")
    lines.append("")
    lines.append("| pred_len | groups | mse_mean | mae_mean |")
    lines.append("| --- | ---: | ---: | ---: |")
    for pl, recs in sorted(by_predlen.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {pl} | {len(recs)} | {fmt(mean(r.get('mean_test_mse') for r in recs))} | "
                     f"{fmt(mean(r.get('mean_test_mae') for r in recs))} |")
    lines.append("")

    lines.append("## Per Group Selection")
    lines.append("")
    hdr = "| " + " | ".join(selection_keys) + " | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |"
    sep = "| " + " | ".join(["---"] * len(selection_keys)) + " | --- | ---: | ---: | ---: |"
    lines.append(hdr)
    lines.append(sep)
    for rec in group_records:
        cells = [str(rec.get(k, "")) for k in selection_keys]
        cells += [rec["selected_arm"], fmt(rec.get("mean_val_score")),
                  fmt(rec.get("mean_test_mse")), fmt(rec.get("mean_test_mae"))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Margin / prefer-order trace: shows whether prefer_arm_order flipped the raw
    # best arm (e.g. weather 336/720 switched back to individual).
    lines.append("## Margin / Prefer-Order Trace")
    lines.append("")
    thdr = "| " + " | ".join(selection_keys) + " | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |"
    tsep = "| " + " | ".join(["---"] * len(selection_keys)) + " | --- | ---: | --- | --- | ---: |"
    lines.append(thdr)
    lines.append(tsep)
    for rec in group_records:
        cells = [str(rec.get(k, "")) for k in selection_keys]
        cells += [rec.get("raw_best_arm", ""), fmt(rec.get("raw_best_score")),
                  "; ".join(rec.get("near_best_arms", [])) or "(none)",
                  rec["selected_arm"], fmt(rec.get("mean_val_score"))]
        lines.append("| " + " | ".join(cells) + " |")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"selected_rows={len(selected_rows)} groups={len(group_records)} "
          f"metric_col={metric_col} output={args.output} summary={args.summary}")


if __name__ == "__main__":
    main()
