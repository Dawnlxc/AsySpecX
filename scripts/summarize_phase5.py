#!/usr/bin/env python3
"""AsySpecX Phase 5-Lockdown summary.

Combines raw results.csv with (optional) validation-selected CSV and external
baselines. Includes paired statistics vs an anchor arm, selected-vs-single-arm
comparison, validation-segment mismatch diagnostics, and optional model
diagnostics. Missing columns never crash.
"""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATES = [
    "phase5_asx_cross",
    "phase5_asx_cross_clip05",
    "phase5_asx_individual",
    "phase5_asx_individual_revin",
    "phase5_asx_period_multi",
    "phase5_asx_individual_period",
]
BASELINES = ["FITS", "PatchTST", "SparseTSF", "DLinear", "TimesNet", "FEDformer"]
DIAG_COLS = ["temporal_gate_mean", "period_gate_mean", "low_freq_energy_ratio",
             "gate_mean", "residual_offdiag_ratio_mean", "clip_active_fraction"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    if v in ("", None):
        return None
    try:
        f = float(v)
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


def cell(row):
    return (row.get("dataset", ""), row.get("seq_len", ""), row.get("pred_len", ""))


def paired_key(row):
    return (row.get("dataset", ""), row.get("seq_len", ""), row.get("pred_len", ""), row.get("seed", ""))


def last_seg_col(header, base="val_mse"):
    pat = re.compile(rf"^{re.escape(base)}_seg(\d+)$")
    idxs = [(int(m.group(1)), c) for c in header for m in [pat.match(c)] if m]
    return max(idxs)[1] if idxs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--selected_csv", default="")
    parser.add_argument("--baseline_csv", default="")
    parser.add_argument("--anchor_arm", default="phase5_asx_cross")
    parser.add_argument("--output_dir", default="phase5_results/main")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    header = list(rows[0].keys()) if rows else []
    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    failed = [r for r in rows if r.get("status", "ok") != "ok"]

    by_arm = defaultdict(list)
    by_cell = defaultdict(list)
    for r in ok:
        by_arm[r.get("arm", "")].append(r)
        by_cell[cell(r)].append(r)

    L = ["# Phase 5-Lockdown Summary", ""]
    L.append("Do not pick arms by test metric. Report a fixed single-arm result AND "
             "the validation-selected result separately.")
    L.append("")
    L.append(f"- total_runs: {len(rows)}")
    L.append(f"- ok_runs: {len(ok)}")
    L.append(f"- failed_runs: {len(failed)}")
    L.append(f"- anchor_arm: {args.anchor_arm}")
    L.append("")

    # arm means
    L += ["## Arm Means", "", "| arm | n | mse_mean | mae_mean | val_mse_mean |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for arm in sorted(by_arm):
        g = by_arm[arm]
        L.append(f"| {arm} | {len(g)} | {fmt(mean(to_float(r.get('mse')) for r in g))} | "
                 f"{fmt(mean(to_float(r.get('mae')) for r in g))} | "
                 f"{fmt(mean(to_float(r.get('val_mse')) for r in g))} |")
    L.append("")

    # best by test (analysis only)
    L += ["## Best Arm Per Dataset/Seq_len/Pred_len BY TEST (analysis only -- not for selection)",
          "", "| dataset | seq_len | pred_len | best_arm | mse_mean | mae_mean |",
          "| --- | ---: | ---: | --- | ---: | ---: |"]
    best_count = Counter()
    for key in sorted(by_cell):
        arm_scores = defaultdict(list)
        for r in by_cell[key]:
            arm_scores[r.get("arm", "")].append(r)
        best_arm, best_mse, best_mae = None, None, None
        for arm, arm_rows in arm_scores.items():
            m = mean(to_float(r.get("mse")) for r in arm_rows)
            if m is not None and (best_mse is None or m < best_mse):
                best_arm, best_mse = arm, m
                best_mae = mean(to_float(r.get("mae")) for r in arm_rows)
        if best_arm is not None:
            best_count[best_arm] += 1
        ds, sl, pl = key
        L.append(f"| {ds} | {sl} | {pl} | {best_arm} | {fmt(best_mse)} | {fmt(best_mae)} |")
    L.append("")
    L += ["## Best-Cell Count (by test, analysis only)", "", "| arm | cells |", "| --- | ---: |"]
    for arm, n in sorted(best_count.items()):
        L.append(f"| {arm} | {n} |")
    L.append("")

    # delta vs anchor (cell-mean)
    anchor_cell = {}
    for key, g in by_cell.items():
        a_rows = [r for r in g if r.get("arm") == args.anchor_arm]
        if a_rows:
            anchor_cell[key] = (mean(to_float(r.get("mse")) for r in a_rows),
                                mean(to_float(r.get("mae")) for r in a_rows))
    L += ["## Delta Versus Anchor (cell-mean)", "",
          "| arm | cells | delta_mse_mean | delta_mae_mean |", "| --- | ---: | ---: | ---: |"]
    for arm in sorted(by_arm):
        if arm == args.anchor_arm:
            continue
        dms, dmas = [], []
        arm_by_cell = defaultdict(list)
        for r in by_arm[arm]:
            arm_by_cell[cell(r)].append(r)
        for key, g in arm_by_cell.items():
            if key not in anchor_cell:
                continue
            am = mean(to_float(r.get("mse")) for r in g)
            ama = mean(to_float(r.get("mae")) for r in g)
            if am is not None and anchor_cell[key][0] is not None:
                dms.append(am - anchor_cell[key][0])
            if ama is not None and anchor_cell[key][1] is not None:
                dmas.append(ama - anchor_cell[key][1])
        L.append(f"| {arm} | {len(dms)} | {fmt(mean(dms))} | {fmt(mean(dmas))} |")
    L.append("")

    # single-arm candidate summary
    L += ["## Single-Arm Candidate Summary", "",
          "| arm | n | mse_mean | mae_mean | val_mse_mean |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in CANDIDATES:
        g = by_arm.get(arm, [])
        if not g:
            L.append(f"| {arm} | 0 | | | |")
            continue
        L.append(f"| {arm} | {len(g)} | {fmt(mean(to_float(r.get('mse')) for r in g))} | "
                 f"{fmt(mean(to_float(r.get('mae')) for r in g))} | "
                 f"{fmt(mean(to_float(r.get('val_mse')) for r in g))} |")
    L.append("")

    # paired statistics vs anchor
    L += ["## Paired Statistics vs Anchor", "",
          "Paired by dataset/seq_len/pred_len/seed.", "",
          "| arm | pairs | dMSE_mean | dMSE_std | dMSE_2sd | win/loss/tie | dMAE_mean | dMAE_std |",
          "| --- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |"]
    anchor_paired = {paired_key(r): r for r in by_arm.get(args.anchor_arm, [])}
    for arm in sorted(by_arm):
        if arm == args.anchor_arm:
            continue
        dmse, dmae = [], []
        w = l = t = 0
        for r in by_arm[arm]:
            a = anchor_paired.get(paired_key(r))
            if a is None:
                continue
            am, bm = to_float(r.get("mse")), to_float(a.get("mse"))
            aa, ba = to_float(r.get("mae")), to_float(a.get("mae"))
            if am is not None and bm is not None:
                d = am - bm
                dmse.append(d)
                w += d < 0
                l += d > 0
                t += d == 0
            if aa is not None and ba is not None:
                dmae.append(aa - ba)
        if not dmse:
            continue
        s = pstd(dmse)
        L.append(f"| {arm} | {len(dmse)} | {fmt(mean(dmse))} | {fmt(s)} | {fmt(2*s)} | "
                 f"{w}/{l}/{t} | {fmt(mean(dmae))} | {fmt(pstd(dmae))} |")
    L.append("")

    # validation-selected summary + selected vs single-arm
    selected = read_rows(args.selected_csv) if args.selected_csv else []
    if selected:
        sel_mse = mean(to_float(r.get("test_mse")) for r in selected)
        sel_mae = mean(to_float(r.get("test_mae")) for r in selected)
        # one representative per group
        by_group = defaultdict(list)
        for r in selected:
            by_group[cell(r)].append(r)
        counts = Counter(g[0].get("selected_arm", "") for g in by_group.values())
        L += ["## Validation-Selected Summary", "",
              f"- selected_test_mse_mean: {fmt(sel_mse)}",
              f"- selected_test_mae_mean: {fmt(sel_mae)}", "",
              "### Selected Arm Counts (per group)", "", "| arm | groups |", "| --- | ---: |"]
        for arm, n in sorted(counts.items()):
            L.append(f"| {arm} | {n} |")
        L.append("")
        # selected vs best single-arm (over the same groups, by test)
        best_single_arm, best_single = None, None
        for arm in CANDIDATES:
            g = by_arm.get(arm, [])
            m = mean(to_float(r.get("mse")) for r in g)
            if m is not None and (best_single is None or m < best_single):
                best_single, best_single_arm = m, arm
        L += ["### Selected vs Best Single-Arm", "",
              f"- selected_mse_mean: {fmt(sel_mse)}",
              f"- best_single_arm: {best_single_arm} (mse_mean={fmt(best_single)})",
              f"- delta(selected - best_single): {fmt((sel_mse - best_single) if (sel_mse is not None and best_single is not None) else None)}",
              ""]

    # per dataset / pred_len / seq_len (selected if present else raw ok means)
    def group_table(title, keyfn, source):
        buckets = defaultdict(list)
        for r in source:
            buckets[keyfn(r)].append(r)
        out = [f"## {title}", "", "| key | n | mse_mean | mae_mean |", "| --- | ---: | ---: | ---: |"]
        mkey = "test_mse" if source is selected else "mse"
        akey = "test_mae" if source is selected else "mae"
        for k, g in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            out.append(f"| {k} | {len(g)} | {fmt(mean(to_float(r.get(mkey)) for r in g))} | "
                       f"{fmt(mean(to_float(r.get(akey)) for r in g))} |")
        out.append("")
        return out

    src = selected if selected else ok
    L += group_table("Per Dataset", lambda r: r.get("dataset", ""), src)
    L += group_table("Per Pred_len", lambda r: r.get("pred_len", ""), src)
    L += group_table("Per Seq_len", lambda r: r.get("seq_len", ""), src)

    # external baselines
    if args.baseline_csv and Path(args.baseline_csv).is_file():
        base = read_rows(args.baseline_csv)
        bhas_sl = base and "seq_len" in base[0]
        bkey = defaultdict(list)
        for r in base:
            if bhas_sl:
                bkey[(r.get("dataset", ""), r.get("seq_len", ""), r.get("pred_len", ""), r.get("model", ""))].append(r)
            else:
                bkey[(r.get("dataset", ""), r.get("pred_len", ""), r.get("model", ""))].append(r)

        def bmatch(ds, sl, pl, model):
            if bhas_sl:
                m = bkey.get((ds, sl, pl, model), [])
                if m:
                    return mean(to_float(r.get("mse")) for r in m)
            return mean(to_float(r.get("mse")) for r in bkey.get((ds, pl, model), []))

        # single-arm (period_multi) and individual_period vs baselines, plus selected
        for label, source, mcol in [("Single-Arm phase5_asx_period_multi", by_arm.get("phase5_asx_period_multi", []), "mse"),
                                    ("Single-Arm phase5_asx_individual_period", by_arm.get("phase5_asx_individual_period", []), "mse"),
                                    ("Validation-Selected", selected, "test_mse")]:
            if not source:
                continue
            agg = defaultdict(list)
            for r in source:
                agg[cell(r)].append(r)
            for model in BASELINES:
                any_row = False
                rows_out = []
                w = l = 0
                for key, g in sorted(agg.items()):
                    ds, sl, pl = key
                    b = bmatch(ds, sl, pl, model)
                    a = mean(to_float(r.get(mcol)) for r in g)
                    if a is None or b is None:
                        continue
                    any_row = True
                    gap = a - b
                    w += gap < 0
                    l += gap > 0
                    rows_out.append(f"| {ds} | {sl} | {pl} | {a:.6g} | {b:.6g} | {gap:.6g} | {100*gap/b:.3g}% |")
                if any_row:
                    L += [f"## {label} vs {model}", "",
                          "| dataset | seq_len | pred_len | asx_mse | base_mse | gap_abs | gap_pct |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
                    L += rows_out
                    L += [f"", f"- vs {model}: wins={w} losses={l}", ""]

    # validation-segment mismatch diagnostics
    seg_col = last_seg_col(header, "val_mse")
    if seg_col:
        L += ["## Validation Segment Mismatch (full val_mse vs last segment)", "",
              f"Last segment column: {seg_col}. How often the arm chosen by mean full "
              "val_mse differs from the arm chosen by mean last-segment val_mse.", "",
              "| dataset | groups | mismatches |", "| --- | ---: | ---: |"]
        groups = defaultdict(lambda: defaultdict(list))
        for r in ok:
            groups[cell(r)][r.get("arm", "")].append(r)
        mismatch_by_ds = Counter()
        total_by_ds = Counter()
        for key, arms in groups.items():
            ds = key[0]
            full = {a: mean(to_float(r.get("val_mse")) for r in rs) for a, rs in arms.items()}
            seg = {a: mean(to_float(r.get(seg_col)) for r in rs) for a, rs in arms.items()}
            full = {a: v for a, v in full.items() if v is not None}
            seg = {a: v for a, v in seg.items() if v is not None}
            if not full or not seg:
                continue
            total_by_ds[ds] += 1
            if min(full, key=full.get) != min(seg, key=seg.get):
                mismatch_by_ds[ds] += 1
        for ds in sorted(total_by_ds):
            L.append(f"| {ds} | {total_by_ds[ds]} | {mismatch_by_ds[ds]} |")
        L.append("")

    # optional diagnostics
    present_diag = [c for c in DIAG_COLS if c in header]
    if present_diag:
        L += ["## Model Diagnostics (means where present)", "",
              "| arm | " + " | ".join(present_diag) + " |",
              "| --- | " + " | ".join(["---:"] * len(present_diag)) + " |"]
        for arm in sorted(by_arm):
            cells = [fmt(mean(to_float(r.get(c)) for r in by_arm[arm])) for c in present_diag]
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## Fairness Note", "",
          "Selection uses validation metrics aggregated over seeds. Test metrics are "
          "reported only after arm selection.", ""]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary_phase5.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"summary={out}")


if __name__ == "__main__":
    main()
