#!/usr/bin/env python3
"""AsySpecX Phase 6 Full-Field summary.

Fixed single-arm + validation-selected + selector audit + TEST oracle upper
bound (analysis only) + optional external baselines. Missing columns/files never
crash.
"""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATES = [
    "phase6_asx_cross", "phase6_asx_cross_clip05", "phase6_asx_individual",
    "phase6_asx_individual_revin", "phase6_asx_period_multi", "phase6_asx_individual_period",
]
BASELINES = ["FITS", "PatchTST", "SparseTSF", "DLinear", "TimesNet", "FEDformer"]
DIAG_COLS = ["temporal_gate_mean", "period_gate_mean", "low_freq_energy_ratio",
             "gate_mean", "residual_offdiag_ratio_mean", "clip_active_fraction"]
FAMILY = {
    "ETTh1": "ETT", "ETTh2": "ETT", "ETTm1": "ETT", "ETTm2": "ETT",
    "electricity": "LargeC", "traffic": "LargeC",
    "PEMS04": "PEMS", "PEMS08": "PEMS", "weather": "Weather",
}


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


def cell(r):
    return (r.get("dataset", ""), r.get("seq_len", ""), r.get("pred_len", ""))


def pkey(r):
    return (r.get("dataset", ""), r.get("seq_len", ""), r.get("pred_len", ""), r.get("seed", ""))


def last_seg_col(header, base="val_mse"):
    pat = re.compile(rf"^{re.escape(base)}_seg(\d+)$")
    idxs = [(int(m.group(1)), c) for c in header for m in [pat.match(c)] if m]
    return max(idxs)[1] if idxs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--selected_csv", default="")
    p.add_argument("--baseline_csv", default="")
    p.add_argument("--anchor_arm", default="phase6_asx_cross")
    p.add_argument("--output_dir", default="phase6_results/fullfield")
    p.add_argument("--selected_csvs", default="", help="comma list for multi-selector audit")
    p.add_argument("--selected_names", default="", help="comma list of names for --selected_csvs")
    args = p.parse_args()

    rows = read_rows(args.csv)
    header = list(rows[0].keys()) if rows else []
    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    failed = [r for r in rows if r.get("status", "ok") != "ok"]

    by_arm = defaultdict(list)
    by_cell = defaultdict(list)
    for r in ok:
        by_arm[r.get("arm", "")].append(r)
        by_cell[cell(r)].append(r)

    # fixed single-arm + oracle
    arm_mse = {a: mean(to_float(r.get("mse")) for r in g) for a, g in by_arm.items()}
    arm_mse = {a: v for a, v in arm_mse.items() if v is not None}
    best_single = min(arm_mse, key=arm_mse.get) if arm_mse else None
    best_single_mse = arm_mse.get(best_single)
    oracle = {}
    for key, g in by_cell.items():
        sc = defaultdict(list)
        for r in g:
            sc[r.get("arm", "")].append(r)
        ba, bm = None, None
        for a, rs in sc.items():
            m = mean(to_float(r.get("mse")) for r in rs)
            if m is not None and (bm is None or m < bm):
                ba, bm = a, m
        oracle[key] = (ba, bm)
    oracle_mse = mean(v[1] for v in oracle.values())

    L = ["# Phase 6 Full-Field Summary", ""]
    L.append("Report fixed single-arm and validation-selected separately. Oracle is "
             "analysis only and must not be reported as a valid selected model.")
    L.append("")
    L += [f"- total_runs: {len(rows)}", f"- ok_runs: {len(ok)}", f"- failed_runs: {len(failed)}",
          f"- anchor_arm: {args.anchor_arm}",
          f"- best_fixed_single_arm: {best_single} (mse_mean={fmt(best_single_mse)})",
          f"- test_oracle_mse_mean (ANALYSIS ONLY): {fmt(oracle_mse)}", ""]

    # arm means
    L += ["## Arm Means", "", "| arm | n | mse_mean | mae_mean | val_mse_mean |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for arm in sorted(by_arm):
        g = by_arm[arm]
        L.append(f"| {arm} | {len(g)} | {fmt(mean(to_float(r.get('mse')) for r in g))} | "
                 f"{fmt(mean(to_float(r.get('mae')) for r in g))} | "
                 f"{fmt(mean(to_float(r.get('val_mse')) for r in g))} |")
    L.append("")

    # best by test (analysis) + best-cell count
    L += ["## Best Arm Per Dataset/Seq_len/Pred_len BY TEST (analysis only -- not for selection)", "",
          "| dataset | seq_len | pred_len | best_arm | mse_mean | mae_mean |",
          "| --- | ---: | ---: | --- | ---: | ---: |"]
    best_count = Counter()
    for key in sorted(by_cell):
        sc = defaultdict(list)
        for r in by_cell[key]:
            sc[r.get("arm", "")].append(r)
        ba, bm, bma = None, None, None
        for a, rs in sc.items():
            m = mean(to_float(r.get("mse")) for r in rs)
            if m is not None and (bm is None or m < bm):
                ba, bm, bma = a, m, mean(to_float(r.get("mae")) for r in rs)
        if ba:
            best_count[ba] += 1
        ds, sl, pl = key
        L.append(f"| {ds} | {sl} | {pl} | {ba} | {fmt(bm)} | {fmt(bma)} |")
    L += ["", "## Best-Cell Count (by test, analysis only)", "", "| arm | cells |", "| --- | ---: |"]
    for a, n in sorted(best_count.items()):
        L.append(f"| {a} | {n} |")
    L.append("")

    # delta vs anchor + paired stats
    anchor_paired = {pkey(r): r for r in by_arm.get(args.anchor_arm, [])}
    L += ["## Paired Statistics vs Anchor", "", "Paired by dataset/seq_len/pred_len/seed.", "",
          "| arm | pairs | dMSE_mean | dMSE_std | dMSE_2sd | win/loss/tie | dMAE_mean |",
          "| --- | ---: | ---: | ---: | ---: | :--- | ---: |"]
    for arm in sorted(by_arm):
        if arm == args.anchor_arm:
            continue
        dmse, dmae = [], []
        w = l = t = 0
        for r in by_arm[arm]:
            a = anchor_paired.get(pkey(r))
            if a is None:
                continue
            am, bm = to_float(r.get("mse")), to_float(a.get("mse"))
            aa, ba = to_float(r.get("mae")), to_float(a.get("mae"))
            if am is not None and bm is not None:
                d = am - bm
                dmse.append(d); w += d < 0; l += d > 0; t += d == 0
            if aa is not None and ba is not None:
                dmae.append(aa - ba)
        if not dmse:
            continue
        s = pstd(dmse)
        L.append(f"| {arm} | {len(dmse)} | {fmt(mean(dmse))} | {fmt(s)} | {fmt(2*s)} | "
                 f"{w}/{l}/{t} | {fmt(mean(dmae))} |")
    L.append("")

    # per dataset / seq_len / pred_len / family
    def table(title, keyfn, source, mkey, akey):
        buckets = defaultdict(list)
        for r in source:
            buckets[keyfn(r)].append(r)
        out = [f"## {title}", "", "| key | n | mse_mean | mae_mean |", "| --- | ---: | ---: | ---: |"]
        for k, g in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            out.append(f"| {k} | {len(g)} | {fmt(mean(to_float(r.get(mkey)) for r in g))} | "
                       f"{fmt(mean(to_float(r.get(akey)) for r in g))} |")
        return out + [""]

    L += table("Per Dataset", lambda r: r.get("dataset", ""), ok, "mse", "mae")
    L += table("Per Seq_len", lambda r: r.get("seq_len", ""), ok, "mse", "mae")
    L += table("Per Pred_len", lambda r: r.get("pred_len", ""), ok, "mse", "mae")
    L += table("Per Dataset Family", lambda r: FAMILY.get(r.get("dataset", ""), "Other"), ok, "mse", "mae")

    # validation-selected
    selected = read_rows(args.selected_csv) if args.selected_csv and Path(args.selected_csv).is_file() else []
    if selected:
        sel_mse = mean(to_float(r.get("test_mse")) for r in selected)
        sel_mae = mean(to_float(r.get("test_mae")) for r in selected)
        byg = defaultdict(list)
        for r in selected:
            byg[cell(r)].append(r)
        counts = Counter(g[0].get("selected_arm", "") for g in byg.values())
        L += ["## Validation-Selected Summary", "",
              f"- selected_test_mse_mean: {fmt(sel_mse)}",
              f"- selected_test_mae_mean: {fmt(sel_mae)}",
              f"- selected vs best_fixed_single ({best_single}): delta_mse="
              f"{fmt((sel_mse - best_single_mse) if (sel_mse is not None and best_single_mse is not None) else None)}",
              f"- selected vs test_oracle (ANALYSIS ONLY): delta_mse="
              f"{fmt((sel_mse - oracle_mse) if (sel_mse is not None and oracle_mse is not None) else None)}",
              "", "### Selected Arm Counts (per group)", "", "| arm | groups |", "| --- | ---: |"]
        for a, n in sorted(counts.items()):
            L.append(f"| {a} | {n} |")
        L.append("")
        L += table("Selected Per Dataset", lambda r: r.get("dataset", ""), selected, "test_mse", "test_mae")
        L += table("Selected Per Pred_len", lambda r: r.get("pred_len", ""), selected, "test_mse", "test_mae")
        L += ["Selector fairness note: selection uses validation metrics aggregated over "
              "seeds; test metrics reported only after selection.", ""]

    # baselines
    if args.baseline_csv and Path(args.baseline_csv).is_file():
        base = read_rows(args.baseline_csv)
        bhas_sl = base and "seq_len" in base[0]
        bkey = defaultdict(list)
        for r in base:
            k = (r.get("dataset", ""), r.get("seq_len", ""), r.get("pred_len", ""), r.get("model", "")) if bhas_sl \
                else (r.get("dataset", ""), r.get("pred_len", ""), r.get("model", ""))
            bkey[k].append(r)

        def bmatch(ds, sl, pl, model):
            if bhas_sl:
                m = bkey.get((ds, sl, pl, model), [])
                if m:
                    return mean(to_float(r.get("mse")) for r in m)
            return mean(to_float(r.get("mse")) for r in bkey.get((ds, pl, model), []))

        srcs = [("Fixed-Single " + str(best_single), by_arm.get(best_single, []), "mse")]
        if selected:
            srcs.append(("Validation-Selected", selected, "test_mse"))
        for label, source, mcol in srcs:
            agg = defaultdict(list)
            for r in source:
                agg[cell(r)].append(r)
            for model in BASELINES:
                out, w, l = [], 0, 0
                for key, g in sorted(agg.items()):
                    ds, sl, pl = key
                    b = bmatch(ds, sl, pl, model)
                    a = mean(to_float(r.get(mcol)) for r in g)
                    if a is None or b is None:
                        continue
                    gap = a - b
                    w += gap < 0; l += gap > 0
                    out.append(f"| {ds} | {sl} | {pl} | {a:.6g} | {b:.6g} | {gap:.6g} | {100*gap/b:.3g}% |")
                if out:
                    L += [f"## {label} vs {model}", "",
                          "| dataset | seq_len | pred_len | asx_mse | base_mse | gap_abs | gap_pct |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"] + out + \
                         ["", f"- vs {model}: wins={w} losses={l}", ""]

    # multi-selector audit
    if args.selected_csvs:
        files = [x.strip() for x in args.selected_csvs.split(",") if x.strip()]
        names = [x.strip() for x in args.selected_names.split(",")] if args.selected_names else []
        L += ["## Selector Comparison", "",
              "| selector | groups | mse_mean | mae_mean | delta_vs_best_single |",
              "| --- | ---: | ---: | ---: | ---: |"]
        for i, fpath in enumerate(files):
            if not Path(fpath).is_file():
                L.append(f"| {names[i] if i < len(names) else fpath} (MISSING) | | | | |")
                continue
            srows = read_rows(fpath)
            byg = defaultdict(list)
            for r in srows:
                byg[cell(r)].append(r)
            m = mean(to_float(r.get("test_mse")) for r in srows)
            ma = mean(to_float(r.get("test_mae")) for r in srows)
            nm = names[i] if i < len(names) else Path(fpath).stem
            dv = (m - best_single_mse) if (m is not None and best_single_mse is not None) else None
            L.append(f"| {nm} | {len(byg)} | {fmt(m)} | {fmt(ma)} | {fmt(dv)} |")
        L.append("")

    # val segment mismatch + diagnostics
    seg_col = last_seg_col(header, "val_mse")
    if seg_col:
        L += ["## Validation Segment Mismatch (full val_mse vs last segment)", "",
              f"Last segment column: {seg_col}.", "",
              "| dataset | groups | mismatches |", "| --- | ---: | ---: |"]
        groups = defaultdict(lambda: defaultdict(list))
        for r in ok:
            groups[cell(r)][r.get("arm", "")].append(r)
        mism, tot = Counter(), Counter()
        for key, arms in groups.items():
            ds = key[0]
            full = {a: mean(to_float(r.get("val_mse")) for r in rs) for a, rs in arms.items()}
            seg = {a: mean(to_float(r.get(seg_col)) for r in rs) for a, rs in arms.items()}
            full = {a: v for a, v in full.items() if v is not None}
            seg = {a: v for a, v in seg.items() if v is not None}
            if not full or not seg:
                continue
            tot[ds] += 1
            if min(full, key=full.get) != min(seg, key=seg.get):
                mism[ds] += 1
        for ds in sorted(tot):
            L.append(f"| {ds} | {tot[ds]} | {mism[ds]} |")
        L.append("")

    present = [c for c in DIAG_COLS if c in header]
    if present:
        L += ["## Model Diagnostics (means where present)", "",
              "| arm | " + " | ".join(present) + " |",
              "| --- | " + " | ".join(["---:"] * len(present)) + " |"]
        for arm in sorted(by_arm):
            L.append(f"| {arm} | " + " | ".join(fmt(mean(to_float(r.get(c)) for r in by_arm[arm])) for c in present) + " |")
        L.append("")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary_phase6_fullfield.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"summary={out}")


if __name__ == "__main__":
    main()
