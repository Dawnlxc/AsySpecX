#!/usr/bin/env python3
"""AsySpecX Phase 7-Breakthrough summary (merged Phase6+Phase7 pool).

Fixed single-arm + validation-selected + paired stats vs phase6_asx_period_multi
+ TEST oracle (analysis only) + which NEW (phase7_*) arms beat the OLD (phase6_*)
per-cell best. Missing columns/files never crash.
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

BASELINES = ["FITS", "PatchTST", "SparseTSF", "DLinear", "TimesNet", "FEDformer"]
DIAG_COLS = ["branch_weight_spec_mean", "branch_weight_period_mean", "branch_weight_patch_mean",
             "branch_weight_linear_mean", "branch_entropy", "linear_gate_mean", "patch_gate_mean",
             "period_weight_mean", "temporal_gate_mean", "eta_mean", "clip_active_fraction"]
FAMILY = {"ETTh1": "ETT", "ETTh2": "ETT", "ETTm1": "ETT", "ETTm2": "ETT",
          "electricity": "LargeC", "traffic": "LargeC",
          "PEMS04": "PEMS", "PEMS08": "PEMS", "weather": "Weather"}


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--selected_csv", default="")
    p.add_argument("--baseline_csv", default="")
    p.add_argument("--anchor_arm", default="phase7_period_multi_auto_acf_patchlinear")
    p.add_argument("--ensemble_csv", default="")
    p.add_argument("--output_dir", default="phase8_results/merged")
    args = p.parse_args()

    rows = read_rows(args.csv)
    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    failed = [r for r in rows if r.get("status", "ok") != "ok"]
    by_arm, by_cell = defaultdict(list), defaultdict(list)
    for r in ok:
        by_arm[r.get("arm", "")].append(r)
        by_cell[cell(r)].append(r)

    arm_mse = {a: mean(to_float(r.get("mse")) for r in g) for a, g in by_arm.items()}
    arm_mse = {a: v for a, v in arm_mse.items() if v is not None}
    best_single = min(arm_mse, key=arm_mse.get) if arm_mse else None

    def cell_arm_mse(g):
        s = defaultdict(list)
        for r in g:
            s[r.get("arm", "")].append(r)
        return {a: mean(to_float(r.get("mse")) for r in rs) for a, rs in s.items()}

    oracle = {}
    for key, g in by_cell.items():
        am = {a: v for a, v in cell_arm_mse(g).items() if v is not None}
        if am:
            ba = min(am, key=am.get)
            oracle[key] = (ba, am[ba])
    oracle_mse = mean(v[1] for v in oracle.values())

    L = ["# Phase 8-Hydra Summary", ""]
    L.append("Report fixed single-arm and validation-selected separately. Oracle is "
             "analysis only and must NOT be reported as a valid selected model.")
    L.append("")
    L += [f"- total_runs: {len(rows)}", f"- ok_runs: {len(ok)}", f"- failed_runs: {len(failed)}",
          f"- anchor_arm: {args.anchor_arm}",
          f"- best_fixed_single_arm: {best_single} (mse_mean={fmt(arm_mse.get(best_single))})",
          f"- test_oracle_mse_mean (ANALYSIS ONLY): {fmt(oracle_mse)}", ""]

    L += ["## Arm Means", "", "| arm | n | mse_mean | mae_mean | val_mse_mean |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for arm in sorted(by_arm):
        g = by_arm[arm]
        L.append(f"| {arm} | {len(g)} | {fmt(mean(to_float(r.get('mse')) for r in g))} | "
                 f"{fmt(mean(to_float(r.get('mae')) for r in g))} | {fmt(mean(to_float(r.get('val_mse')) for r in g))} |")
    L.append("")

    # best-cell count (test, analysis)
    best_count = Counter()
    for key, g in by_cell.items():
        am = {a: v for a, v in cell_arm_mse(g).items() if v is not None}
        if am:
            best_count[min(am, key=am.get)] += 1
    L += ["## Best-Cell Count (by test, analysis only)", "", "| arm | cells |", "| --- | ---: |"]
    for a, n in sorted(best_count.items(), key=lambda kv: -kv[1]):
        L.append(f"| {a} | {n} |")
    L.append("")

    # paired vs anchor
    anchor_paired = {pkey(r): r for r in by_arm.get(args.anchor_arm, [])}
    L += ["## Paired Statistics vs Anchor", "", "Paired by dataset/seq_len/pred_len/seed.", "",
          "| arm | pairs | dMSE_mean | dMSE_std | dMSE_2sd | win/loss/tie |",
          "| --- | ---: | ---: | ---: | ---: | :--- |"]
    for arm in sorted(by_arm):
        if arm == args.anchor_arm:
            continue
        dmse, w, l, t = [], 0, 0, 0
        for r in by_arm[arm]:
            a = anchor_paired.get(pkey(r))
            if a is None:
                continue
            am, bm = to_float(r.get("mse")), to_float(a.get("mse"))
            if am is not None and bm is not None:
                d = am - bm; dmse.append(d); w += d < 0; l += d > 0; t += d == 0
        if not dmse:
            continue
        s = pstd(dmse)
        L.append(f"| {arm} | {len(dmse)} | {fmt(mean(dmse))} | {fmt(s)} | {fmt(2*s)} | {w}/{l}/{t} |")
    L.append("")

    # which phase8 arms beat the phase6/7 per-cell best
    L += ["## Phase8 Arms Improving Phase6/7 Best Cells", "",
          "| dataset | seq_len | pred_len | old_best_arm | old_mse | new_best_arm | new_mse | improvement |",
          "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |"]
    n_improved = 0
    for key in sorted(by_cell):
        am = {a: v for a, v in cell_arm_mse(by_cell[key]).items() if v is not None}
        old = {a: v for a, v in am.items() if a.startswith("phase6_") or a.startswith("phase7_")}
        new = {a: v for a, v in am.items() if a.startswith("phase8_")}
        if not old or not new:
            continue
        oa = min(old, key=old.get); na = min(new, key=new.get)
        if new[na] < old[oa]:
            n_improved += 1
            ds, sl, pl = key
            L.append(f"| {ds} | {sl} | {pl} | {oa} | {old[oa]:.6g} | {na} | {new[na]:.6g} | {old[oa]-new[na]:.6g} |")
    L += ["", f"- cells where a phase8 arm beats the phase6/7 best: {n_improved}", ""]

    # per dataset/seq_len/pred_len/family
    def table(title, keyfn, source, mkey, akey):
        b = defaultdict(list)
        for r in source:
            b[keyfn(r)].append(r)
        out = [f"## {title}", "", "| key | n | mse_mean | mae_mean |", "| --- | ---: | ---: | ---: |"]
        for k, g in sorted(b.items(), key=lambda kv: str(kv[0])):
            out.append(f"| {k} | {len(g)} | {fmt(mean(to_float(r.get(mkey)) for r in g))} | "
                       f"{fmt(mean(to_float(r.get(akey)) for r in g))} |")
        return out + [""]

    L += table("Per Dataset", lambda r: r.get("dataset", ""), ok, "mse", "mae")
    L += table("Per Seq_len", lambda r: r.get("seq_len", ""), ok, "mse", "mae")
    L += table("Per Pred_len", lambda r: r.get("pred_len", ""), ok, "mse", "mae")
    L += table("Per Dataset Family", lambda r: FAMILY.get(r.get("dataset", ""), "Other"), ok, "mse", "mae")

    # selected
    selected = read_rows(args.selected_csv) if args.selected_csv and Path(args.selected_csv).is_file() else []
    if selected:
        sm = mean(to_float(r.get("test_mse")) for r in selected)
        sa = mean(to_float(r.get("test_mae")) for r in selected)
        byg = defaultdict(list)
        for r in selected:
            byg[cell(r)].append(r)
        counts = Counter(g[0].get("selected_arm", "") for g in byg.values())
        L += ["## Validation-Selected Summary", "",
              f"- selected_test_mse_mean: {fmt(sm)}", f"- selected_test_mae_mean: {fmt(sa)}",
              f"- selected vs best_fixed_single ({best_single}): delta_mse="
              f"{fmt((sm - arm_mse.get(best_single)) if (sm is not None and arm_mse.get(best_single) is not None) else None)}",
              f"- selected vs test_oracle (ANALYSIS ONLY): delta_mse="
              f"{fmt((sm - oracle_mse) if (sm is not None and oracle_mse is not None) else None)}",
              "", "### Selected Arm Counts (per group)", "", "| arm | groups |", "| --- | ---: |"]
        for a, n in sorted(counts.items()):
            L.append(f"| {a} | {n} |")
        L.append("")
        L += table("Selected Per Dataset", lambda r: r.get("dataset", ""), selected, "test_mse", "test_mae")
        L += table("Selected Per Pred_len", lambda r: r.get("pred_len", ""), selected, "test_mse", "test_mae")

    if args.baseline_csv and Path(args.baseline_csv).is_file():
        base = read_rows(args.baseline_csv)
        bhas_sl = base and "seq_len" in base[0]
        bkey = defaultdict(list)
        for r in base:
            k = (r.get("dataset", ""), r.get("seq_len", ""), r.get("pred_len", ""), r.get("model", "")) if bhas_sl \
                else (r.get("dataset", ""), r.get("pred_len", ""), r.get("model", ""))
            bkey[k].append(r)

        def bmatch(ds, sl, pl, model):
            if bhas_sl and bkey.get((ds, sl, pl, model)):
                return mean(to_float(r.get("mse")) for r in bkey[(ds, sl, pl, model)])
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
                    gap = a - b; w += gap < 0; l += gap > 0
                    out.append(f"| {ds} | {sl} | {pl} | {a:.6g} | {b:.6g} | {gap:.6g} | {100*gap/b:.3g}% |")
                if out:
                    L += [f"## {label} vs {model}", "",
                          "| dataset | seq_len | pred_len | asx_mse | base_mse | gap_abs | gap_pct |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"] + out + \
                         ["", f"- vs {model}: wins={w} losses={l}", ""]

    # branch / gate diagnostics if the CSV carries them
    header = list(rows[0].keys()) if rows else []
    present = [c for c in DIAG_COLS if c in header]
    if present:
        L += ["## Branch / Gate Diagnostics (means where present)", "",
              "| arm | " + " | ".join(present) + " |",
              "| --- | " + " | ".join(["---:"] * len(present)) + " |"]
        for arm in sorted(by_arm):
            L.append(f"| {arm} | " + " | ".join(fmt(mean(to_float(r.get(c)) for r in by_arm[arm])) for c in present) + " |")
        L.append("")

    # offline ensemble summary if provided
    if args.ensemble_csv and Path(args.ensemble_csv).is_file():
        erows = read_rows(args.ensemble_csv)
        em = mean(to_float(r.get("ens_mse")) for r in erows)
        ema = mean(to_float(r.get("ens_mae")) for r in erows)
        bs = mean(to_float(r.get("best_single_mse")) for r in erows)
        L += ["## Offline Ensemble (AsySpecX-Ensemble, analysis)", "",
              f"- ensemble_mse_mean: {fmt(em)}", f"- ensemble_mae_mean: {fmt(ema)}",
              f"- per-group best-single_mse_mean: {fmt(bs)}",
              "Weights fit on validation only; reported separately from any single model.", ""]

    L += ["## Fairness / Oracle Note", "",
          "Validation selection uses validation metrics aggregated over seeds. Test "
          "metrics reported only after selection. The oracle is a per-cell test-best "
          "upper bound for analysis only and must not be reported as a valid model.", ""]

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary_phase8.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"summary={out}")


if __name__ == "__main__":
    main()
