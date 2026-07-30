#!/usr/bin/env python3
"""Offline validation convex ensemble over trained AsySpecX arms (Phase 7).

Analysis/performance result (AsySpecX-Ensemble), reported separately from the
single model. Learns per-(dataset,seq_len,pred_len) weights on VALIDATION
predictions only, then applies them to TEST predictions. Test targets are never
used to fit weights.

Prediction npz files (from run.py --save_predictions 1) are named:
  <arm>__<dataset>__sl<seq_len>__pl<pred_len>__sd<seed>.npz
with arrays val_pred, val_true, test_pred, test_true.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np


def parse_tag(path):
    base = os.path.basename(path)
    base = base[:-4] if base.endswith(".npz") else base
    parts = base.split("__")
    d = {"arm": parts[0] if parts else base}
    for tok in parts[1:]:
        m = re.match(r"^sl(\d+)$", tok)
        if m:
            d["seq_len"] = m.group(1); continue
        m = re.match(r"^pl(\d+)$", tok)
        if m:
            d["pred_len"] = m.group(1); continue
        m = re.match(r"^sd(\w+)$", tok)
        if m:
            d["seed"] = m.group(1); continue
        d.setdefault("dataset", tok)
    return d


def mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def fit_simplex(preds, y, iters=500, lr=0.5):
    """Projected gradient on the probability simplex minimizing MSE. preds: [K, ...]."""
    K = preds.shape[0]
    w = np.full(K, 1.0 / K)
    P = preds.reshape(K, -1)
    t = y.reshape(-1)
    for _ in range(iters):
        pred = w @ P
        grad = 2.0 * (P @ (pred - t)) / t.size
        w = w - lr * grad
        w = project_simplex(w)
    return w


def project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def fit_ridge(preds, y, lam=1e-3, nonneg=True):
    K = preds.shape[0]
    P = preds.reshape(K, -1).T  # [N, K]
    t = y.reshape(-1)
    A = P.T @ P + lam * np.eye(K)
    w = np.linalg.solve(A, P.T @ t)
    if nonneg:
        w = np.maximum(w, 0.0)
        s = w.sum()
        w = w / s if s > 0 else np.full(K, 1.0 / K)
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--results_csv", default="")
    p.add_argument("--selection_keys", default="dataset,seq_len,pred_len")
    p.add_argument("--replicate_key", default="seed")
    p.add_argument("--arm_key", default="arm")
    p.add_argument("--arms", default="")
    p.add_argument("--mode", default="simplex_val", choices=["uniform", "simplex_val", "ridge_val"])
    p.add_argument("--output_csv", default="ensemble_results.csv")
    p.add_argument("--summary", default="ensemble_summary.md")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz prediction files in {args.pred_dir}")
    arm_filter = set(a.strip() for a in args.arms.split(",") if a.strip())

    # group -> arm -> list of (seed, path)
    groups = defaultdict(lambda: defaultdict(list))
    for f in files:
        meta = parse_tag(f)
        if arm_filter and meta.get("arm") not in arm_filter:
            continue
        gkey = (meta.get("dataset", ""), meta.get("seq_len", ""), meta.get("pred_len", ""))
        groups[gkey][meta.get("arm", "")].append((meta.get("seed", ""), f))

    import csv
    rows_out = []
    lines = ["# AsySpecX-Ensemble (offline validation convex ensemble)", "",
             "Analysis/performance result. Weights are fit on VALIDATION only; "
             "test targets are never used to fit weights.", "",
             f"- mode: {args.mode}", ""]

    overall_mse, overall_mae, single_best_mse = [], [], []
    lines.append("## Per Group")
    lines.append("")
    lines.append("| dataset | seq_len | pred_len | arms | ens_mse | ens_mae | best_single_mse | weights |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for gkey, arms in sorted(groups.items()):
        arm_names = sorted(arms.keys())
        if len(arm_names) < 1:
            continue
        # concatenate replicate seeds per arm (group-level weights, mode (a))
        val_preds, test_preds, val_true, test_true = [], [], None, None
        usable = []
        for arm in arm_names:
            vp, tp, vt, tt = [], [], None, None
            for seed, path in sorted(arms[arm]):
                z = np.load(path)
                vp.append(z["val_pred"]); tp.append(z["test_pred"])
                vt = z["val_true"]; tt = z["test_true"]
            if not vp:
                continue
            val_preds.append(np.concatenate(vp, axis=0))
            test_preds.append(np.concatenate(tp, axis=0))
            usable.append(arm)
            # targets: use the first arm's concatenated truths (same across arms/seeds order)
            if val_true is None:
                val_true = np.concatenate([np.load(pp)["val_true"] for _, pp in sorted(arms[arm])], axis=0)
                test_true = np.concatenate([np.load(pp)["test_true"] for _, pp in sorted(arms[arm])], axis=0)
        if len(usable) < 1:
            continue
        # align shapes (arms must share sample count)
        n_val = min(v.shape[0] for v in val_preds)
        n_test = min(t.shape[0] for t in test_preds)
        VP = np.stack([v[:n_val] for v in val_preds], axis=0)   # [K, n_val, H, C]
        TP = np.stack([t[:n_test] for t in test_preds], axis=0)
        VY = val_true[:n_val]; TY = test_true[:n_test]

        if args.mode == "uniform":
            w = np.full(len(usable), 1.0 / len(usable))
        elif args.mode == "simplex_val":
            w = fit_simplex(VP, VY)
        else:
            w = fit_ridge(VP, VY)

        ens_test = np.tensordot(w, TP, axes=(0, 0))
        e_mse, e_mae = mse(ens_test, TY), mae(ens_test, TY)
        best_single = min(mse(TP[i], TY) for i in range(len(usable)))
        overall_mse.append(e_mse); overall_mae.append(e_mae); single_best_mse.append(best_single)
        ds, sl, pl = gkey
        wstr = "; ".join(f"{a}:{w[i]:.3f}" for i, a in enumerate(usable))
        lines.append(f"| {ds} | {sl} | {pl} | {len(usable)} | {e_mse:.6g} | {e_mae:.6g} | {best_single:.6g} | {wstr} |")
        rows_out.append({"dataset": ds, "seq_len": sl, "pred_len": pl, "n_arms": len(usable),
                         "ens_mse": e_mse, "ens_mae": e_mae, "best_single_mse": best_single, "weights": wstr})

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seq_len", "pred_len", "n_arms",
                                          "ens_mse", "ens_mae", "best_single_mse", "weights"])
        w.writeheader(); w.writerows(rows_out)

    def m(v):
        return sum(v) / len(v) if v else float("nan")
    lines += ["", "## Overall", "",
              f"- ensemble_mse_mean: {m(overall_mse):.6g}",
              f"- ensemble_mae_mean: {m(overall_mae):.6g}",
              f"- best_single_per_group_mse_mean: {m(single_best_mse):.6g}", ""]
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"groups={len(rows_out)} ensemble_mse_mean={m(overall_mse):.6g} "
          f"output={args.output_csv} summary={args.summary}")


if __name__ == "__main__":
    main()
