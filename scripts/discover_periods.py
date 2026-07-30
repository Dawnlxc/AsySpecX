#!/usr/bin/env python3
"""Auto-period discovery for the AsySpecX sparse-period adapter (Phase 7).

Estimates dominant periods from the TRAIN split only (never val/test), using
either autocorrelation (auto_acf) or the rFFT power spectrum (auto_fft), and
writes a JSON with the top-k periods. The train split is obtained through the
repo's own Dataset class (data_provider, flag='train') so the split convention
matches training exactly; the series used is the standardized train input
(data_x), which is documented behavior.

Usage mirrors run.py args:
  python scripts/discover_periods.py --data custom --root_path ./dataset/weather/ \
    --data_path weather.csv --seq_len 720 --enc_in 21 --method auto_acf \
    --topk 3 --period_min 4 --period_max 0 --output auto_periods.json
"""

import argparse
import contextlib
import json
import os
import sys
from argparse import Namespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_train_series(args):
    """Return standardized train series [L, C] via the repo Dataset (train split)."""
    from data_provider.data_factory import data_provider
    ns = Namespace(
        data=args.data, root_path=args.root_path, data_path=args.data_path,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        features=args.features, target=args.target, embed=args.embed, freq=args.freq,
        cycle=args.cycle, batch_size=1, num_workers=0,
    )
    data_set, _ = data_provider(ns, flag="train")
    x = np.asarray(data_set.data_x, dtype=np.float64)  # [L, C], standardized on train
    return x


def _topk_local_maxima(score, pmin, pmax, k):
    lo, hi = max(pmin, 1), min(pmax, len(score) - 1)
    cands = []
    for p in range(lo, hi + 1):
        left = score[p - 1] if p - 1 >= 0 else -np.inf
        right = score[p + 1] if p + 1 < len(score) else -np.inf
        if score[p] >= left and score[p] >= right:
            cands.append((p, float(score[p])))
    # Rank by (rounded) score desc, tie-break toward the SMALLER lag so the
    # fundamental period wins over its harmonics on clean periodic signals.
    cands.sort(key=lambda t: (-round(t[1], 4), t[0]))
    return cands[:k]


def discover_acf(x, pmin, pmax, k, max_rows=100000):
    x = x[:max_rows]
    L, C = x.shape
    x = x - x.mean(axis=0, keepdims=True)
    var = (x ** 2).mean(axis=0) + 1e-12
    score = np.full(pmax + 1, -np.inf)
    for lag in range(pmin, pmax + 1):
        if lag >= L:
            break
        prod = (x[:-lag] * x[lag:]).mean(axis=0) / var  # per-channel autocorr at lag
        # SIGNED mean over channels: a true period gives +1 (phase-independent),
        # while the half-period gives -1, so anti-correlated lags are not peaks.
        score[lag] = float(np.mean(prod))
    peaks = _topk_local_maxima(score, pmin, pmax, k)
    return [p for p, _ in peaks], [s for _, s in peaks]


def discover_fft(x, pmin, pmax, k, max_rows=100000):
    x = x[:max_rows]
    L, C = x.shape
    x = x - x.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(x, axis=0)) ** 2  # [F, C]
    power = power.mean(axis=1)  # aggregate over channels
    period_score = {}
    for f in range(1, len(power)):  # skip DC
        p = int(round(L / f))
        if pmin <= p <= pmax:
            period_score[p] = period_score.get(p, 0.0) + float(power[f])
    ranked = sorted(period_score.items(), key=lambda t: t[1], reverse=True)[:k]
    return [p for p, _ in ranked], [s for _, s in ranked]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="")   # informational label only
    p.add_argument("--data", default="custom")
    p.add_argument("--root_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--label_len", type=int, default=0)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--features", default="M")
    p.add_argument("--target", default="OT")
    p.add_argument("--embed", default="timeF")
    p.add_argument("--freq", default="h")
    p.add_argument("--cycle", type=int, default=24)
    p.add_argument("--enc_in", type=int, default=0)
    p.add_argument("--method", default="auto_acf", choices=["auto_acf", "auto_fft", "union_auto"])
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--period_min", type=int, default=4)
    p.add_argument("--period_max", type=int, default=0)
    p.add_argument("--fallback_periods", default="24")
    p.add_argument("--manual_periods", default="", help="union_auto: manual periods (priority first); defaults to fallback")
    p.add_argument("--max_periods", type=int, default=5, help="union_auto: cap on number of periods")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    pmax = args.period_max if args.period_max > 0 else min(args.seq_len, 672)
    pmin = max(2, args.period_min)

    try:
        # data loaders may print progress (e.g. "train <N>") to stdout; keep stdout
        # clean so a runner capturing $(...) gets only the comma-joined periods line.
        with contextlib.redirect_stdout(sys.stderr):
            x = load_train_series(args)
        if x.shape[0] <= pmin:
            raise ValueError(f"train length {x.shape[0]} too short for period_min {pmin}")

        def valid_dedup(seq):
            seen, out = set(), []
            for p_ in seq:
                p_ = int(p_)
                if pmin <= p_ <= pmax and p_ not in seen:
                    seen.add(p_); out.append(p_)
            return out

        if args.method == "union_auto":
            # priority: manual first, then auto_acf, then auto_fft; capped to max_periods.
            manual = [int(v) for v in (args.manual_periods or args.fallback_periods).replace("+", ",").split(",") if v.strip()]
            acf_p, _ = discover_acf(x, pmin, pmax, args.topk)
            fft_p, _ = discover_fft(x, pmin, pmax, args.topk)
            periods = valid_dedup(manual + acf_p + fft_p)[: max(1, args.max_periods)]
            scores = []
        elif args.method == "auto_acf":
            periods, scores = discover_acf(x, pmin, pmax, args.topk)
            periods = valid_dedup(periods)
        else:
            periods, scores = discover_fft(x, pmin, pmax, args.topk)
            periods = valid_dedup(periods)
        if not periods:
            raise ValueError("no valid periods found")
    except Exception as exc:
        periods = [int(v) for v in args.fallback_periods.replace("+", ",").split(",") if v.strip()]
        scores = []
        print(f"[discover_periods] WARNING: {exc}; falling back to {periods}", file=sys.stderr)

    out = {
        "dataset": args.dataset or args.data_path,
        "seq_len": args.seq_len,
        "method": args.method,
        "periods": periods,
        "scores": scores,
        "period_min": pmin,
        "period_max": pmax,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    # print the comma-joined periods to stdout so a runner can capture them
    print(",".join(str(p) for p in periods))


if __name__ == "__main__":
    main()
