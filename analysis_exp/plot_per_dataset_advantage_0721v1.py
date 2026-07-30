#!/usr/bin/env python3
"""Per-dataset efficiency panels + cell-level advantage plot (0721v1 table).

Figure A (one per input length): 11 small multiples, one per dataset.
x = median params within the dataset (log), y = MSE averaged over that
dataset's four horizons (same scale within a panel, so the average is fair).
Ours is the accent bubble; the per-panel best baseline is named in muted ink.

Figure B: one dot per audited cell (dataset, L, H): x = params of the
per-cell best baseline divided by Ours params (log; >1 means Ours is
smaller), y = Ours MSE relative to that best baseline in percent (<0 means
Ours is strictly better than every baseline in that cell).
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "COMPACT_ECHO_ALL_BASELINES_L96_L720_0721V1_RESULTS.csv"
OUTDIR = REPO / "Figures"

MODELS = ["Compact + Echo", "TQNet", "CycleNet", "FITS", "SparseTSF", "FreTS",
          "FilterNet", "iTransformer", "PatchTST", "DLinear", "MixLinear",
          "PhaseFormer", "FreqCycle"]
OURS = "Compact + Echo"
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity",
            "traffic", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]

ACCENT = "#2a78d6"
NEUTRAL = "#9c9c97"
INK = "#33322f"
INK_MUTED = "#6f6e6a"
GRID = "#e6e5e2"
SURFACE = "#ffffff"
WIN_FILL = "#e3edf9"   # light tint of the accent for the winning region


def load():
    cells = {}
    for r in csv.DictReader(open(SRC)):
        if r["status"] != "ok" or r["model"] not in MODELS:
            continue
        key = (r["model"], r["dataset"], int(r["seq_len"]), int(r["pred_len"]))
        cells[key] = (float(r["mse"]), int(float(r["param_count"])))
    return cells


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=7.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def horizons(ds):
    return [12, 24, 48, 96] if ds.startswith("PEMS") else [96, 192, 336, 720]


def figure_a(cells, L):
    fig, axes = plt.subplots(3, 4, figsize=(12.6, 8.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    flat = axes.ravel()
    for i, ds in enumerate(DATASETS):
        ax = flat[i]
        pts = {}
        for m in MODELS:
            vals = [cells[(m, ds, L, h)] for h in horizons(ds)]
            mses = [v[0] for v in vals]
            ps = sorted(v[1] for v in vals)
            pts[m] = (0.5 * (ps[1] + ps[2]), sum(mses) / len(mses))
        base_best = min((m for m in MODELS if m != OURS), key=lambda m: pts[m][1])
        for m in MODELS:
            x, y = pts[m]
            if m == OURS:
                ax.scatter(x, y, s=64, facecolor=ACCENT, edgecolor=SURFACE,
                           linewidth=1.2, zorder=4)
            else:
                ax.scatter(x, y, s=26, facecolor=NEUTRAL, alpha=0.75,
                           edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ox, oy = pts[OURS]
        bx, by = pts[base_best]
        ratio = bx / ox
        gap = (oy - by) / by * 100
        ax.annotate("Ours", (ox, oy), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.8, color=INK,
                    fontweight="bold", zorder=5)
        ax.annotate(base_best, (bx, by), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=7, color=INK_MUTED,
                    zorder=5)
        tag = (f"{gap:+.1f}% MSE, {ratio:.0f}× smaller" if ratio >= 1.5
               else f"{gap:+.1f}% MSE vs best")
        ax.set_title(f"{ds}   ({tag})", fontsize=8.6, color=INK, pad=4)
        ax.set_xscale("log")
        ax.margins(x=0.15, y=0.22)
        style_axes(ax)
    flat[len(DATASETS)].axis("off")
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=ACCENT,
               markeredgecolor=SURFACE, markersize=9, label="Compact + Echo (Ours)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=NEUTRAL,
               markeredgecolor=SURFACE, markersize=6.5, label="12 baselines"),
    ]
    flat[len(DATASETS)].legend(handles=handles, loc="center left", frameon=False,
                               fontsize=9, labelcolor=INK)
    fig.suptitle(
        f"Per-dataset accuracy vs model size at L = {L} "
        "(y: test MSE averaged over the dataset's four horizons; x: median params, log)",
        fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.012,
             "0721v1 audit · per-panel y scales differ (per-dataset MSE scale) · "
             "panel tag: Ours vs the best baseline of that dataset · baselines mostly "
             "single-seed — descriptive comparison",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"perdataset_efficiency_L{L}_0721v1.{ext}",
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote perdataset_efficiency_L{L}_0721v1.png/.pdf")


def figure_b(cells):
    std = [d for d in DATASETS if not d.startswith("PEMS")]
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    wins = within = 0
    xs_all = []
    rows = []
    for L in (96, 720):
        for ds in std:
            for h in horizons(ds):
                ours_mse, ours_p = cells[(OURS, ds, L, h)]
                bm = min((m for m in MODELS if m != OURS),
                         key=lambda m: cells[(m, ds, L, h)][0])
                b_mse, b_p = cells[(bm, ds, L, h)]
                gap = (ours_mse - b_mse) / b_mse * 100
                ratio = b_p / ours_p
                rows.append((L, ds, h, ratio, gap, bm))
                xs_all.append(ratio)
                wins += gap < 0
                within += gap < 3
    ax.axhspan(-100, 0, color=WIN_FILL, zorder=0)
    ax.axhline(0, color=INK_MUTED, linewidth=0.9, zorder=1)
    ax.axvline(1, color=GRID, linewidth=0.9, zorder=1)
    for L, marker in ((96, "o"), (720, "^")):
        sub = [r for r in rows if r[0] == L]
        ax.scatter([r[3] for r in sub], [r[4] for r in sub], s=34, marker=marker,
                   facecolor=ACCENT, alpha=0.75, edgecolor=SURFACE,
                   linewidth=0.8, zorder=3,
                   label=f"L = {L}  (one dot per dataset × horizon)")
    worst = max(rows, key=lambda r: r[4])
    ax.annotate(f"{worst[1]} {worst[0]}→{worst[2]} (vs {worst[5]})",
                (worst[3], worst[4]), textcoords="offset points",
                xytext=(-8, -9), ha="right", fontsize=7.5, color=INK_MUTED)
    best = min(rows, key=lambda r: r[4])
    ax.annotate(f"{best[1]} {best[0]}→{best[2]}", (best[3], best[4]),
                textcoords="offset points", xytext=(8, -2), ha="left",
                fontsize=7.5, color=INK_MUTED)
    n = len(rows)
    ax.text(0.015, 0.83,
            f"{wins}/{n} cells strictly better than every baseline\n"
            f"{within}/{n} within +3% of the per-cell best\n"
            f"median size gap {sorted(xs_all)[len(xs_all)//2]:.0f}×",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
            color=INK, linespacing=1.6)
    ax.text(0.015, 0.03, "shaded: Ours beats the best baseline outright",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color=INK_MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Params of the per-cell best baseline ÷ Ours params (log; >1 → Ours smaller)",
                  fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("Ours MSE vs per-cell best baseline (%)", fontsize=9.5,
                  color=INK_MUTED)
    ax.set_ylim(min(r[4] for r in rows) - 2.5, max(r[4] for r in rows) + 2.5)
    style_axes(ax)
    ax.tick_params(labelsize=8.5)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                    labelcolor=INK, bbox_to_anchor=(0.0, 0.99))
    fig.suptitle("How close is Compact + Echo to the strongest baseline of each cell — "
                 "and how much smaller is it (7 standard LTSF datasets)",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.015,
             "0721v1 audit · 56 cells: ETTh1/2, ETTm1/2, weather, electricity, traffic × {96,720} × 4 horizons · "
             "best baseline chosen per cell by test MSE across all 12 baselines (oracle-strength reference)\n"
             "PEMS excluded: spatial-graph regime where channel-mixing transformers lead every "
             "channel-independent model (see per-dataset figure) · baselines mostly single-seed — descriptive",
             fontsize=6.8, color=INK_MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"advantage_vs_best_baseline_0721v1.{ext}",
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote advantage_vs_best_baseline_0721v1.png/.pdf")


def main():
    cells = load()
    OUTDIR.mkdir(exist_ok=True)
    figure_a(cells, 96)
    figure_a(cells, 720)
    figure_b(cells)


if __name__ == "__main__":
    main()
