#!/usr/bin/env python3
"""TQNet-Figure-7-style efficiency panels for Compact + Echo (0721v1 data).

Each panel is one audited cell (dataset, L=720, H): x = total training time
(log, seconds), y = held-out test MSE, bubble area = parameter count
(log-mapped; the size legend uses the same mapping). Eight representative
models, fixed color per model across panels; every bubble carries its name in
ink so color never works alone.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "COMPACT_ECHO_ALL_BASELINES_L96_L720_0721V1_RESULTS.csv"
OUTDIR = REPO / "Figures"

# fixed categorical slots (light mode); identity also carried by text labels
COLORS = {
    "Compact + Echo": "#2a78d6",
    "TQNet": "#eb6834",
    "CycleNet": "#1baf7a",
    "FITS": "#eda100",
    "PhaseFormer": "#e87ba4",
    "iTransformer": "#008300",
    "PatchTST": "#4a3aa7",
    "FreqCycle": "#e34948",
}
MODELS = list(COLORS)
OURS = "Compact + Echo"

PANELS = [("weather", 720, 720), ("electricity", 720, 96), ("ETTm1", 720, 720)]

INK = "#33322f"
INK_MUTED = "#6f6e6a"
GRID = "#e6e5e2"
SURFACE = "#ffffff"

# per-panel label offsets (points): model -> (dx, dy, ha)
OFFSETS = {
    ("weather", 720): {
        "Compact + Echo": (0, 12, "center"),
        "TQNet": (-15, -6, "right"),
        "CycleNet": (13, 6, "left"),
        "FITS": (0, 12, "center"),
        "PhaseFormer": (0, 12, "center"),
        "iTransformer": (0, 14, "center"),
        "PatchTST": (0, -26, "center"),
        "FreqCycle": (0, -18, "center"),
    },
    ("electricity", 96): {
        "Compact + Echo": (-8, 12, "right"),
        "TQNet": (0, -20, "center"),
        "CycleNet": (-4, -18, "center"),
        "FITS": (0, 12, "center"),
        "PhaseFormer": (0, 12, "center"),
        "iTransformer": (0, 14, "center"),
        "PatchTST": (0, -26, "center"),
        "FreqCycle": (0, 14, "center"),
    },
    ("ETTm1", 720): {
        "Compact + Echo": (0, -18, "center"),
        "TQNet": (0, -20, "center"),
        "CycleNet": (14, 4, "left"),
        "FITS": (0, 12, "center"),
        "PhaseFormer": (0, 12, "center"),
        "iTransformer": (0, 14, "center"),
        "PatchTST": (0, -26, "center"),
        "FreqCycle": (0, 14, "center"),
    },
}


def bubble_area(params):
    return (math.log10(max(params, 10)) + 0.8) ** 2 * 20


def load():
    cells = {}
    for r in csv.DictReader(open(SRC)):
        if r["status"] != "ok" or r["model"] not in MODELS:
            continue
        key = (r["model"], r["dataset"], int(r["seq_len"]), int(r["pred_len"]))
        cells[key] = (float(r["mse"]), int(float(r["param_count"])),
                      float(r["train_seconds"]))
    return cells


def fmt_params(p):
    if p >= 1e6:
        return f"{p/1e6:.1f}M"
    if p >= 1e3:
        return f"{p/1e3:.0f}K"
    return str(p)


def main():
    cells = load()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ds, L, H) in zip(axes, PANELS):
        for m in MODELS:
            mse, p, t = cells[(m, ds, L, H)]
            is_ours = m == OURS
            ax.scatter(t, mse, s=bubble_area(p), facecolor=COLORS[m],
                       alpha=0.92 if is_ours else 0.72,
                       edgecolor=SURFACE, linewidth=1.4,
                       zorder=4 if is_ours else 3)
            dx, dy, ha = OFFSETS[(ds, H)][m]
            label = f"Ours ({fmt_params(p)})" if is_ours else m
            ax.annotate(label, (t, mse), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=8, color=INK,
                        fontweight="bold" if is_ours else "normal", zorder=5)
        ours_mse = cells[(OURS, ds, L, H)][0]
        ax.axhline(ours_mse, color=COLORS[OURS], linewidth=0.7,
                   linestyle=(0, (5, 4)), alpha=0.5, zorder=1)
        ax.set_xscale("log")
        ax.margins(x=0.16, y=0.20)
        ax.set_title(f"{ds}  {L} → {H}", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("Total training time, seconds (log)", fontsize=9,
                      color=INK_MUTED)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test MSE  (lower is better)", fontsize=9.5,
                       color=INK_MUTED)
    size_handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
               markeredgecolor=INK_MUTED, markersize=math.sqrt(bubble_area(p)),
               label=lab)
        for p, lab in ((1e4, "10K params"), (1e6, "1M"), (1e7, "10M"))
    ]
    leg = axes[0].legend(handles=size_handles, loc="upper right", frameon=False,
                         fontsize=7.5, labelcolor=INK_MUTED, handletextpad=2.2,
                         labelspacing=2.4, borderaxespad=0.3,
                         title="bubble area = params", title_fontsize=7.5)
    leg.get_title().set_color(INK_MUTED)
    fig.suptitle("Prediction accuracy vs training cost vs model size — "
                 "Compact + Echo is the accuracy leader in each cell",
                 fontsize=12.5, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.015,
             "0721v1 audit, one panel = one (dataset, input, horizon) cell at L=720 · "
             "dashed line: Ours MSE · 8 of 13 models shown for readability — full "
             "13-model evidence in the 0721v1 table · Ours 3-seed mean, baselines "
             "mostly single-seed (descriptive)",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig7_style_efficiency_0721v1.{ext}",
                    facecolor=SURFACE, bbox_inches="tight")
    print("wrote", OUTDIR / "fig7_style_efficiency_0721v1.png", "and .pdf")


if __name__ == "__main__":
    main()
