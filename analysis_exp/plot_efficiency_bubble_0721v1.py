#!/usr/bin/env python3
"""Bubble charts: accuracy vs model size / training cost (0721v1 table).

Marks: one bubble per model; area encodes median parameter count (log-mapped,
with a reference-size legend using the same mapping). Ours is the single accent
hue; baselines are neutral gray; identity is carried by direct text labels, so
color never carries meaning alone. Y is macro test MSE over the 44 audited
cells per input length. A dashed step line traces the Pareto frontier
(lower-left is better).
"""

import csv
import math
import statistics
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

ACCENT = "#2a78d6"      # validated categorical slot 1 (light surface)
NEUTRAL = "#8b8b86"     # de-emphasis neutral; identity comes from labels
INK = "#33322f"
INK_MUTED = "#6f6e6a"
GRID = "#e6e5e2"
SURFACE = "#ffffff"

# hand-tuned label offsets (points): (dx, dy, ha)
OFFSETS = {
    (96, "params"): {
        "Compact + Echo": (0, 11, "center"),
        "TQNet": (6, -16, "center"),
        "CycleNet": (0, 10, "center"),
        "FITS": (8, 6, "left"),
        "SparseTSF": (8, -3, "left"),
        "FreTS": (0, 11, "center"),
        "FilterNet": (0, 12, "center"),
        "iTransformer": (0, -16, "center"),
        "PatchTST": (0, 11, "center"),
        "DLinear": (8, 2, "left"),
        "MixLinear": (8, 0, "left"),
        "PhaseFormer": (8, 2, "left"),
        "FreqCycle": (-14, -15, "center"),
    },
    (720, "params"): {
        "Compact + Echo": (0, 11, "center"),
        "TQNet": (0, -15, "center"),
        "CycleNet": (-11, 2, "right"),
        "FITS": (9, 4, "left"),
        "SparseTSF": (9, -6, "left"),
        "FreTS": (-4, -16, "right"),
        "FilterNet": (0, 13, "center"),
        "iTransformer": (10, -2, "left"),
        "PatchTST": (10, -3, "left"),
        "DLinear": (-11, 2, "right"),
        "MixLinear": (0, 10, "center"),
        "PhaseFormer": (0, -14, "center"),
        "FreqCycle": (-2, -16, "right"),
    },
    (96, "time"): {
        "Compact + Echo": (0, 11, "center"),
        "TQNet": (0, -15, "center"),
        "CycleNet": (0, 10, "center"),
        "FITS": (8, 4, "left"),
        "SparseTSF": (-9, -4, "right"),
        "FreTS": (0, 11, "center"),
        "FilterNet": (0, 12, "center"),
        "iTransformer": (0, -16, "center"),
        "PatchTST": (0, 11, "center"),
        "DLinear": (8, 2, "left"),
        "MixLinear": (8, 0, "left"),
        "PhaseFormer": (0, 10, "center"),
        "FreqCycle": (-14, -3, "right"),
    },
    (720, "time"): {
        "Compact + Echo": (-14, 4, "right"),
        "TQNet": (-10, -5, "right"),
        "CycleNet": (0, 11, "center"),
        "FITS": (8, 4, "left"),
        "SparseTSF": (8, -4, "left"),
        "FreTS": (0, -16, "center"),
        "FilterNet": (0, -15, "center"),
        "iTransformer": (10, 3, "left"),
        "PatchTST": (0, -15, "center"),
        "DLinear": (-10, 2, "right"),
        "MixLinear": (0, 10, "center"),
        "PhaseFormer": (0, -14, "center"),
        "FreqCycle": (0, 12, "center"),
    },
}


def load():
    agg = {}
    for r in csv.DictReader(open(SRC)):
        if r["status"] != "ok" or r["model"] not in MODELS:
            continue
        key = (r["model"], int(r["seq_len"]))
        d = agg.setdefault(key, {"mse": [], "p": [], "t": []})
        d["mse"].append(float(r["mse"]))
        d["p"].append(int(float(r["param_count"])))
        if r["train_seconds"]:
            d["t"].append(float(r["train_seconds"]))
    stats = {}
    for (m, L), d in agg.items():
        assert len(d["mse"]) == 44, (m, L, len(d["mse"]))
        stats.setdefault(L, {})[m] = dict(
            mse=sum(d["mse"]) / len(d["mse"]),
            p=statistics.median(d["p"]),
            t=statistics.median(d["t"]),
        )
    return stats


def bubble_area(params):
    # radius ~ log10(params); legend uses the same mapping so it stays honest
    return (math.log10(max(params, 10)) + 0.8) ** 2 * 14


def pareto(points):
    pts = sorted(points, key=lambda q: (q[0], q[1]))
    front, best = [], math.inf
    for x, y in pts:
        if y < best:
            front.append((x, y))
            best = y
    return front


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def draw(stats, xkey, xlabel, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    for ax, L in zip(axes, (96, 720)):
        s = stats[L]
        xs = {m: s[m][xkey] for m in MODELS}
        ys = {m: s[m]["mse"] for m in MODELS}
        front = pareto([(xs[m], ys[m]) for m in MODELS])
        fx = [q[0] for q in front]
        fy = [q[1] for q in front]
        ax.plot(fx, fy, drawstyle="steps-post", color=INK_MUTED, linewidth=0.9,
                linestyle=(0, (4, 3)), zorder=1, alpha=0.7)
        for m in MODELS:
            is_ours = m == OURS
            ax.scatter(xs[m], ys[m], s=bubble_area(s[m]["p"]),
                       facecolor=ACCENT if is_ours else NEUTRAL,
                       alpha=0.95 if is_ours else 0.55,
                       edgecolor=SURFACE, linewidth=1.6, zorder=3 if is_ours else 2)
            dx, dy, ha = OFFSETS[(L, "params" if xkey == "p" else "time")][m]
            ax.annotate("Ours" if is_ours else m, (xs[m], ys[m]),
                        textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=8.2, color=INK,
                        fontweight="bold" if is_ours else "normal", zorder=4)
        ax.set_xscale("log")
        ax.margins(x=0.10, y=0.14)
        ax.set_title(f"L = {L}", fontsize=11, color=INK, pad=8)
        ax.set_xlabel(xlabel, fontsize=9.5, color=INK_MUTED)
        style_axes(ax)
    axes[0].set_ylabel("Macro test MSE over 44 cells  (lower is better)",
                       fontsize=9.5, color=INK_MUTED)
    size_handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
               markeredgecolor=INK_MUTED,
               markersize=math.sqrt(bubble_area(p)),
               label=lab)
        for p, lab in ((1e3, "1K params"), (1e5, "100K"), (1e7, "10M"))
    ]
    leg = axes[0].legend(handles=size_handles, loc="upper right", frameon=False,
                         fontsize=8, labelcolor=INK_MUTED, handletextpad=1.2,
                         borderaxespad=0.4, labelspacing=1.3,
                         title="bubble area = median params", title_fontsize=8)
    leg.get_title().set_color(INK_MUTED)
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.015,
             "0721v1 audit · 11 datasets × 4 horizons per L · Ours: 3-seed mean "
             "(72/88 cells true Compact) · baselines: mostly single-seed local "
             "reproductions — descriptive comparison",
             fontsize=7.2, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{fname}.{ext}", facecolor=SURFACE,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUTDIR / f"{fname}.png", "and .pdf")


def main():
    stats = load()
    draw(stats, "p", "Median parameter count (log scale)",
         "efficiency_bubble_params_0721v1",
         "Accuracy vs model size — Compact + Echo and 12 baselines")
    draw(stats, "t", "Median training time per run, seconds (log scale)",
         "efficiency_bubble_traintime_0721v1",
         "Accuracy vs training cost — Compact + Echo and 12 baselines")


if __name__ == "__main__":
    main()
