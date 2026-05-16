"""Channel-coupling graph for AsySpecX time-delay ranks.

Visualises the directional channel relations learned by AsySpecX's cross-block,
overlaid on a physically-meaningful node layout. For each time-delay rank
(identified automatically by phase-ramp R²), the top-K |a_k[i] · b_k[j]| pairs
are drawn as directed edges j → i, colour-coded by rank, with edge thickness
proportional to coupling magnitude and labelled by the estimated lag τ̂ in
hours (assuming hourly data; configurable).

Default layout for ETTh1 places the load channels in their voltage hierarchy
(High / Middle / Low rows × Useful / Useless cols) with OT at the centre. This
makes the physical clustering of the learned modes visually obvious.

Usage:
    python analysis_exp/vis_channel_coupling.py \
        --ckpt /path/to/checkpoint.pth \
        --dataset ETTh1
"""
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Per-dataset channel layout. For each dataset we hand-pick a 2-D coordinate
# for every channel that reflects its physical/structural role.
# ---------------------------------------------------------------------------
DATASET_LAYOUTS = {
    'ETTh1': {
        'channels': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'descriptions': {
            'HUFL': 'High UseFul Load',
            'HULL': 'High UseLess Load',
            'MUFL': 'Middle UseFul Load',
            'MULL': 'Middle UseLess Load',
            'LUFL': 'Low UseFul Load',
            'LULL': 'Low UseLess Load',
            'OT':   'Oil Temperature',
        },
        'positions': {
            'HUFL': (-1.5,  1.0),
            'HULL': ( 1.5,  1.0),
            'MUFL': (-1.5,  0.0),
            'MULL': ( 1.5,  0.0),
            'LUFL': (-1.5, -1.0),
            'LULL': ( 1.5, -1.0),
            'OT':   ( 0.0,  0.0),
        },
        'node_colors': {
            'HUFL': '#fce5c4', 'HULL': '#fce5c4',
            'MUFL': '#a5d8ff', 'MULL': '#a5d8ff',
            'LUFL': '#d3f9d8', 'LULL': '#d3f9d8',
            'OT':   '#ffd1d1',
        },
        'samples_per_unit_time': 1,  # 1 hour = 1 sample (hourly data)
        'time_unit': 'h',
    },
    'ETTh2': 'ETTh1',  # alias
    'ETTm1': {
        'channels': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'descriptions': {  # same as ETTh1
            'HUFL': 'High UseFul Load',
            'HULL': 'High UseLess Load',
            'MUFL': 'Middle UseFul Load',
            'MULL': 'Middle UseLess Load',
            'LUFL': 'Low UseFul Load',
            'LULL': 'Low UseLess Load',
            'OT':   'Oil Temperature',
        },
        'positions': {
            'HUFL': (-1.5,  1.0),
            'HULL': ( 1.5,  1.0),
            'MUFL': (-1.5,  0.0),
            'MULL': ( 1.5,  0.0),
            'LUFL': (-1.5, -1.0),
            'LULL': ( 1.5, -1.0),
            'OT':   ( 0.0,  0.0),
        },
        'node_colors': {
            'HUFL': '#fce5c4', 'HULL': '#fce5c4',
            'MUFL': '#a5d8ff', 'MULL': '#a5d8ff',
            'LUFL': '#d3f9d8', 'LULL': '#d3f9d8',
            'OT':   '#ffd1d1',
        },
        'samples_per_unit_time': 4,  # 15 min = 1/4 hour
        'time_unit': 'h',
    },
    'ETTm2': 'ETTm1',
}


def resolve_layout(dataset):
    layout = DATASET_LAYOUTS.get(dataset)
    if isinstance(layout, str):
        return DATASET_LAYOUTS[layout]
    return layout


# ---------------------------------------------------------------------------
# Lag computation
# ---------------------------------------------------------------------------
def per_rank_phase_stats(g_re, g_im):
    """Return list of dicts with phase-ramp R², slope, circ_var per rank."""
    phase = np.angle(g_re + 1j * g_im)
    n_bands, n_rank = phase.shape
    out = []
    for k in range(n_rank):
        ph_un = np.unwrap(phase[:, k])
        b = np.arange(n_bands, dtype=float)
        slope, intercept = np.polyfit(b, ph_un, 1)
        pred = slope * b + intercept
        ss_res = float(np.sum((ph_un - pred) ** 2))
        ss_tot = float(np.sum((ph_un - ph_un.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        z = np.exp(1j * phase[:, k])
        circ_var = 1.0 - abs(z.mean())
        out.append({
            'rank': k, 'slope': float(slope), 'r2': float(r2),
            'circ_var': float(circ_var),
        })
    return out


def slope_to_tau_samples(slope_per_band, n_bands, F_out, N_total):
    """Convert phase slope (rad/band) to time delay τ in samples."""
    slope_per_bin = slope_per_band / (F_out / n_bands)
    return -slope_per_bin * N_total / (2 * np.pi)


def select_time_delay_ranks(stats, cos_AB, r2_thresh=0.55, sym_cos_thresh=0.7):
    """Pick ranks that look like time-delay modes:
       - phase-ramp R² high
       - spatial pattern not collapsed to symmetric (|cos(A,B)| not too high)
    """
    selected = []
    for s in stats:
        if s['r2'] >= r2_thresh and abs(cos_AB[s['rank']]) <= sym_cos_thresh:
            selected.append(s)
    return selected


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_coupling_graph(A, B, time_delay_ranks, lag_per_rank, layout,
                        out_path, top_k=4, skip_self_loops=True,
                        rank_colors=None, dataset='ETTh1'):
    """Draw the channel-coupling graph."""
    channels = layout['channels']
    positions = layout['positions']
    descriptions = layout['descriptions']
    node_colors = layout['node_colors']
    n_ch = len(channels)
    if rank_colors is None:
        # Default categorical colors
        palette = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd',
                   '#8c564b', '#e377c2']
        rank_colors = {r: palette[i % len(palette)] for i, r in enumerate(time_delay_ranks)}

    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad_x = 1.0
    pad_y = 0.7
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect('equal')
    ax.axis('off')

    # Collect edges per rank (top-K)
    all_edges = []  # (i, j, rank, weight, sign, lag)
    rank_top_pairs = {}
    for r in time_delay_ranks:
        a = A[:, r]
        b = B[:, r]
        outer = np.outer(a, b)  # outer[i, j] = a[i] * b[j]; represents j → i
        # Rank top |outer[i,j]| pairs
        flat = np.argsort(np.abs(outer).flatten())[::-1]
        kept = []
        for idx in flat:
            i, j = idx // n_ch, idx % n_ch
            if skip_self_loops and i == j:
                continue
            kept.append((i, j, r, abs(outer[i, j]), np.sign(outer[i, j]), lag_per_rank[r]))
            if len(kept) >= top_k:
                break
        all_edges.extend(kept)
        rank_top_pairs[r] = kept

    if not all_edges:
        print('  WARNING: no edges to draw')
        return

    max_weight = max(e[3] for e in all_edges)

    # Group edges by (i, j) so multiple ranks on same pair fan out into arcs.
    edge_groups = defaultdict(list)
    for e in all_edges:
        edge_groups[(e[0], e[1])].append(e)

    # Per-rank base curvature; we offset within group.
    base_rad = {r: c for r, c in zip(time_delay_ranks, [0.20, -0.20, 0.05, -0.05])}

    for (i, j), group in edge_groups.items():
        pi = positions[channels[i]]
        pj = positions[channels[j]]
        n_in_group = len(group)
        for idx_in_group, (_, _, r, w, sign, lag) in enumerate(group):
            color = rank_colors.get(r, 'gray')
            thickness = 1.5 + 5.0 * (w / max_weight)
            rad = base_rad.get(r, 0) + 0.06 * (idx_in_group - (n_in_group - 1) / 2)
            ls = 'solid' if sign > 0 else (0, (3, 2))  # dashed via tuple

            ax.annotate(
                '',
                xy=pi, xytext=pj,
                arrowprops=dict(
                    arrowstyle='-|>,head_length=0.7,head_width=0.45',
                    color=color,
                    lw=thickness,
                    ls=ls,
                    connectionstyle=f'arc3,rad={rad}',
                    shrinkA=28, shrinkB=28,
                    alpha=0.85,
                ),
                zorder=2,
            )
            # Lag label at curve apex (offset perpendicular to line by rad).
            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            mid_x = (pi[0] + pj[0]) / 2 - rad * dy * 0.5
            mid_y = (pi[1] + pj[1]) / 2 + rad * dx * 0.5
            ax.text(
                mid_x, mid_y, f'{abs(lag):.1f}h',
                fontsize=9, color=color, fontweight='bold',
                ha='center', va='center', zorder=4,
                bbox=dict(facecolor='white', edgecolor=color,
                          boxstyle='round,pad=0.15', alpha=0.95, lw=0.8),
            )

    # Draw nodes (on top of edge ends)
    for ch in channels:
        x, y = positions[ch]
        color = node_colors.get(ch, 'white')
        rect = patches.FancyBboxPatch(
            (x - 0.45, y - 0.18), 0.9, 0.36,
            boxstyle='round,pad=0.04',
            facecolor=color, edgecolor='black', linewidth=1.4, zorder=5,
        )
        ax.add_patch(rect)
        ax.text(x, y, ch, ha='center', va='center',
                fontsize=11.5, fontweight='bold', zorder=6)

    # Title and legend
    ax.set_title(
        f'AsySpecX learned cross-channel directional graph — {dataset}\n'
        f'Edges: top-{top_k} channel pairs per time-delay rank '
        f'(direction: input b_k → output a_k; label: lag τ̂)',
        fontsize=13, pad=18,
    )

    legend_elements = []
    rank_descriptions = describe_ranks(time_delay_ranks, lag_per_rank, rank_top_pairs, channels)
    for r in time_delay_ranks:
        legend_elements.append(
            Line2D([0], [0], color=rank_colors[r], lw=3,
                   label=rank_descriptions[r])
        )
    legend_elements.append(
        Line2D([0], [0], color='gray', lw=2, ls='solid', label='positive coupling')
    )
    legend_elements.append(
        Line2D([0], [0], color='gray', lw=2, ls=(0, (3, 2)), label='negative coupling')
    )
    legend_elements.append(
        Line2D([0], [0], color='none', label='edge thickness ∝ |a_k[i] · b_k[j]|')
    )
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=10, frameon=True)

    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def describe_ranks(time_delay_ranks, lag_per_rank, rank_top_pairs, channels):
    """One-line label per rank for the legend."""
    out = {}
    for r in time_delay_ranks:
        lag = abs(lag_per_rank[r])
        # Top 1 channel pair
        if rank_top_pairs.get(r):
            i, j, _, w, sign, _ = rank_top_pairs[r][0]
            top_pair = f'{channels[j]}→{channels[i]}'
        else:
            top_pair = '?'
        out[r] = f'rank {r}: τ̂≈{lag:.1f}h, top: {top_pair}'
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset', default='ETTh1',
                   choices=list(DATASET_LAYOUTS.keys()))
    p.add_argument('--seq_len', type=int, default=336)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--top_k', type=int, default=4,
                   help='top-K channel pairs per time-delay rank')
    p.add_argument('--r2_thresh', type=float, default=0.55,
                   help='phase-ramp R² above which a rank is treated as time-delay')
    p.add_argument('--out_dir', default='figures/AsySpecX_cross_vis')
    args = p.parse_args()

    layout = resolve_layout(args.dataset)
    if layout is None:
        raise ValueError(f'no layout defined for dataset {args.dataset!r}; '
                         f'add one to DATASET_LAYOUTS')

    sd = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    A = sd['cross_block.A'].numpy()
    B = sd['cross_block.B'].numpy()
    g_re = sd['cross_block.g_re'].numpy()
    g_im = sd['cross_block.g_im'].numpy()
    n_bands, n_rank = g_re.shape

    n_ch = A.shape[0]
    if n_ch != len(layout['channels']):
        print(f'WARNING: ckpt has {n_ch} channels but layout has '
              f'{len(layout["channels"])}; using first {n_ch} layout entries')

    # Phase analysis & lag conversion
    stats = per_rank_phase_stats(g_re, g_im)
    cos_AB = []
    for k in range(n_rank):
        a, b = A[:, k], B[:, k]
        cos_AB.append(float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    F_out = int(round((args.seq_len + args.pred_len) /
                      (args.seq_len) *
                      ((args.seq_len // 4 + 1) if args.seq_len <= 96 else (args.seq_len // 4 + 1))))
    # Read actual F_out from g_re band_ids if available — but cross_block stores
    # band_ids of length F_out, so use that:
    F_out = int(sd['cross_block.band_ids'].numel())
    N_total = args.seq_len + args.pred_len

    samples_per_h = layout['samples_per_unit_time']
    lag_per_rank = {}
    for k in range(n_rank):
        tau_samples = slope_to_tau_samples(stats[k]['slope'], n_bands, F_out, N_total)
        lag_per_rank[k] = tau_samples / samples_per_h  # convert to hours

    # Select time-delay ranks
    selected_stats = select_time_delay_ranks(stats, cos_AB, r2_thresh=args.r2_thresh)
    selected_ranks = [s['rank'] for s in selected_stats]
    print(f'Selected time-delay ranks (R²≥{args.r2_thresh}, '
          f'|cos(A,B)|≤0.7): {selected_ranks}')
    for r in selected_ranks:
        print(f'  rank {r}: R²={stats[r]["r2"]:.3f}  slope={stats[r]["slope"]:+.3f}  '
              f'cos(A,B)={cos_AB[r]:+.3f}  τ̂={lag_per_rank[r]:+.2f}h')

    if not selected_ranks:
        print('No time-delay ranks satisfy thresholds; lower --r2_thresh to include more.')
        return

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'channel_coupling_graph.png'
    plot_coupling_graph(
        A, B, selected_ranks, lag_per_rank, layout,
        out_path, top_k=args.top_k, dataset=args.dataset,
    )

    # Also print the channel-pair report
    rank_top_pairs = {}
    print()
    print('=== Channel-pair detail per time-delay rank ===')
    for r in selected_ranks:
        a = A[:, r]
        b = B[:, r]
        outer = np.outer(a, b)
        flat = np.argsort(np.abs(outer).flatten())[::-1]
        kept = []
        for idx in flat:
            i, j = idx // n_ch, idx % n_ch
            if i == j:
                continue
            kept.append((i, j, abs(outer[i, j]), np.sign(outer[i, j])))
            if len(kept) >= args.top_k:
                break
        ch = layout['channels']
        desc = layout['descriptions']
        print(f'\nRank {r}  (τ̂={lag_per_rank[r]:+.2f}h)')
        for i, j, w, s in kept:
            arrow = '→' if s > 0 else '⊣'
            print(f'  {ch[j]:5s} {arrow} {ch[i]:5s}   '
                  f'(|coeff|={w:.3f}, sign={"+" if s > 0 else "-"})  '
                  f'[{desc.get(ch[j], ""):s} → {desc.get(ch[i], ""):s}]')


if __name__ == '__main__':
    main()
