"""Visualize AsySpecX cross-channel module to demonstrate the *intended*
asymmetry of its effective channel transfer matrix.

Hypothesis under test (and the central claim of AsySpecX):
    AsySpecX's cross-channel residual is built around H_b = A diag(g_b) Bᵀ
    with A and B initialised orthogonally and trained INDEPENDENTLY. This
    parameterisation is generically asymmetric — H_b is symmetric only in the
    degenerate cases A = B (collapse to inner-product) or A ⊥ B exactly.

    In contrast:
      - FreTS' channel module (vis_FreTS.py) is mathematically symmetric in
        its dominant linear regime (~87% sym² energy on ETTh1).
      - iTransformer's softmax(QKᵀ/√d) is empirically near-symmetric after
        training (>93% sym² head-averaged).

    AsySpecX should sit on the OPPOSITE side: anti² ≫ FreTS / iTransformer.

Empirical procedure:
    1. Load a trained AsySpecX (or AsySpecXClean) checkpoint — these are
       structurally identical: keys cross_block.{A, B, g_re, g_im, gate_logit,
       band_ids}.
    2. Read the cross-block parameters and report:
         - cos(A_:k, B_:k) for each rank-k column → if all ≈ 1, A == B and the
           module collapses to symmetric.
         - ‖A − B‖_F  vs  ‖A‖_F as another asymmetry proxy.
    3. For each frequency band b, build the effective transfer matrix:
            H_b = A · diag(g_re[b] + i·g_im[b]) · Bᵀ                ∈ ℂ^{C×C}
       and decompose H_b = H_b,sym + H_b,anti where the (anti)Hermitian part
            H_sym  = (H + Hᴴ) / 2
            H_anti = (H − Hᴴ) / 2.
       Report ‖H_sym‖² / ‖H‖² per band.
    4. Compute the global effective transfer (gated, frequency-averaged real
       transfer that the residual contributes) and its sym²/anti² split.
    5. If a JSON of FreTS/iTransformer sym² results exists at
       figures/cross_model_symmetry_summary.json, emit a bar chart comparing
       all three.

Outputs (figures/AsySpecX_cross_vis/):
  A_B_heatmaps.png      — side-by-side heatmaps of A and B, cosine similarity
  per_band_transfer.png — H_b heatmaps (top: real, bottom: imag) per band
  per_band_symmetry.png — bar chart: ‖H_sym‖²/‖H‖² per band
  global_transfer.png   — effective real transfer matrix (gated, freq-avg)
  cross_model_compare.png — bar chart of sym² across FreTS/iTransformer/AsySpecX
                            (only if peer summaries are present)
  symmetry_report.txt   — numeric scores

Usage:
    cd /scratch3/lin250/bldgFM/AsySpecX
    conda activate tsfm

    python analysis_exp/vis_AsySpecX.py \\
        --ckpt /scratch3/lin250/bldgFM/Scope/checkpoints/asyspecx_clean/real_ETTh1_H96_s2026_clean_AsySpecXClean_ETTh1_ftM_sl336_pl96_cycle24_seed2026/checkpoint.pth \\
        --enc_in 7
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_cross_params(ckpt_path):
    """Returns dict of np arrays: A, B, g_re, g_im, band_ids, gate_logit."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    needed = ['cross_block.A', 'cross_block.B', 'cross_block.g_re',
              'cross_block.g_im', 'cross_block.band_ids',
              'cross_block.gate_logit']
    out = {}
    for k in needed:
        if k not in sd:
            raise KeyError(f'checkpoint missing {k}; this is not an AsySpecX-style ckpt')
        out[k.split('.', 1)[1]] = sd[k].cpu().numpy()
    return out


def hermitian_split(H):
    """Decompose square complex matrix H into Hermitian + anti-Hermitian:
        H_h = (H + Hᴴ)/2,  H_a = (H − Hᴴ)/2 (purely imaginary along the diag).
    Energy ratios use Frobenius norm."""
    Hh = 0.5 * (H + H.conj().T)
    Ha = 0.5 * (H - H.conj().T)
    n_tot = float(np.linalg.norm(H, 'fro'))
    n_h = float(np.linalg.norm(Hh, 'fro'))
    n_a = float(np.linalg.norm(Ha, 'fro'))
    return Hh, Ha, n_tot, n_h, n_a


def real_sym_split(W):
    Ws = 0.5 * (W + W.T)
    Wa = 0.5 * (W - W.T)
    n_tot = float(np.linalg.norm(W, 'fro'))
    n_s = float(np.linalg.norm(Ws, 'fro'))
    n_a = float(np.linalg.norm(Wa, 'fro'))
    return Ws, Wa, n_tot, n_s, n_a


def plot_AB_heatmaps(A, B, out_path):
    C, r = A.shape
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    vmax = max(abs(A).max(), abs(B).max())
    im = axes[0, 0].imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[0, 0].set_title(f'A  (channels × rank, shape={A.shape})')
    axes[0, 0].set_xlabel('rank k')
    axes[0, 0].set_ylabel('channel c')
    plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(B, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[0, 1].set_title(f'B  (channels × rank, shape={B.shape})')
    axes[0, 1].set_xlabel('rank k')
    axes[0, 1].set_ylabel('channel c')
    plt.colorbar(im, ax=axes[0, 1])
    diff = A - B
    vmax_d = abs(diff).max() + 1e-12
    im = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, aspect='auto')
    axes[0, 2].set_title(f'A − B   (‖A−B‖/‖A‖ = {np.linalg.norm(diff) / max(np.linalg.norm(A), 1e-12):.3f})')
    axes[0, 2].set_xlabel('rank k')
    plt.colorbar(im, ax=axes[0, 2])

    # Per-rank cosine similarity
    cos = []
    for k in range(r):
        a = A[:, k]
        b = B[:, k]
        c = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
        cos.append(c)
    ax = axes[1, 0]
    ax.bar(range(r), cos, color=['tab:red' if abs(c) > 0.95 else 'tab:blue' for c in cos])
    ax.axhline(1.0, ls='--', c='k', lw=0.5, alpha=0.5)
    ax.axhline(-1.0, ls='--', c='k', lw=0.5, alpha=0.5)
    ax.axhline(0.0, ls='-', c='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('rank k')
    ax.set_ylabel('cos(A_k, B_k)')
    ax.set_title('Per-rank A–B alignment\n(|cos|≈1 ⇒ that rank degenerates to symmetric)')
    ax.set_ylim(-1.1, 1.1)

    # Singular spectra of A and B
    sa = np.linalg.svd(A, compute_uv=False)
    sb = np.linalg.svd(B, compute_uv=False)
    ax = axes[1, 1]
    ax.plot(sa, 'o-', label='A')
    ax.plot(sb, 's-', label='B')
    ax.set_xlabel('rank k')
    ax.set_ylabel('singular value')
    ax.set_title('Singular spectra of A, B')
    ax.legend()
    ax.grid(alpha=0.3)

    # Principal-angle subspace alignment (small angle ⇒ same row-space)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1, 1)
    angles = np.degrees(np.arccos(s))
    ax = axes[1, 2]
    ax.bar(range(len(angles)), angles, color='tab:purple')
    ax.set_xlabel('principal-angle index')
    ax.set_ylabel('angle (°)')
    ax.set_title(f'Principal angles between span(A) and span(B)\n'
                 f'(0° ⇒ same subspace; 90° ⇒ orthogonal subspaces)')
    ax.set_ylim(0, 95)

    fig.suptitle('AsySpecX cross-block: A vs B', y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')
    return cos, angles


def analyze_polar_g(g_re, g_im):
    """Polar analysis of the per-band complex gain g_{b,k} ∈ ℂ.

    Returns a dict with:
      mag         [B, K] — |g|
      phase       [B, K] — arg(g) in (-π, π]
      mean_abs_phase  scalar — E[|arg g|] across all (b,k); compare to π/2
                              (≈1.57) which is the expected value under
                              uniform-random phase (= no learned structure).
      circ_var_per_rank  [K] — 1 - |E[e^{iφ_b}]| for each rank k across bands;
                              0 = perfectly coherent across bands, 1 = random.
      phase_ramp_r2_per_rank  [K] — R² of linear regression φ_b ~ b
                              (high R² ⇒ pure time-delay signature).
      phase_slope_per_rank    [K] — slope of that fit (≈ -2π·τ/N → estimated
                              effective time delay τ in samples).
      rayleigh_p   scalar — p-value of Rayleigh uniformity test on the pooled
                            phase distribution. Low p ⇒ reject uniform → phase
                            is concentrated → learned signal.
    """
    g = g_re + 1j * g_im
    mag = np.abs(g)        # [B, K]
    phase = np.angle(g)    # [B, K] in (-π, π]
    mean_abs_phase = float(np.mean(np.abs(phase)))

    n_bands, n_rank = phase.shape
    circ_var = np.zeros(n_rank)
    ramp_r2 = np.zeros(n_rank)
    ramp_slope = np.zeros(n_rank)

    for k in range(n_rank):
        # Circular variance across bands for rank k
        z = np.exp(1j * phase[:, k])
        circ_var[k] = 1.0 - abs(z.mean())
        # Phase ramp via linear regression on unwrapped phase
        ph_un = np.unwrap(phase[:, k])
        b_idx = np.arange(n_bands, dtype=float)
        # Simple OLS
        slope, intercept = np.polyfit(b_idx, ph_un, 1)
        pred = slope * b_idx + intercept
        ss_res = float(np.sum((ph_un - pred) ** 2))
        ss_tot = float(np.sum((ph_un - ph_un.mean()) ** 2))
        ramp_r2[k] = 1.0 - ss_res / max(ss_tot, 1e-12)
        ramp_slope[k] = slope

    # Rayleigh test for uniformity (pooled across all (b,k))
    phases_pooled = phase.flatten()
    n = len(phases_pooled)
    z_bar = np.mean(np.exp(1j * phases_pooled))
    R = abs(z_bar)            # length of mean resultant vector
    Z = n * R ** 2            # Rayleigh's Z
    # asymptotic p-value (Mardia & Jupp 2000, Eq 6.3.3 with first-order correction)
    p = float(np.exp(-Z) * (1.0 + (2 * Z - Z ** 2) / (4 * n) - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2)))
    p = max(min(p, 1.0), 0.0)

    return dict(
        mag=mag, phase=phase,
        mean_abs_phase=mean_abs_phase,
        circ_var_per_rank=circ_var,
        phase_ramp_r2_per_rank=ramp_r2,
        phase_slope_per_rank=ramp_slope,
        rayleigh_p=p, rayleigh_R=float(R), rayleigh_Z=float(Z),
    )


def classify_rank(cos_AB, circ_var, ramp_r2, slope, mag_mean):
    """Classify a rank's mode based on its statistics. Returns (label, color)."""
    # Spatial symmetric: cos(A,B) close to ±1 ⇒ b=±a ⇒ outer product symmetric
    if abs(cos_AB) > 0.7:
        return 'Symmetric (spatial)', 'tab:gray'
    # Phase concentrated (low var) but no ramp ⇒ constant non-zero phase
    if circ_var < 0.35 and ramp_r2 < 0.3:
        return 'Const-phase directional', 'tab:orange'
    # Linear phase ramp ⇒ time-delay mode
    if ramp_r2 > 0.55:
        return f'Time-delay (τ̂={-slope * 8 / (2 * np.pi):+.2f})', 'tab:red'
    # Magnitude small or all noisy
    if mag_mean < 0.05 or (circ_var > 0.7 and ramp_r2 < 0.3):
        return 'Noise / unused', 'tab:olive'
    return 'Mixed / weak', 'tab:purple'


def plot_polar_g(g_re, g_im, polar_stats, out_path, cos_AB=None):
    """Per-rank polar trajectories — each rank gets its own subplot so the
    band-trajectory shape is clearly visible."""
    g = g_re + 1j * g_im
    mag = polar_stats['mag']
    phase = polar_stats['phase']
    n_bands, n_rank = g.shape
    cos_AB = cos_AB if cos_AB is not None else np.zeros(n_rank)

    # Layout: 2 rows of n_rank subplots (top: per-rank polar, bottom: phase ramp)
    # plus 1 full-width row at the bottom for diagnostics.
    fig = plt.figure(figsize=(2.6 * n_rank, 9.5))
    gs = fig.add_gridspec(3, n_rank, hspace=0.6, wspace=0.4,
                          height_ratios=[2.6, 1.6, 1.6])

    band_cmap = plt.cm.viridis
    band_colors = [band_cmap(b / max(n_bands - 1, 1)) for b in range(n_bands)]

    lim = max(abs(g.real).max(), abs(g.imag).max()) * 1.15

    classifications = []
    for k in range(n_rank):
        # Top row: complex-plane trajectory with arrows from band b → b+1
        ax = fig.add_subplot(gs[0, k])
        ax.axhline(0, color='gray', lw=0.4)
        ax.axvline(0, color='gray', lw=0.4)
        # reference circles
        for r in [lim * 0.33, lim * 0.66, lim * 0.99]:
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(r * np.cos(th), r * np.sin(th), 'k:', lw=0.3, alpha=0.4)
        # trajectory line connecting bands
        gk_re = g[:, k].real
        gk_im = g[:, k].imag
        ax.plot(gk_re, gk_im, '-', color='tab:blue', lw=1.0, alpha=0.5)
        # arrows between consecutive bands to show direction of travel
        for b in range(n_bands - 1):
            dx = gk_re[b + 1] - gk_re[b]
            dy = gk_im[b + 1] - gk_im[b]
            if abs(dx) + abs(dy) > 1e-8:
                ax.annotate(
                    '', xy=(gk_re[b + 1], gk_im[b + 1]),
                    xytext=(gk_re[b], gk_im[b]),
                    arrowprops=dict(arrowstyle='->', color='tab:blue',
                                    alpha=0.5, lw=0.7),
                )
        # markers per band, colored by band index
        for b in range(n_bands):
            ax.scatter(gk_re[b], gk_im[b], color=band_colors[b],
                       s=80, edgecolors='k', linewidths=0.5, zorder=3)
            ax.annotate(str(b), (gk_re[b], gk_im[b]), fontsize=7,
                        ha='center', va='center', zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

        # Classification
        mag_mean = float(mag[:, k].mean())
        circ_var = float(polar_stats['circ_var_per_rank'][k])
        r2 = float(polar_stats['phase_ramp_r2_per_rank'][k])
        slope = float(polar_stats['phase_slope_per_rank'][k])
        label, color = classify_rank(cos_AB[k], circ_var, r2, slope, mag_mean)
        classifications.append((k, label, color, cos_AB[k], circ_var, r2, slope, mag_mean))
        ax.set_title(f'rank {k}\n{label}', fontsize=9, color=color, fontweight='bold')
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_xlabel('Re(g)', fontsize=8)
            ax.set_ylabel('Im(g)', fontsize=8)

    # Middle row: phase vs band (linear) per rank
    for k in range(n_rank):
        ax = fig.add_subplot(gs[1, k])
        ph = phase[:, k]
        ph_un = np.unwrap(ph)
        b_idx = np.arange(n_bands, dtype=float)
        slope, intercept = np.polyfit(b_idx, ph_un, 1)
        ax.plot(b_idx, ph_un, 'o-', color='tab:blue', markersize=5)
        ax.plot(b_idx, slope * b_idx + intercept, '--', color='tab:red',
                alpha=0.6, lw=1.0)
        ax.axhline(0, color='gray', lw=0.4)
        r2 = float(polar_stats['phase_ramp_r2_per_rank'][k])
        ax.set_title(f'φ vs band\nR²={r2:.2f}, slope={slope:+.2f}', fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_xlabel('band b', fontsize=8)
            ax.set_ylabel('unwrapped φ', fontsize=8)

    # Bottom row: |g| vs band per rank (magnitude profile)
    for k in range(n_rank):
        ax = fig.add_subplot(gs[2, k])
        ax.bar(np.arange(n_bands), mag[:, k], color=band_colors)
        ax.set_title(f'|g| profile\n‖g_{{:,k}}‖={np.linalg.norm(g[:, k]):.3f}',
                     fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_xlabel('band b', fontsize=8)
            ax.set_ylabel('|g|', fontsize=8)

    fig.suptitle(
        'AsySpecX per-rank g_b complex trajectory\n'
        'Top: complex-plane path band 0→7 (arrows mark direction). Middle: '
        'unwrapped phase vs band (red dashed = linear fit, slope ≠ 0 ⇒ time delay). '
        'Bottom: magnitude per band.',
        y=1.0, fontsize=11,
    )
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')

    # Print classification summary
    print('  per-rank classification:')
    for k, label, color, ca, cv, r2, sl, mm in classifications:
        print(f'    rank {k}: {label:30s} '
              f'cos(A,B)={ca:+.3f} circ_var={cv:.3f} R²={r2:.3f} '
              f'slope={sl:+.3f} <|g|>={mm:.3f}')

    return classifications


def plot_polar_summary_overlay(g_re, g_im, classifications, out_path):
    """One additional figure: all 7 trajectories overlaid in one complex plane,
    color-coded by classification (so you can spot which ranks are doing what)."""
    g = g_re + 1j * g_im
    n_bands, n_rank = g.shape
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    lim = max(abs(g.real).max(), abs(g.imag).max()) * 1.15
    th = np.linspace(0, 2 * np.pi, 100)
    for r in [lim * 0.33, lim * 0.66, lim * 0.99]:
        ax.plot(r * np.cos(th), r * np.sin(th), 'k:', lw=0.3, alpha=0.4)

    seen_labels = set()
    for k, label, color, *_ in classifications:
        gk = g[:, k]
        ax.plot(gk.real, gk.imag, 'o-', color=color, lw=1.5,
                markersize=6, alpha=0.85, zorder=2,
                label=label if label not in seen_labels else None)
        seen_labels.add(label)
        # band 0 marker
        ax.scatter([gk[0].real], [gk[0].imag], s=120, marker='s',
                   color=color, edgecolors='k', linewidths=1.0, zorder=3)
        # band 7 marker
        ax.scatter([gk[-1].real], [gk[-1].imag], s=120, marker='*',
                   color=color, edgecolors='k', linewidths=1.0, zorder=3)
        ax.annotate(f' r{k}', (gk[-1].real, gk[-1].imag), fontsize=8,
                    color=color, fontweight='bold')

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Re(g)')
    ax.set_ylabel('Im(g)')
    ax.set_title(
        'AsySpecX g_b trajectories — all ranks overlaid\n'
        '(■ = band 0 start, ★ = band 7 end, line = trajectory)\n'
        'colors classify mode type'
    )
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')


def plot_per_band_transfer(A, B, g_re, g_im, out_path):
    """Build H_b = A diag(g_b) Bᵀ for each band and plot real+imag heatmaps."""
    n_bands = g_re.shape[0]
    C = A.shape[0]
    fig, axes = plt.subplots(2, n_bands, figsize=(2.4 * n_bands, 5.0))
    if n_bands == 1:
        axes = axes[:, None]
    H_list = []
    for b in range(n_bands):
        g_b = g_re[b] + 1j * g_im[b]
        H_b = A @ np.diag(g_b) @ B.T  # complex [C, C]
        H_list.append(H_b)
        # plot real + imag side by side per band
        vmax = max(abs(H_b.real).max(), abs(H_b.imag).max()) + 1e-12
        im = axes[0, b].imshow(H_b.real, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, b].set_title(f'b{b}: Re(H)\n‖g_b‖={np.linalg.norm(g_b):.3f}', fontsize=8)
        axes[0, b].set_xticks([]); axes[0, b].set_yticks([])
        im2 = axes[1, b].imshow(H_b.imag, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, b].set_title('Im(H)', fontsize=8)
        axes[1, b].set_xticks([]); axes[1, b].set_yticks([])
        if b == 0:
            axes[0, 0].set_ylabel('Re(H_b)')
            axes[1, 0].set_ylabel('Im(H_b)')
    fig.suptitle('Per-band effective channel transfer  H_b = A · diag(g_b) · Bᵀ', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')
    return H_list


def plot_per_band_symmetry(H_list, out_path):
    """Bar chart: ‖H_h‖²/‖H‖² per band (Hermitian = correlation-like)."""
    n_bands = len(H_list)
    sym_ratios = []
    anti_ratios = []
    norms = []
    for H in H_list:
        _, _, n_tot, n_h, n_a = hermitian_split(H)
        sym_ratios.append((n_h / max(n_tot, 1e-12)) ** 2)
        anti_ratios.append((n_a / max(n_tot, 1e-12)) ** 2)
        norms.append(n_tot)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(n_bands)
    w = 0.4
    ax = axes[0]
    ax.bar(x - w / 2, sym_ratios, w, label='Hermitian (corr)', color='tab:blue')
    ax.bar(x + w / 2, anti_ratios, w, label='anti-Hermitian (directional)', color='tab:red')
    ax.axhline(0.5, ls='--', c='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('frequency band b')
    ax.set_ylabel('energy fraction (squared Frobenius)')
    ax.set_title('AsySpecX H_b symmetry decomposition')
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.bar(x, norms, color='tab:gray')
    ax.set_xlabel('frequency band b')
    ax.set_ylabel('‖H_b‖_F')
    ax.set_title('Per-band transfer magnitude')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')
    return sym_ratios, anti_ratios, norms


def plot_global_transfer(A, B, g_re, g_im, gate_max, gate_logit, band_ids, out_path):
    """The residual that AsySpecX adds is gate * Σ_f H_b(f) * U[f] in freq.
    Treating this as an aggregate channel transfer, the magnitude-weighted
    real transfer is gate * Σ_f Re(H_b(f)) (per-channel input mass spread)."""
    n_freqs = len(band_ids)
    gate = gate_max * (1 / (1 + np.exp(-gate_logit)))  # sigmoid scalar
    # Average over all freq bins by mapping each freq to its band
    H_total = np.zeros((A.shape[0], A.shape[0]), dtype=complex)
    for f in range(n_freqs):
        b = int(band_ids[f])
        g_b = g_re[b] + 1j * g_im[b]
        H_total = H_total + (A @ np.diag(g_b) @ B.T)
    H_total = H_total / max(n_freqs, 1) * gate  # avg per-freq transfer scaled by gate

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    H_real = H_total.real
    Ws, Wa, n_tot, n_s, n_a = real_sym_split(H_real)
    vmax = abs(H_real).max() + 1e-12
    vmax_a = abs(Wa).max() + 1e-12

    im = axes[0].imshow(H_real, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title(f'Re(H_global) (gated, freq-avg)\n'
                      f'gate = gate_max·σ(logit) = {gate:.3f}')
    axes[0].set_xlabel('input channel j')
    axes[0].set_ylabel('output channel i')
    plt.colorbar(im, ax=axes[0])

    im = axes[1].imshow(Ws, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('W_sym = (W + Wᵀ)/2')
    plt.colorbar(im, ax=axes[1])

    im = axes[2].imshow(Wa, cmap='RdBu_r', vmin=-vmax_a, vmax=vmax_a)
    axes[2].set_title('W_anti = (W − Wᵀ)/2')
    plt.colorbar(im, ax=axes[2])

    ax = axes[3]
    ax.scatter(H_real.flatten(), H_real.T.flatten(), s=8, alpha=0.6)
    lim = abs(H_real).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('W[i,j]')
    ax.set_ylabel('W[j,i]')
    ax.set_aspect('equal')
    ax.set_title(f'W vs Wᵀ\nsym²/total² = {(n_s / n_tot) ** 2:.3f}')

    fig.suptitle(
        f'AsySpecX effective real channel transfer (cross-block residual contribution)\n'
        f'‖W_sym‖² / ‖W‖² = {(n_s / n_tot) ** 2:.3f}    '
        f'‖W_anti‖² / ‖W‖² = {(n_a / n_tot) ** 2:.3f}',
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}  '
          f'sym²={n_s ** 2 / n_tot ** 2:.3f}  anti²={n_a ** 2 / n_tot ** 2:.3f}')
    return (n_s / n_tot) ** 2, (n_a / n_tot) ** 2, gate


def plot_cross_model_compare(asyspec_sym2, peer_path, out_path):
    """If a peer-summary JSON exists, draw a bar chart of sym² across models."""
    if not peer_path.exists():
        print(f'(no peer summary at {peer_path}; skipping cross-model bar chart. '
              f'Run vis_FreTS / vis_iTransformer first to populate it.)')
        return None
    with peer_path.open() as f:
        peers = json.load(f)
    models = list(peers.keys()) + ['AsySpecX']
    sym2 = [peers[m]['sym2'] for m in peers] + [asyspec_sym2]
    anti2 = [1 - s for s in sym2]
    x = np.arange(len(models))
    w = 0.4
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.bar(x - w / 2, sym2, w, label='symmetric energy (correlation)', color='tab:blue')
    ax.bar(x + w / 2, anti2, w, label='antisymmetric energy (directional)', color='tab:red')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('energy fraction')
    ax.set_title('Cross-model channel-transfer symmetry')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to AsySpecX/AsySpecXClean checkpoint.pth')
    p.add_argument('--enc_in', type=int, default=None,
                   help='Channel count, used only for sanity check')
    p.add_argument('--gate_max', type=float, default=0.2,
                   help='gate_max from training (default 0.2 per AsySpecX runs)')
    p.add_argument('--out_dir', type=str, default='figures/AsySpecX_cross_vis')
    p.add_argument('--peer_summary', type=str,
                   default='figures/cross_model_symmetry_summary.json')
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output dir: {out_dir}')

    params = load_cross_params(args.ckpt)
    A = params['A']
    B = params['B']
    g_re = params['g_re']
    g_im = params['g_im']
    band_ids = params['band_ids']
    gate_logit = float(params['gate_logit'])
    print(f'cross-block: A shape={A.shape}, B shape={B.shape}, '
          f'g shape={g_re.shape} (bands × rank), '
          f'gate_logit={gate_logit:.4f} -> gate={args.gate_max / (1 + np.exp(-gate_logit)):.4f}')
    if args.enc_in and A.shape[0] != args.enc_in:
        print(f'WARNING: --enc_in={args.enc_in} but ckpt has C={A.shape[0]}')

    # 1. A vs B alignment
    cos, angles = plot_AB_heatmaps(A, B, out_dir / 'A_B_heatmaps.png')

    # 1b. Polar analysis of complex gain g — frequency-domain phase = time-domain lag
    polar_stats = analyze_polar_g(g_re, g_im)
    classifications = plot_polar_g(g_re, g_im, polar_stats,
                                   out_dir / 'polar_g.png',
                                   cos_AB=np.array(cos))
    plot_polar_summary_overlay(g_re, g_im, classifications,
                               out_dir / 'polar_g_overlay.png')
    print(f'  polar: mean|φ|={polar_stats["mean_abs_phase"]:.3f} '
          f'(uniform-rand baseline=1.571)  '
          f'Rayleigh R={polar_stats["rayleigh_R"]:.3f}  p={polar_stats["rayleigh_p"]:.2e}')

    # 2. Per-band complex transfer H_b
    H_list = plot_per_band_transfer(A, B, g_re, g_im, out_dir / 'per_band_transfer.png')

    # 3. Per-band symmetry decomposition
    sym_per_b, anti_per_b, norms_per_b = plot_per_band_symmetry(
        H_list, out_dir / 'per_band_symmetry.png')

    # 4. Global gated real transfer
    sym2_global, anti2_global, gate_val = plot_global_transfer(
        A, B, g_re, g_im, args.gate_max, gate_logit, band_ids,
        out_dir / 'global_transfer.png',
    )

    # 5. Cross-model bar chart (optional)
    plot_cross_model_compare(
        sym2_global,
        ROOT / args.peer_summary,
        out_dir / 'cross_model_compare.png',
    )

    # 6. Numeric report
    rep = out_dir / 'symmetry_report.txt'
    rep.write_text(
        'AsySpecX cross-block asymmetry analysis\n'
        '========================================\n\n'
        f'Checkpoint: {args.ckpt}\n'
        f'Channels (C) : {A.shape[0]}\n'
        f'Rank     (r) : {A.shape[1]}\n'
        f'Bands         : {g_re.shape[0]}\n'
        f'Frequencies   : {len(band_ids)}\n'
        f'gate_max      : {args.gate_max}\n'
        f'gate_logit    : {gate_logit:.4f}\n'
        f'gate (σ-applied): {gate_val:.4f}\n\n'
        f'A vs B alignment:\n'
        f'  ‖A − B‖_F / ‖A‖_F = {np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-12):.4f}\n'
        f'  per-rank cos(A_k, B_k) = {[f"{c:+.3f}" for c in cos]}\n'
        f'  principal angles (deg) between span(A), span(B) = '
        f'{[f"{a:.1f}" for a in angles]}\n\n'
        f'Per-band Hermitian (correlation) energy fraction ‖H_h‖²/‖H‖²:\n'
        + '\n'.join(f'  band {b}: sym² = {sym_per_b[b]:.4f}    anti² = {anti_per_b[b]:.4f}    '
                    f'‖H_b‖ = {norms_per_b[b]:.4f}'
                    for b in range(len(sym_per_b)))
        + '\n\n'
        f'Global real transfer (gated, freq-avg):\n'
        f'  ‖W_sym‖² / ‖W‖² = {sym2_global:.4f}\n'
        f'  ‖W_anti‖²/ ‖W‖² = {anti2_global:.4f}\n\n'
        f'Polar analysis of complex gain g_{{b,k}} (Fourier-shift signature):\n'
        f'  E[|arg g|]            = {polar_stats["mean_abs_phase"]:.4f} rad   '
        f'(uniform-random baseline = π/2 ≈ 1.5708)\n'
        f'  Rayleigh R̄           = {polar_stats["rayleigh_R"]:.4f}\n'
        f'  Rayleigh p (vs uniform)= {polar_stats["rayleigh_p"]:.3e}\n'
        f'  per-rank circular var = '
        + ' '.join(f'{c:.3f}' for c in polar_stats['circ_var_per_rank']) + '\n'
        f'  per-rank ramp R²       = '
        + ' '.join(f'{c:.3f}' for c in polar_stats['phase_ramp_r2_per_rank']) + '\n'
        f'  per-rank phase slope  = '
        + ' '.join(f'{c:+.3f}' for c in polar_stats['phase_slope_per_rank']) + '\n\n'
        f'Interpretation:\n'
        f'  - Per-rank cos(A_k, B_k) far from ±1 ⇒ A and B point to genuinely\n'
        f'    different subspaces; cross transfer is NOT a degenerate inner\n'
        f'    product. Compare to FreTS, where the equivalent collapse is\n'
        f'    forced symmetric by tied per-freq weights.\n'
        f'  - anti² > 0.3 (per-band or global) ⇒ AsySpecX has captured a\n'
        f'    directional structure baselines architecturally cannot model.\n'
        f'  - phase(g) ≠ 0 with low Rayleigh p ⇒ frequency-domain phase shift\n'
        f'    is statistically significant. By the Fourier shift theorem this\n'
        f'    is the canonical signature of a TIME-DOMAIN LAG between channels:\n'
        f'        x_i(t) ≈ α · x_j(t - τ)   ⇔   X_i[k] = α e^{{-i·2πk τ/N}} · X_j[k].\n'
        f'    Symmetric (real) channel transfers (FreTS, iTransformer-empirical)\n'
        f'    cannot produce non-zero phase. Falsify via ablation: train with\n'
        f'    g_im ≡ 0 (real-only g) and compare mse on directional-rich datasets.\n'
    )
    print(f'  saved {rep}')

    # 7. Append to peer summary so cross-model bar chart can include AsySpecX
    summary_path = ROOT / args.peer_summary
    if summary_path.exists():
        try:
            with summary_path.open() as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    else:
        summary = {}
    summary['AsySpecX'] = {
        'sym2': float(sym2_global),
        'anti2': float(anti2_global),
        'source_ckpt': args.ckpt,
    }
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)
    print(f'  updated {summary_path}')


if __name__ == '__main__':
    main()
