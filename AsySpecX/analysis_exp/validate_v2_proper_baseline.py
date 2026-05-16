"""Re-do AsySpecX validation with two more appropriate baselines:

  Baseline A — Cross-correlation peak (time-domain, linear, lag-explicit)
    For each ordered channel pair (i, j), compute lagged cross-correlation
    R_ij(τ) for τ ∈ [-max_lag, +max_lag] hours, find peak |R*|, signed peak τ*.
    Same mathematical class as AsySpecX's per-mode time-shift, so this is the
    apples-to-apples lag indicator.

  Baseline B — Coherence + Phase Slope Index (frequency-domain, banded)
    Compute Welch cross-spectrum per (i, j), aggregate into the SAME 8 bands
    AsySpecX uses, fit phase slope per band. PSI_ij = mean phase slope across
    bands ⇒ frequency-domain ground truth for "j leads i with linear phase
    ramp", directly parallel to our g_b polar analysis.

  Both are compared against AsySpecX's aggregated coupling
  (Σ_{k ∈ TD} |a_k[i] · b_k[j]|, averaged across 3 seeds) via Spearman/Pearson.

  Plus a third diagnostic — Procrustes alignment between seed pairs at the
  *aggregate* matrix level — to test whether the per-seed coupling MATRICES
  (not labels) are consistent.

Output (figures/AsySpecX_validation/):
  v2_cross_correlation_matrices.png    — XC peak |R*| and peak τ heatmaps
  v2_coherence_psi_matrices.png        — coherence and phase-slope heatmaps
  v2_three_way_scatter.png             — AsySpecX vs each ground truth
  v2_procrustes_seed_distance.txt      — pairwise seed distances at aggregate level
  v2_summary.txt                       — final verdict
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load same constants from earlier validation
DATA_PATH = ROOT / "dataset" / "ETT-small" / "ETTh1.csv"
CKPT_PATHS = {
    2026: "/scratch3/lin250/bldgFM/Scope/checkpoints/asyspecx_clean/real_ETTh1_H96_s2026_clean_AsySpecXClean_ETTh1_ftM_sl336_pl96_cycle24_seed2026/checkpoint.pth",
    2027: "/scratch3/lin250/bldgFM/Scope/checkpoints/asyspecx_clean/real_ETTh1_H96_s2027_clean_AsySpecXClean_ETTh1_ftM_sl336_pl96_cycle24_seed2027/checkpoint.pth",
    2028: "/scratch3/lin250/bldgFM/Scope/checkpoints/asyspecx_clean/real_ETTh1_H96_s2028_clean_AsySpecXClean_ETTh1_ftM_sl336_pl96_cycle24_seed2028/checkpoint.pth",
}
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
N_CH = 7
SEQ_LEN, PRED_LEN = 336, 96
N_BANDS = 8
F_OUT = 108
N_TOTAL = SEQ_LEN + PRED_LEN  # 432h

OUT_DIR = ROOT / "figures" / "AsySpecX_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# AsySpecX side
# ---------------------------------------------------------------------------
def load_cross_block(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return {
        "A": sd["cross_block.A"].numpy(),
        "B": sd["cross_block.B"].numpy(),
        "g_re": sd["cross_block.g_re"].numpy(),
        "g_im": sd["cross_block.g_im"].numpy(),
    }


def per_rank_stats(g_re, g_im, A, B):
    n_rank = A.shape[1]
    phase = np.angle(g_re + 1j * g_im)
    out = []
    for k in range(n_rank):
        ph_un = np.unwrap(phase[:, k])
        b = np.arange(N_BANDS, dtype=float)
        slope = np.polyfit(b, ph_un, 1)[0]
        pred = slope * b + np.polyfit(b, ph_un, 1)[1]
        ss_res = float(np.sum((ph_un - pred) ** 2))
        ss_tot = float(np.sum((ph_un - ph_un.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        cos_ab = float(A[:, k] @ B[:, k]) / max(
            np.linalg.norm(A[:, k]) * np.linalg.norm(B[:, k]), 1e-12
        )
        out.append({"rank": k, "r2": float(r2), "slope": float(slope), "cos_ab": cos_ab})
    return out


def is_time_delay_rank(s, r2_thresh=0.55, cos_thresh=0.7):
    return s["r2"] > r2_thresh and abs(s["cos_ab"]) <= cos_thresh


def aggregate_asyspecx(seeds_dict):
    """Return [N×N] of mean |a_k[i]·b_k[j]| over time-delay ranks, averaged
    over seeds."""
    accum = np.zeros((N_CH, N_CH))
    n_used = 0
    for seed, params in seeds_dict.items():
        stats = per_rank_stats(params["g_re"], params["g_im"],
                               params["A"], params["B"])
        td = [s["rank"] for s in stats if is_time_delay_rank(s)]
        if not td:
            continue
        coupling = np.zeros((N_CH, N_CH))
        for k in td:
            coupling += np.abs(np.outer(params["A"][:, k], params["B"][:, k]))
        accum += coupling
        n_used += 1
    return accum / max(n_used, 1)


# ---------------------------------------------------------------------------
# Baseline A — cross-correlation peak
# ---------------------------------------------------------------------------
def detrend_deseason(x, season=24):
    """Remove linear trend then daily seasonal mean for hourly data."""
    n = len(x)
    t = np.arange(n)
    slope, intercept = np.polyfit(t, x, 1)
    x = x - (slope * t + intercept)
    # Subtract per-hour-of-day mean
    if season > 0:
        idx = np.arange(n) % season
        seasonal = np.zeros(season)
        for h in range(season):
            seasonal[h] = x[idx == h].mean()
        x = x - seasonal[idx]
    return x


def cross_correlation_matrix(data, max_lag=12):
    """For each (i, j), compute cross-correlation R_ij(τ) for τ in [-max_lag,
    +max_lag], record peak magnitude and signed peak lag."""
    # data: [T, N]
    n = data.shape[1]
    # Standardize so correlation is normalized
    x = (data - data.mean(0)) / (data.std(0) + 1e-9)
    R_peak = np.zeros((n, n))
    tau_peak = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi = x[:, i]
            xj = x[:, j]
            # corrcoef at various lags: R(τ) = corr(xi[τ:], xj[:T-τ]) for τ>0
            #                          R(-τ) = corr(xi[:T-τ], xj[τ:])
            taus = list(range(-max_lag, max_lag + 1))
            rs = []
            for tau in taus:
                if tau >= 0:
                    a = xi[tau:]
                    b = xj[:len(xj) - tau] if tau > 0 else xj
                else:
                    a = xi[:tau]  # tau is negative, so [:tau] drops last |tau|
                    b = xj[-tau:]
                m = min(len(a), len(b))
                if m < 100:
                    rs.append(0.0)
                    continue
                r = float(np.corrcoef(a[:m], b[:m])[0, 1])
                rs.append(r)
            rs = np.array(rs)
            # Pick peak by max |r|
            peak_idx = np.argmax(np.abs(rs))
            R_peak[i, j] = rs[peak_idx]
            tau_peak[i, j] = taus[peak_idx]
    return R_peak, tau_peak


# ---------------------------------------------------------------------------
# Baseline B — coherence + phase slope index (frequency banded)
# ---------------------------------------------------------------------------
def coherence_psi_matrix(data, n_bands=N_BANDS, fs=1.0, nperseg=256):
    """Compute coherence and per-band phase slope for each (i, j) pair using
    Welch cross-spectrum.

    Returns:
      coh_mean [N×N]   mean coherence over freq bins (broadband strength)
      psi_mean [N×N]   mean phase slope over bands (signed; lag-like)
      psi_lag  [N×N]   converted to hours under same convention as AsySpecX
    """
    from scipy.signal import csd
    n = data.shape[1]
    # Pre-process per channel
    proc = np.column_stack([detrend_deseason(data[:, c], season=24) for c in range(n)])
    # Standardize again post-detrend
    proc = (proc - proc.mean(0)) / (proc.std(0) + 1e-9)

    coh = np.zeros((n, n))
    psi = np.zeros((n, n))

    # First freq bins: compute one csd to get freq vector
    f, _ = csd(proc[:, 0], proc[:, 1], fs=fs, nperseg=nperseg)
    # Partition freq into n_bands contiguous bins
    band_edges = np.linspace(0, len(f), n_bands + 1).astype(int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            f_, Pxy = csd(proc[:, i], proc[:, j], fs=fs, nperseg=nperseg)
            f_, Pxx = csd(proc[:, i], proc[:, i], fs=fs, nperseg=nperseg)
            f_, Pyy = csd(proc[:, j], proc[:, j], fs=fs, nperseg=nperseg)
            coh_f = np.abs(Pxy) ** 2 / (np.abs(Pxx) * np.abs(Pyy) + 1e-12)
            phase_f = np.unwrap(np.angle(Pxy))
            # Per-band phase slope
            band_phase = np.zeros(n_bands)
            for b in range(n_bands):
                lo, hi = band_edges[b], band_edges[b + 1]
                if hi - lo >= 2:
                    band_phase[b] = np.mean(phase_f[lo:hi])
            # Slope of band_phase vs band index = the per-band phase slope.
            slope_per_band = np.polyfit(np.arange(n_bands), band_phase, 1)[0]
            coh[i, j] = float(coh_f.mean())
            psi[i, j] = float(slope_per_band)

    # Convert psi (rad / band) to hours (using same conversion as AsySpecX)
    # τ_hours = -slope_per_band / freq_per_band * N_TOTAL / (2π)   (sl=336 default)
    bins_per_band = F_OUT / N_BANDS
    psi_lag_h = -psi / bins_per_band * N_TOTAL / (2 * np.pi)
    return coh, psi, psi_lag_h


# ---------------------------------------------------------------------------
# Procrustes alignment for seed-pair distance at aggregate level
# ---------------------------------------------------------------------------
def aggregate_distance_between_seeds(seed_params_a, seed_params_b):
    """Compute Frobenius distance between aggregate coupling matrices of two
    seeds (no rank-permutation required because we already aggregated)."""
    G_a = aggregate_asyspecx({0: seed_params_a})
    G_b = aggregate_asyspecx({1: seed_params_b})
    diff = np.linalg.norm(G_a - G_b, 'fro')
    return diff, np.linalg.norm(G_a, 'fro'), np.linalg.norm(G_b, 'fro')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1: Build AsySpecX aggregated coupling (3 seeds)")
    print("=" * 70)
    seeds_dict = {}
    for seed, ckpt in CKPT_PATHS.items():
        if os.path.exists(ckpt):
            seeds_dict[seed] = load_cross_block(ckpt)
    print(f"  loaded {len(seeds_dict)} seed checkpoints")

    G_asy = aggregate_asyspecx(seeds_dict)
    print(f"  AsySpecX aggregated matrix:  ‖G‖_F = {np.linalg.norm(G_asy, 'fro'):.4f}")

    # Aggregate-level cross-seed distance
    seeds_l = list(seeds_dict.keys())
    print("\n  Pairwise aggregate-coupling distance (Frobenius):")
    pair_dists = []
    for i in range(len(seeds_l)):
        for j in range(i + 1, len(seeds_l)):
            d, na, nb = aggregate_distance_between_seeds(
                seeds_dict[seeds_l[i]], seeds_dict[seeds_l[j]]
            )
            rel = d / (0.5 * (na + nb))
            print(f"    seed {seeds_l[i]} vs {seeds_l[j]}: "
                  f"‖G_a − G_b‖ = {d:.4f}  ({rel*100:.1f}% of mean ‖G‖)")
            pair_dists.append((seeds_l[i], seeds_l[j], d, rel))

    print("\n" + "=" * 70)
    print("STEP 2: Cross-correlation ground-truth")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH)
    data = df[CHANNELS].values[:12000]
    # Apply detrend + deseason for stationarity
    data_clean = np.column_stack([detrend_deseason(data[:, c], season=24)
                                   for c in range(N_CH)])
    print("  data: 12000 hourly samples, detrended + de-daily-seasoned")
    R_peak, tau_peak = cross_correlation_matrix(data_clean, max_lag=12)
    pd.DataFrame(R_peak, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "v2_xc_peak_R.csv")
    pd.DataFrame(tau_peak, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "v2_xc_peak_tau.csv")
    print("  saved v2_xc_peak_R.csv and v2_xc_peak_tau.csv")

    print("\n" + "=" * 70)
    print("STEP 3: Coherence + phase slope ground-truth")
    print("=" * 70)
    coh, psi_band, psi_h = coherence_psi_matrix(data, n_bands=N_BANDS)
    pd.DataFrame(coh, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "v2_coherence.csv")
    pd.DataFrame(psi_band, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "v2_phase_slope_per_band.csv")
    pd.DataFrame(psi_h, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "v2_psi_lag_h.csv")
    print(f"  mean coherence (off-diag): {coh[~np.eye(N_CH, dtype=bool)].mean():.4f}")
    print(f"  mean |psi_lag_h|: {np.abs(psi_h[~np.eye(N_CH, dtype=bool)]).mean():.3f} h")

    print("\n" + "=" * 70)
    print("STEP 4: Compare AsySpecX vs ground-truths")
    print("=" * 70)
    from scipy.stats import spearmanr, pearsonr
    mask = ~np.eye(N_CH, dtype=bool)
    asy_flat = G_asy[mask]

    # Compare to |R_peak| (cross-correlation strength)
    xc_flat = np.abs(R_peak[mask])
    rho_s_xc, p_s_xc = spearmanr(xc_flat, asy_flat)
    rho_p_xc, p_p_xc = pearsonr(xc_flat, asy_flat)

    # Compare to coherence (broadband alignment)
    coh_flat = coh[mask]
    rho_s_co, p_s_co = spearmanr(coh_flat, asy_flat)
    rho_p_co, p_p_co = pearsonr(coh_flat, asy_flat)

    # Compare to |PSI lag| (phase-slope-derived lag)
    psi_flat = np.abs(psi_h[mask])
    rho_s_ps, p_s_ps = spearmanr(psi_flat, asy_flat)
    rho_p_ps, p_p_ps = pearsonr(psi_flat, asy_flat)

    # Lag agreement: tau_peak vs the dominant AsySpecX lag for each pair.
    # Per pair, AsySpecX's "primary lag" = sum_k |a_k[i]·b_k[j]| · τ̂_k / sum
    # weighted by coupling strength.
    # We need per-rank τ̂_k → averaged across seeds for ranks identified as TD.
    primary_lag_h = np.zeros((N_CH, N_CH))
    for seed, params in seeds_dict.items():
        stats = per_rank_stats(params["g_re"], params["g_im"],
                               params["A"], params["B"])
        td_ranks = [s["rank"] for s in stats if is_time_delay_rank(s)]
        if not td_ranks:
            continue
        for k in td_ranks:
            slope_per_band = stats[k]["slope"]
            slope_per_bin = slope_per_band / (F_OUT / N_BANDS)
            tau = -slope_per_bin * N_TOTAL / (2 * np.pi)  # hours
            outer = np.abs(np.outer(params["A"][:, k], params["B"][:, k]))
            primary_lag_h += outer * tau
    # Weighted lag = primary_lag_h / G_asy (if non-zero)
    weighted_lag = np.where(G_asy > 1e-6, primary_lag_h / np.where(G_asy > 1e-6, G_asy * len(seeds_dict), 1), 0)
    pd.DataFrame(weighted_lag, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "v2_asyspecx_weighted_lag_h.csv")

    # Lag-agreement: |τ_xc| vs |asyspec τ̂|
    tau_xc_flat = np.abs(tau_peak[mask])
    asy_tau_flat = np.abs(weighted_lag[mask])
    rho_lag_s, p_lag_s = spearmanr(tau_xc_flat, asy_tau_flat)

    print(f"\nAsySpecX |a·b| vs:")
    print(f"  |R_peak|           Spearman ρ={rho_s_xc:+.3f} p={p_s_xc:.3e}  "
          f"Pearson ρ={rho_p_xc:+.3f} p={p_p_xc:.3e}")
    print(f"  Coherence (mean)   Spearman ρ={rho_s_co:+.3f} p={p_s_co:.3e}  "
          f"Pearson ρ={rho_p_co:+.3f} p={p_p_co:.3e}")
    print(f"  |PSI lag (h)|      Spearman ρ={rho_s_ps:+.3f} p={p_s_ps:.3e}  "
          f"Pearson ρ={rho_p_ps:+.3f} p={p_p_ps:.3e}")
    print(f"\nLag agreement:")
    print(f"  |τ_xc| vs |τ̂_AsySpecX|  Spearman ρ={rho_lag_s:+.3f} p={p_lag_s:.3e}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    # Row 1: matrices
    def heatmap(ax, M, title, cmap='viridis', vmin=None, vmax=None, signed=False):
        Mp = M.copy()
        np.fill_diagonal(Mp, np.nan)
        if signed:
            v = np.nanmax(np.abs(Mp))
            im = ax.imshow(Mp, cmap='RdBu_r', vmin=-v, vmax=v)
        else:
            im = ax.imshow(Mp, cmap=cmap, vmin=vmin or 0, vmax=vmax or np.nanmax(Mp))
        ax.set_xticks(range(N_CH))
        ax.set_yticks(range(N_CH))
        ax.set_xticklabels(CHANNELS, rotation=45, fontsize=8)
        ax.set_yticklabels(CHANNELS, fontsize=8)
        ax.set_title(title, fontsize=9.5)
        plt.colorbar(im, ax=ax, fraction=0.046)

    heatmap(axes[0, 0], G_asy, "AsySpecX |a·b| (3-seed avg)")
    heatmap(axes[0, 1], np.abs(R_peak), "Cross-corr |R_peak|")
    heatmap(axes[0, 2], coh, "Coherence (mean over freq)")
    heatmap(axes[0, 3], np.abs(psi_h), "|PSI lag| (h)")

    # Row 2: scatter plots
    for ax, (gt_flat, name, rho_s, p_s, rho_p, p_p) in zip(
        axes[1],
        [
            (xc_flat, "|R_peak|", rho_s_xc, p_s_xc, rho_p_xc, p_p_xc),
            (coh_flat, "Coherence", rho_s_co, p_s_co, rho_p_co, p_p_co),
            (psi_flat, "|PSI lag|", rho_s_ps, p_s_ps, rho_p_ps, p_p_ps),
            (tau_xc_flat, "|τ_xc| (lag)", rho_lag_s, p_lag_s, None, None),
        ],
    ):
        if name == "|τ_xc| (lag)":
            ax.scatter(gt_flat, asy_tau_flat, s=40, alpha=0.7)
            ax.set_xlabel("|τ_xc| ground truth (h)")
            ax.set_ylabel("AsySpecX |τ̂| (h, weighted)")
            ax.set_title(f"Lag agreement\nSpearman ρ={rho_s:+.3f}  p={p_s:.2e}")
        else:
            ax.scatter(gt_flat, asy_flat, s=40, alpha=0.7)
            ax.set_xlabel(f"{name} ground truth")
            ax.set_ylabel("AsySpecX |a·b|")
            ax.set_title(f"AsySpecX vs {name}\n"
                         f"Spearman ρ={rho_s:+.3f} p={p_s:.2e}\n"
                         f"Pearson  ρ={rho_p:+.3f} p={p_p:.2e}")
        ax.grid(alpha=0.3)

    fig.suptitle("AsySpecX directional structure vs proper baselines (ETTh1)",
                 y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v2_three_way_scatter.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  saved {OUT_DIR / 'v2_three_way_scatter.png'}")

    # Summary file
    summary = [
        "AsySpecX directional structure validation (v2: proper baselines)",
        "=" * 70,
        "",
        "Aggregate-level cross-seed stability:",
    ]
    for i, j, d, rel in pair_dists:
        summary.append(f"  seed {i} vs {j}: distance = {d:.4f}  ({rel*100:.1f}% of mean ‖G‖)")
    summary.append("")
    summary.append("AsySpecX coupling vs ground truths (off-diagonal pairs, n=42):")
    summary.append(f"  |R_peak| (cross-corr):   Spearman ρ={rho_s_xc:+.3f} p={p_s_xc:.3e}")
    summary.append(f"                           Pearson  ρ={rho_p_xc:+.3f} p={p_p_xc:.3e}")
    summary.append(f"  Coherence (broadband):   Spearman ρ={rho_s_co:+.3f} p={p_s_co:.3e}")
    summary.append(f"                           Pearson  ρ={rho_p_co:+.3f} p={p_p_co:.3e}")
    summary.append(f"  |PSI lag| (freq-banded): Spearman ρ={rho_s_ps:+.3f} p={p_s_ps:.3e}")
    summary.append(f"                           Pearson  ρ={rho_p_ps:+.3f} p={p_p_ps:.3e}")
    summary.append("")
    summary.append("Lag agreement (cross-corr peak τ vs AsySpecX weighted τ̂):")
    summary.append(f"  Spearman ρ = {rho_lag_s:+.3f}  p = {p_lag_s:.3e}")
    summary.append("")
    summary.append("Verdict (heuristic):")
    best_rho = max(rho_s_xc, rho_s_co, rho_s_ps)
    if best_rho > 0.5:
        summary.append(f"  ✓ STRONG: at least one ground truth shows ρ > 0.5")
    elif best_rho > 0.3:
        summary.append(f"  ~ MODERATE: best ρ = {best_rho:+.3f}")
    elif best_rho > 0.0:
        summary.append(f"  ⚠ WEAK: best ρ = {best_rho:+.3f}")
    else:
        summary.append(f"  ✗ NO evidence")

    (OUT_DIR / "v2_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"  saved {OUT_DIR / 'v2_summary.txt'}")
    print("\n".join(summary[-15:]))


if __name__ == "__main__":
    main()
