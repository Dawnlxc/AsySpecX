"""Validate AsySpecX's directional channel-coupling claim with two tests:

  (1) Cross-seed stability — run the rank classification on 3 independently-
      trained checkpoints (different seeds, same dataset/horizon). If the
      time-delay rank assignments and lag values are consistent across seeds,
      the structure is a learned property, not random noise.

  (2) Granger ground-truth — pairwise Granger causality on the RAW ETTh1
      time series gives an independent 7×7 channel-pair directional matrix.
      Compare to AsySpecX's aggregated coupling (sum of |a_k[i]·b_k[j]|
      across time-delay ranks). Spearman ρ between the two matrices
      quantifies how much AsySpecX recovers Granger-style structure.

Output (figures/AsySpecX_validation/):
  cross_seed_summary.txt        — per-seed rank classification & lag table
  cross_seed_overlap.txt        — count of channel-pair overlaps across seeds
  granger_vs_asyspecx.png       — side-by-side heatmaps + Spearman ρ
  granger_p_matrix.csv          — Granger -log10(p) matrix
  asyspecx_aggregated.csv       — AsySpecX coupling magnitude per (i, j)

Usage:
    python analysis_exp/validate_asyspecx_directional.py
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
N_TOTAL = SEQ_LEN + PRED_LEN  # 432 samples (hours)

OUT_DIR = ROOT / "figures" / "AsySpecX_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# AsySpecX side: per-seed rank classification
# ---------------------------------------------------------------------------
def load_cross_block(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return {
        "A": sd["cross_block.A"].numpy(),
        "B": sd["cross_block.B"].numpy(),
        "g_re": sd["cross_block.g_re"].numpy(),
        "g_im": sd["cross_block.g_im"].numpy(),
        "gate_logit": float(sd["cross_block.gate_logit"]),
    }


def per_rank_stats(A, B, g_re, g_im):
    n_rank = A.shape[1]
    phase = np.angle(g_re + 1j * g_im)
    mag = np.abs(g_re + 1j * g_im)
    out = []
    for k in range(n_rank):
        ph_un = np.unwrap(phase[:, k])
        b = np.arange(N_BANDS, dtype=float)
        slope, intercept = np.polyfit(b, ph_un, 1)
        pred = slope * b + intercept
        ss_res = float(np.sum((ph_un - pred) ** 2))
        ss_tot = float(np.sum((ph_un - ph_un.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        cos_ab = float(A[:, k] @ B[:, k]) / max(
            np.linalg.norm(A[:, k]) * np.linalg.norm(B[:, k]), 1e-12
        )
        slope_per_bin = slope / (F_OUT / N_BANDS)
        tau_h = -slope_per_bin * N_TOTAL / (2 * np.pi)
        circ_var = 1.0 - abs(np.exp(1j * phase[:, k]).mean())
        out.append({
            "rank": k,
            "r2": float(r2),
            "slope": float(slope),
            "tau_h": float(tau_h),
            "cos_ab": cos_ab,
            "circ_var": float(circ_var),
            "g_norm": float(np.linalg.norm(g_re[:, k] + 1j * g_im[:, k])),
            "mag_mean": float(mag[:, k].mean()),
        })
    return out


def classify_rank(s):
    cos_ab = s["cos_ab"]
    r2 = s["r2"]
    cv = s["circ_var"]
    if abs(cos_ab) > 0.7:
        return "Symmetric"
    if r2 > 0.55:
        return "Time-delay"
    if cv < 0.35 and r2 < 0.3:
        return "Const-phase"
    if cv > 0.7 and r2 < 0.3:
        return "Noise"
    return "Mixed"


def aggregate_coupling(A, B, time_delay_ranks):
    """Build a 7×7 matrix of summed |a_k[i] · b_k[j]| over time-delay ranks.
    Entry (i, j) measures total directional coupling j → i in those ranks."""
    G = np.zeros((N_CH, N_CH))
    for k in time_delay_ranks:
        G += np.abs(np.outer(A[:, k], B[:, k]))
    return G


# ---------------------------------------------------------------------------
# Granger side: pairwise Granger F on raw ETTh1
# ---------------------------------------------------------------------------
def compute_granger_matrix(max_lag=12):
    """Returns an [N×N] matrix where (i, j) entry is -log10(p) of the Granger
    test of "j Granger-causes i" at the lag minimizing p, plus the optimal lag.

    Because we operate on detrended/normalized data, the magnitude of -log10(p)
    is comparable across pairs."""
    from statsmodels.tsa.stattools import grangercausalitytests

    df = pd.read_csv(DATA_PATH)
    cols = CHANNELS  # ETT-h column order
    data = df[cols].values  # [T, 7]
    # Use the first 12000 hourly samples (training portion of ETTh1)
    data = data[:12000]
    # Standardize per-channel for numerical stability
    data = (data - data.mean(0)) / (data.std(0) + 1e-8)

    G_p = np.zeros((N_CH, N_CH))   # -log10(min p-value)
    G_lag = np.zeros((N_CH, N_CH))  # lag at which min p occurs

    print("Computing Granger causality matrix (this may take ~1-2 min)...")
    for i in range(N_CH):
        for j in range(N_CH):
            if i == j:
                continue
            # Test: does cols[j] Granger-cause cols[i]?
            # statsmodels takes [target, predictor] columns
            x = np.column_stack([data[:, i], data[:, j]])
            try:
                res = grangercausalitytests(x, max_lag, verbose=False)
                ps = []
                for lag in range(1, max_lag + 1):
                    p = res[lag][0]["ssr_ftest"][1]
                    ps.append((lag, p))
                best_lag, best_p = min(ps, key=lambda t: t[1])
                G_p[i, j] = -np.log10(max(best_p, 1e-300))
                G_lag[i, j] = best_lag
            except Exception as exc:
                print(f"  warn: Granger ({CHANNELS[j]}→{CHANNELS[i]}) failed: {exc}")
        print(f"  done channel {i+1}/{N_CH}")
    return G_p, G_lag, data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # === STEP 1: Per-seed AsySpecX rank classification ===
    print("=" * 70)
    print("STEP 1: cross-seed rank classification")
    print("=" * 70)

    seed_results = {}
    seed_lines = []
    for seed, ckpt in CKPT_PATHS.items():
        if not os.path.exists(ckpt):
            print(f"  skip seed {seed}: ckpt not found")
            continue
        params = load_cross_block(ckpt)
        stats = per_rank_stats(params["A"], params["B"],
                               params["g_re"], params["g_im"])
        td_ranks = []
        seed_lines.append(f"\nSeed {seed}:")
        for s in stats:
            label = classify_rank(s)
            seed_lines.append(
                f"  rank {s['rank']}: {label:13s} "
                f"R²={s['r2']:.3f}  τ̂={s['tau_h']:+.2f}h  "
                f"cos(A,B)={s['cos_ab']:+.3f}  "
                f"|g|={s['g_norm']:.3f}"
            )
            if label == "Time-delay":
                td_ranks.append(s["rank"])
        seed_results[seed] = {
            "params": params,
            "stats": stats,
            "td_ranks": td_ranks,
        }
        seed_lines.append(f"  → time-delay ranks: {td_ranks}")
    print("\n".join(seed_lines))

    # Cross-seed agreement metrics
    if len(seed_results) >= 2:
        # 1. Number of time-delay ranks per seed
        # 2. Set overlap of "lag bucket" — quantize τ̂ to nearest hour and see
        #    how often two seeds find a TD rank in the same bucket
        lag_buckets_per_seed = {}
        for seed, sr in seed_results.items():
            buckets = []
            for k in sr["td_ranks"]:
                tau = sr["stats"][k]["tau_h"]
                buckets.append(round(abs(tau)))
            lag_buckets_per_seed[seed] = sorted(buckets)
        seed_lines.append("\nLag buckets (rounded hours, abs values) per seed:")
        for seed, buckets in lag_buckets_per_seed.items():
            seed_lines.append(f"  seed {seed}: {buckets}")
        # Pairwise overlap
        seeds_l = list(lag_buckets_per_seed.keys())
        if len(seeds_l) >= 2:
            seed_lines.append("\nLag-bucket overlap between seed pairs (counts):")
            for i in range(len(seeds_l)):
                for j in range(i + 1, len(seeds_l)):
                    a = lag_buckets_per_seed[seeds_l[i]]
                    b = lag_buckets_per_seed[seeds_l[j]]
                    common = []
                    bcopy = list(b)
                    for x in a:
                        # match within ±1 hour
                        matched = next((y for y in bcopy if abs(x - y) <= 1), None)
                        if matched is not None:
                            common.append((x, matched))
                            bcopy.remove(matched)
                    seed_lines.append(
                        f"  seed {seeds_l[i]} ∩ {seeds_l[j]}: "
                        f"{len(common)} matched (within ±1h)  → {common}"
                    )

    # Save cross-seed report
    (OUT_DIR / "cross_seed_summary.txt").write_text("\n".join(seed_lines) + "\n")
    print(f"\nsaved {OUT_DIR / 'cross_seed_summary.txt'}")

    # === STEP 2: Granger ground-truth ===
    print("\n" + "=" * 70)
    print("STEP 2: Granger ground-truth on raw ETTh1")
    print("=" * 70)

    G_p, G_lag, _ = compute_granger_matrix(max_lag=12)
    pd.DataFrame(G_p, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "granger_p_matrix.csv"
    )
    pd.DataFrame(G_lag, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "granger_optimal_lag.csv"
    )
    print(f"  saved granger_p_matrix.csv (entry [i,j] = -log10(p) of j→i)")
    print(f"  saved granger_optimal_lag.csv (best lag in samples = hours)")

    # === STEP 3: AsySpecX aggregated coupling, averaged over seeds ===
    asy_per_seed = {}
    for seed, sr in seed_results.items():
        if not sr["td_ranks"]:
            continue
        G_asy = aggregate_coupling(sr["params"]["A"], sr["params"]["B"],
                                   sr["td_ranks"])
        asy_per_seed[seed] = G_asy
    if asy_per_seed:
        G_asy_avg = np.mean(list(asy_per_seed.values()), axis=0)
    else:
        G_asy_avg = np.zeros((N_CH, N_CH))
    pd.DataFrame(G_asy_avg, index=CHANNELS, columns=CHANNELS).to_csv(
        OUT_DIR / "asyspecx_aggregated.csv"
    )
    print(f"  saved asyspecx_aggregated.csv (mean over {len(asy_per_seed)} seeds)")

    # === STEP 4: Compare ===
    from scipy.stats import spearmanr, pearsonr
    # Mask out diagonal (self-pairs)
    mask = ~np.eye(N_CH, dtype=bool)
    g_flat = G_p[mask]
    asy_flat = G_asy_avg[mask]
    rho_s, p_s = spearmanr(g_flat, asy_flat)
    rho_p, p_p = pearsonr(g_flat, asy_flat)
    print(f"\nSpearman ρ (Granger -log10 p vs AsySpecX |a·b|): "
          f"{rho_s:+.4f}  p={p_s:.4e}")
    print(f"Pearson  ρ:                                       "
          f"{rho_p:+.4f}  p={p_p:.4e}")

    # === STEP 5: Visualisation ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Granger matrix
    ax = axes[0]
    G_plot = G_p.copy()
    np.fill_diagonal(G_plot, np.nan)
    vmax = np.nanmax(G_plot)
    im = ax.imshow(G_plot, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(N_CH))
    ax.set_yticks(range(N_CH))
    ax.set_xticklabels(CHANNELS, rotation=45)
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("predictor j (Granger-cause)")
    ax.set_ylabel("target i")
    ax.set_title("Granger ground-truth\n−log₁₀(p) of j→i")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 2: AsySpecX aggregated
    ax = axes[1]
    A_plot = G_asy_avg.copy()
    np.fill_diagonal(A_plot, np.nan)
    vmax2 = np.nanmax(A_plot)
    im = ax.imshow(A_plot, cmap="viridis", vmin=0, vmax=vmax2)
    ax.set_xticks(range(N_CH))
    ax.set_yticks(range(N_CH))
    ax.set_xticklabels(CHANNELS, rotation=45)
    ax.set_yticklabels(CHANNELS)
    ax.set_xlabel("input channel j")
    ax.set_ylabel("output channel i")
    ax.set_title(f"AsySpecX aggregated coupling\n"
                 f"Σ_k∈TD |a_k[i]·b_k[j]|  (avg over {len(asy_per_seed)} seeds)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 3: scatter
    ax = axes[2]
    ax.scatter(g_flat, asy_flat, s=40, alpha=0.7)
    # Annotate top channel pairs
    pair_labels = [(i, j) for i in range(N_CH) for j in range(N_CH) if i != j]
    for k, (i, j) in enumerate(pair_labels):
        if g_flat[k] > 100 or asy_flat[k] > 0.5 * np.max(asy_flat):
            ax.annotate(f"{CHANNELS[j]}→{CHANNELS[i]}",
                        (g_flat[k], asy_flat[k]), fontsize=7,
                        ha="left", va="bottom")
    ax.set_xlabel("Granger −log₁₀(p)")
    ax.set_ylabel("AsySpecX |a·b| (mean over seeds)")
    ax.set_title(f"Spearman ρ = {rho_s:+.3f}  (p = {p_s:.2e})\n"
                 f"Pearson  ρ = {rho_p:+.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "AsySpecX learned coupling vs Granger causality on raw ETTh1\n"
        "(higher ρ ⇒ AsySpecX recovers Granger-style directional structure)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "granger_vs_asyspecx.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR / 'granger_vs_asyspecx.png'}")

    # === STEP 6: Summary verdict ===
    verdict_lines = ["", "=" * 70, "VALIDATION VERDICT", "=" * 70]
    # Cross-seed: how many seeds find the same lag clusters?
    n_seeds = len(seed_results)
    n_td_per_seed = [len(sr["td_ranks"]) for sr in seed_results.values()]
    verdict_lines.append(f"Cross-seed stability:")
    verdict_lines.append(f"  Seeds tested: {list(seed_results.keys())}")
    verdict_lines.append(f"  Time-delay rank counts per seed: {n_td_per_seed}")
    verdict_lines.append(f"  Mean TD ranks: {np.mean(n_td_per_seed):.1f}, "
                          f"std: {np.std(n_td_per_seed):.2f}")
    verdict_lines.append(f"")
    verdict_lines.append(f"Granger ground-truth comparison:")
    verdict_lines.append(f"  Spearman ρ (off-diagonal pairs): {rho_s:+.4f}  "
                         f"p={p_s:.4e}")
    verdict_lines.append(f"  Pearson  ρ: {rho_p:+.4f}  p={p_p:.4e}")
    verdict_lines.append(f"")
    verdict_lines.append(f"Verdict (heuristic):")
    if rho_s > 0.5 and p_s < 0.01:
        verdict_lines.append(f"  ✓ STRONG evidence: AsySpecX coupling structure")
        verdict_lines.append(f"    correlates with raw-data Granger causality")
        verdict_lines.append(f"    (ρ > 0.5, p < 0.01)")
    elif rho_s > 0.3 and p_s < 0.05:
        verdict_lines.append(f"  ~ MODERATE evidence: positive but not strong")
        verdict_lines.append(f"    correlation; frame is suggestive")
    elif rho_s > 0:
        verdict_lines.append(f"  ⚠ WEAK evidence: positive but small correlation;")
        verdict_lines.append(f"    physics-grounded claim is hard to defend")
    else:
        verdict_lines.append(f"  ✗ NO evidence: AsySpecX coupling does NOT")
        verdict_lines.append(f"    correlate with Granger; frame collapses")

    print("\n".join(verdict_lines))
    with (OUT_DIR / "cross_seed_summary.txt").open("a") as f:
        f.write("\n".join(verdict_lines) + "\n")


if __name__ == "__main__":
    main()
