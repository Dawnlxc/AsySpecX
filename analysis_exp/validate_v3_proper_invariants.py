"""V3: re-do cross-seed validation using ONLY permutation-invariant quantities.

Earlier attempts compared seed-pairs by aggregating over the TD-classified rank
subset. Different seeds classify different ranks as TD, so that aggregate is
over different rank sets — instability there does NOT prove the underlying
function is unstable, only that the threshold-based slicing differs.

Truly permutation-invariant tests (rank-decomposition-independent):

  Test A — full-rank aggregate of mode magnitudes
            G_full[i,j] = Σ_k |a_k[i] · b_k[j]|     (sum over ALL k=1..r)
            depends only on the unsigned outer products; permutation-invariant.

  Test B — true effective transfer matrix
            H_total[i,j] = | Σ_k a_k[i] · g_band·k mean · b_k[j] |    (sum first, then |·|)
            and per-band |H_b|. These are matrix-valued functions of the
            cross-block parameters; rank decomposition is internal.

  Test C — forward equivalence
            Take fixed deterministic test input U_test (real Gaussian batch),
            run through each seed's cross_block, compare residual outputs
            (gate * H * U_test). If two cross_blocks compute the same function
            up to numerical precision, this output is identical.
            Pearson r between flattened outputs is the cleanest invariance
            score.

Output: figures/AsySpecX_validation/v3_*.png and v3_summary.txt
"""
from pathlib import Path
import os, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

CKPT_PATHS = {
    2026: "/scratch3/lin250/bldgFM/Scope/checkpoints/asyspecx_clean/real_ETTh1_H96_s2026_clean_AsySpecXClean_ETTh1_ftM_sl336_pl96_cycle24_seed2026/checkpoint.pth",
    2027: "/scratch3/lin250/bldgFM/Scope/checkpoints/asyspecx_clean/real_ETTh1_H96_s2027_clean_AsySpecXClean_ETTh1_ftM_sl336_pl96_cycle24_seed2027/checkpoint.pth",
    2028: "/scratch3/lin250/bldgFM/Scope/checkpoints/asyspecx_clean/real_ETTh1_H96_s2028_clean_AsySpecXClean_ETTh1_ftM_sl336_pl96_cycle24_seed2028/checkpoint.pth",
}
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
N_CH, N_BANDS, F_OUT, N_TOTAL = 7, 8, 108, 432

OUT = ROOT / "figures" / "AsySpecX_validation"
OUT.mkdir(parents=True, exist_ok=True)


def load_p(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "A": sd["cross_block.A"].numpy(),
        "B": sd["cross_block.B"].numpy(),
        "g_re": sd["cross_block.g_re"].numpy(),
        "g_im": sd["cross_block.g_im"].numpy(),
        "gate_logit": float(sd["cross_block.gate_logit"]),
        "band_ids": sd["cross_block.band_ids"].numpy(),
    }


def G_full(params):
    """Permutation-invariant aggregate: Σ_k |a_k · b_k^T|."""
    A, B = params["A"], params["B"]
    G = np.zeros((N_CH, N_CH))
    for k in range(A.shape[1]):
        G += np.abs(np.outer(A[:, k], B[:, k]))
    return G


def H_per_band(params):
    """Returns 8 complex matrices H_b = A diag(g_b) B^T, one per band."""
    A, B = params["A"], params["B"]
    g = params["g_re"] + 1j * params["g_im"]
    out = []
    for b in range(N_BANDS):
        out.append(A @ np.diag(g[b]) @ B.T)
    return out


def H_integrated_magnitude(params):
    """|sum_b H_b| element-wise — fully decomposition-invariant."""
    H_bands = H_per_band(params)
    H_total = sum(H_bands)  # ∈ ℂ^{N×N}
    return np.abs(H_total)


def gate(params):
    return 0.2 / (1.0 + np.exp(-params["gate_logit"]))


def cross_forward(params, U):
    """Apply gate * H_b U for the relevant band per freq bin (mimics
    AsymCross.forward). U: complex [B, C, F]. Returns the residual R
    (without adding U back), so two seeds with the SAME function will give
    the SAME R."""
    A, B = params["A"], params["B"]
    g = params["g_re"] + 1j * params["g_im"]   # [n_bands, r]
    band_ids = params["band_ids"]              # [F]
    g_per_freq = g[band_ids]                   # [F, r]
    Bt = B  # we use B^T below
    # S = B^T U  → [B, r, F]
    S = np.einsum('cr,bcf->brf', Bt, U)
    # Apply per-freq complex gain to each rank
    Sg = S * g_per_freq.T[None, :, :]          # [B, r, F]
    # Project back: R = A Sg → [B, C, F]
    R = np.einsum('cr,brf->bcf', A, Sg)
    g_val = gate(params)
    return g_val * R


def forward_equivalence(seeds_params, n_test=128, freqs=F_OUT, seed_test=42):
    """Generate a fixed complex test signal, push through each seed's cross
    block, compute pairwise Pearson r between flattened (real+imag) outputs."""
    rng = np.random.default_rng(seed_test)
    Ur = rng.standard_normal((n_test, N_CH, freqs))
    Ui = rng.standard_normal((n_test, N_CH, freqs))
    U = Ur + 1j * Ui
    outputs = {}
    for seed, p in seeds_params.items():
        out = cross_forward(p, U)
        outputs[seed] = np.concatenate([out.real.flatten(), out.imag.flatten()])
    return outputs


def main():
    print("Loading 3 seed checkpoints...")
    P = {s: load_p(c) for s, c in CKPT_PATHS.items() if os.path.exists(c)}

    print("\n=== Test A: full-rank aggregate G_full ===")
    G = {s: G_full(p) for s, p in P.items()}
    seeds = sorted(G.keys())
    print("Pairwise Frobenius distance (full-rank aggregate, permutation-INVARIANT):")
    A_dists = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            d = np.linalg.norm(G[seeds[i]] - G[seeds[j]], 'fro')
            mean_norm = 0.5 * (np.linalg.norm(G[seeds[i]], 'fro') + np.linalg.norm(G[seeds[j]], 'fro'))
            rel = d / mean_norm
            print(f"  {seeds[i]} vs {seeds[j]}:  dist = {d:.3f}   ({rel*100:.1f}% of mean ‖G‖)")
            A_dists.append((seeds[i], seeds[j], d, rel))

    print("\n=== Test B: integrated transfer |Σ_b H_b| ===")
    H = {s: H_integrated_magnitude(p) for s, p in P.items()}
    print("Pairwise Frobenius distance (integrated transfer matrix):")
    B_dists = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            d = np.linalg.norm(H[seeds[i]] - H[seeds[j]], 'fro')
            mean_norm = 0.5 * (np.linalg.norm(H[seeds[i]], 'fro') + np.linalg.norm(H[seeds[j]], 'fro'))
            rel = d / mean_norm
            print(f"  {seeds[i]} vs {seeds[j]}:  dist = {d:.3f}   ({rel*100:.1f}% of mean ‖H‖)")
            B_dists.append((seeds[i], seeds[j], d, rel))

    print("\n=== Test C: forward output equivalence ===")
    outs = forward_equivalence(P, n_test=128)
    print("Pairwise Pearson r and Frobenius distance between cross-block forward outputs:")
    print("(if two cross-blocks compute the same function, r should ≈ 1.)")
    C_metrics = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            o_a, o_b = outs[seeds[i]], outs[seeds[j]]
            r_p, p_p = pearsonr(o_a, o_b)
            d = np.linalg.norm(o_a - o_b)
            n = 0.5 * (np.linalg.norm(o_a) + np.linalg.norm(o_b))
            rel = d / n
            print(f"  {seeds[i]} vs {seeds[j]}:  Pearson r = {r_p:+.4f}  "
                  f"(p = {p_p:.2e}, rel-dist = {rel*100:.1f}% of mean ‖output‖)")
            C_metrics.append((seeds[i], seeds[j], r_p, p_p, rel))

    print("\n=== Test D: per-band Hermitian/anti-Hermitian energy ===")
    print("(scalar quantities — automatically permutation-invariant; computed before)")
    for s, p in P.items():
        H_bands = H_per_band(p)
        sum_sym = 0
        sum_total = 0
        for Hb in H_bands:
            Hh = 0.5 * (Hb + Hb.conj().T)
            sum_sym += np.linalg.norm(Hh, 'fro') ** 2
            sum_total += np.linalg.norm(Hb, 'fro') ** 2
        sym_ratio = sum_sym / max(sum_total, 1e-12)
        print(f"  seed {s}: ‖H_sym‖² / ‖H‖² = {sym_ratio:.4f}   "
              f"anti² = {1 - sym_ratio:.4f}")

    # === Plot side-by-side aggregates ===
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, s in enumerate(seeds):
        ax = axes[0, col]
        Mp = G[s].copy(); np.fill_diagonal(Mp, np.nan)
        im = ax.imshow(Mp, cmap='viridis')
        ax.set_xticks(range(N_CH)); ax.set_yticks(range(N_CH))
        ax.set_xticklabels(CHANNELS, rotation=45, fontsize=8)
        ax.set_yticklabels(CHANNELS, fontsize=8)
        ax.set_title(f'G_full seed {s}\n(Σ_k |a_k·b_k^T|)', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, col]
        Mp = H[s].copy(); np.fill_diagonal(Mp, np.nan)
        im = ax.imshow(Mp, cmap='viridis')
        ax.set_xticks(range(N_CH)); ax.set_yticks(range(N_CH))
        ax.set_xticklabels(CHANNELS, rotation=45, fontsize=8)
        ax.set_yticklabels(CHANNELS, fontsize=8)
        ax.set_title(f'|Σ_b H_b|  seed {s}', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Cross-seed comparison of permutation-invariant quantities',
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'v3_invariant_aggregates.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\nsaved {OUT / 'v3_invariant_aggregates.png'}")

    # === Summary ===
    summary = ["v3: permutation-invariant cross-seed comparison", "=" * 60, ""]
    summary.append("Test A — full-rank aggregate G_full[i,j] = Σ_k |a_k[i]·b_k[j]|:")
    for s1, s2, d, rel in A_dists:
        summary.append(f"  {s1} vs {s2}: rel-dist = {rel*100:.1f}% of mean ‖G‖")
    summary.append("")
    summary.append("Test B — integrated transfer |Σ_b H_b|:")
    for s1, s2, d, rel in B_dists:
        summary.append(f"  {s1} vs {s2}: rel-dist = {rel*100:.1f}% of mean ‖H‖")
    summary.append("")
    summary.append("Test C — forward equivalence (Pearson r of output on shared test input):")
    for s1, s2, r, p, rel in C_metrics:
        summary.append(f"  {s1} vs {s2}: Pearson r = {r:+.4f}  rel-dist = {rel*100:.1f}%")
    summary.append("")
    summary.append("Verdict:")
    avg_C_r = np.mean([m[2] for m in C_metrics])
    if avg_C_r > 0.9:
        summary.append(f"  ✓ Cross-blocks compute essentially the SAME function (avg r = {avg_C_r:+.3f})")
        summary.append(f"     → previous instability was decomposition-permutation only")
    elif avg_C_r > 0.5:
        summary.append(f"  ~ Cross-blocks PARTIALLY agree (avg r = {avg_C_r:+.3f})")
        summary.append(f"     → real function differs across seeds, not just labels")
    elif avg_C_r > 0.0:
        summary.append(f"  ⚠ Cross-blocks weakly correlate (avg r = {avg_C_r:+.3f})")
        summary.append(f"     → seeds learn different functions; frame is not robust")
    else:
        summary.append(f"  ✗ Cross-blocks are uncorrelated/anti-correlated (avg r = {avg_C_r:+.3f})")
        summary.append(f"     → seeds learn fundamentally different functions")

    (OUT / 'v3_summary.txt').write_text("\n".join(summary) + "\n")
    print("\n".join(summary[-7:]))


if __name__ == "__main__":
    main()
