"""Confidence calibration and bounded SafeRoute decisions."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def calibrate_lcb_quantile(
    predicted_advantage: np.ndarray,
    actual_advantage: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Calibrate q where LCB = predicted_advantage - q."""

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    predicted = np.asarray(predicted_advantage, dtype=np.float64)
    actual = np.asarray(actual_advantage, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.size == 0:
        raise ValueError("predicted and actual advantages need equal non-empty shapes")
    residual = predicted - actual
    return float(np.quantile(residual, 1.0 - alpha))


def safe_scores(
    predicted_advantage: np.ndarray,
    quantiles: np.ndarray | float,
    seed_variance: np.ndarray,
    uncertainty_beta: float = 0.1,
    eps: float = 1e-8,
) -> np.ndarray:
    predicted = np.asarray(predicted_advantage, dtype=np.float64)
    variance = np.asarray(seed_variance, dtype=np.float64)
    if predicted.shape != variance.shape:
        raise ValueError(f"advantage/uncertainty shape mismatch: {predicted.shape} vs {variance.shape}")
    lcb = predicted - np.asarray(quantiles, dtype=np.float64)
    score = lcb - float(uncertainty_beta) * np.log(np.maximum(variance, eps))
    return np.where(np.isfinite(score), score, -np.inf)


def safe_route(
    anchor: np.ndarray,
    alternatives: np.ndarray,
    predicted_advantage: np.ndarray,
    quantiles: np.ndarray | float,
    seed_variance: np.ndarray,
    decision: str = "safe_top1_blend",
    min_gain: float = 0.0,
    full_gain: float = 0.02,
    uncertainty_beta: float = 0.1,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Route one horizon block with strict bounded fallback to anchor.

    Shapes are ``anchor=[N,H,C]``, ``alternatives=[N,K,H,C]`` and router
    values ``[N,K]``. The returned prediction is exactly equal to anchor for
    every uncertain sample.
    """

    anchor_arr = np.asarray(anchor)
    alt = np.asarray(alternatives)
    if anchor_arr.ndim != 3 or alt.ndim != 4 or alt.shape[0] != anchor_arr.shape[0]:
        raise ValueError("anchor must be [N,H,C] and alternatives [N,K,H,C]")
    if alt.shape[2:] != anchor_arr.shape[1:]:
        raise ValueError("expert prediction shapes do not align")
    score = safe_scores(
        predicted_advantage,
        quantiles,
        seed_variance,
        uncertainty_beta=uncertainty_beta,
        eps=eps,
    )
    if score.shape != alt.shape[:2]:
        raise ValueError("router score shape does not match alternatives")
    if decision not in {"hard_top1", "safe_top1_blend", "safe_multi_mix"}:
        raise ValueError(f"unknown router decision {decision!r}")
    if full_gain <= min_gain:
        raise ValueError("full_gain must be greater than min_gain")
    if uncertainty_beta < 0.0:
        raise ValueError("uncertainty_beta must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    top_index = np.argmax(score, axis=1)
    top_score = score[np.arange(score.shape[0]), top_index]
    active = top_score > float(min_gain)
    alpha = np.zeros(score.shape[0], dtype=np.float64)
    if decision == "hard_top1":
        alpha[active] = 1.0
        chosen = alt[np.arange(alt.shape[0]), top_index]
    elif decision == "safe_top1_blend":
        alpha = np.clip(
            (top_score - float(min_gain)) / (float(full_gain) - float(min_gain) + eps),
            0.0,
            1.0,
        )
        alpha[~active] = 0.0
        chosen = alt[np.arange(alt.shape[0]), top_index]
    else:
        eligible = score > float(min_gain)
        logits = score / float(temperature)
        logits = np.where(eligible, logits, -np.inf)
        row_max = np.max(np.where(np.isfinite(logits), logits, -np.inf), axis=1, keepdims=True)
        safe_row_max = np.where(np.isfinite(row_max), row_max, 0.0)
        stable = np.where(np.isfinite(logits), logits - safe_row_max, -np.inf)
        weights = np.where(np.isfinite(stable), np.exp(stable), 0.0)
        denominator = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
        chosen = np.einsum("nk,nkhc->nhc", weights, alt)
        alpha = np.clip(
            (top_score - float(min_gain)) / (float(full_gain) - float(min_gain) + eps),
            0.0,
            1.0,
        )
        alpha[~active] = 0.0

    routed = anchor_arr + alpha[:, None, None] * (chosen - anchor_arr)
    routed[~active] = anchor_arr[~active]
    diagnostics = {
        "active": active,
        "alpha": alpha,
        "top_index": top_index,
        "top_score": top_score,
        "safe_scores": score,
    }
    return routed.astype(anchor_arr.dtype, copy=False), diagnostics


def activation_diagnostics(
    anchor_loss: np.ndarray,
    alternative_losses: np.ndarray,
    diagnostics: Dict[str, np.ndarray],
    catastrophic_threshold: float = 0.01,
) -> Dict[str, float]:
    anchor = np.asarray(anchor_loss, dtype=np.float64)
    alternatives = np.asarray(alternative_losses, dtype=np.float64)
    active = np.asarray(diagnostics["active"], dtype=bool)
    top = np.asarray(diagnostics["top_index"], dtype=np.int64)
    chosen = alternatives[np.arange(alternatives.shape[0]), top]
    regret = chosen - anchor
    activated = max(int(active.sum()), 1)
    return {
        "fallback_fraction": float((~active).mean()),
        "mean_alpha": float(np.asarray(diagnostics["alpha"]).mean()),
        "false_activation_rate": float(((regret > 0) & active).sum() / activated),
        "catastrophic_activation_rate": float(
            ((regret > catastrophic_threshold) & active).sum() / activated
        ),
        "mean_actual_advantage_activated": float((-regret[active]).mean()) if active.any() else 0.0,
    }
