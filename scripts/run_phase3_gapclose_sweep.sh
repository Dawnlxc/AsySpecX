#!/bin/bash
# Optional Phase 3-GapClose sweep helper. Disabled unless RUN_SWEEP=1.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

if [ "${RUN_SWEEP:-0}" != "1" ]; then
    cat <<'EOF'
Phase 3-GapClose sweep disabled.

Weather priority:
  RUN_ONLY=phase3_fits_shared
  RUN_ONLY=phase3_fits_individual
  RUN_ONLY=phase3_fits_shared_revin_affine
  RUN_ONLY=phase3_fits_shared_subtract_last
  then diag_only / anchor / sparse_period

Electricity priority:
  RUN_ONLY=phase3_anchor_sparse_period PERIOD=24
  RUN_ONLY=phase3_fits_sparse_period PERIOD=24
  RUN_ONLY=phase3_individual_sparse_period PERIOD=24
  then cut_freq 32/48/64/96
  then validation selection

Available sweeps:
  CUT_FREQ: 16 24 32 48 64 96 128
  lift_sharing: shared individual
  norm_mode: rin_noaffine revin_affine subtract_last
  electricity periods: 24, 168 optional
  weather periods: 144, 24 sanity
  temporal_gate_init_logit: -6 -4 -2
  temporal_fusion: convex additive
  residual: none offdiag_only split diag_only

Example:
  RUN_SWEEP=1 DATASET=electricity SEQ_LEN=720 PRED_LEN=96 bash scripts/run_phase3_gapclose_sweep.sh
EOF
    exit 0
fi

for cf in 32 48 64 96; do
    CUT_FREQ="$cf" RUN_ONLY=phase3_fits_shared bash scripts/run_phase3_gapclose.sh
done

for period in 24 168; do
    PERIOD="$period" RUN_ONLY=phase3_fits_sparse_period bash scripts/run_phase3_gapclose.sh
done

for arm in phase3_fits_shared phase3_fits_individual phase3_fits_shared_revin_affine phase3_fits_shared_subtract_last; do
    RUN_ONLY="$arm" bash scripts/run_phase3_gapclose.sh
done
