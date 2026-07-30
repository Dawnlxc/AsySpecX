#!/bin/bash
# Optional AsySpecX Phase 2 sweep helper.
# This file documents recommended small sweeps and only runs when RUN_SWEEP=1.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

if [ "${RUN_SWEEP:-0}" != "1" ]; then
    cat <<'EOF'
Phase 2 sweep is disabled by default.

Examples:
  RUN_SWEEP=1 DATASET=ETTm1 SEQ_LEN=720 PRED_LEN=96 bash scripts/run_phase2_sweep.sh

Sweeps:
  gate_init_logit: -8 -6 -4
  gate_lr_mult:    1 5 10
  residual_clip:   -1 0.5 1.0
  backcast_weight: 0.0 0.01 0.03

Backcast remains optional; do not use it as default.
EOF
    exit 0
fi

for gate_init in -8 -6 -4; do
    GATE_INIT_LOGIT="$gate_init" RUN_ONLY=phase2_global_all bash scripts/run_phase2_asyspecx.sh
done

for mult in 1 5 10; do
    GATE_LR_MULT="$mult" RUN_ONLY=phase2_hier_all bash scripts/run_phase2_asyspecx.sh
done

for clip in -1 0.5 1.0; do
    RESIDUAL_CLIP_ETA="$clip" RUN_ONLY=phase2_global_all bash scripts/run_phase2_asyspecx.sh
done

for backcast in 0.0 0.01 0.03; do
    BACKCAST_LOSS_WEIGHT="$backcast" RUN_ONLY=phase2_global_all bash scripts/run_phase2_asyspecx.sh
done
