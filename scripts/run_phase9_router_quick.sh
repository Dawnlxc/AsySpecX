#!/bin/bash
# Validation-adapted quick router. Test labels are used only after decisions.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

: "${EXPERT_MANIFEST:?EXPERT_MANIFEST is required}"
OUTROOT="${OUTROOT:-phase9_results/quick_cell}"
PYTHON="${PYTHON:-python}"
COMMON=(--expert_manifest "$EXPERT_MANIFEST" --expert_seeds "${EXPERT_SEEDS:-2024,2025,2026}" --experts "${EXPERTS:-}")

"$PYTHON" scripts/build_router_meta.py "${COMMON[@]}" \
  --split val --router_meta_source val \
  --router_num_horizon_blocks "${ROUTER_NUM_HORIZON_BLOCKS:-4}" \
  --router_scope "${ROUTER_SCOPE:-cell}" \
  --batch_size "${ROUTER_BATCH_SIZE:-0}" --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}" --expert_device_policy "${EXPERT_DEVICE_POLICY:-resident}" \
  --output "$OUTROOT/meta_val" --overwrite "${OVERWRITE:-0}"

"$PYTHON" scripts/train_safe_router.py \
  --meta "$OUTROOT/meta_val" --output_dir "$OUTROOT/router" \
  --router_backend "${ROUTER_BACKEND:-xgboost}" \
  --router_scope "${ROUTER_SCOPE:-cell}" --router_target "${ROUTER_TARGET:-advantage}" \
  --router_min_samples "${ROUTER_MIN_SAMPLES:-256}" \
  --router_cv_folds "${ROUTER_CV_FOLDS:-4}" --router_purge_steps "${ROUTER_PURGE_STEPS:-0}" \
  --router_confidence_alpha "${ROUTER_CONFIDENCE_ALPHA:-0.1}"

"$PYTHON" scripts/evaluate_safe_router.py "${COMMON[@]}" \
  --router "$OUTROOT/router" --output_dir "$OUTROOT" \
  --router_num_horizon_blocks "${ROUTER_NUM_HORIZON_BLOCKS:-4}" \
  --router_decision "${ROUTER_DECISION:-safe_top1_blend}" \
  --router_min_gain "${ROUTER_MIN_GAIN:-0.0}" --router_full_gain "${ROUTER_FULL_GAIN:-0.02}" \
  --router_uncertainty_beta "${ROUTER_UNCERTAINTY_BETA:-0.1}" \
  --router_temperature "${ROUTER_TEMPERATURE:-0.1}" \
  --batch_size "${ROUTER_BATCH_SIZE:-0}" --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}" --expert_device_policy "${EXPERT_DEVICE_POLICY:-resident}"
