#!/bin/bash
# Rolling-OOF router. Refuses to run unless the quick result cleared 0.002 MSE.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

: "${EXPERT_MANIFEST:?EXPERT_MANIFEST is required}"
: "${QUICK_RESULT:?QUICK_RESULT routed_results.csv is required}"
OUTROOT="${OUTROOT:-phase9_results/oof_cell}"
PYTHON="${PYTHON:-python}"

if [ "${GLOBAL_QUICK_GATE_PASSED:-0}" != "1" ]; then
"$PYTHON" - "$QUICK_RESULT" <<'PY'
import csv, sys
row = next(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
gain = float(row["anchor_mse"]) - float(row["routed_mse"])
if gain < 0.002:
    raise SystemExit(f"quick router gain {gain:.6f} < 0.002; rolling OOF is locked")
print(f"quick gate passed: gain={gain:.6f}")
PY
else
  echo "global quick gate already passed; running cell OOF"
fi

EXPERT_MANIFEST="$EXPERT_MANIFEST" OUTROOT="$OUTROOT" PYTHON="$PYTHON" \
  EXPERTS="${EXPERTS:-}" ROUTER_OOF_SEED="${ROUTER_OOF_SEED:-2024}" \
  ROUTER_NUM_HORIZON_BLOCKS="${ROUTER_NUM_HORIZON_BLOCKS:-4}" \
  ROUTER_SCOPE="${ROUTER_SCOPE:-cell}" ROUTER_PURGE_STEPS="${ROUTER_PURGE_STEPS:-0}" \
  OOF_EPOCHS="${OOF_EPOCHS:-0}" ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-0}" \
  NUM_WORKERS="${NUM_WORKERS:-0}" DEVICE="${DEVICE:-auto}" \
  EXPERT_DEVICE_POLICY="${EXPERT_DEVICE_POLICY:-resident}" OVERWRITE="${OVERWRITE:-0}" \
  bash scripts/run_phase9_oof_experts.sh

# Official validation calibrates q; it never trains on test labels.
"$PYTHON" scripts/build_router_meta.py \
  --expert_manifest "$EXPERT_MANIFEST" --expert_seeds "${EXPERT_SEEDS:-2024,2025,2026}" \
  --experts "${EXPERTS:-}" --split val --router_meta_source val \
  --router_num_horizon_blocks "${ROUTER_NUM_HORIZON_BLOCKS:-4}" \
  --router_scope "${ROUTER_SCOPE:-cell}" --batch_size "${ROUTER_BATCH_SIZE:-0}" \
  --num_workers "${NUM_WORKERS:-0}" --device "${DEVICE:-auto}" \
  --expert_device_policy "${EXPERT_DEVICE_POLICY:-resident}" \
  --output "$OUTROOT/meta_val_calibration" --overwrite "${OVERWRITE:-0}"

"$PYTHON" scripts/train_safe_router.py \
  --meta "$OUTROOT/meta_oof" --calibration_meta "$OUTROOT/meta_val_calibration" \
  --output_dir "$OUTROOT/router" --router_backend "${ROUTER_BACKEND:-xgboost}" \
  --router_scope "${ROUTER_SCOPE:-cell}" --router_target "${ROUTER_TARGET:-advantage}" \
  --router_min_samples "${ROUTER_MIN_SAMPLES:-256}" \
  --router_cv_folds "${ROUTER_CV_FOLDS:-4}" --router_purge_steps "${ROUTER_PURGE_STEPS:-0}" \
  --router_confidence_alpha "${ROUTER_CONFIDENCE_ALPHA:-0.1}"

"$PYTHON" scripts/evaluate_safe_router.py \
  --expert_manifest "$EXPERT_MANIFEST" --expert_seeds "${EXPERT_SEEDS:-2024,2025,2026}" \
  --experts "${EXPERTS:-}" --router "$OUTROOT/router" --output_dir "$OUTROOT" \
  --router_num_horizon_blocks "${ROUTER_NUM_HORIZON_BLOCKS:-4}" \
  --router_decision "${ROUTER_DECISION:-safe_top1_blend}" \
  --router_min_gain "${ROUTER_MIN_GAIN:-0.0}" --router_full_gain "${ROUTER_FULL_GAIN:-0.02}" \
  --router_uncertainty_beta "${ROUTER_UNCERTAINTY_BETA:-0.1}" \
  --router_temperature "${ROUTER_TEMPERATURE:-0.1}" \
  --batch_size "${ROUTER_BATCH_SIZE:-0}" --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}" --expert_device_policy "${EXPERT_DEVICE_POLICY:-resident}"
