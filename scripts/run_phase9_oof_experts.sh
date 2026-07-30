#!/bin/bash
# Chronology-safe rolling OOF expert retraining. Never run before the quick gate.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

: "${EXPERT_MANIFEST:?EXPERT_MANIFEST is required}"
OUTROOT="${OUTROOT:-phase9_results/oof}"
PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/build_router_oof_meta.py \
  --expert_manifest "$EXPERT_MANIFEST" \
  --experts "${EXPERTS:-}" \
  --router_oof_seed "${ROUTER_OOF_SEED:-2024}" \
  --router_num_horizon_blocks "${ROUTER_NUM_HORIZON_BLOCKS:-4}" \
  --router_scope "${ROUTER_SCOPE:-cell}" \
  --router_purge_steps "${ROUTER_PURGE_STEPS:-0}" \
  --oof_epochs "${OOF_EPOCHS:-0}" \
  --batch_size "${ROUTER_BATCH_SIZE:-0}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}" \
  --expert_device_policy "${EXPERT_DEVICE_POLICY:-one_at_a_time}" \
  --output "$OUTROOT/meta_oof" \
  --overwrite "${OVERWRITE:-0}"
