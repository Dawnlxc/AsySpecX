#!/bin/bash
# Frozen-expert test headroom audit. All oracle outputs are analysis only.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

: "${EXPERT_MANIFEST:?EXPERT_MANIFEST is required}"
OUTROOT="${OUTROOT:-phase9_results/headroom_cell}"
PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/build_router_meta.py \
  --expert_manifest "$EXPERT_MANIFEST" \
  --expert_seeds "${EXPERT_SEEDS:-2024,2025,2026}" \
  --experts "${EXPERTS:-}" \
  --split test --router_meta_source test_analysis \
  --router_num_horizon_blocks "${ROUTER_NUM_HORIZON_BLOCKS:-4}" \
  --router_scope "${ROUTER_SCOPE:-cell}" \
  --router_channel_groups "${ROUTER_CHANNEL_GROUPS:-1}" \
  --batch_size "${ROUTER_BATCH_SIZE:-0}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}" \
  --expert_device_policy "${EXPERT_DEVICE_POLICY:-resident}" \
  --save_full_predictions "${SAVE_FULL_PREDICTIONS:-0}" \
  --output "$OUTROOT/meta_test" --overwrite "${OVERWRITE:-0}"

"$PYTHON" scripts/audit_router_headroom.py \
  --meta "$OUTROOT/meta_test" \
  --current_validation_selected "${CURRENT_VALIDATION_SELECTED:-0.333805}" \
  --output_dir "$OUTROOT/audit"
