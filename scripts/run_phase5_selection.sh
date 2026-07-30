#!/bin/bash
# AsySpecX Phase 5-Lockdown: run the three validation-selection variants.
#   1. unrestricted_mean          -- cleanest validation-selected result.
#   2. unrestricted_last_segment  -- uses last val segment (needs val_num_segments>1).
#   3. policy_family_mean_plus_std -- analysis policy (per-dataset pools + prefer order).
# None of these use test metrics for selection.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

ROOT="${ROOT:-phase5_results/main}"
CSV="${CSV:-${ROOT}/results.csv}"
PYTHON="${PYTHON:-python}"
UNRESTRICTED="${UNRESTRICTED_JSON:-configs/selection/unrestricted.json}"
POLICY="${POLICY_JSON:-configs/selection/policy_family.json}"

mkdir -p "$ROOT"

echo "== 1. unrestricted_mean =="
"$PYTHON" scripts/select_by_validation.py \
    --csv "$CSV" \
    --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
    --select_metric val_mse --metric_mode mean \
    --arm_allowlist_json "$UNRESTRICTED" \
    --output "$ROOT/selected_unrestricted_mean.csv" \
    --summary "$ROOT/selected_unrestricted_mean.md"

echo "== 2. unrestricted_last_segment =="
if "$PYTHON" scripts/select_by_validation.py \
    --csv "$CSV" \
    --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
    --select_metric val_mse --metric_mode last_segment \
    --arm_allowlist_json "$UNRESTRICTED" \
    --output "$ROOT/selected_unrestricted_last_segment.csv" \
    --summary "$ROOT/selected_unrestricted_last_segment.md"; then
    echo "last_segment selection ok"
else
    echo "last_segment selection skipped (val_mse_seg* columns absent -> run with --val_num_segments>1)"
fi

echo "== 3. policy_family_mean_plus_std =="
"$PYTHON" scripts/select_by_validation.py \
    --csv "$CSV" \
    --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
    --select_metric val_mse --metric_mode mean_plus_std \
    --std_weight 0.5 --selection_margin_pct 0.002 \
    --prefer_arm_order "phase5_asx_individual_revin,phase5_asx_individual,phase5_asx_cross,phase5_asx_cross_clip05,phase5_asx_period_multi,phase5_asx_individual_period" \
    --arm_allowlist_json "$POLICY" \
    --output "$ROOT/selected_policy_family.csv" \
    --summary "$ROOT/selected_policy_family.md"

echo "Phase 5 selection done. Unrestricted_mean is the main validation-selected result; policy_family is an analysis policy."
