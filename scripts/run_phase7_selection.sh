#!/bin/bash
# Phase 7-Breakthrough selection over the merged Phase6+Phase7 candidate pool.
# 4 validation-only variants. Test metrics never drive selection.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

ROOT="${ROOT:-phase7_results/merged}"
CSV="${CSV:-${ROOT}/results.csv}"
PYTHON="${PYTHON:-python}"
UNRESTRICTED="${UNRESTRICTED_JSON:-configs/selection/phase7_unrestricted.json}"
POLICY="${POLICY_JSON:-configs/selection/phase6_policy_family.json}"
PREFER="phase6_asx_individual_revin,phase6_asx_individual,phase6_asx_cross,phase6_asx_cross_clip05,phase7_period_multi_split_clip05,phase7_period_multi,phase6_asx_period_multi,phase7_period_multi_patchlinear,phase6_asx_individual_period"

mkdir -p "$ROOT"

echo "== 1. unrestricted_mean =="
"$PYTHON" scripts/select_by_validation.py --csv "$CSV" \
    --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
    --select_metric val_mse --metric_mode mean --arm_allowlist_json "$UNRESTRICTED" \
    --output "$ROOT/selected_unrestricted_mean.csv" --summary "$ROOT/selected_unrestricted_mean.md"

echo "== 2. unrestricted_segment_robust =="
if "$PYTHON" scripts/select_by_validation.py --csv "$CSV" \
    --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
    --select_metric val_mse --metric_mode segment_mean_plus_std --std_weight 0.5 \
    --arm_allowlist_json "$UNRESTRICTED" \
    --output "$ROOT/selected_unrestricted_segment_robust.csv" --summary "$ROOT/selected_unrestricted_segment_robust.md"; then
    echo "segment_robust ok"
else
    echo "segment_robust skipped (no val_mse_seg* columns)"
fi

echo "== 3. unrestricted_margin_prefer_simple =="
"$PYTHON" scripts/select_by_validation.py --csv "$CSV" \
    --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
    --select_metric val_mse --metric_mode mean_plus_std --std_weight 0.25 \
    --selection_margin_pct 0.002 --prefer_arm_order "$PREFER" --arm_allowlist_json "$UNRESTRICTED" \
    --output "$ROOT/selected_unrestricted_margin_prefer_simple.csv" --summary "$ROOT/selected_unrestricted_margin_prefer_simple.md"

if [ -f "$POLICY" ]; then
    echo "== 4. policy_family =="
    "$PYTHON" scripts/select_by_validation.py --csv "$CSV" \
        --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
        --select_metric val_mse --metric_mode mean_plus_std --std_weight 0.5 \
        --selection_margin_pct 0.002 --prefer_arm_order "$PREFER" --arm_allowlist_json "$POLICY" \
        --output "$ROOT/selected_policy_family.csv" --summary "$ROOT/selected_policy_family.md" || echo "[warn] policy_family failed"
fi

echo "Phase 7 selection done. unrestricted_mean is the main result."
