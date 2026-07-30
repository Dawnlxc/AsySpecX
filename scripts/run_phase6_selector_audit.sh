#!/bin/bash
# Phase 6-Protocol: run 5 selection variants on existing Phase 5 results, then audit.
# All selection is validation-only; test metrics never drive selection.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"

ROOT="${ROOT:-phase5_results/main}"
CSV="${CSV:-${ROOT}/results.csv}"
PYTHON="${PYTHON:-python}"
UNRESTRICTED="${UNRESTRICTED_JSON:-configs/selection/unrestricted.json}"
POLICY="${POLICY_JSON:-configs/selection/policy_family.json}"
PREFER="phase5_asx_individual_revin,phase5_asx_individual,phase5_asx_cross,phase5_asx_cross_clip05,phase5_asx_period_multi,phase5_asx_individual_period"

mkdir -p "$ROOT"

sel() {  # sel <name> <extra args...>
    local name="$1"; shift
    echo "== $name =="
    "$PYTHON" scripts/select_by_validation.py \
        --csv "$CSV" \
        --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
        --select_metric val_mse \
        --arm_allowlist_json "$UNRESTRICTED" \
        --output "$ROOT/selected_${name}.csv" \
        --summary "$ROOT/selected_${name}.md" \
        "$@" || echo "[warn] selection $name failed (see stderr)"
}

sel unrestricted_mean --metric_mode mean
sel unrestricted_last_segment --metric_mode last_segment
sel unrestricted_segment_robust --metric_mode segment_mean_plus_std --std_weight 0.5
sel unrestricted_margin_prefer_simple --metric_mode mean_plus_std --std_weight 0.25 \
    --selection_margin_pct 0.002 --prefer_arm_order "$PREFER"

echo "== policy_family =="
"$PYTHON" scripts/select_by_validation.py \
    --csv "$CSV" \
    --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
    --select_metric val_mse --metric_mode mean_plus_std --std_weight 0.5 \
    --selection_margin_pct 0.002 --prefer_arm_order "$PREFER" \
    --arm_allowlist_json "$POLICY" \
    --output "$ROOT/selected_policy_family.csv" \
    --summary "$ROOT/selected_policy_family.md" || echo "[warn] policy_family failed"

echo "== audit =="
"$PYTHON" scripts/audit_phase5_selectors.py \
    --csv "$CSV" \
    --selected_files "selected_unrestricted_mean.csv,selected_unrestricted_last_segment.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv" \
    --output_dir "$ROOT"

echo "Phase 6 selector audit done. See $ROOT/selector_audit.md"
