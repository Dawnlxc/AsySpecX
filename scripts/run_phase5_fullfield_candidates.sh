#!/bin/bash
# AsySpecX Phase 5-Lockdown full-field locked candidates (local matrix runner).
#
# Locked pool (frozen -- do NOT add new structures here):
#   phase5_asx_cross, phase5_asx_cross_clip05, phase5_asx_individual,
#   phase5_asx_individual_revin, phase5_asx_period_multi, phase5_asx_individual_period
#   (+ phase5_asx_period_multi_gate_l1 only if ENABLE_PERIOD_REG=1)
#
# Interpretation: pick arms by validation only (scripts/run_phase5_selection.sh),
# never by best test cell. Report one fixed single-arm result AND the
# validation-selected result separately.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASETS="${DATASETS:-ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08}"
SEQ_LENS="${SEQ_LENS:-96 720}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-2024 2025 2026}"
RUN_PEMS_SEQ720="${RUN_PEMS_SEQ720:-0}"
ENABLE_PERIOD_REG="${ENABLE_PERIOD_REG:-0}"
VAL_NUM_SEGMENTS="${VAL_NUM_SEGMENTS:-4}"
PRED_LENS_OVERRIDE="${PRED_LENS:-}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"
RUN_ONLY="${RUN_ONLY:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

ARMS="phase5_asx_cross phase5_asx_cross_clip05 phase5_asx_individual phase5_asx_individual_revin phase5_asx_period_multi phase5_asx_individual_period"
if [ "$ENABLE_PERIOD_REG" = "1" ]; then
    ARMS="$ARMS phase5_asx_period_multi_gate_l1"
fi

mkdir -p logs/AsySpecX_phase5 checkpoints results

for dataset in $DATASETS; do
    load_dataset "$dataset"
    apply_asyspecx_overrides "$data_key" 720   # sets rank/num_bands/lr/bs/patience/epochs; cut_freq recomputed per sl below
    [ -n "$EPOCHS" ] && epochs="$EPOCHS"
    [ -n "$PATIENCE" ] && patience="$PATIENCE"

    if [ -n "${PERIODS:-}" ]; then periods="$PERIODS"; else periods="$(phase5_periods_for "$dataset")"; fi
    first_period="${periods%%,*}"; first_period="${first_period%%+*}"

    # pred_len set per dataset (respect PRED_LENS override).
    if [ -n "$PRED_LENS_OVERRIDE" ]; then pls="$PRED_LENS_OVERRIDE"; else pls="$pred_lens"; fi

    for sl in $SEQ_LENS; do
        case "$dataset" in
            PEMS*)
                if [ "$sl" != "96" ] && [ "$RUN_PEMS_SEQ720" != "1" ]; then
                    echo "[skip] $dataset seq_len=$sl (set RUN_PEMS_SEQ720=1 to enable)"; continue
                fi ;;
        esac
        apply_asyspecx_overrides "$data_key" "$sl"   # refresh cut_freq for this seq_len
        [ -n "$EPOCHS" ] && epochs="$EPOCHS"
        [ -n "$PATIENCE" ] && patience="$PATIENCE"
        for arm in $ARMS; do
            [ -n "$RUN_ONLY" ] && [ "$RUN_ONLY" != "$arm" ] && continue
            mapfile -t FLAGS < <(phase5_arm_flags "$arm" "$periods" "$first_period")
            for seed in $SEEDS; do
                for pl in $pls; do
                    model_id="AsySpecX_${arm}_${data_key}_L${sl}_H${pl}"
                    log_file="logs/AsySpecX_phase5/${model_id}_sd${seed}.log"
                    echo "[$(date '+%F %T')] $arm $data_key L=$sl H=$pl sd=$seed periods=$periods log=$log_file"
                    python -u run.py \
                        --is_training 1 --random_seed "$seed" \
                        --root_path "./dataset/$subdir/" --data_path "$data_path" \
                        --model_id "$model_id" \
                        --model AsySpecX --data "$data_name" --features M \
                        --seq_len "$sl" --pred_len "$pl" --enc_in "$enc_in" \
                        --train_epochs "$epochs" --patience "$patience" \
                        --batch_size "$bs" --learning_rate "$lr" --num_workers 4 \
                        --itr 1 --cut_freq "$cut_freq" --individual 0 \
                        --rank "$rank" --num_bands "$num_bands" \
                        --val_num_segments "$VAL_NUM_SEGMENTS" \
                        "${FLAGS[@]}" > "$log_file" 2>&1
                done
            done
        done
    done
done

echo "Phase 5 full-field candidates done. Select by validation: bash scripts/run_phase5_selection.sh"
