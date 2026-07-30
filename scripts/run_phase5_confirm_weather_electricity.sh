#!/bin/bash
# AsySpecX Phase 5-Lockdown confirmation: re-run weather/electricity with more
# seeds to confirm the small Phase 4 differences before claiming final.
#
# weather arms:      individual, individual_revin, period_multi, individual_period
# electricity arms:  cross, period_multi, individual_period (+ period_multi_gate_l1 if ENABLE_PERIOD_REG=1)
# Uses --val_num_segments 4. Does NOT run the whole full-field pool.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASETS="${DATASETS:-weather electricity}"
SEQ_LEN="${SEQ_LEN:-720}"
SEEDS="${SEEDS:-2021 2022 2023 2024 2025}"
GPU="${GPU:-0}"
ENABLE_PERIOD_REG="${ENABLE_PERIOD_REG:-0}"
VAL_NUM_SEGMENTS="${VAL_NUM_SEGMENTS:-4}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

mkdir -p logs/AsySpecX_phase5 checkpoints results

arms_for() {
    case "$1" in
        weather)     echo "phase5_asx_individual phase5_asx_individual_revin phase5_asx_period_multi phase5_asx_individual_period" ;;
        electricity)
            local a="phase5_asx_cross phase5_asx_period_multi phase5_asx_individual_period"
            [ "$ENABLE_PERIOD_REG" = "1" ] && a="$a phase5_asx_period_multi_gate_l1"
            echo "$a" ;;
        *) echo "phase5_asx_individual phase5_asx_period_multi" ;;
    esac
}

for dataset in $DATASETS; do
    load_dataset "$dataset"
    apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
    [ -n "$EPOCHS" ] && epochs="$EPOCHS"
    [ -n "$PATIENCE" ] && patience="$PATIENCE"
    if [ -n "${PERIODS:-}" ]; then periods="$PERIODS"; else periods="$(phase5_periods_for "$dataset")"; fi
    first_period="${periods%%,*}"; first_period="${first_period%%+*}"

    for arm in $(arms_for "$dataset"); do
        mapfile -t FLAGS < <(phase5_arm_flags "$arm" "$periods" "$first_period")
        for seed in $SEEDS; do
            for pl in $PRED_LENS; do
                model_id="AsySpecX_${arm}_${data_key}_L${SEQ_LEN}_H${pl}"
                log_file="logs/AsySpecX_phase5/${model_id}_sd${seed}.log"
                echo "[$(date '+%F %T')] confirm $arm $data_key H=$pl sd=$seed log=$log_file"
                python -u run.py \
                    --is_training 1 --random_seed "$seed" \
                    --root_path "./dataset/$subdir/" --data_path "$data_path" \
                    --model_id "$model_id" \
                    --model AsySpecX --data "$data_name" --features M \
                    --seq_len "$SEQ_LEN" --pred_len "$pl" --enc_in "$enc_in" \
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

echo "Phase 5 confirmation done."
