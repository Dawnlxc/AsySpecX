#!/bin/bash
# AsySpecX Phase 2 single dataset/length runner.
# Recommended order:
#   1) mechanism attribution: global_all, diag_only, offdiag_only, split, self_band_gain
#   2) safe gate unlock: hier_all, hier_split, hier_all_clip05
#   3) small sweeps: gate_lr_mult 5/10, clip -1/0.5/1.0, gate_init -6/-4
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASET="${DATASET:-ETTh1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LEN="${PRED_LEN:-96}"
GPU="${GPU:-0}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"
RUN_ONLY="${RUN_ONLY:-}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:--6.0}"
GATE_LR_MULT="${GATE_LR_MULT:-5.0}"
RESIDUAL_CLIP_ETA_OVERRIDE="${RESIDUAL_CLIP_ETA:-}"
BACKCAST_LOSS_WEIGHT="${BACKCAST_LOSS_WEIGHT:-0.0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
if [ -n "$EPOCHS" ]; then epochs="$EPOCHS"; fi
if [ -n "$PATIENCE" ]; then patience="$PATIENCE"; fi

mkdir -p logs/AsySpecX_phase2 checkpoints results

run_arm() {
    local arm="$1"
    shift
    if [ -n "$RUN_ONLY" ] && [ "$RUN_ONLY" != "$arm" ]; then
        return
    fi
    local model_id="AsySpecX_${arm}_${data_key}_L${SEQ_LEN}_H${PRED_LEN}"
    local log_file="logs/AsySpecX_phase2/${model_id}_sd${SEED}.log"
    echo "[$(date '+%F %T')] $arm dataset=$data_key L=$SEQ_LEN H=$PRED_LEN seed=$SEED log=$log_file"
    python -u run.py \
        --is_training 1 --random_seed "$SEED" \
        --root_path "./dataset/$subdir/" --data_path "$data_path" \
        --model_id "$model_id" \
        --model AsySpecX --data "$data_name" --features M \
        --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" --enc_in "$enc_in" \
        --train_epochs "$epochs" --patience "$patience" \
        --batch_size "$bs" --learning_rate "$lr" --num_workers 4 \
        --itr 1 --cut_freq "$cut_freq" --individual 0 \
        --rank "$rank" --num_bands "$num_bands" \
        "$@" > "$log_file" 2>&1
}

clip_arg() {
    local default="$1"
    if [ -n "$RESIDUAL_CLIP_ETA_OVERRIDE" ]; then
        echo "$RESIDUAL_CLIP_ETA_OVERRIDE"
    else
        echo "$default"
    fi
}

COMMON=(--spectral_lift fits_linear --gate_init_logit "$GATE_INIT_LOGIT" --gate_max 1.0 --backcast_loss_weight "$BACKCAST_LOSS_WEIGHT")

run_arm phase2_global_all \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part all --gate_type global --residual_clip_eta "$(clip_arg -1)"

run_arm phase2_global_diag_only \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part diag_only --gate_type global --residual_clip_eta "$(clip_arg -1)"

run_arm phase2_global_offdiag_only \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part offdiag_only --gate_type global --residual_clip_eta "$(clip_arg -1)"

run_arm phase2_global_split \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part split --gate_type global --residual_clip_eta "$(clip_arg -1)"

run_arm phase2_hier_all \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part all --gate_type hier_channel_band --residual_clip_eta "$(clip_arg -1)" --gate_lr_mult "$GATE_LR_MULT"

run_arm phase2_hier_split \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --residual_clip_eta "$(clip_arg -1)" --gate_lr_mult "$GATE_LR_MULT"

run_arm phase2_self_band_gain_global \
    "${COMMON[@]}" --cross_mode self_band_gain --gate_type global --residual_clip_eta "$(clip_arg -1)"

run_arm phase2_global_all_clip05 \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part all --gate_type global --residual_clip_eta "$(clip_arg 0.5)"

run_arm phase2_hier_all_clip05 \
    "${COMMON[@]}" --cross_mode asym_lowrank --residual_part all --gate_type hier_channel_band --residual_clip_eta "$(clip_arg 0.5)" --gate_lr_mult "$GATE_LR_MULT"
