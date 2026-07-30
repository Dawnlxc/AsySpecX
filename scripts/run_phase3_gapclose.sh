#!/bin/bash
# AsySpecX Phase 3-GapClose single dataset/length runner.
# electricity: start PERIOD=24; PERIOD=168 optional.
# weather: if 10-minute sampled, try PERIOD=144; PERIOD=24 sanity check.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASET="${DATASET:-weather}"
SEQ_LEN="${SEQ_LEN:-720}"
PRED_LEN="${PRED_LEN:-96}"
GPU="${GPU:-0}"
SEED="${SEED:-2024}"
PERIOD="${PERIOD:-24}"
RUN_ONLY="${RUN_ONLY:-}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"
CUT_FREQ_OVERRIDE="${CUT_FREQ:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
if [ -n "$EPOCHS" ]; then epochs="$EPOCHS"; fi
if [ -n "$PATIENCE" ]; then patience="$PATIENCE"; fi
if [ -n "$CUT_FREQ_OVERRIDE" ]; then cut_freq="$CUT_FREQ_OVERRIDE"; fi

mkdir -p logs/AsySpecX_phase3 checkpoints results

run_arm() {
    local arm="$1"
    shift
    if [ -n "$RUN_ONLY" ] && [ "$RUN_ONLY" != "$arm" ]; then return; fi
    local model_id="AsySpecX_${arm}_${data_key}_L${SEQ_LEN}_H${PRED_LEN}"
    local log_file="logs/AsySpecX_phase3/${model_id}_sd${SEED}.log"
    echo "[$(date '+%F %T')] $arm dataset=$data_key L=$SEQ_LEN H=$PRED_LEN seed=$SEED period=$PERIOD log=$log_file"
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

ANCHOR=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none)
FITS_SHARED=(--spectral_lift fits_linear --lift_sharing shared --cross_mode none --norm_mode rin_noaffine --temporal_adapter none --backcast_loss_weight 0.0)
PERIODIC=(--temporal_adapter sparse_period --period "$PERIOD" --periodic_init seasonal_naive --temporal_fusion convex --temporal_gate_type global --temporal_gate_init_logit -4.0)

run_arm phase3_anchor_hier_split "${ANCHOR[@]}"
run_arm phase3_fits_shared "${FITS_SHARED[@]}"
run_arm phase3_fits_individual --spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine --temporal_adapter none --backcast_loss_weight 0.0
run_arm phase3_individual_hier_split --spectral_lift fits_linear --lift_sharing individual --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none
run_arm phase3_fits_shared_revin_affine --spectral_lift fits_linear --lift_sharing shared --cross_mode none --norm_mode revin_affine --temporal_adapter none --backcast_loss_weight 0.0
run_arm phase3_anchor_revin_affine --spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode revin_affine --temporal_adapter none
run_arm phase3_fits_shared_subtract_last --spectral_lift fits_linear --lift_sharing shared --cross_mode none --norm_mode subtract_last --temporal_adapter none --backcast_loss_weight 0.0
run_arm phase3_anchor_sparse_period "${ANCHOR[@]}" "${PERIODIC[@]}"
run_arm phase3_fits_sparse_period "${FITS_SHARED[@]}" "${PERIODIC[@]}"
run_arm phase3_individual_sparse_period --spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine --backcast_loss_weight 0.0 "${PERIODIC[@]}"
run_arm phase3_diag_only_weather_guard --spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part diag_only --gate_type global --gate_init_logit -6.0 --gate_max 1.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none
run_arm phase3_offdiag_only_anchor --spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part offdiag_only --gate_type global --gate_init_logit -6.0 --gate_max 1.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none
