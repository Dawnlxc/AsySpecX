#!/bin/bash
# AsySpecX Phase 4-Finalize final candidate arms (single dataset/length/pred/seed).
#
# Interpretation guide:
#   weather:      look first at phase4_asx_individual (pure channel-specific FITS).
#   electricity:  look first at phase4_asx_period_single / phase4_asx_period_multi.
#   final table:  ALWAYS via validation selection (scripts/select_by_validation.py),
#                 never by picking the best test cell.
#
# Candidate pool is intentionally <= ~7 arms so validation selection stays fair.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASET="${DATASET:-weather}"
SEQ_LEN="${SEQ_LEN:-720}"
PRED_LEN="${PRED_LEN:-96}"
GPU="${GPU:-0}"
SEED="${SEED:-2024}"
PERIODS="${PERIODS:-}"
RUN_ONLY="${RUN_ONLY:-}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"
CUT_FREQ_OVERRIDE="${CUT_FREQ:-}"

# Per-dataset default period list (overridable via PERIODS env var). Comma OK
# locally; the slurm submit path converts to '+' to survive sbatch --export.
if [ -z "$PERIODS" ]; then
    case "$DATASET" in
        weather)              PERIODS="144" ;;   # 10-min sampling -> daily=144 (override if preprocessed differently)
        electricity)          PERIODS="24,168" ;; # daily + weekly
        ETTh1|ETTh2)          PERIODS="24,168" ;;
        ETTm1|ETTm2)          PERIODS="96,672" ;;
        traffic)              PERIODS="24,168" ;;
        PEMS04|PEMS08)        PERIODS="24" ;;    # sampling period uncertain; try 12/24/96 manually
        *)                    PERIODS="24" ;;
    esac
fi
FIRST_PERIOD="${PERIODS%%,*}"; FIRST_PERIOD="${FIRST_PERIOD%%+*}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
if [ -n "$EPOCHS" ]; then epochs="$EPOCHS"; fi
if [ -n "$PATIENCE" ]; then patience="$PATIENCE"; fi
if [ -n "$CUT_FREQ_OVERRIDE" ]; then cut_freq="$CUT_FREQ_OVERRIDE"; fi

mkdir -p logs/AsySpecX_phase4 checkpoints results

echo "Phase4 final candidates: dataset=$data_key L=$SEQ_LEN H=$PRED_LEN seed=$SEED PERIODS=$PERIODS (first=$FIRST_PERIOD)"

run_arm() {
    local arm="$1"; shift
    if [ -n "$RUN_ONLY" ] && [ "$RUN_ONLY" != "$arm" ]; then return; fi
    local model_id="AsySpecX_${arm}_${data_key}_L${SEQ_LEN}_H${PRED_LEN}"
    local log_file="logs/AsySpecX_phase4/${model_id}_sd${SEED}.log"
    echo "[$(date '+%F %T')] $arm log=$log_file"
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

CROSS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none)

# 1. Phase 2/3 cross anchor
run_arm phase4_asx_cross "${CROSS[@]}"

# 2. Weather-best individual FITS
run_arm phase4_asx_individual --spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine --temporal_adapter none --backcast_loss_weight 0.0

# 3. Existing single-period sparse adapter (first period)
run_arm phase4_asx_period_single \
    --spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split \
    --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 \
    --backcast_loss_weight 0.0 --norm_mode rin_noaffine \
    --temporal_adapter sparse_period --period "$FIRST_PERIOD" --periodic_init seasonal_naive \
    --temporal_fusion convex --temporal_gate_type global --temporal_gate_init_logit -4.0

# 4. New multi-period sparse adapter
run_arm phase4_asx_period_multi \
    --spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split \
    --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 \
    --backcast_loss_weight 0.0 --norm_mode rin_noaffine \
    --temporal_adapter sparse_period --periods "$PERIODS" --periodic_init seasonal_naive \
    --period_fusion sum_gated --period_gate_type period --period_gate_init_logit 0.0 \
    --temporal_fusion convex --temporal_gate_type horizon --temporal_gate_init_logit -4.0 \
    --periodic_l1_weight 0.0 --periodic_l2_weight 0.0

# 5. Robust compromise: individual backbone + multi-period adapter
run_arm phase4_asx_individual_period \
    --spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine \
    --temporal_adapter sparse_period --periods "$PERIODS" --periodic_init seasonal_naive \
    --period_fusion sum_gated --period_gate_type period --period_gate_init_logit 0.0 \
    --temporal_fusion convex --temporal_gate_type horizon --temporal_gate_init_logit -4.0 \
    --backcast_loss_weight 0.0

# 6. Weather tuning candidate: individual + revin_affine
run_arm phase4_asx_individual_revin --spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode revin_affine --temporal_adapter none --backcast_loss_weight 0.0

# 7. Normalization candidate: cross + revin_affine
run_arm phase4_asx_cross_revin \
    --spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split \
    --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 \
    --backcast_loss_weight 0.0 --norm_mode revin_affine --temporal_adapter none
