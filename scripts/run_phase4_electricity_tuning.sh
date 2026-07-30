#!/bin/bash
# AsySpecX Phase 4-Finalize electricity tuning: close 192/336 gap to PatchTST/SparseTSF.
#
# Compact by default (FULL_SWEEP=0): a small, curated set of anchor+period arms.
# FULL_SWEEP=1 opens the larger sweep (cut_freq x gate x fusion x reg).
# Do NOT run the full Cartesian product unless FULL_SWEEP=1.
#
# Select best arm via validation (scripts/select_by_validation.py), never by test.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASET="${DATASET:-electricity}"
SEQ_LEN="${SEQ_LEN:-720}"
GPU="${GPU:-0}"
SEED="${SEED:-2024}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
FULL_SWEEP="${FULL_SWEEP:-0}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
if [ -n "$EPOCHS" ]; then epochs="$EPOCHS"; fi
if [ -n "$PATIENCE" ]; then patience="$PATIENCE"; fi

mkdir -p logs/AsySpecX_phase4 checkpoints results

ANCHOR=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine)

run_one() {
    local arm="$1" pl="$2" cf="$3"; shift 3
    local model_id="AsySpecX_${arm}_${data_key}_L${SEQ_LEN}_H${pl}_cf${cf}"
    local log_file="logs/AsySpecX_phase4/${model_id}_sd${SEED}.log"
    echo "[$(date '+%F %T')] $arm pl=$pl cut_freq=$cf log=$log_file"
    python -u run.py \
        --is_training 1 --random_seed "$SEED" \
        --root_path "./dataset/$subdir/" --data_path "$data_path" \
        --model_id "$model_id" \
        --model AsySpecX --data "$data_name" --features M \
        --seq_len "$SEQ_LEN" --pred_len "$pl" --enc_in "$enc_in" \
        --train_epochs "$epochs" --patience "$patience" \
        --batch_size "$bs" --learning_rate "$lr" --num_workers 4 \
        --itr 1 --cut_freq "$cf" --individual 0 \
        --rank "$rank" --num_bands "$num_bands" \
        "$@" > "$log_file" 2>&1
}

if [ "$FULL_SWEEP" != "1" ]; then
    # Compact, curated set.
    for pl in $PRED_LENS; do
        # anchor single period=24, global gate
        run_one "anchor_p24_global" "$pl" "$cut_freq" "${ANCHOR[@]}" \
            --temporal_adapter sparse_period --period 24 --periodic_init seasonal_naive \
            --temporal_fusion convex --temporal_gate_type global --temporal_gate_init_logit -4.0
        # anchor multi period 24+168, sum_gated, horizon temporal gate
        run_one "anchor_p24+168_multi" "$pl" "$cut_freq" "${ANCHOR[@]}" \
            --temporal_adapter sparse_period --periods 24+168 --periodic_init seasonal_naive \
            --period_fusion sum_gated --period_gate_type period --period_gate_init_logit 0.0 \
            --temporal_fusion convex --temporal_gate_type horizon --temporal_gate_init_logit -4.0
        # individual + multi period
        run_one "individual_p24+168" "$pl" "$cut_freq" \
            --spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine \
            --backcast_loss_weight 0.0 \
            --temporal_adapter sparse_period --periods 24+168 --periodic_init seasonal_naive \
            --period_fusion sum_gated --period_gate_type period --period_gate_init_logit 0.0 \
            --temporal_fusion convex --temporal_gate_type horizon --temporal_gate_init_logit -4.0
    done
    echo "electricity compact tuning done. Set FULL_SWEEP=1 for the larger sweep."
    exit 0
fi

# ---- FULL_SWEEP=1 ----
CUT_FREQS="32 48 64 96"
for pl in $PRED_LENS; do
    # 1. anchor single period=24
    for cf in $CUT_FREQS; do
        for tg in global horizon channel; do
            for gl in -6 -4 -2; do
                run_one "anchor_p24_${tg}_g${gl}" "$pl" "$cf" "${ANCHOR[@]}" \
                    --temporal_adapter sparse_period --period 24 --periodic_init seasonal_naive \
                    --temporal_fusion convex --temporal_gate_type "$tg" --temporal_gate_init_logit "$gl"
            done
        done
    done
    # 2. anchor multi period=24,168
    for pf in sum_gated softmax; do
        for pg in period period_horizon; do
            for tg in global horizon; do
                for gl in -6 -4 -2; do
                    run_one "anchor_p24+168_${pf}_${pg}_${tg}_g${gl}" "$pl" "$cut_freq" "${ANCHOR[@]}" \
                        --temporal_adapter sparse_period --periods 24+168 --periodic_init seasonal_naive \
                        --period_fusion "$pf" --period_gate_type "$pg" --period_gate_init_logit 0.0 \
                        --temporal_fusion convex --temporal_gate_type "$tg" --temporal_gate_init_logit "$gl"
                done
            done
        done
    done
    # 3. individual + multi period, optional regularization
    for tg in global horizon; do
        for l1 in 0 1e-5 1e-4; do
            run_one "individual_p24+168_${tg}_l1${l1}" "$pl" "$cut_freq" \
                --spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine \
                --backcast_loss_weight 0.0 \
                --temporal_adapter sparse_period --periods 24+168 --periodic_init seasonal_naive \
                --period_fusion sum_gated --period_gate_type period --period_gate_init_logit 0.0 \
                --temporal_fusion convex --temporal_gate_type "$tg" --temporal_gate_init_logit -4.0 \
                --periodic_l1_weight "$l1"
        done
    done
done

echo "electricity FULL sweep done. Select by validation before reading any baseline gap."
