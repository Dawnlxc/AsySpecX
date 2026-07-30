#!/bin/bash
# AsySpecX Phase 4-Finalize weather tuning: close the last small gap to FITS.
#
# Compact by default (FULL_SWEEP=0): fits_individual over cut_freq {64,96,128}
# x norm_mode {rin_noaffine, revin_affine}, no sparse_period.
# FULL_SWEEP=1 adds cut_freq {32,48,192}, subtract_last, and a sparse_period arm.
#
# Select best cut_freq / norm via validation (scripts/select_by_validation.py),
# never by best test cell.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASET="${DATASET:-weather}"
SEQ_LEN="${SEQ_LEN:-720}"
GPU="${GPU:-0}"
SEED="${SEED:-2024}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
FULL_SWEEP="${FULL_SWEEP:-0}"
PERIODS="${PERIODS:-144}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
if [ -n "$EPOCHS" ]; then epochs="$EPOCHS"; fi
if [ -n "$PATIENCE" ]; then patience="$PATIENCE"; fi

mkdir -p logs/AsySpecX_phase4 checkpoints results

if [ "$FULL_SWEEP" = "1" ]; then
    CUT_FREQS="32 48 64 96 128 192"
    NORMS="rin_noaffine revin_affine subtract_last"
else
    CUT_FREQS="64 96 128"
    NORMS="rin_noaffine revin_affine"
fi

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
        --spectral_lift fits_linear --lift_sharing individual --cross_mode none \
        --backcast_loss_weight 0.0 "$@" > "$log_file" 2>&1
}

for pl in $PRED_LENS; do
    for cf in $CUT_FREQS; do
        for nm in $NORMS; do
            run_one "fits_individual_${nm}" "$pl" "$cf" --norm_mode "$nm" --temporal_adapter none
        done
    done
done

if [ "$FULL_SWEEP" = "1" ]; then
    # fits_individual + sparse_period (PERIODS=144), gate init sweep, cut_freq {64,96,128}
    for pl in $PRED_LENS; do
        for cf in 64 96 128; do
            for gl in -6 -4 -2; do
                run_one "fits_individual_sparse_p${PERIODS//,/+}_g${gl}" "$pl" "$cf" \
                    --norm_mode rin_noaffine \
                    --temporal_adapter sparse_period --periods "$PERIODS" --periodic_init seasonal_naive \
                    --period_fusion sum_gated --period_gate_type period --period_gate_init_logit 0.0 \
                    --temporal_fusion convex --temporal_gate_type horizon --temporal_gate_init_logit "$gl"
            done
        done
    done
fi

echo "weather tuning done. Summarize cut_freq: python scripts/summarize_cut_freq.py --csv <results.csv>"
echo "Select by validation before reading any FITS gap."
