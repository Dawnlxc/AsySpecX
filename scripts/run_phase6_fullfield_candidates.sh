#!/bin/bash
# Phase 6-Protocol TRUE full-field candidate runner (local matrix).
#
# Fixes the Phase 5 gap where only weather/electricity seq_len=720 were run.
# 6 locked arms x all target datasets/seq_lens/pred_lens/seeds.
# PEMS runs seq_len=96 only unless RUN_PEMS_SEQ720=1.
# DRY_RUN=1 prints commands + the run-count estimate without launching.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASETS="${DATASETS:-ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08}"
SEQ_LENS="${SEQ_LENS:-96 720}"
SEEDS="${SEEDS:-2024 2025 2026}"
GPU="${GPU:-0}"
RUN_PEMS_SEQ720="${RUN_PEMS_SEQ720:-0}"
OUT_ROOT="${OUT_ROOT:-phase6_results/fullfield}"
VAL_NUM_SEGMENTS="${VAL_NUM_SEGMENTS:-4}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ONLY="${RUN_ONLY:-}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

ARMS="phase6_asx_cross phase6_asx_cross_clip05 phase6_asx_individual phase6_asx_individual_revin phase6_asx_period_multi phase6_asx_individual_period"
NARMS=6

mkdir -p logs/AsySpecX_phase6 checkpoints results "$OUT_ROOT"

# ---- run count estimate ----
count=0
for dataset in $DATASETS; do
    load_dataset "$dataset" >/dev/null
    case "$dataset" in
        PEMS*) sls="96"; [ "$RUN_PEMS_SEQ720" = "1" ] && sls="$SEQ_LENS" ;;
        *)     sls="$SEQ_LENS" ;;
    esac
    npl=$(printf '%s\n' $pred_lens | wc -w)
    nsl=$(printf '%s\n' $sls | wc -w)
    nsd=$(printf '%s\n' $SEEDS | wc -w)
    count=$((count + nsl * npl * nsd * NARMS))
done
echo "Phase 6 full-field estimate: $count runs (arms=$NARMS, datasets=[$DATASETS], seq_lens=[$SEQ_LENS], seeds=[$SEEDS], RUN_PEMS_SEQ720=$RUN_PEMS_SEQ720)"
echo "OUT_ROOT=$OUT_ROOT VAL_NUM_SEGMENTS=$VAL_NUM_SEGMENTS DRY_RUN=$DRY_RUN"

for dataset in $DATASETS; do
    load_dataset "$dataset"
    if [ -n "${PERIODS:-}" ]; then periods="$PERIODS"; else periods="$(phase5_periods_for "$dataset")"; fi
    first_period="${periods%%,*}"; first_period="${first_period%%+*}"
    case "$dataset" in
        PEMS*) sls="96"; [ "$RUN_PEMS_SEQ720" = "1" ] && sls="$SEQ_LENS" ;;
        *)     sls="$SEQ_LENS" ;;
    esac
    for sl in $sls; do
        apply_asyspecx_overrides "$data_key" "$sl"
        [ -n "$EPOCHS" ] && epochs="$EPOCHS"
        [ -n "$PATIENCE" ] && patience="$PATIENCE"
        for arm in $ARMS; do
            [ -n "$RUN_ONLY" ] && [ "$RUN_ONLY" != "$arm" ] && continue
            mapfile -t FLAGS < <(phase6_arm_flags "$arm" "$periods" "$first_period")
            for seed in $SEEDS; do
                for pl in $pred_lens; do
                    model_id="AsySpecX_${arm}_${data_key}_L${sl}_H${pl}"
                    log_file="logs/AsySpecX_phase6/${model_id}_sd${seed}.log"
                    cmd=(python -u run.py
                        --is_training 1 --random_seed "$seed"
                        --root_path "./dataset/$subdir/" --data_path "$data_path"
                        --model_id "$model_id"
                        --model AsySpecX --data "$data_name" --features M
                        --seq_len "$sl" --pred_len "$pl" --enc_in "$enc_in"
                        --train_epochs "$epochs" --patience "$patience"
                        --batch_size "$bs" --learning_rate "$lr" --num_workers 4
                        --itr 1 --cut_freq "$cut_freq" --individual 0
                        --rank "$rank" --num_bands "$num_bands"
                        --val_num_segments "$VAL_NUM_SEGMENTS"
                        "${FLAGS[@]}")
                    if [ "$DRY_RUN" = "1" ]; then
                        printf '[dry]'; printf ' %q' "${cmd[@]}"; printf '\n'
                    else
                        echo "[$(date '+%F %T')] $arm $data_key L=$sl H=$pl sd=$seed periods=$periods"
                        "${cmd[@]}" > "$log_file" 2>&1
                    fi
                done
            done
        done
    done
done

echo "Phase 6 full-field candidates done (dry_run=$DRY_RUN). Aggregate + select next."
