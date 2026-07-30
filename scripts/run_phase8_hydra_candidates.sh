#!/bin/bash
# Phase 8-Hydra candidate runner (local matrix).
#
# 7 arms (COMPACT=1 -> 4). Auto/union-period arms resolve PERIODS via
# scripts/discover_periods.py (TRAIN split only), cached in OUT_ROOT/auto_periods.
# DRY_RUN=1 prints commands + run-count estimate without launching.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/.." && pwd)"
source scripts/_common.sh

DATASETS="${DATASETS:-ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08}"
SEQ_LENS="${SEQ_LENS:-96 720}"
SEEDS="${SEEDS:-2024 2025 2026}"
GPU="${GPU:-0}"
RUN_PEMS_SEQ720="${RUN_PEMS_SEQ720:-0}"
OUT_ROOT="${OUT_ROOT:-phase8_results/hydra}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
VAL_NUM_SEGMENTS="${VAL_NUM_SEGMENTS:-4}"
COMPACT="${COMPACT:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ONLY="${RUN_ONLY:-}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"

if [ "$COMPACT" = "1" ]; then ARMS="$PHASE8_ARMS_COMPACT"; else ARMS="$PHASE8_ARMS_FULL"; fi
NARMS=$(printf '%s\n' $ARMS | wc -w)
mkdir -p logs/AsySpecX_phase8 checkpoints results "$OUT_ROOT" "$OUT_ROOT/auto_periods"

count=0
for dataset in $DATASETS; do
    load_dataset "$dataset" >/dev/null
    case "$dataset" in PEMS*) sls="96"; [ "$RUN_PEMS_SEQ720" = "1" ] && sls="$SEQ_LENS" ;; *) sls="$SEQ_LENS" ;; esac
    npl=$(printf '%s\n' $pred_lens | wc -w); nsl=$(printf '%s\n' $sls | wc -w); nsd=$(printf '%s\n' $SEEDS | wc -w)
    count=$((count + nsl * npl * nsd * NARMS))
done
echo "Phase 8 estimate: $count runs (arms=$NARMS [COMPACT=$COMPACT], datasets=[$DATASETS], seq_lens=[$SEQ_LENS], seeds=[$SEEDS], RUN_PEMS_SEQ720=$RUN_PEMS_SEQ720)"
echo "OUT_ROOT=$OUT_ROOT SAVE_PREDICTIONS=$SAVE_PREDICTIONS VAL_NUM_SEGMENTS=$VAL_NUM_SEGMENTS DRY_RUN=$DRY_RUN"

resolve_periods() {  # <dataset> <sl> <manual> <method>
    local dataset="$1" sl="$2" manual="$3" method="$4"
    [ -z "$method" ] && { echo "$manual"; return; }
    local cache="$OUT_ROOT/auto_periods/${dataset}_sl${sl}_${method}.json" out
    if [ -f "$cache" ]; then
        out=$("$PYTHON" -c "import json,sys;print(','.join(str(p) for p in json.load(open(sys.argv[1]))['periods']))" "$cache" 2>/dev/null || true)
    else
        out=$("$PYTHON" scripts/discover_periods.py --dataset "$dataset" --data "$data_name" \
            --root_path "./dataset/$subdir/" --data_path "$data_path" --seq_len "$sl" --enc_in "$enc_in" \
            --cycle "$cycle" --method "$method" --topk 3 --period_min 4 --period_max 0 \
            --manual_periods "$manual" --max_periods 5 --fallback_periods "$manual" --output "$cache" \
            2>>logs/AsySpecX_phase8/discover.log | grep -E '^[0-9]+(,[0-9]+)*$' | tail -n1)
    fi
    [ -z "$out" ] && out="$manual"
    echo "$out"
}

for dataset in $DATASETS; do
    load_dataset "$dataset"
    if [ -n "${PERIODS:-}" ]; then manual_periods="$PERIODS"; else manual_periods="$(phase5_periods_for "$dataset")"; fi
    case "$dataset" in PEMS*) sls="96"; [ "$RUN_PEMS_SEQ720" = "1" ] && sls="$SEQ_LENS" ;; *) sls="$SEQ_LENS" ;; esac
    for sl in $sls; do
        apply_asyspecx_overrides "$data_key" "$sl"
        [ -n "$EPOCHS" ] && epochs="$EPOCHS"; [ -n "$PATIENCE" ] && patience="$PATIENCE"
        for arm in $ARMS; do
            [ -n "$RUN_ONLY" ] && [ "$RUN_ONLY" != "$arm" ] && continue
            method="$(phase8_arm_is_auto "$arm")"
            periods="$(resolve_periods "$dataset" "$sl" "$manual_periods" "$method")"
            first_period="${periods%%,*}"; first_period="${first_period%%+*}"
            mapfile -t FLAGS < <(phase8_arm_flags "$arm" "$periods" "$first_period")
            for seed in $SEEDS; do
                for pl in $pred_lens; do
                    model_id="AsySpecX_${arm}_${data_key}_L${sl}_H${pl}"
                    log_file="logs/AsySpecX_phase8/${model_id}_sd${seed}.log"
                    cmd=(python -u run.py --is_training 1 --random_seed "$seed"
                        --root_path "./dataset/$subdir/" --data_path "$data_path"
                        --model_id "$model_id" --model AsySpecX --data "$data_name" --features M
                        --seq_len "$sl" --pred_len "$pl" --enc_in "$enc_in"
                        --train_epochs "$epochs" --patience "$patience"
                        --batch_size "$bs" --learning_rate "$lr" --num_workers 4
                        --itr 1 --cut_freq "$cut_freq" --individual 0 --rank "$rank" --num_bands "$num_bands"
                        --val_num_segments "$VAL_NUM_SEGMENTS" --periods "$periods" "${FLAGS[@]}")
                    if [ "$SAVE_PREDICTIONS" = "1" ]; then
                        cmd+=(--save_predictions 1 --pred_save_dir "$OUT_ROOT/predictions" --pred_tag "${arm}__${data_key}__sl${sl}__pl${pl}__sd${seed}")
                    fi
                    if [ "$DRY_RUN" = "1" ]; then printf '[dry]'; printf ' %q' "${cmd[@]}"; printf '\n'
                    else echo "[$(date '+%F %T')] $arm $data_key L=$sl H=$pl sd=$seed periods=$periods method=${method:-manual}"
                         "${cmd[@]}" > "$log_file" 2>&1; fi
                done
            done
        done
    done
done
echo "Phase 8 candidates done (dry_run=$DRY_RUN)."
