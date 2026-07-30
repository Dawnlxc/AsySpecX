#!/bin/bash
# Submit AsySpecX Phase 5-Lockdown jobs to Slurm.
#
# PERIODS caveat: sbatch --export splits on comma -> a "24,168" list would be
# truncated. We convert commas to '+' before exporting; run.py --periods parses '+'.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase5_results/main}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0704v1}"
BASELINE_CSV="${BASELINE_CSV:-}"
VAL_NUM_SEGMENTS="${VAL_NUM_SEGMENTS:-4}"
DRY_RUN="${DRY_RUN:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"
ENABLE_PERIOD_REG="${ENABLE_PERIOD_REG:-0}"
RUN_PEMS_SEQ720="${RUN_PEMS_SEQ720:-0}"

DEFAULT_ARMS="phase5_asx_cross phase5_asx_cross_clip05 phase5_asx_individual phase5_asx_individual_revin phase5_asx_period_multi phase5_asx_individual_period"
[ "$ENABLE_PERIOD_REG" = "1" ] && DEFAULT_ARMS="$DEFAULT_ARMS phase5_asx_period_multi_gate_l1"

DEFAULT_DATASETS="weather electricity"
DEFAULT_SEEDS="2024 2025 2026"

ARMS="${ARMS:-$DEFAULT_ARMS}"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
SEQ_LENS="${SEQ_LENS:-720}"
PRED_LENS="${PRED_LENS:-auto}"
ONLY="${ONLY:-}"
EPOCHS="${EPOCHS:-}"

mkdir -p logs/slurm "$OUTROOT"

submitted=0
echo "AsySpecX phase5-lockdown submit"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION OUTROOT=$OUTROOT VAL_NUM_SEGMENTS=$VAL_NUM_SEGMENTS"
echo "ARMS=$ARMS"
echo "DATASETS=$DATASETS SEEDS=$SEEDS SEQ_LENS=$SEQ_LENS PRED_LENS=$PRED_LENS"

for arm in $ARMS; do
    if [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ]; then continue; fi
    for dataset in $DATASETS; do
        load_dataset "$dataset"
        if [ "$PRED_LENS" = "auto" ]; then pls="$pred_lens"; else pls="$PRED_LENS"; fi
        if [ -n "${PERIODS:-}" ]; then periods="$PERIODS"; else periods="$(phase5_periods_for "$dataset")"; fi
        periods_export="${periods//,/+}"
        for seed in $SEEDS; do
            for sl in $SEQ_LENS; do
                case "$dataset" in
                    PEMS*) if [ "$sl" != "96" ] && [ "$RUN_PEMS_SEQ720" != "1" ]; then continue; fi ;;
                esac
                for pl in $pls; do
                    jobname="asx5_${arm}_${dataset}_s${sl}_p${pl}_d${seed}"
                    export_args="ALL,ARM=$arm,DATASET=$dataset,SEED=$seed,SEQ_LEN=$sl,PRED_LEN=$pl,PERIODS=$periods_export,VAL_NUM_SEGMENTS=$VAL_NUM_SEGMENTS,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV"
                    [ -n "$EPOCHS" ] && export_args="$export_args,EPOCHS=$EPOCHS"
                    for var in BATCH_SIZE LR CUT_FREQ RANK NUM_BANDS PATIENCE; do
                        if [ -n "${!var:-}" ]; then export_args="$export_args,$var=${!var}"; fi
                    done
                    cmd=(sbatch -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$export_args" scripts/slurm/asyspecx_phase5_run.sbatch)
                    printf '[submit]'; printf ' %q' "${cmd[@]}"; printf '\n'
                    if [ "$DRY_RUN" != "1" ]; then "${cmd[@]}"; fi
                    submitted=$((submitted + 1))
                done
            done
        done
    done
done

echo "Submitted jobs: $submitted (dry_run=$DRY_RUN)"
[ "$DRY_RUN" = "1" ] && exit 0

if [ "$NO_WATCH" != "1" ]; then
    mkdir -p logs
    echo "Starting watcher: logs/autopush_asyspecx_phase5.log"
    BASELINE_CSV="$BASELINE_CSV" MAIL_SUBJECT="$MAIL_SUBJECT" \
    nohup bash scripts/slurm/autopush_asyspecx_phase5_watch.sh "$OUTROOT" "$MAIL_TO" "$NO_PUSH" \
        > logs/autopush_asyspecx_phase5.nohup 2>&1 &
fi
