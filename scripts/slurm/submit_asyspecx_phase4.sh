#!/bin/bash
# Submit AsySpecX Phase 4-Finalize jobs to Slurm.
#
# PERIODS caveat: sbatch --export splits on comma, which would truncate a
# "24,168" period list. We convert commas to '+' before exporting; run.py
# --periods parses '+' back to a list.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase4_results/main}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
DRY_RUN="${DRY_RUN:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"

DEFAULT_ARMS="phase4_asx_cross phase4_asx_individual phase4_asx_period_single phase4_asx_period_multi phase4_asx_individual_period phase4_asx_individual_revin phase4_asx_cross_revin"
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

# Default period list per dataset (comma form; converted to '+' for export).
default_periods() {
    case "$1" in
        weather)       echo "144" ;;
        electricity)   echo "24,168" ;;
        ETTh1|ETTh2)   echo "24,168" ;;
        ETTm1|ETTm2)   echo "96,672" ;;
        traffic)       echo "24,168" ;;
        PEMS04|PEMS08) echo "24" ;;
        *)             echo "24" ;;
    esac
}

submitted=0
echo "AsySpecX phase4-finalize submit"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION OUTROOT=$OUTROOT"
echo "ARMS=$ARMS"
echo "DATASETS=$DATASETS SEEDS=$SEEDS SEQ_LENS=$SEQ_LENS PRED_LENS=$PRED_LENS"

for arm in $ARMS; do
    if [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ]; then continue; fi
    for dataset in $DATASETS; do
        load_dataset "$dataset"
        if [ "$PRED_LENS" = "auto" ]; then pls="$pred_lens"; else pls="$PRED_LENS"; fi
        if [ -n "${PERIODS:-}" ]; then periods="$PERIODS"; else periods="$(default_periods "$dataset")"; fi
        periods_export="${periods//,/+}"   # comma -> plus for sbatch --export
        for seed in $SEEDS; do
            for sl in $SEQ_LENS; do
                for pl in $pls; do
                    jobname="asx4_${arm}_${dataset}_s${sl}_p${pl}_d${seed}"
                    export_args="ALL,ARM=$arm,DATASET=$dataset,SEED=$seed,SEQ_LEN=$sl,PRED_LEN=$pl,PERIODS=$periods_export,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV"
                    if [ -n "$EPOCHS" ]; then export_args="$export_args,EPOCHS=$EPOCHS"; fi
                    for var in BATCH_SIZE LR CUT_FREQ RANK NUM_BANDS PATIENCE; do
                        if [ -n "${!var:-}" ]; then export_args="$export_args,$var=${!var}"; fi
                    done
                    cmd=(sbatch -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$export_args" scripts/slurm/asyspecx_phase4_run.sbatch)
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
    echo "Starting watcher: logs/autopush_asyspecx_phase4.log"
    nohup bash scripts/slurm/autopush_asyspecx_phase4_watch.sh "$OUTROOT" "$MAIL_TO" "$NO_PUSH" \
        > logs/autopush_asyspecx_phase4.nohup 2>&1 &
fi
