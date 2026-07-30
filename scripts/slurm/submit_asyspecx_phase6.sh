#!/bin/bash
# Submit AsySpecX Phase 6 Full-Field jobs to Slurm.
#
# TRUE full-field: 6 arms x all target datasets/seq_lens/pred_lens/seeds.
# PEMS runs seq_len=96 only unless RUN_PEMS_SEQ720=1.
# PERIODS comma->plus for sbatch --export (run.py --periods parses '+').
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase6_results/fullfield}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0707v1}"
BASELINE_CSV="${BASELINE_CSV:-}"
VAL_NUM_SEGMENTS="${VAL_NUM_SEGMENTS:-4}"
DRY_RUN="${DRY_RUN:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"
RUN_PEMS_SEQ720="${RUN_PEMS_SEQ720:-0}"

DEFAULT_ARMS="phase6_asx_cross phase6_asx_cross_clip05 phase6_asx_individual phase6_asx_individual_revin phase6_asx_period_multi phase6_asx_individual_period"
DEFAULT_DATASETS="ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08"
DEFAULT_SEEDS="2024 2025 2026"

ARMS="${ARMS:-$DEFAULT_ARMS}"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
SEQ_LENS="${SEQ_LENS:-96 720}"
PRED_LENS="${PRED_LENS:-auto}"
ONLY="${ONLY:-}"
EPOCHS="${EPOCHS:-}"

mkdir -p logs/slurm "$OUTROOT"

submitted=0
echo "AsySpecX phase6 full-field submit"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION OUTROOT=$OUTROOT VAL_NUM_SEGMENTS=$VAL_NUM_SEGMENTS"
echo "ARMS=$ARMS"
echo "DATASETS=$DATASETS SEEDS=$SEEDS SEQ_LENS=$SEQ_LENS PRED_LENS=$PRED_LENS RUN_PEMS_SEQ720=$RUN_PEMS_SEQ720"

for arm in $ARMS; do
    if [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ]; then continue; fi
    for dataset in $DATASETS; do
        load_dataset "$dataset"
        if [ "$PRED_LENS" = "auto" ]; then pls="$pred_lens"; else pls="$PRED_LENS"; fi
        if [ -n "${PERIODS:-}" ]; then periods="$PERIODS"; else periods="$(phase5_periods_for "$dataset")"; fi
        periods_export="${periods//,/+}"
        case "$dataset" in
            PEMS*) sls="96"; [ "$RUN_PEMS_SEQ720" = "1" ] && sls="$SEQ_LENS" ;;
            *)     sls="$SEQ_LENS" ;;
        esac
        for seed in $SEEDS; do
            for sl in $sls; do
                for pl in $pls; do
                    jobname="asx6_${arm}_${dataset}_s${sl}_p${pl}_d${seed}"
                    export_args="ALL,ARM=$arm,DATASET=$dataset,SEED=$seed,SEQ_LEN=$sl,PRED_LEN=$pl,PERIODS=$periods_export,VAL_NUM_SEGMENTS=$VAL_NUM_SEGMENTS,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV"
                    [ -n "$EPOCHS" ] && export_args="$export_args,EPOCHS=$EPOCHS"
                    for var in BATCH_SIZE LR CUT_FREQ RANK NUM_BANDS PATIENCE; do
                        if [ -n "${!var:-}" ]; then export_args="$export_args,$var=${!var}"; fi
                    done
                    cmd=(sbatch -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$export_args" scripts/slurm/asyspecx_phase6_run.sbatch)
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
    echo "Starting watcher: logs/autopush_asyspecx_phase6.log"
    BASELINE_CSV="$BASELINE_CSV" MAIL_SUBJECT="$MAIL_SUBJECT" \
    nohup bash scripts/slurm/autopush_asyspecx_phase6_watch.sh "$OUTROOT" "$MAIL_TO" "$NO_PUSH" \
        > logs/autopush_asyspecx_phase6.nohup 2>&1 &
fi
