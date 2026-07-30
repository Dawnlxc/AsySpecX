#!/bin/bash
# Submit AsySpecX phase-1 ablation jobs to Slurm.
#
# Defaults are Petrichor-oriented and overrideable via env:
#   ACCOUNT=od-241336 PARTITION=h24gpu CONDA_ROOT=/scratch3/lin250/conda_envs CONDA_ENV=tsfm
#
# Canary:
#   NO_PUSH=1 ONLY=phase1_fits_only DATASETS=ETTh1 SEEDS=2026 SEQ_LENS=96 PRED_LENS=96 EPOCHS=1 \
#     OUTROOT=phase1_results/canary bash scripts/slurm/submit_asyspecx_phase1.sh
#
# Full default:
#   bash scripts/slurm/submit_asyspecx_phase1.sh
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase1_results/main}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
DRY_RUN="${DRY_RUN:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"

DEFAULT_ARMS="phase1_fits_only phase1_cross_zero_global phase1_safe_cross phase1_safe_cross_backcast"
DEFAULT_DATASETS="ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08"
DEFAULT_SEEDS="2026 2027"

ARMS="${ARMS:-$DEFAULT_ARMS}"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
SEQ_LENS="${SEQ_LENS:-auto}"
PRED_LENS="${PRED_LENS:-auto}"
ONLY="${ONLY:-}"
EPOCHS="${EPOCHS:-}"

mkdir -p logs/slurm "$OUTROOT"

submitted=0
skipped=0

echo "AsySpecX phase1 submit"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION CONDA_ROOT=$CONDA_ROOT CONDA_ENV=$CONDA_ENV OUTROOT=$OUTROOT"
echo "ARMS=$ARMS"
echo "DATASETS=$DATASETS"
echo "SEEDS=$SEEDS"
echo "SEQ_LENS=$SEQ_LENS PRED_LENS=$PRED_LENS EPOCHS=${EPOCHS:-default}"

for arm in $ARMS; do
    if [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    for dataset in $DATASETS; do
        load_dataset "$dataset"
        if [ "$SEQ_LENS" = "auto" ]; then
            case "$dataset" in
                PEMS*) seqs="96" ;;
                *) seqs="96 720" ;;
            esac
        else
            seqs="$SEQ_LENS"
        fi
        if [ "$PRED_LENS" = "auto" ]; then
            pls="$pred_lens"
        else
            pls="$PRED_LENS"
        fi

        for seed in $SEEDS; do
            for sl in $seqs; do
                for pl in $pls; do
                    jobname="asx1_${arm}_${dataset}_s${sl}_p${pl}_d${seed}"
                    export_args="ALL,ARM=$arm,DATASET=$dataset,SEED=$seed,SEQ_LEN=$sl,PRED_LEN=$pl,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV"
                    if [ -n "$EPOCHS" ]; then export_args="$export_args,EPOCHS=$EPOCHS"; fi
                    if [ -n "${BATCH_SIZE:-}" ]; then export_args="$export_args,BATCH_SIZE=$BATCH_SIZE"; fi
                    if [ -n "${LR:-}" ]; then export_args="$export_args,LR=$LR"; fi
                    if [ -n "${CUT_FREQ:-}" ]; then export_args="$export_args,CUT_FREQ=$CUT_FREQ"; fi
                    if [ -n "${RANK:-}" ]; then export_args="$export_args,RANK=$RANK"; fi
                    if [ -n "${NUM_BANDS:-}" ]; then export_args="$export_args,NUM_BANDS=$NUM_BANDS"; fi
                    if [ -n "${RESIDUAL_CLIP_ETA:-}" ]; then export_args="$export_args,RESIDUAL_CLIP_ETA=$RESIDUAL_CLIP_ETA"; fi
                    if [ -n "${BACKCAST_LOSS_WEIGHT:-}" ]; then export_args="$export_args,BACKCAST_LOSS_WEIGHT=$BACKCAST_LOSS_WEIGHT"; fi
                    if [ -n "${GATE_INIT_LOGIT:-}" ]; then export_args="$export_args,GATE_INIT_LOGIT=$GATE_INIT_LOGIT"; fi

                    cmd=(sbatch -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$export_args" scripts/slurm/asyspecx_phase1_run.sbatch)
                    printf '[submit]'
                    printf ' %q' "${cmd[@]}"
                    printf '\n'
                    if [ "$DRY_RUN" != "1" ]; then
                        "${cmd[@]}"
                    fi
                    submitted=$((submitted + 1))
                done
            done
        done
    done
done

echo "Submitted jobs: $submitted (dry_run=$DRY_RUN)"

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

if [ "$NO_WATCH" != "1" ]; then
    mkdir -p logs
    echo "Starting watcher: logs/autopush_asyspecx_phase1.log"
    nohup bash scripts/slurm/autopush_asyspecx_phase1_watch.sh "$OUTROOT" "$MAIL_TO" "$NO_PUSH" \
        > logs/autopush_asyspecx_phase1.nohup 2>&1 &
fi
