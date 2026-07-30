#!/bin/bash
# Submit AsySpecX Phase 3-GapClose jobs to Slurm.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase3_gapclose_results/main}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
DRY_RUN="${DRY_RUN:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"

DEFAULT_ARMS="phase3_anchor_hier_split phase3_fits_shared phase3_fits_individual phase3_individual_hier_split phase3_fits_shared_revin_affine phase3_anchor_revin_affine phase3_fits_shared_subtract_last phase3_anchor_sparse_period phase3_fits_sparse_period phase3_individual_sparse_period phase3_diag_only_weather_guard phase3_offdiag_only_anchor"
DEFAULT_DATASETS="weather electricity"
DEFAULT_SEEDS="2026 2027"

ARMS="${ARMS:-$DEFAULT_ARMS}"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
SEQ_LENS="${SEQ_LENS:-720}"
PRED_LENS="${PRED_LENS:-auto}"
ONLY="${ONLY:-}"
EPOCHS="${EPOCHS:-}"

mkdir -p logs/slurm "$OUTROOT"

submitted=0
echo "AsySpecX phase3-gapclose submit"
echo "ACCOUNT=$ACCOUNT PARTITION=$PARTITION CONDA_ROOT=$CONDA_ROOT CONDA_ENV=$CONDA_ENV OUTROOT=$OUTROOT"
echo "ARMS=$ARMS"
echo "DATASETS=$DATASETS"
echo "SEEDS=$SEEDS"
echo "SEQ_LENS=$SEQ_LENS PRED_LENS=$PRED_LENS EPOCHS=${EPOCHS:-default}"

for arm in $ARMS; do
    if [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ]; then
        continue
    fi
    for dataset in $DATASETS; do
        load_dataset "$dataset"
        if [ "$PRED_LENS" = "auto" ]; then
            pls="$pred_lens"
        else
            pls="$PRED_LENS"
        fi
        if [ -n "${PERIOD:-}" ]; then
            period="$PERIOD"
        else
            case "$dataset" in
                weather) period=144 ;;
                electricity) period=24 ;;
                *) period=24 ;;
            esac
        fi
        for seed in $SEEDS; do
            for sl in $SEQ_LENS; do
                for pl in $pls; do
                    jobname="asx3_${arm}_${dataset}_s${sl}_p${pl}_d${seed}"
                    export_args="ALL,ARM=$arm,DATASET=$dataset,SEED=$seed,SEQ_LEN=$sl,PRED_LEN=$pl,PERIOD=$period,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV"
                    if [ -n "$EPOCHS" ]; then export_args="$export_args,EPOCHS=$EPOCHS"; fi
                    for var in BATCH_SIZE LR CUT_FREQ RANK NUM_BANDS PATIENCE; do
                        if [ -n "${!var:-}" ]; then export_args="$export_args,$var=${!var}"; fi
                    done
                    cmd=(sbatch -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$export_args" scripts/slurm/asyspecx_phase3_run.sbatch)
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
    echo "Starting watcher: logs/autopush_asyspecx_phase3_gapclose.log"
    nohup bash scripts/slurm/autopush_asyspecx_phase3_gapclose_watch.sh "$OUTROOT" "$MAIL_TO" "$NO_PUSH" \
        > logs/autopush_asyspecx_phase3_gapclose.nohup 2>&1 &
fi
