#!/bin/bash
# Submit AsySpecX Phase 2 jobs to Slurm.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase2_results/main}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
DRY_RUN="${DRY_RUN:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"

DEFAULT_ARMS="phase2_global_all phase2_global_diag_only phase2_global_offdiag_only phase2_global_split phase2_hier_all phase2_hier_split phase2_self_band_gain_global phase2_global_all_clip05 phase2_hier_all_clip05"
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
echo "AsySpecX phase2 submit"
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
                    jobname="asx2_${arm}_${dataset}_s${sl}_p${pl}_d${seed}"
                    export_args="ALL,ARM=$arm,DATASET=$dataset,SEED=$seed,SEQ_LEN=$sl,PRED_LEN=$pl,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV"
                    if [ -n "$EPOCHS" ]; then export_args="$export_args,EPOCHS=$EPOCHS"; fi
                    for var in BATCH_SIZE LR CUT_FREQ RANK NUM_BANDS RESIDUAL_CLIP_ETA BACKCAST_LOSS_WEIGHT GATE_INIT_LOGIT GATE_LR_MULT; do
                        if [ -n "${!var:-}" ]; then export_args="$export_args,$var=${!var}"; fi
                    done
                    cmd=(sbatch -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$export_args" scripts/slurm/asyspecx_phase2_run.sbatch)
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
    echo "Starting watcher: logs/autopush_asyspecx_phase2.log"
    nohup bash scripts/slurm/autopush_asyspecx_phase2_watch.sh "$OUTROOT" "$MAIL_TO" "$NO_PUSH" \
        > logs/autopush_asyspecx_phase2.nohup 2>&1 &
fi
