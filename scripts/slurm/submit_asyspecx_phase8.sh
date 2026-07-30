#!/bin/bash
# Submit AsySpecX Phase 8-Hydra jobs to Slurm.
# 7 arms (COMPACT=1 -> 4). Auto/union-period resolved in-job (train only).
# Sends ONE start email now; the watcher sends ONE done email with duration,
# GPU-job count, and result summary (no intermediate mails).
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase8_results/hydra}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0709v1}"
BASELINE_CSV="${BASELINE_CSV:-}"
PHASE6_CSV="${PHASE6_CSV:-phase6_results/fullfield/results.csv}"
PHASE7_CSV="${PHASE7_CSV:-phase7_results/merged/results.csv}"
VAL_NUM_SEGMENTS="${VAL_NUM_SEGMENTS:-4}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
COMPACT="${COMPACT:-1}"
DRY_RUN="${DRY_RUN:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"
NO_MAIL="${NO_MAIL:-0}"
RUN_PEMS_SEQ720="${RUN_PEMS_SEQ720:-0}"

if [ "$COMPACT" = "1" ]; then DEFAULT_ARMS="$PHASE8_ARMS_COMPACT"; else DEFAULT_ARMS="$PHASE8_ARMS_FULL"; fi
ARMS="${ARMS:-$DEFAULT_ARMS}"
DATASETS="${DATASETS:-ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08}"
SEEDS="${SEEDS:-2024 2025 2026}"
SEQ_LENS="${SEQ_LENS:-96 720}"
PRED_LENS="${PRED_LENS:-auto}"
ONLY="${ONLY:-}"
EPOCHS="${EPOCHS:-}"

mkdir -p logs/slurm "$OUTROOT"
submitted=0
echo "AsySpecX phase8 hydra submit (COMPACT=$COMPACT)"
echo "ARMS=$ARMS"
echo "DATASETS=$DATASETS SEEDS=$SEEDS SEQ_LENS=$SEQ_LENS RUN_PEMS_SEQ720=$RUN_PEMS_SEQ720 SAVE_PREDICTIONS=$SAVE_PREDICTIONS"

for arm in $ARMS; do
    [ -n "$ONLY" ] && [ "$ONLY" != "$arm" ] && continue
    for dataset in $DATASETS; do
        load_dataset "$dataset"
        if [ "$PRED_LENS" = "auto" ]; then pls="$pred_lens"; else pls="$PRED_LENS"; fi
        if [ -n "${PERIODS:-}" ]; then periods="$PERIODS"; else periods="$(phase5_periods_for "$dataset")"; fi
        periods_export="${periods//,/+}"
        case "$dataset" in PEMS*) sls="96"; [ "$RUN_PEMS_SEQ720" = "1" ] && sls="$SEQ_LENS" ;; *) sls="$SEQ_LENS" ;; esac
        for seed in $SEEDS; do
            for sl in $sls; do
                for pl in $pls; do
                    jobname="asx8_${arm}_${dataset}_s${sl}_p${pl}_d${seed}"
                    export_args="ALL,ARM=$arm,DATASET=$dataset,SEED=$seed,SEQ_LEN=$sl,PRED_LEN=$pl,PERIODS=$periods_export,VAL_NUM_SEGMENTS=$VAL_NUM_SEGMENTS,SAVE_PREDICTIONS=$SAVE_PREDICTIONS,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV"
                    [ -n "$EPOCHS" ] && export_args="$export_args,EPOCHS=$EPOCHS"
                    for var in BATCH_SIZE LR CUT_FREQ RANK NUM_BANDS PATIENCE; do
                        if [ -n "${!var:-}" ]; then export_args="$export_args,$var=${!var}"; fi
                    done
                    cmd=(sbatch -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$export_args" scripts/slurm/asyspecx_phase8_run.sbatch)
                    printf '[submit]'; printf ' %q' "${cmd[@]}"; printf '\n'
                    [ "$DRY_RUN" != "1" ] && "${cmd[@]}"
                    submitted=$((submitted + 1))
                done
            done
        done
    done
done

echo "Submitted jobs: $submitted (dry_run=$DRY_RUN)"
[ "$DRY_RUN" = "1" ] && exit 0

# Record start marker (epoch + job count) for the watcher's duration/card report.
start_epoch=$(date +%s)
mkdir -p "$OUTROOT"
printf 'start_epoch=%s\njobs=%s\npartition=%s\n' "$start_epoch" "$submitted" "$PARTITION" > "$OUTROOT/.run_meta"

# ONE start email.
if [ "$NO_MAIL" != "1" ] && command -v mail >/dev/null 2>&1; then
    {
        echo "AsySpecX Phase 8-Hydra STARTED on $(hostname) at $(date)."
        echo "Submitted jobs: $submitted (1 GPU each), partition=$PARTITION, COMPACT=$COMPACT"
        echo "Arms: $ARMS"
        echo "Datasets: $DATASETS  seeds: $SEEDS  seq_lens: $SEQ_LENS"
        echo "OUTROOT: $PWD/$OUTROOT"
        echo "You will receive ONE more email when everything finishes (duration + cards + results)."
    } | mail -s "[$MAIL_SUBJECT] AsySpecX Phase8 START ($submitted jobs)" "$MAIL_TO" || true
fi

if [ "$NO_WATCH" != "1" ]; then
    mkdir -p logs
    echo "Starting watcher: logs/autopush_asyspecx_phase8.log"
    BASELINE_CSV="$BASELINE_CSV" MAIL_SUBJECT="$MAIL_SUBJECT" PHASE6_CSV="$PHASE6_CSV" PHASE7_CSV="$PHASE7_CSV" \
    SAVE_PREDICTIONS="$SAVE_PREDICTIONS" NO_MAIL="$NO_MAIL" \
    nohup bash scripts/slurm/autopush_asyspecx_phase8_watch.sh "$OUTROOT" "$MAIL_TO" "$NO_PUSH" \
        > logs/autopush_asyspecx_phase8.nohup 2>&1 &
fi
