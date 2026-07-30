#!/bin/bash
# Submit one Phase 9 stage. Top-level default=headroom sends the sole START mail.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh

STAGE="${STAGE:-headroom}"
ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
OUTROOT="${OUTROOT:-phase9_results/main}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0711v1}"
DATASETS="${DATASETS:-ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08}"
SEQ_LENS="${SEQ_LENS:-96 720}"
PRED_LENS="${PRED_LENS:-auto}"
EXPERT_SEEDS="${EXPERT_SEEDS:-2024,2025,2026}"
EXPERTS="${EXPERTS:-anchor,dlinear,split_clip,individual_revin,individual_period}"
ROUTER_BACKEND="${ROUTER_BACKEND:-xgboost}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-0}"
EXPERT_DEVICE_POLICY="${EXPERT_DEVICE_POLICY:-resident}"
ROUTER_NUM_HORIZON_BLOCKS="${ROUTER_NUM_HORIZON_BLOCKS:-4}"
ROUTER_CHANNEL_GROUPS="${ROUTER_CHANNEL_GROUPS:-1}"
ROUTER_SCOPE="${ROUTER_SCOPE:-cell}"
ROUTER_TARGET="${ROUTER_TARGET:-advantage}"
ROUTER_MIN_SAMPLES="${ROUTER_MIN_SAMPLES:-256}"
ROUTER_CV_FOLDS="${ROUTER_CV_FOLDS:-4}"
ROUTER_PURGE_STEPS="${ROUTER_PURGE_STEPS:-0}"
ROUTER_CONFIDENCE_ALPHA="${ROUTER_CONFIDENCE_ALPHA:-0.1}"
ROUTER_DECISION="${ROUTER_DECISION:-safe_top1_blend}"
ROUTER_MIN_GAIN="${ROUTER_MIN_GAIN:-0.0}"
ROUTER_FULL_GAIN="${ROUTER_FULL_GAIN:-0.02}"
ROUTER_UNCERTAINTY_BETA="${ROUTER_UNCERTAINTY_BETA:-0.1}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}"
ROUTER_OOF_SEED="${ROUTER_OOF_SEED:-2024}"
OOF_EPOCHS="${OOF_EPOCHS:-0}"
DRY_RUN="${DRY_RUN:-0}"
NO_MAIL="${NO_MAIL:-0}"
NO_WATCH="${NO_WATCH:-0}"
NO_PUSH="${NO_PUSH:-0}"
RUN_PEMS_SEQ720="${RUN_PEMS_SEQ720:-0}"

case "$STAGE" in headroom) prefix=asx9h ;; quick) prefix=asx9q ;; oof) prefix=asx9o ;; *) echo "bad STAGE=$STAGE" >&2; exit 2 ;; esac
if [ "$STAGE" = headroom ] && [ "$DRY_RUN" != 1 ] && [ "$NO_MAIL" != 1 ] && ! command -v mail >/dev/null 2>&1; then
  echo "[error] mail command is required before the formal Phase 9 submission" >&2
  exit 2
fi
mkdir -p logs/slurm logs/phase9 "$OUTROOT"
expert_seeds_export="${EXPERT_SEEDS//,/+}"
experts_export="${EXPERTS//,/+}"
stamp="$(date +%Y%m%d_%H%M%S)"
manifest_csv="logs/phase9/slurm_phase9_${STAGE}_${stamp}.csv"
printf 'stage,job_id,dataset,seq_len,pred_len\n' > "$manifest_csv"
submitted=0

for dataset in $DATASETS; do
  load_dataset "$dataset"
  [ "$PRED_LENS" = auto ] && pls="$pred_lens" || pls="$PRED_LENS"
  case "$dataset" in PEMS*) sls=96; [ "$RUN_PEMS_SEQ720" = 1 ] && sls="$SEQ_LENS" ;; *) sls="$SEQ_LENS" ;; esac
  for sl in $sls; do
    for pl in $pls; do
      jobname="${prefix}_${dataset}_s${sl}_p${pl}"
      cell_tag="${dataset}_sl${sl}_pl${pl}"
      if [ "$DRY_RUN" != 1 ]; then
        mkdir -p "$OUTROOT/job_status/$STAGE"
        rm -f "$OUTROOT/job_status/$STAGE/${cell_tag}.json"
      fi
      exports="ALL,STAGE=$STAGE,DATASET=$dataset,SEQ_LEN=$sl,PRED_LEN=$pl,OUTROOT=$OUTROOT,CONDA_ROOT=$CONDA_ROOT,CONDA_ENV=$CONDA_ENV,EXPERT_SEEDS=$expert_seeds_export,EXPERTS=$experts_export"
      for var in ROUTER_BACKEND ROUTER_BATCH_SIZE EXPERT_DEVICE_POLICY ROUTER_NUM_HORIZON_BLOCKS ROUTER_CHANNEL_GROUPS ROUTER_SCOPE ROUTER_TARGET ROUTER_MIN_SAMPLES ROUTER_CV_FOLDS ROUTER_PURGE_STEPS ROUTER_CONFIDENCE_ALPHA ROUTER_DECISION ROUTER_MIN_GAIN ROUTER_FULL_GAIN ROUTER_UNCERTAINTY_BETA ROUTER_TEMPERATURE ROUTER_OOF_SEED OOF_EPOCHS GLOBAL_QUICK_GATE_PASSED; do
        [ -n "${!var:-}" ] && exports="$exports,$var=${!var}"
      done
      cmd=(sbatch --parsable -J "$jobname" --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --export="$exports" scripts/slurm/asyspecx_phase9_run.sbatch)
      printf '[submit]'; printf ' %q' "${cmd[@]}"; printf '\n'
      if [ "$DRY_RUN" = 1 ]; then job_id=dryrun; else job_id="$("${cmd[@]}")"; fi
      printf '%s,%s,%s,%s,%s\n' "$STAGE" "$job_id" "$dataset" "$sl" "$pl" >> "$manifest_csv"
      submitted=$((submitted + 1))
    done
  done
done
echo "Submitted jobs: $submitted stage=$STAGE dry_run=$DRY_RUN manifest=$manifest_csv"
[ "$DRY_RUN" = 1 ] && exit 0

if [ "$STAGE" = headroom ]; then
  start_epoch="$(date +%s)"
  printf 'start_epoch=%s\njobs_headroom=%s\npartition=%s\n' "$start_epoch" "$submitted" "$PARTITION" > "$OUTROOT/.run_meta"
  if [ "$NO_MAIL" != 1 ] && [ ! -f "$OUTROOT/.start_mail_sent" ] && command -v mail >/dev/null 2>&1; then
    {
      echo "AsySpecX Phase 9 SafeRoute STARTED on $(hostname) at $(date)."
      echo "Initial headroom jobs: $submitted (1 GPU each), partition=$PARTITION"
      echo "Pipeline gates: headroom >=0.004 -> quick; quick >=0.002 -> rolling OOF."
      echo "Datasets: $DATASETS; seq_lens: $SEQ_LENS; experts: $EXPERTS; seeds: $EXPERT_SEEDS"
      echo "No intermediate emails will be sent."
    } | mail -s "[$MAIL_SUBJECT] Phase9 SafeRoute START ($submitted jobs)" "$MAIL_TO" && touch "$OUTROOT/.start_mail_sent"
  fi
  if [ "$NO_WATCH" != 1 ]; then
    ACCOUNT="$ACCOUNT" PARTITION="$PARTITION" CONDA_ROOT="$CONDA_ROOT" CONDA_ENV="$CONDA_ENV" \
      OUTROOT="$OUTROOT" MAIL_TO="$MAIL_TO" MAIL_SUBJECT="$MAIL_SUBJECT" NO_PUSH="$NO_PUSH" NO_MAIL="$NO_MAIL" \
      HEADROOM_MANIFEST="$manifest_csv" \
      DATASETS="$DATASETS" SEQ_LENS="$SEQ_LENS" PRED_LENS="$PRED_LENS" EXPERT_SEEDS="$EXPERT_SEEDS" EXPERTS="$EXPERTS" \
      ROUTER_BACKEND="$ROUTER_BACKEND" ROUTER_BATCH_SIZE="$ROUTER_BATCH_SIZE" ROUTER_NUM_HORIZON_BLOCKS="$ROUTER_NUM_HORIZON_BLOCKS" \
      ROUTER_CHANNEL_GROUPS="$ROUTER_CHANNEL_GROUPS" \
      EXPERT_DEVICE_POLICY="$EXPERT_DEVICE_POLICY" \
      ROUTER_SCOPE="$ROUTER_SCOPE" ROUTER_TARGET="$ROUTER_TARGET" ROUTER_MIN_SAMPLES="$ROUTER_MIN_SAMPLES" \
      ROUTER_CV_FOLDS="$ROUTER_CV_FOLDS" ROUTER_PURGE_STEPS="$ROUTER_PURGE_STEPS" ROUTER_CONFIDENCE_ALPHA="$ROUTER_CONFIDENCE_ALPHA" \
      ROUTER_DECISION="$ROUTER_DECISION" ROUTER_MIN_GAIN="$ROUTER_MIN_GAIN" ROUTER_FULL_GAIN="$ROUTER_FULL_GAIN" \
      ROUTER_UNCERTAINTY_BETA="$ROUTER_UNCERTAINTY_BETA" ROUTER_TEMPERATURE="$ROUTER_TEMPERATURE" \
      ROUTER_OOF_SEED="$ROUTER_OOF_SEED" OOF_EPOCHS="$OOF_EPOCHS" AUTO_QUICK="${AUTO_QUICK:-1}" AUTO_OOF="${AUTO_OOF:-1}" \
      nohup bash scripts/slurm/autopush_asyspecx_phase9_watch.sh > logs/autopush_asyspecx_phase9.nohup 2>&1 &
    echo "Watcher started: logs/autopush_asyspecx_phase9.log"
  fi
fi
