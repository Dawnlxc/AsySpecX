#!/bin/bash -l
#SBATCH --job-name=asx9_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/../.." && pwd)}"

STAGE="${STAGE:?STAGE required}"
DATASET="${DATASET:?DATASET required}"
SEQ_LEN="${SEQ_LEN:?SEQ_LEN required}"
PRED_LEN="${PRED_LEN:?PRED_LEN required}"
OUTROOT="${OUTROOT:-phase9_results/main}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
EXPERT_SEEDS="${EXPERT_SEEDS:-2024,2025,2026}"
EXPERTS="${EXPERTS:-anchor,dlinear,split_clip,individual_revin,individual_period}"
EXPERT_SEEDS="${EXPERT_SEEDS//+/,}"
EXPERTS="${EXPERTS//+/,}"
ROUTER_BACKEND="${ROUTER_BACKEND:-xgboost}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
cell_tag="${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}"
manifest="$OUTROOT/manifests/${cell_tag}.json"
status_dir="$OUTROOT/job_status/$STAGE"
mkdir -p logs/slurm "$OUTROOT/manifests" "$status_dir"

write_status() {
  local rc="$1" state=failed
  [ "$rc" = "0" ] && state=ok
  printf '{"stage":"%s","cell":"%s","status":"%s","exit_code":%s,"job_id":"%s"}\n' \
    "$STAGE" "$cell_tag" "$state" "$rc" "${SLURM_JOB_ID:-local}" > "$status_dir/${cell_tag}.json"
}
trap 'rc=$?; write_status "$rc"' EXIT

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [ -d "$CONDA_ROOT/$CONDA_ENV" ]; then conda activate "$CONDA_ROOT/$CONDA_ENV"; else conda activate "$CONDA_ENV"; fi
elif [ -x "$CONDA_ROOT/$CONDA_ENV/bin/python" ]; then
  export PATH="$CONDA_ROOT/$CONDA_ENV/bin:$PATH"
else
  echo "[error] cannot find conda environment" >&2; exit 2
fi

python scripts/build_phase9_manifest.py \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" \
  --expert_seeds "$EXPERT_SEEDS" --experts "$EXPERTS" \
  --checkpoint_root checkpoints --data_root dataset --repo_root . \
  --output "$manifest" --strict 1

common_env=(
  "EXPERT_MANIFEST=$manifest" "EXPERT_SEEDS=$EXPERT_SEEDS" "EXPERTS=$EXPERTS"
  "ROUTER_NUM_HORIZON_BLOCKS=${ROUTER_NUM_HORIZON_BLOCKS:-4}"
  "ROUTER_CHANNEL_GROUPS=${ROUTER_CHANNEL_GROUPS:-1}"
  "ROUTER_SCOPE=${ROUTER_SCOPE:-cell}" "ROUTER_BATCH_SIZE=$ROUTER_BATCH_SIZE"
  "NUM_WORKERS=$NUM_WORKERS" "DEVICE=${DEVICE:-auto}" \
  "EXPERT_DEVICE_POLICY=${EXPERT_DEVICE_POLICY:-resident}" "OVERWRITE=1"
)

case "$STAGE" in
  headroom)
    env "${common_env[@]}" OUTROOT="$OUTROOT/headroom_cells/$cell_tag" \
      SAVE_FULL_PREDICTIONS=0 bash scripts/run_phase9_headroom.sh
    ;;
  quick)
    env "${common_env[@]}" OUTROOT="$OUTROOT/quick/$cell_tag" \
      ROUTER_BACKEND="$ROUTER_BACKEND" ROUTER_TARGET="${ROUTER_TARGET:-advantage}" \
      ROUTER_MIN_SAMPLES="${ROUTER_MIN_SAMPLES:-256}" ROUTER_CV_FOLDS="${ROUTER_CV_FOLDS:-4}" \
      ROUTER_PURGE_STEPS="${ROUTER_PURGE_STEPS:-0}" ROUTER_CONFIDENCE_ALPHA="${ROUTER_CONFIDENCE_ALPHA:-0.1}" \
      ROUTER_DECISION="${ROUTER_DECISION:-safe_top1_blend}" ROUTER_MIN_GAIN="${ROUTER_MIN_GAIN:-0.0}" \
      ROUTER_FULL_GAIN="${ROUTER_FULL_GAIN:-0.02}" ROUTER_UNCERTAINTY_BETA="${ROUTER_UNCERTAINTY_BETA:-0.1}" \
      ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}" bash scripts/run_phase9_router_quick.sh
    ;;
  oof)
    quick_result="$OUTROOT/quick/$cell_tag/routed_results.csv"
    env "${common_env[@]}" OUTROOT="$OUTROOT/oof/$cell_tag" QUICK_RESULT="$quick_result" \
      GLOBAL_QUICK_GATE_PASSED="${GLOBAL_QUICK_GATE_PASSED:-0}" \
      ROUTER_BACKEND="$ROUTER_BACKEND" ROUTER_TARGET="${ROUTER_TARGET:-advantage}" \
      ROUTER_OOF_SEED="${ROUTER_OOF_SEED:-2024}" OOF_EPOCHS="${OOF_EPOCHS:-0}" \
      ROUTER_MIN_SAMPLES="${ROUTER_MIN_SAMPLES:-256}" ROUTER_CV_FOLDS="${ROUTER_CV_FOLDS:-4}" \
      ROUTER_PURGE_STEPS="${ROUTER_PURGE_STEPS:-0}" ROUTER_CONFIDENCE_ALPHA="${ROUTER_CONFIDENCE_ALPHA:-0.1}" \
      ROUTER_DECISION="${ROUTER_DECISION:-safe_top1_blend}" ROUTER_MIN_GAIN="${ROUTER_MIN_GAIN:-0.0}" \
      ROUTER_FULL_GAIN="${ROUTER_FULL_GAIN:-0.02}" ROUTER_UNCERTAINTY_BETA="${ROUTER_UNCERTAINTY_BETA:-0.1}" \
      ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.1}" bash scripts/run_phase9_router_oof.sh
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
