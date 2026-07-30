#!/bin/bash -l
#SBATCH --job-name=asx1_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# In SLURM $0 is a spooled path, so dirname-based cd breaks; prefer SLURM_SUBMIT_DIR
# (the repo root the job was submitted from), with a fallback for direct execution.
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/../.." && pwd)}"
source scripts/_common.sh

ARM="${ARM:?ARM required}"
DATASET="${DATASET:?DATASET required}"
SEED="${SEED:?SEED required}"
SEQ_LEN="${SEQ_LEN:?SEQ_LEN required}"
PRED_LEN="${PRED_LEN:?PRED_LEN required}"
OUTROOT="${OUTROOT:-phase1_results/main}"
EPOCHS="${EPOCHS:-}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p logs/slurm logs/AsySpecX_phase1 checkpoints results "$OUTROOT"

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [ -d "$CONDA_ROOT/$CONDA_ENV" ]; then
        conda activate "$CONDA_ROOT/$CONDA_ENV"
    else
        conda activate "$CONDA_ENV"
    fi
elif [ -x "$CONDA_ROOT/$CONDA_ENV/bin/python" ]; then
    export PATH="$CONDA_ROOT/$CONDA_ENV/bin:$PATH"
else
    echo "[error] cannot find conda env: CONDA_ROOT=$CONDA_ROOT CONDA_ENV=$CONDA_ENV" >&2
    exit 2
fi

load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
if [ -n "$EPOCHS" ]; then epochs="$EPOCHS"; fi
if [ -n "${BATCH_SIZE:-}" ]; then bs="$BATCH_SIZE"; fi
if [ -n "${LR:-}" ]; then lr="$LR"; fi
if [ -n "${CUT_FREQ:-}" ]; then cut_freq="$CUT_FREQ"; fi
if [ -n "${RANK:-}" ]; then rank="$RANK"; fi
if [ -n "${NUM_BANDS:-}" ]; then num_bands="$NUM_BANDS"; fi

case "$ARM" in
    phase1_fits_only)
        PHASE_FLAGS=(--spectral_lift fits_linear --cross_mode none --backcast_loss_weight 0.0)
        ;;
    phase1_cross_zero_global)
        PHASE_FLAGS=(--spectral_lift fits_linear --cross_mode asym_lowrank --gate_type global --gate_init_logit -6.0 --gate_max 1.0 --mask_self_transfer 0 --residual_clip_eta -1 --backcast_loss_weight 0.0)
        ;;
    phase1_safe_cross)
        PHASE_FLAGS=(--spectral_lift fits_linear --cross_mode asym_lowrank --gate_type channel_band --gate_init_logit -6.0 --gate_max 1.0 --mask_self_transfer 1 --residual_clip_eta 0.2 --backcast_loss_weight 0.0)
        ;;
    phase1_safe_cross_backcast)
        PHASE_FLAGS=(--spectral_lift fits_linear --cross_mode asym_lowrank --gate_type channel_band --gate_init_logit -6.0 --gate_max 1.0 --mask_self_transfer 1 --residual_clip_eta 0.2 --backcast_loss_weight 0.1)
        ;;
    *)
        echo "[error] unknown ARM=$ARM" >&2
        exit 2
        ;;
esac

if [ -n "${RESIDUAL_CLIP_ETA:-}" ]; then
    PHASE_FLAGS+=(--residual_clip_eta "$RESIDUAL_CLIP_ETA")
fi
if [ -n "${BACKCAST_LOSS_WEIGHT:-}" ]; then
    PHASE_FLAGS+=(--backcast_loss_weight "$BACKCAST_LOSS_WEIGHT")
fi
if [ -n "${GATE_INIT_LOGIT:-}" ]; then
    PHASE_FLAGS+=(--gate_init_logit "$GATE_INIT_LOGIT")
fi

run_id="${ARM}_${data_key}_sl${SEQ_LEN}_pl${PRED_LEN}_sd${SEED}"
run_dir="$OUTROOT/$ARM/$data_key/sl${SEQ_LEN}_pl${PRED_LEN}_sd${SEED}"
log_file="logs/AsySpecX_phase1/${run_id}.log"
summary_file="$run_dir/run_summary.json"
mkdir -p "$run_dir"

echo "==== AsySpecX phase1 run ===="
echo "host=$(hostname) job_id=${SLURM_JOB_ID:-local} start=$(date)"
echo "arm=$ARM dataset=$data_key seed=$SEED seq_len=$SEQ_LEN pred_len=$PRED_LEN"
echo "epochs=$epochs patience=$patience bs=$bs lr=$lr rank=$rank num_bands=$num_bands cut_freq=$cut_freq"
echo "log=$PWD/$log_file"

set +e
python -u run.py \
    --is_training 1 --random_seed "$SEED" \
    --root_path "./dataset/$subdir/" --data_path "$data_path" \
    --model_id "$run_id" \
    --model AsySpecX --data "$data_name" --features M \
    --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" --enc_in "$enc_in" \
    --train_epochs "$epochs" --patience "$patience" \
    --batch_size "$bs" --learning_rate "$lr" --num_workers "$NUM_WORKERS" \
    --itr 1 --cut_freq "$cut_freq" --individual 0 \
    --rank "$rank" --num_bands "$num_bands" \
    "${PHASE_FLAGS[@]}" \
    > "$log_file" 2>&1
status=$?
set -e

python - "$summary_file" "$log_file" "$status" <<'PY'
import json
import math
import os
import re
import sys

summary_path, log_path, status_s = sys.argv[1:4]
status = int(status_s)
text = ""
try:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
except FileNotFoundError:
    pass

matches = re.findall(r"mse:([0-9eE+\-.]+),\s*mae:([0-9eE+\-.]+)", text)
mse = mae = None
if matches:
    mse, mae = (float(matches[-1][0]), float(matches[-1][1]))

payload = {
    "arm": os.environ.get("ARM"),
    "dataset": os.environ.get("DATASET"),
    "seed": int(os.environ.get("SEED", "0")),
    "seq_len": int(os.environ.get("SEQ_LEN", "0")),
    "pred_len": int(os.environ.get("PRED_LEN", "0")),
    "status": "ok" if status == 0 and mse is not None and mae is not None else "failed",
    "exit_code": status,
    "mse": mse,
    "mae": mae,
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "log_file": log_path,
}
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "end=$(date) status=$status summary=$PWD/$summary_file"
exit "$status"
