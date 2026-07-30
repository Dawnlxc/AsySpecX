#!/bin/bash -l
#SBATCH --job-name=asx3_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/../.." && pwd)}"
source scripts/_common.sh

ARM="${ARM:?ARM required}"
DATASET="${DATASET:?DATASET required}"
SEED="${SEED:?SEED required}"
SEQ_LEN="${SEQ_LEN:?SEQ_LEN required}"
PRED_LEN="${PRED_LEN:?PRED_LEN required}"
OUTROOT="${OUTROOT:-phase3_gapclose_results/main}"
PERIOD="${PERIOD:-24}"
EPOCHS="${EPOCHS:-}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p logs/slurm logs/AsySpecX_phase3 checkpoints results "$OUTROOT"

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
if [ -n "${PATIENCE:-}" ]; then patience="$PATIENCE"; fi
if [ -n "${BATCH_SIZE:-}" ]; then bs="$BATCH_SIZE"; fi
if [ -n "${LR:-}" ]; then lr="$LR"; fi
if [ -n "${CUT_FREQ:-}" ]; then cut_freq="$CUT_FREQ"; fi
if [ -n "${RANK:-}" ]; then rank="$RANK"; fi
if [ -n "${NUM_BANDS:-}" ]; then num_bands="$NUM_BANDS"; fi

case "$ARM" in
    phase3_anchor_hier_split)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none)
        ;;
    phase3_fits_shared)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode none --norm_mode rin_noaffine --temporal_adapter none --backcast_loss_weight 0.0)
        ;;
    phase3_fits_individual)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine --temporal_adapter none --backcast_loss_weight 0.0)
        ;;
    phase3_individual_hier_split)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing individual --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none)
        ;;
    phase3_fits_shared_revin_affine)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode none --norm_mode revin_affine --temporal_adapter none --backcast_loss_weight 0.0)
        ;;
    phase3_anchor_revin_affine)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode revin_affine --temporal_adapter none)
        ;;
    phase3_fits_shared_subtract_last)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode none --norm_mode subtract_last --temporal_adapter none --backcast_loss_weight 0.0)
        ;;
    phase3_anchor_sparse_period)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part split --gate_type hier_channel_band --gate_init_logit -6.0 --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter sparse_period --period "$PERIOD" --periodic_init seasonal_naive --temporal_fusion convex --temporal_gate_type global --temporal_gate_init_logit -4.0)
        ;;
    phase3_fits_sparse_period)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode none --norm_mode rin_noaffine --temporal_adapter sparse_period --period "$PERIOD" --periodic_init seasonal_naive --temporal_fusion convex --temporal_gate_type global --temporal_gate_init_logit -4.0 --backcast_loss_weight 0.0)
        ;;
    phase3_individual_sparse_period)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing individual --cross_mode none --norm_mode rin_noaffine --temporal_adapter sparse_period --period "$PERIOD" --periodic_init seasonal_naive --temporal_fusion convex --temporal_gate_type global --temporal_gate_init_logit -4.0 --backcast_loss_weight 0.0)
        ;;
    phase3_diag_only_weather_guard)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part diag_only --gate_type global --gate_init_logit -6.0 --gate_max 1.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none)
        ;;
    phase3_offdiag_only_anchor)
        PHASE_FLAGS=(--spectral_lift fits_linear --lift_sharing shared --cross_mode asym_lowrank --residual_part offdiag_only --gate_type global --gate_init_logit -6.0 --gate_max 1.0 --residual_clip_eta -1 --backcast_loss_weight 0.0 --norm_mode rin_noaffine --temporal_adapter none)
        ;;
    *)
        echo "[error] unknown ARM=$ARM" >&2
        exit 2
        ;;
esac

run_id="${ARM}_${data_key}_sl${SEQ_LEN}_pl${PRED_LEN}_sd${SEED}"
run_dir="$OUTROOT/$ARM/$data_key/sl${SEQ_LEN}_pl${PRED_LEN}_sd${SEED}"
log_file="logs/AsySpecX_phase3/${run_id}.log"
summary_file="$run_dir/run_summary.json"
mkdir -p "$run_dir"

echo "==== AsySpecX phase3 gapclose run ===="
echo "host=$(hostname) job_id=${SLURM_JOB_ID:-local} start=$(date)"
echo "arm=$ARM dataset=$data_key seed=$SEED seq_len=$SEQ_LEN pred_len=$PRED_LEN period=$PERIOD"
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
import json, os, re, sys
summary_path, log_path, status_s = sys.argv[1:4]
status = int(status_s)
text = ""
try:
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
except FileNotFoundError:
    pass
matches = re.findall(r"mse:([0-9eE+\-.]+),\s*mae:([0-9eE+\-.]+)", text)
val_matches = re.findall(r"Vali Loss:\s*([0-9eE+\-.]+)", text)
mse = mae = val_mse = None
if matches:
    mse, mae = float(matches[-1][0]), float(matches[-1][1])
if val_matches:
    val_mse = min(float(v) for v in val_matches)
payload = {
    "arm": os.environ.get("ARM"),
    "dataset": os.environ.get("DATASET"),
    "seed": int(os.environ.get("SEED", "0")),
    "seq_len": int(os.environ.get("SEQ_LEN", "0")),
    "pred_len": int(os.environ.get("PRED_LEN", "0")),
    "status": "ok" if status == 0 and mse is not None and mae is not None else "failed",
    "exit_code": status,
    "val_mse": val_mse,
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
