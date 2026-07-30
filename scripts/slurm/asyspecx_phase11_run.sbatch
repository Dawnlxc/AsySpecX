#!/bin/bash
#SBATCH --job-name=asx11_cell
#SBATCH --account=od-241336
#SBATCH --partition=h24gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO="${REPO:-/scratch3/lin250/bldgFM/DUBABA/AsySpecX}"
cd "$REPO"
source scripts/_common.sh

PYTHON="${PYTHON:-/scratch3/lin250/conda_envs/tsfm/bin/python}"
RUN_TAG="${RUN_TAG:-phase11_forecastability_0715v1}"
OUTROOT="${OUTROOT:-phase11_results/screen}"
ARM="${ARM:?ARM is required}"
BASE_PROFILE="${BASE_PROFILE:-ind_cycle_full}"
DATASET="${DATASET:?DATASET is required}"
SEQ_LEN="${SEQ_LEN:?SEQ_LEN is required}"
PRED_LEN="${PRED_LEN:?PRED_LEN is required}"
SEED="${SEED:?SEED is required}"
CUT_FREQ="${CUT_FREQ:?CUT_FREQ is required}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEFER_TEST="${DEFER_TEST:-1}"
INIT_MAX_WINDOWS="${INIT_MAX_WINDOWS:-2048}"
INIT_RIDGE="${INIT_RIDGE:-0.01}"
INIT_ARTIFACT_RANK="${INIT_ARTIFACT_RANK:-16}"

test -x "$PYTHON"
load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
[ -n "${BATCH_SIZE:-}" ] && bs="$BATCH_SIZE"
[ -n "${LR:-}" ] && lr="$LR"

periods="${PERIODS:-$(phase5_periods_for "$DATASET")}"
periods="${periods//,/+}"
individual_flags=(
    --spectral_lift fits_linear --lift_sharing individual
    --cross_mode none --norm_mode revin_affine
    --temporal_adapter none --backcast_loss_weight 0.0
)
cross_flags=(
    --spectral_lift fits_linear --lift_sharing shared
    --cross_mode asym_lowrank --residual_part split
    --gate_type hier_channel_band --gate_init_logit -6.0
    --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1
    --backcast_loss_weight 0.0 --norm_mode rin_noaffine
    --temporal_adapter none
)
case "$BASE_PROFILE" in
    ind_cycle_full)
        base_flags=("${individual_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0)
        ;;
    compact_period_cycle_full)
        base_flags=(
            "${cross_flags[@]}"
            --temporal_adapter compact_period --periods "$periods"
            --periodic_init seasonal_naive --period_fusion sum_gated
            --period_gate_type period --period_gate_init_logit 0.0
            --temporal_fusion convex --temporal_gate_type horizon
            --temporal_gate_init_logit -4.0
            --cycle_residual 1 --cycle_residual_rank 0
        )
        ;;
    *)
        echo "[phase11] unknown BASE_PROFILE=$BASE_PROFILE" >&2
        exit 2
        ;;
esac
dense_flags=(
    --linear_adapter direct_linear --linear_sharing shared
    --linear_init zeros --linear_fusion additive
    --linear_gate_type horizon --linear_gate_init_logit 4.0
)
forecast_common=(
    --forecast_kernel lowrank_time
    --forecast_kernel_fusion convex
    --forecast_kernel_gate_type horizon
    --forecast_kernel_gate_init_logit -4.0
    --gate_lr_mult 5.0
)

forecast_rank=0
forecast_init=none
forecast_channel_scale=0
forecast_sm_components=0
forecast_sm_sharing=none
forecast_sm_base_trainable=1
forecast_extension_shrink=none
forecast_phase_basis_dim=0
forecast_phase_max=0
case "$ARM" in
    anchor)
        flags=("${base_flags[@]}")
        ;;
    dense_direct)
        flags=("${base_flags[@]}" "${dense_flags[@]}")
        ;;
    fk_r4)
        forecast_rank=4; forecast_init=small_random; forecast_channel_scale=0
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 4 --forecast_kernel_init small_random)
        ;;
    fk_r8)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=0
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random)
        ;;
    fk_r8_cs)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1)
        ;;
    fk_sm2_shared)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=2; forecast_sm_sharing=shared
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 2 --forecast_kernel_sm_sharing shared)
        ;;
    fk_sm2_mode)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=2; forecast_sm_sharing=mode
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 2 --forecast_kernel_sm_sharing mode)
        ;;
    fk_sm2_tail2)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=2; forecast_sm_sharing=mode
        forecast_extension_shrink=tail2_linear
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 2 --forecast_kernel_sm_sharing mode --forecast_kernel_extension_shrink tail2_linear)
        ;;
    fk_sm4_mode)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=4; forecast_sm_sharing=mode
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode)
        ;;
    fk_sm4_ph2_q)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=4; forecast_sm_sharing=mode
        forecast_phase_basis_dim=2; forecast_phase_max=0.7853981633974483
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_phase_basis_dim 2 --forecast_kernel_phase_max 0.7853981633974483)
        ;;
    fk_sm4_ph4_q)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=4; forecast_sm_sharing=mode
        forecast_phase_basis_dim=4; forecast_phase_max=0.7853981633974483
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_phase_basis_dim 4 --forecast_kernel_phase_max 0.7853981633974483)
        ;;
    fk_sm4_ph4_h)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=4; forecast_sm_sharing=mode
        forecast_phase_basis_dim=4; forecast_phase_max=1.5707963267948966
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_phase_basis_dim 4 --forecast_kernel_phase_max 1.5707963267948966)
        ;;
    fk_sm4_frozen)
        forecast_rank=8; forecast_init=small_random; forecast_channel_scale=1
        forecast_sm_components=4; forecast_sm_sharing=mode; forecast_sm_base_trainable=0
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_sm_base_trainable 0)
        ;;
    fk_r8_svd)
        forecast_rank=8; forecast_init=ridge_svd; forecast_channel_scale=1
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init ridge_svd --forecast_kernel_channel_scale 1)
        ;;
    fk_r16_svd)
        forecast_rank=16; forecast_init=ridge_svd; forecast_channel_scale=1
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 16 --forecast_kernel_init ridge_svd --forecast_kernel_channel_scale 1)
        ;;
    *)
        echo "[phase11] unknown ARM=$ARM" >&2
        exit 2
        ;;
esac

init_path="${INIT_PATH:-$OUTROOT/artifacts/${data_key}_sl${SEQ_LEN}_pl${PRED_LEN}_r${INIT_ARTIFACT_RANK}_ridge${INIT_RIDGE}.pt}"
init_build_seconds=0
if [ "$forecast_init" = "ridge_svd" ]; then
    mkdir -p "$(dirname "$init_path")"
    init_start="$(date +%s)"
    (
        flock -x 9
        if [ ! -s "$init_path" ]; then
            echo "[phase11-init] building train-only artifact $init_path"
            "$PYTHON" -u scripts/build_phase11_forecastability_init.py \
                --dataset "$DATASET" --data "$data_name" \
                --root_path "./dataset/$subdir/" --data_path "$data_path" \
                --features M --target OT --embed timeF --freq h \
                --cycle "$cycle" --enc_in "$enc_in" \
                --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" \
                --rank "$INIT_ARTIFACT_RANK" --ridge "$INIT_RIDGE" \
                --norm_mode revin_affine --max_windows "$INIT_MAX_WINDOWS" \
                --batch_windows 64 --output "$init_path"
        else
            echo "[phase11-init] reusing $init_path"
        fi
    ) 9>"${init_path}.lock"
    init_build_seconds=$(( $(date +%s) - init_start ))
    flags+=(--forecast_kernel_init_path "$init_path")
fi

run_id="${RUN_TAG}_${ARM}_${data_key}_sl${SEQ_LEN}_pl${PRED_LEN}_cf${CUT_FREQ}_sd${SEED}"
run_dir="$OUTROOT/$ARM/$data_key/sl${SEQ_LEN}_pl${PRED_LEN}_cf${CUT_FREQ}_sd${SEED}"
log_file="logs/AsySpecX_phase11/${run_id}.log"
summary_file="$run_dir/run_summary.json"
mkdir -p logs/slurm logs/AsySpecX_phase11 checkpoints results "$run_dir"

echo "[phase11] host=$(hostname) job=${SLURM_JOB_ID:-local} start=$(date --iso-8601=seconds)"
echo "[phase11] run_id=$run_id arm=$ARM profile=$BASE_PROFILE dataset=$data_key sl=$SEQ_LEN pl=$PRED_LEN seed=$SEED cf=$CUT_FREQ periods=$periods"
echo "[phase11] epochs=$EPOCHS patience=$PATIENCE bs=$bs lr=$lr defer_test=$DEFER_TEST"
echo "[phase11] kernel_rank=$forecast_rank init=$forecast_init channel_scale=$forecast_channel_scale sm_components=$forecast_sm_components sm_sharing=$forecast_sm_sharing sm_base_trainable=$forecast_sm_base_trainable extension_shrink=$forecast_extension_shrink phase_basis=$forecast_phase_basis_dim phase_max=$forecast_phase_max init_build_seconds=$init_build_seconds"

set +e
"$PYTHON" -u run.py \
    --is_training 1 --random_seed "$SEED" \
    --root_path "./dataset/$subdir/" --data_path "$data_path" \
    --model_id "$run_id" --model AsySpecX --data "$data_name" --features M \
    --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" --enc_in "$enc_in" --cycle "$cycle" \
    --train_epochs "$EPOCHS" --patience "$PATIENCE" \
    --batch_size "$bs" --learning_rate "$lr" --num_workers "$NUM_WORKERS" \
    --itr 1 --cut_freq "$CUT_FREQ" --individual 0 \
    --rank "$rank" --num_bands "$num_bands" --val_num_segments 4 \
    --eval_test_during_train 0 --defer_test "$DEFER_TEST" \
    "${flags[@]}" >"$log_file" 2>&1
status=$?
set -e

export ARM BASE_PROFILE DATASET SEQ_LEN PRED_LEN SEED CUT_FREQ RUN_TAG OUTROOT periods
export forecast_rank forecast_init forecast_channel_scale init_path init_build_seconds DEFER_TEST
export forecast_sm_components forecast_sm_sharing forecast_sm_base_trainable
export forecast_extension_shrink
export forecast_phase_basis_dim forecast_phase_max
"$PYTHON" - "$summary_file" "$log_file" "$status" <<'PY'
import json
import os
import re
import sys

summary_path, log_path, status_text = sys.argv[1:4]
status = int(status_text)
try:
    text = open(log_path, encoding="utf-8", errors="replace").read()
except FileNotFoundError:
    text = ""

def last_float(pattern):
    found = re.findall(pattern, text)
    return float(found[-1]) if found else None

metrics = re.findall(r"mse:([0-9eE+.-]+),\s*mae:([0-9eE+.-]+)", text)
val = [float(x) for x in re.findall(r"Vali Loss:\s*([0-9eE+.-]+)", text)]
diag_matches = re.findall(r"^\[ASYSPECX_FINAL_DIAGNOSTICS\]\s+(\{.*\})$", text, re.M)
diagnostics = json.loads(diag_matches[-1]) if diag_matches else {}
test_deferred = bool(int(os.environ["DEFER_TEST"]))
payload = {
    "run_tag": os.environ["RUN_TAG"],
    "arm": os.environ["ARM"],
    "base_profile": os.environ["BASE_PROFILE"],
    "dataset": os.environ["DATASET"],
    "seq_len": int(os.environ["SEQ_LEN"]),
    "pred_len": int(os.environ["PRED_LEN"]),
    "seed": int(os.environ["SEED"]),
    "cut_freq": int(os.environ["CUT_FREQ"]),
    "periods": os.environ.get("periods", ""),
    "forecast_rank": int(os.environ["forecast_rank"]),
    "forecast_init": os.environ["forecast_init"],
    "forecast_channel_scale": int(os.environ["forecast_channel_scale"]),
    "forecast_sm_components": int(os.environ["forecast_sm_components"]),
    "forecast_sm_sharing": os.environ["forecast_sm_sharing"],
    "forecast_sm_base_trainable": int(os.environ["forecast_sm_base_trainable"]),
    "forecast_extension_shrink": os.environ["forecast_extension_shrink"],
    "forecast_phase_basis_dim": int(os.environ["forecast_phase_basis_dim"]),
    "forecast_phase_max": float(os.environ["forecast_phase_max"]),
    "test_deferred": test_deferred,
    "status": "ok" if status == 0 and val else "failed",
    "exit_code": status,
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "n_param": int(last_float(r"n_param:\s*([0-9]+)") or 0),
    "val_mse": min(val) if val else None,
    "mse": float(metrics[-1][0]) if metrics else None,
    "mae": float(metrics[-1][1]) if metrics else None,
    "t_train": last_float(r"t_train:\s*([0-9eE+.-]+)"),
    "t_inf": last_float(r"t_inf:\s*([0-9eE+.-]+)"),
    "peak_cuda_mb": last_float(r"peak_cuda_mb:\s*([0-9eE+.-]+)"),
    "init_path": os.environ.get("init_path", ""),
    "init_build_seconds": int(os.environ.get("init_build_seconds", "0")),
    "diagnostics": diagnostics,
    "log_file": log_path,
}
for key in (
    "forecast_kernel_gate_mean", "forecast_kernel_gate_min",
    "forecast_kernel_gate_max", "forecast_kernel_effective_rank",
    "forecast_kernel_raw_rms", "forecast_kernel_delta_rms",
    "forecast_kernel_sm_enabled", "forecast_kernel_sm_components",
    "forecast_kernel_sm_gate_mean", "forecast_kernel_sm_gate_abs_max",
    "forecast_kernel_sm_effective_gate_abs_max",
    "forecast_kernel_sm_factor_min", "forecast_kernel_sm_factor_max",
    "forecast_kernel_extension_scale", "forecast_kernel_extension_identity",
    "forecast_kernel_phase_enabled", "forecast_kernel_phase_basis_dim",
    "forecast_kernel_phase_abs_mean", "forecast_kernel_phase_abs_max",
    "forecast_kernel_phase_rms",
):
    payload[key] = diagnostics.get(key)
for line in reversed(text.splitlines()):
    if line.startswith("[VAL_SEG]"):
        for key, value in re.findall(r"(val_(?:mse|mae)(?:_seg\d+)?)=([0-9eE+.-]+|nan)", line):
            payload[key] = None if value == "nan" else float(value)
        break
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "[phase11] end=$(date --iso-8601=seconds) status=$status summary=$summary_file"
exit "$status"
