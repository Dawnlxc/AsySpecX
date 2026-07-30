#!/bin/bash
#SBATCH --job-name=asx11_eval
#SBATCH --account=od-241336
#SBATCH --partition=h24gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO="${REPO:-/scratch3/lin250/bldgFM/DUBABA/AsySpecX}"
cd "$REPO"
source scripts/_common.sh

PYTHON="${PYTHON:-/scratch3/lin250/conda_envs/tsfm/bin/python}"
SCREEN_RUN_TAG="${SCREEN_RUN_TAG:-phase11_screen_0715v1}"
SCREEN_OUTROOT="${SCREEN_OUTROOT:-phase11_results/screen}"
EVAL_OUTROOT="${EVAL_OUTROOT:-phase11_results/eval_selected}"
ARM="${ARM:?ARM is required}"
DATASET="${DATASET:?DATASET is required}"
SEQ_LEN="${SEQ_LEN:?SEQ_LEN is required}"
PRED_LEN="${PRED_LEN:?PRED_LEN is required}"
SEED="${SEED:?SEED is required}"
CUT_FREQ="${CUT_FREQ:?CUT_FREQ is required}"
NUM_WORKERS="${NUM_WORKERS:-4}"
INIT_RIDGE="${INIT_RIDGE:-0.01}"
INIT_ARTIFACT_RANK="${INIT_ARTIFACT_RANK:-16}"

load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"

base_flags=(
    --spectral_lift fits_linear --lift_sharing individual
    --cross_mode none --norm_mode revin_affine
    --temporal_adapter none --backcast_loss_weight 0.0
    --cycle_residual 1 --cycle_residual_rank 0
)
dense_flags=(
    --linear_adapter direct_linear --linear_sharing shared
    --linear_init zeros --linear_fusion additive
    --linear_gate_type horizon --linear_gate_init_logit 4.0
)
forecast_common=(
    --forecast_kernel lowrank_time --forecast_kernel_fusion convex
    --forecast_kernel_gate_type horizon --forecast_kernel_gate_init_logit -4.0
    --gate_lr_mult 5.0
)
case "$ARM" in
    anchor) flags=("${base_flags[@]}") ;;
    dense_direct) flags=("${base_flags[@]}" "${dense_flags[@]}") ;;
    fk_r4) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 4 --forecast_kernel_init small_random) ;;
    fk_r8) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random) ;;
    fk_r8_cs) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1) ;;
    fk_sm2_shared) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 2 --forecast_kernel_sm_sharing shared) ;;
    fk_sm2_mode) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 2 --forecast_kernel_sm_sharing mode) ;;
    fk_sm2_tail2) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 2 --forecast_kernel_sm_sharing mode --forecast_kernel_extension_shrink tail2_linear) ;;
    fk_sm4_mode) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode) ;;
    fk_sm4_ph2_q) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_phase_basis_dim 2 --forecast_kernel_phase_max 0.7853981633974483) ;;
    fk_sm4_ph4_q) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_phase_basis_dim 4 --forecast_kernel_phase_max 0.7853981633974483) ;;
    fk_sm4_ph4_h) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_phase_basis_dim 4 --forecast_kernel_phase_max 1.5707963267948966) ;;
    fk_sm4_frozen) flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank 8 --forecast_kernel_init small_random --forecast_kernel_channel_scale 1 --forecast_kernel_spectral_mixtures 4 --forecast_kernel_sm_sharing mode --forecast_kernel_sm_base_trainable 0) ;;
    fk_r8_svd|fk_r16_svd)
        if [ "$ARM" = "fk_r8_svd" ]; then kernel_rank=8; else kernel_rank=16; fi
        init_path="$SCREEN_OUTROOT/artifacts/${data_key}_sl${SEQ_LEN}_pl${PRED_LEN}_r${INIT_ARTIFACT_RANK}_ridge${INIT_RIDGE}.pt"
        test -s "$init_path"
        flags=("${base_flags[@]}" "${forecast_common[@]}" --forecast_kernel_rank "$kernel_rank" --forecast_kernel_init ridge_svd --forecast_kernel_channel_scale 1 --forecast_kernel_init_path "$init_path")
        ;;
    *) echo "[phase11-eval] unknown ARM=$ARM" >&2; exit 2 ;;
esac

run_id="${SCREEN_RUN_TAG}_${ARM}_${data_key}_sl${SEQ_LEN}_pl${PRED_LEN}_cf${CUT_FREQ}_sd${SEED}"
setting="${run_id}_AsySpecX_${data_name}_ftM_sl${SEQ_LEN}_pl${PRED_LEN}_cycle${cycle}_seed${SEED}"
test -s "checkpoints/$setting/checkpoint.pth"
source_summary="$SCREEN_OUTROOT/$ARM/$data_key/sl${SEQ_LEN}_pl${PRED_LEN}_cf${CUT_FREQ}_sd${SEED}/run_summary.json"
test -s "$source_summary"
eval_dir="$EVAL_OUTROOT/$ARM/$data_key/sl${SEQ_LEN}_pl${PRED_LEN}_cf${CUT_FREQ}_sd${SEED}"
log_file="logs/AsySpecX_phase11_eval/${run_id}.log"
summary_file="$eval_dir/test_summary.json"
mkdir -p logs/slurm logs/AsySpecX_phase11_eval "$eval_dir"

set +e
"$PYTHON" -u run.py \
    --is_training 0 --random_seed "$SEED" \
    --root_path "./dataset/$subdir/" --data_path "$data_path" \
    --model_id "$run_id" --model AsySpecX --data "$data_name" --features M \
    --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" --enc_in "$enc_in" --cycle "$cycle" \
    --batch_size "$bs" --learning_rate "$lr" --num_workers "$NUM_WORKERS" \
    --itr 1 --cut_freq "$CUT_FREQ" --individual 0 \
    --rank "$rank" --num_bands "$num_bands" --val_num_segments 4 \
    "${flags[@]}" >"$log_file" 2>&1
status=$?
set -e

"$PYTHON" - "$summary_file" "$source_summary" "$log_file" "$status" <<'PY'
import json, os, re, sys
out, source_path, log_path, status_text = sys.argv[1:5]
source = json.load(open(source_path, encoding="utf-8"))
text = open(log_path, encoding="utf-8", errors="replace").read()
metrics = re.findall(r"mse:([0-9eE+.-]+),\s*mae:([0-9eE+.-]+)", text)
t_inf = re.findall(r"t_inf:\s*([0-9eE+.-]+)", text)
status = int(status_text)
payload = {
    "arm": source["arm"], "dataset": source["dataset"],
    "seq_len": source["seq_len"], "pred_len": source["pred_len"],
    "seed": source["seed"], "cut_freq": source["cut_freq"],
    "selection_val_mse": source["val_mse"],
    "n_param": source["n_param"], "t_train": source["t_train"],
    "peak_cuda_mb": source["peak_cuda_mb"],
    "forecast_kernel_gate_mean": source.get("forecast_kernel_gate_mean"),
    "forecast_kernel_sm_gate_mean": source.get("forecast_kernel_sm_gate_mean"),
    "forecast_kernel_sm_gate_abs_max": source.get("forecast_kernel_sm_gate_abs_max"),
    "forecast_kernel_sm_effective_gate_abs_max": source.get("forecast_kernel_sm_effective_gate_abs_max"),
    "forecast_kernel_sm_factor_min": source.get("forecast_kernel_sm_factor_min"),
    "forecast_kernel_sm_factor_max": source.get("forecast_kernel_sm_factor_max"),
    "forecast_kernel_extension_scale": source.get("forecast_kernel_extension_scale"),
    "forecast_kernel_extension_identity": source.get("forecast_kernel_extension_identity"),
    "forecast_kernel_phase_abs_mean": source.get("forecast_kernel_phase_abs_mean"),
    "forecast_kernel_phase_abs_max": source.get("forecast_kernel_phase_abs_max"),
    "forecast_kernel_phase_rms": source.get("forecast_kernel_phase_rms"),
    "mse": float(metrics[-1][0]) if metrics else None,
    "mae": float(metrics[-1][1]) if metrics else None,
    "t_inf": float(t_inf[-1]) if t_inf else None,
    "test_opened_after_validation_selection": True,
    "status": "ok" if status == 0 and metrics else "failed",
    "exit_code": status, "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "source_summary": source_path, "log_file": log_path,
}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

exit "$status"
