#!/bin/bash
#SBATCH --job-name=asx10_cell
#SBATCH --account=od-241336
#SBATCH --partition=h24gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO="${REPO:-/scratch3/lin250/bldgFM/DUBABA/AsySpecX}"
cd "$REPO"
source scripts/_common.sh

PYTHON="${PYTHON:-/scratch3/lin250/conda_envs/tsfm/bin/python}"
RUN_TAG="${RUN_TAG:-phase10_0712v1}"
OUTROOT="${OUTROOT:-phase10_results/screen}"
ARM="${ARM:?ARM is required}"
DATASET="${DATASET:?DATASET is required}"
SEQ_LEN="${SEQ_LEN:?SEQ_LEN is required}"
PRED_LEN="${PRED_LEN:?PRED_LEN is required}"
SEED="${SEED:?SEED is required}"
CUT_FREQ="${CUT_FREQ:?CUT_FREQ is required}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"

test -x "$PYTHON"
load_dataset "$DATASET"
apply_asyspecx_overrides "$data_key" "$SEQ_LEN"
[ -n "${BATCH_SIZE:-}" ] && bs="$BATCH_SIZE"
[ -n "${LR:-}" ] && lr="$LR"
[ -n "${RANK:-}" ] && rank="$RANK"
[ -n "${NUM_BANDS:-}" ] && num_bands="$NUM_BANDS"

periods="${PERIODS:-$(phase5_periods_for "$DATASET")}"
periods="${periods//,/+}"

cross_flags=(
    --spectral_lift fits_linear --lift_sharing shared
    --cross_mode asym_lowrank --residual_part split
    --gate_type hier_channel_band --gate_init_logit -6.0
    --gate_max 1.0 --gate_lr_mult 5.0 --residual_clip_eta -1
    --backcast_loss_weight 0.0 --norm_mode rin_noaffine
    --temporal_adapter none
)
individual_flags=(
    --spectral_lift fits_linear --lift_sharing individual
    --cross_mode none --norm_mode revin_affine
    --temporal_adapter none --backcast_loss_weight 0.0
)
factor_r2_flags=(
    --spectral_lift fits_linear --lift_sharing lowrank_channel --lift_rank 2
    --cross_mode none --norm_mode revin_affine
    --temporal_adapter none --backcast_loss_weight 0.0
    --cycle_residual 1 --cycle_residual_rank 0
)
factor_r4_flags=(
    --spectral_lift fits_linear --lift_sharing lowrank_channel --lift_rank 4
    --cross_mode none --norm_mode revin_affine
    --temporal_adapter none --backcast_loss_weight 0.0
    --cycle_residual 1 --cycle_residual_rank 0
)
period_flags=(
    "${cross_flags[@]}"
    --temporal_adapter sparse_period --periods "$periods"
    --periodic_init seasonal_naive --period_fusion sum_gated
    --period_gate_type period --period_gate_init_logit 0.0
    --temporal_fusion convex --temporal_gate_type horizon
    --temporal_gate_init_logit -4.0
)
compact_period_flags=(
    "${cross_flags[@]}"
    --temporal_adapter compact_period --periods "$periods"
    --periodic_init seasonal_naive --period_fusion sum_gated
    --period_gate_type period --period_gate_init_logit 0.0
    --temporal_fusion convex --temporal_gate_type horizon
    --temporal_gate_init_logit -4.0
)
direct_residual_flags=(
    --linear_adapter direct_linear --linear_sharing shared
    --linear_init zeros --linear_fusion additive
    --linear_gate_type horizon --linear_gate_init_logit 4.0
)
dlinear_residual_flags=(
    --linear_adapter dlinear_decomp --linear_sharing shared
    --linear_init zeros --linear_fusion additive
    --linear_gate_type horizon --linear_gate_init_logit 4.0
)

case "$ARM" in
    ind) flags=("${individual_flags[@]}") ;;
    ind_cycle_full) flags=("${individual_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0) ;;
    ind_cycle_r8) flags=("${individual_flags[@]}" --cycle_residual 1 --cycle_residual_rank 8) ;;
    ind_direct_res) flags=("${individual_flags[@]}" "${direct_residual_flags[@]}") ;;
    ind_dlinear_res) flags=("${individual_flags[@]}" "${dlinear_residual_flags[@]}") ;;
    ind_cycle_full_direct_res)
        flags=("${individual_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0 "${direct_residual_flags[@]}") ;;
    factor_r2_cycle) flags=("${factor_r2_flags[@]}") ;;
    factor_r4_cycle) flags=("${factor_r4_flags[@]}") ;;
    factor_r2_cycle_direct_res)
        flags=("${factor_r2_flags[@]}" "${direct_residual_flags[@]}") ;;
    factor_r4_cycle_direct_res)
        flags=("${factor_r4_flags[@]}" "${direct_residual_flags[@]}") ;;
    cross) flags=("${cross_flags[@]}") ;;
    cross_cycle_full) flags=("${cross_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0) ;;
    cross_cycle_r8) flags=("${cross_flags[@]}" --cycle_residual 1 --cycle_residual_rank 8) ;;
    cross_cycle_full_direct_res)
        flags=("${cross_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0 "${direct_residual_flags[@]}") ;;
    cross_cycle_full_dlinear_res)
        flags=("${cross_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0 "${dlinear_residual_flags[@]}") ;;
    period) flags=("${period_flags[@]}") ;;
    period_cycle_full) flags=("${period_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0) ;;
    period_cycle_r8) flags=("${period_flags[@]}" --cycle_residual 1 --cycle_residual_rank 8) ;;
    compact_period) flags=("${compact_period_flags[@]}") ;;
    compact_period_cycle_full)
        flags=("${compact_period_flags[@]}" --cycle_residual 1 --cycle_residual_rank 0) ;;
    *) echo "[error] unknown ARM=$ARM" >&2; exit 2 ;;
esac

run_id="${RUN_TAG}_${ARM}_${data_key}_sl${SEQ_LEN}_pl${PRED_LEN}_cf${CUT_FREQ}_sd${SEED}"
run_dir="$OUTROOT/$ARM/$data_key/sl${SEQ_LEN}_pl${PRED_LEN}_cf${CUT_FREQ}_sd${SEED}"
log_file="logs/AsySpecX_phase10/${run_id}.log"
summary_file="$run_dir/run_summary.json"
mkdir -p logs/slurm logs/AsySpecX_phase10 checkpoints results "$run_dir"

echo "[phase10] host=$(hostname) job=${SLURM_JOB_ID:-local} start=$(date --iso-8601=seconds)"
echo "[phase10] run_id=$run_id arm=$ARM dataset=$data_key sl=$SEQ_LEN pl=$PRED_LEN seed=$SEED cf=$CUT_FREQ cycle=$cycle"
echo "[phase10] epochs=$EPOCHS patience=$PATIENCE bs=$bs lr=$lr rank=$rank bands=$num_bands periods=$periods"

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
    "${flags[@]}" >"$log_file" 2>&1
status=$?
set -e

export ARM DATASET SEQ_LEN PRED_LEN SEED CUT_FREQ RUN_TAG OUTROOT periods
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
payload = {
    "run_tag": os.environ["RUN_TAG"],
    "arm": os.environ["ARM"],
    "dataset": os.environ["DATASET"],
    "seq_len": int(os.environ["SEQ_LEN"]),
    "pred_len": int(os.environ["PRED_LEN"]),
    "seed": int(os.environ["SEED"]),
    "cut_freq": int(os.environ["CUT_FREQ"]),
    "cycle": None,
    "periods": os.environ.get("periods", ""),
    "status": "ok" if status == 0 and metrics else "failed",
    "exit_code": status,
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "n_param": int(last_float(r"n_param:\s*([0-9]+)") or 0),
    "val_mse": min(val) if val else None,
    "mse": float(metrics[-1][0]) if metrics else None,
    "mae": float(metrics[-1][1]) if metrics else None,
    "t_train": last_float(r"t_train:\s*([0-9eE+.-]+)"),
    "t_inf": last_float(r"t_inf:\s*([0-9eE+.-]+)"),
    "peak_cuda_mb": last_float(r"peak_cuda_mb:\s*([0-9eE+.-]+)"),
    "log_file": log_path,
}
args = re.search(r"Namespace\((.*?)\)\n", text, re.S)
if args:
    m = re.search(r"cycle=([0-9]+)", args.group(1))
    if m:
        payload["cycle"] = int(m.group(1))
for line in reversed(text.splitlines()):
    if not line.startswith("[VAL_SEG]"):
        continue
    for key, value in re.findall(r"(val_(?:mse|mae)_seg\d+)=([0-9eE+.-]+|nan)", line):
        payload[key] = None if value == "nan" else float(value)
    break
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "[phase10] end=$(date --iso-8601=seconds) status=$status summary=$summary_file"
exit "$status"
