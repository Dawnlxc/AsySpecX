#!/bin/bash
#SBATCH --job-name=asx11_bench
#SBATCH --account=od-241336
#SBATCH --partition=h24gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail
cd "${REPO:-/scratch3/lin250/bldgFM/DUBABA/AsySpecX}"
PYTHON="${PYTHON:-/scratch3/lin250/conda_envs/tsfm/bin/python}"
OUTPUT="${OUTPUT:-phase11_results/screen/resource_audit/fixed_work_optimized.json}"
ARMS="${ARMS:-dense_direct,fk_r8_cs,fk_r8,fk_r4}"
# Slurm --export uses commas as separators, so callers may provide a
# colon-delimited arm list without ambiguous escaping.
ARMS="${ARMS//:/,}"
RUN_TAG="${RUN_TAG:-phase11_screen_0715v1}"
BASE_PROFILE="${BASE_PROFILE:-ind_cycle_full}"
DATASET="${DATASET:-weather}"
DATA_NAME="${DATA_NAME:-custom}"
DATA_PATH="${DATA_PATH:-weather.csv}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LEN="${PRED_LEN:-720}"
CHANNELS="${CHANNELS:-21}"
CYCLE="${CYCLE:-144}"
CUT_FREQ="${CUT_FREQ:-13}"
SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CROSS_RANK="${CROSS_RANK:-8}"
NUM_BANDS="${NUM_BANDS:-8}"
PERIODS="${PERIODS:-144}"
mkdir -p logs/slurm "$(dirname "$OUTPUT")"
"$PYTHON" -u scripts/benchmark_phase11_fixed_work.py \
    --arms "$ARMS" \
    --checkpoint_root checkpoints --run_tag "$RUN_TAG" \
    --base_profile "$BASE_PROFILE" --dataset "$DATASET" \
    --data_name "$DATA_NAME" --data_path "$DATA_PATH" \
    --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" --channels "$CHANNELS" \
    --cycle "$CYCLE" --cut_freq "$CUT_FREQ" --cross_rank "$CROSS_RANK" \
    --num_bands "$NUM_BANDS" --periods "$PERIODS" \
    --seed "$SEED" --batch_size "$BATCH_SIZE" --warmup 5 \
    --inference_repeats 50 --train_repeats 20 \
    --output "$OUTPUT"
