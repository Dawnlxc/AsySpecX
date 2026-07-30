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
mkdir -p logs/slurm phase11_results/screen/resource_audit
"$PYTHON" -u scripts/benchmark_phase11_fixed_work.py \
    --arms dense_direct,fk_r8_cs,fk_r8,fk_r4 \
    --checkpoint_root checkpoints --run_tag phase11_screen_0715v1 \
    --seq_len 96 --pred_len 720 --channels 21 --cycle 144 --cut_freq 13 \
    --seed 2026 --batch_size 64 --warmup 5 \
    --inference_repeats 50 --train_repeats 20 \
    --output phase11_results/screen/resource_audit/fixed_work.json
