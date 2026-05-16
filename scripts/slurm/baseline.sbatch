#!/bin/bash -l
#SBATCH --job-name=AsySpecX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --account=OD-241336
#SBATCH --output=logs/slurm/slurm-%x-%j.out

# Usage: sbatch -J AsySpecX_<DATASET> scripts/slurm/baseline.sbatch <DATASET>
# Optional env: SMOKE=1 to restrict the sweep inside the dataset script.

set -euo pipefail
DATASET="${1:?dataset key required}"

source ~/.bashrc
conda activate tsfm
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# resolve repo root from this script's location so the job is portable
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
mkdir -p logs/AsySpecX logs/slurm checkpoints results

echo "==== AsySpecX on ${DATASET} (account=OD-241336, env=tsfm, seed=2026) ===="
echo "SMOKE=${SMOKE:-0}  Start: $(date)"

bash scripts/AsySpecX/${DATASET}.sh

echo "End: $(date)"
