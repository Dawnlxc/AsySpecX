#!/bin/bash -l
#SBATCH --job-name=JointMLP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --account=OD-241336
#SBATCH --output=logs/slurm/slurm-%x-%j.out

# Usage: sbatch -J JointMLP_<DATASET> scripts/slurm/baseline.sbatch <DATASET>
# Optional env: SMOKE=1 to restrict the sweep inside the dataset script.

set -euo pipefail
DATASET="${1:?dataset key required}"

source ~/.bashrc
conda activate tsfm
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# resolve repo root from this script's location so the job is portable
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
mkdir -p logs/JointMLP logs/slurm checkpoints results

echo "==== JointMLP (JA v4) on ${DATASET} (account=OD-241336, env=tsfm) ===="
echo "SMOKE=${SMOKE:-0}  Start: $(date)"

bash scripts/JointMLP/${DATASET}.sh

echo "End: $(date)"
