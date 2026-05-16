#!/bin/bash
# Submit AsySpecX / JointMLP jobs to slurm.
# Usage:
#   bash scripts/slurm/submit_all.sh --model AsySpecX --smoke
#   bash scripts/slurm/submit_all.sh --model JointMLP --full
#   bash scripts/slurm/submit_all.sh --model AsySpecX --dataset ETTh1
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic PEMS03 PEMS04 PEMS07 PEMS08)
SMOKE_DATASETS=(ETTh1 electricity)

mode=""
single_dataset=""
model=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model)   model="$2"; shift 2 ;;
        --smoke)   mode=smoke; shift ;;
        --full)    mode=full; shift ;;
        --dataset) single_dataset="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

: "${model:?--model required (AsySpecX | JointMLP)}"
if [ ! -d "scripts/${model}" ]; then
    echo "No scripts/${model}/ directory; valid models: AsySpecX, JointMLP" >&2; exit 2
fi

if [ -n "$single_dataset" ]; then
    DATASETS=("$single_dataset")
    smoke_env=""
elif [ "$mode" = "smoke" ]; then
    DATASETS=("${SMOKE_DATASETS[@]}")
    smoke_env="SMOKE=1"
elif [ "$mode" = "full" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
    smoke_env=""
else
    echo "Specify one of: --smoke, --full, or --dataset Y" >&2
    exit 2
fi

mkdir -p logs/slurm
echo "Model: $model  Mode: ${mode:-single}  Datasets: ${#DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
    script="scripts/${model}/${dataset}.sh"
    if [ ! -f "$script" ]; then
        echo "[skip] missing $script" >&2; continue
    fi
    jobname="${model}_${dataset}"
    echo "sbatch -J $jobname (${smoke_env:-no-smoke})  scripts/slurm/baseline.sbatch $model $dataset"
    if [ -n "$smoke_env" ]; then
        sbatch -J "$jobname" --export=ALL,SMOKE=1 scripts/slurm/baseline.sbatch "$model" "$dataset"
    else
        sbatch -J "$jobname" scripts/slurm/baseline.sbatch "$model" "$dataset"
    fi
done

echo "Done. Inspect with: squeue -u \$USER  /  tail -f logs/slurm/slurm-*.out"
