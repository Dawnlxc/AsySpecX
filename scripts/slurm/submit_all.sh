#!/bin/bash
# Submit AsySpecX / JointMLP / baseline jobs to slurm.
#
# Single model:
#   bash scripts/slurm/submit_all.sh --model AsySpecX --smoke
#   bash scripts/slurm/submit_all.sh --model JointMLP --full
#   bash scripts/slurm/submit_all.sh --model TQNet    --dataset ETTh1
#
# All 10 baselines (no AsySpecX / JointMLP):
#   bash scripts/slurm/submit_all.sh --all-baselines --smoke    # 5 reps × 2 datasets = 10 jobs
#   bash scripts/slurm/submit_all.sh --all-baselines --full     # 10 × 11 = 110 jobs
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic PEMS03 PEMS04 PEMS07 PEMS08)
SMOKE_DATASETS=(ETTh1 electricity)
ALL_BASELINES=(TQNet CycleNet DLinear iTransformer PatchTST FITS FreTS FilterNet SparseTSF MixLinear)
SMOKE_BASELINES=(TQNet CycleNet PatchTST iTransformer FITS)

mode=""
single_dataset=""
model=""
all_baselines=0
while [ $# -gt 0 ]; do
    case "$1" in
        --model)          model="$2"; shift 2 ;;
        --all-baselines)  all_baselines=1; shift ;;
        --smoke)          mode=smoke; shift ;;
        --full)           mode=full; shift ;;
        --dataset)        single_dataset="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# resolve model list
if [ "$all_baselines" = "1" ]; then
    if [ "$mode" = "smoke" ]; then
        MODELS=("${SMOKE_BASELINES[@]}")
    else
        MODELS=("${ALL_BASELINES[@]}")
    fi
else
    : "${model:?--model required (AsySpecX | JointMLP | <baseline>) — or use --all-baselines}"
    if [ ! -d "scripts/${model}" ]; then
        echo "No scripts/${model}/ directory" >&2; exit 2
    fi
    MODELS=("$model")
fi

# resolve dataset list
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
total=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
echo "Models: ${#MODELS[@]}  Datasets: ${#DATASETS[@]}  Total: $total"

for m in "${MODELS[@]}"; do
    if [ ! -d "scripts/${m}" ]; then
        echo "[skip-model] no scripts/${m}/ directory" >&2; continue
    fi
    for dataset in "${DATASETS[@]}"; do
        script="scripts/${m}/${dataset}.sh"
        if [ ! -f "$script" ]; then
            echo "[skip] missing $script" >&2; continue
        fi
        jobname="${m}_${dataset}"
        echo "sbatch -J $jobname (${smoke_env:-no-smoke})  scripts/slurm/baseline.sbatch $m $dataset"
        if [ -n "$smoke_env" ]; then
            sbatch -J "$jobname" --export=ALL,SMOKE=1 scripts/slurm/baseline.sbatch "$m" "$dataset"
        else
            sbatch -J "$jobname" scripts/slurm/baseline.sbatch "$m" "$dataset"
        fi
    done
done

echo "Done. Inspect with: squeue -u \$USER  /  tail -f logs/slurm/slurm-*.out"
