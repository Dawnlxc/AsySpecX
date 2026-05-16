#!/bin/bash
# JointMLP (JA v4) — TQNet MLP backbone + JointAxisTWCMv4 cross-channel.
# JA v4 = per-bin per-frame gain g_{k, t'}; see models/JointAxisTWCMv4.py.
# This template is sourced by sibling per-dataset scripts via DATASET_KEY.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh
load_dataset "${DATASET_KEY:?DATASET_KEY must be set}"
apply_jmlp_overrides "$data_key"   # sets jmlp_rank per (high-C / low-C) class

MODEL=JointMLP
SEEDS=(2026 2027)
mkdir -p logs/$MODEL checkpoints results
TAG=${TAG:-jav4}

# Defaults match jmlp_v4.sbatch (production launcher for JA v4).
JMLP_WINDOW=${JMLP_WINDOW:--1}          # -1 → _auto_window(seq_len)
JMLP_STRIDE=${JMLP_STRIDE:--1}          # -1 → W//2
JMLP_RANK=${JMLP_RANK:-$jmlp_rank}      # high-C → 2, low-C → 8 (apply_jmlp_overrides)
JMLP_DELTA_HIDDEN=${JMLP_DELTA_HIDDEN:-64}
JMLP_GATE_INIT=${JMLP_GATE_INIT:--1.0}
JMLP_USE_ENTROPY_GATE=${JMLP_USE_ENTROPY_GATE:-1}
JMLP_INNOVATION_ONLY=${JMLP_INNOVATION_ONLY:-0}
EFFECTIVE_REVIN=${REVIN_OVERRIDE:-$revin}

case "$data_key" in
  PEMS*)        DEFAULT_SEQ_LENS=(96) ;;
  illness)      DEFAULT_SEQ_LENS=(60) ;;
  *)            DEFAULT_SEQ_LENS=(96 336 720) ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  SEQ_LENS=(96); PRED_LENS_OVERRIDE="96"
else
  SEQ_LENS=("${DEFAULT_SEQ_LENS[@]}"); PRED_LENS_OVERRIDE=""
fi

for SEED in "${SEEDS[@]}"; do
  for sl in "${SEQ_LENS[@]}"; do
    PL_LIST=$pred_lens
    if [ -n "$PRED_LENS_OVERRIDE" ]; then PL_LIST=$PRED_LENS_OVERRIDE; fi
    for pl in $PL_LIST; do
      log_file=logs/$MODEL/${MODEL}_${TAG}_${data_key}_sl${sl}_pl${pl}_sd${SEED}.log
      if [ -f "$log_file" ] && grep -q "mse:" "$log_file" 2>/dev/null; then
        echo "[skip] $log_file already complete"; continue
      fi
      echo "[$(date '+%F %T')] $MODEL[$TAG] $data_key sl=$sl pl=$pl sd=$SEED"
      python -u run.py \
        --is_training 1 --random_seed $SEED \
        --root_path ./dataset/$subdir/ --data_path $data_path \
        --model_id ${data_key}_${sl}_${pl}_${TAG} \
        --model $MODEL --data $data_name --features M \
        --seq_len $sl --pred_len $pl --enc_in $enc_in \
        --train_epochs 30 --patience 5 \
        --batch_size $bs --learning_rate $lr --num_workers 4 \
        --itr 1 \
        --cycle $cycle --use_revin $EFFECTIVE_REVIN --dropout 0.5 \
        --jmlp_window $JMLP_WINDOW --jmlp_stride $JMLP_STRIDE \
        --jmlp_rank $JMLP_RANK --jmlp_delta_hidden $JMLP_DELTA_HIDDEN \
        --jmlp_gate_init $JMLP_GATE_INIT \
        --jmlp_use_entropy_gate $JMLP_USE_ENTROPY_GATE \
        --jmlp_innovation_only $JMLP_INNOVATION_ONLY \
        > "$log_file" 2>&1 || echo "[fail] see $log_file"
    done
  done
done
