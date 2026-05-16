#!/bin/bash
# AsySpecX benchmark — AsySpecX self.
# Probe-driven: gate_init=0, gate_max=1.0; rank=2 for high-C, rank=8 for small-C.
# Hyperparams: lr=5e-4, bs=64, patience=10, epochs=30.
# Sweep: seeds {2026,2027} × sl {96,720} × pred_lens.
# This template is sourced by sibling per-dataset scripts via DATASET_KEY.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh
load_dataset "${DATASET_KEY:?DATASET_KEY must be set}"

MODEL=AsySpecX
SEEDS=(2026 2027)
mkdir -p logs/$MODEL checkpoints results

# PEMS uses fixed sl=96; others sweep sl ∈ {96, 720} (extremes only).
case "$data_key" in
  PEMS*)  DEFAULT_SEQ_LENS=(96) ;;
  *)      DEFAULT_SEQ_LENS=(96 720) ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  SEQ_LENS=(96)
  PRED_LENS_OVERRIDE="96"
else
  SEQ_LENS=("${DEFAULT_SEQ_LENS[@]}")
  PRED_LENS_OVERRIDE=""
fi

for SEED in "${SEEDS[@]}"; do
  for sl in "${SEQ_LENS[@]}"; do
    PL_LIST=$pred_lens
    if [ -n "$PRED_LENS_OVERRIDE" ]; then PL_LIST=$PRED_LENS_OVERRIDE; fi
    for pl in $PL_LIST; do
      apply_asyspecx_overrides $data_key $sl
      log_file=logs/$MODEL/${MODEL}_${data_key}_sl${sl}_pl${pl}_sd${SEED}.log
      if [ -f "$log_file" ] && grep -q "mse:" "$log_file" 2>/dev/null; then
        echo "[skip] $log_file already complete"
        continue
      fi
      echo "[$(date '+%F %T')] $MODEL $data_key sl=$sl pl=$pl sd=$SEED rank=$rank cut_freq=$cut_freq gate_init=$gate_init gate_max=$gate_max"
      python -u run.py \
        --is_training 1 --random_seed $SEED \
        --root_path ./dataset/$subdir/ --data_path $data_path \
        --model_id ${data_key}_${sl}_${pl} \
        --model $MODEL --data $data_name --features M \
        --seq_len $sl --pred_len $pl --enc_in $enc_in \
        --train_epochs $epochs --patience $patience \
        --batch_size $bs --learning_rate $lr --num_workers 4 \
        --itr 1 --cut_freq $cut_freq --individual 0 \
        --rank $rank --num_bands $num_bands \
        --gate_init $gate_init --gate_max $gate_max \
        > "$log_file" 2>&1 || echo "[fail] see $log_file"
    done
  done
done
