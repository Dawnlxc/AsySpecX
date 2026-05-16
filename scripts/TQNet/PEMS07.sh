#!/bin/bash
# AsySpecX benchmark — TQNet on PEMS07.
# Sweeps seq_len ∈ {96,336,512,720} (or {96,720} under SMOKE=1) × paper pred_lens.
# Logs per run: logs/TQNet/TQNet_PEMS07_sl<sl>_pl<pl>_sd2026.log
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh
load_dataset PEMS07

SEED=2026
MODEL=TQNet
mkdir -p logs/$MODEL checkpoints results

if [ "${SMOKE:-0}" = "1" ]; then
  SEQ_LENS=(96)
  PRED_LENS_OVERRIDE="96 720"
else
  SEQ_LENS=(96)  # PEMS uses fixed sl=96 (paper convention)
  PRED_LENS_OVERRIDE=""
fi

for sl in "${SEQ_LENS[@]}"; do
  PL_LIST=$pred_lens
  if [ -n "$PRED_LENS_OVERRIDE" ]; then PL_LIST=$PRED_LENS_OVERRIDE; fi
  for pl in $PL_LIST; do
    log_file=logs/$MODEL/${MODEL}_${data_key}_sl${sl}_pl${pl}_sd${SEED}.log
    echo "[$(date '+%F %T')] $MODEL $data_key sl=$sl pl=$pl seed=$SEED -> $log_file"
    python -u run.py \
      --is_training 1 --random_seed $SEED \
      --root_path ./dataset/$subdir/ --data_path $data_path \
      --model_id ${data_key}_${sl}_${pl} \
      --model $MODEL --data $data_name --features M \
      --seq_len $sl --pred_len $pl --enc_in $enc_in \
      --train_epochs 30 --patience 5 \
      --batch_size $bs --learning_rate $lr --num_workers 4 \
      --itr 1 \
      --cycle $cycle --use_revin $revin --dropout 0.5 \
      > "$log_file" 2>&1 || echo "[fail] see $log_file"
  done
done
