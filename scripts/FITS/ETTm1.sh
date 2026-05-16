#!/bin/bash
# AsySpecX benchmark — FITS on ETTm1.
# Hyperparams from VEWOXIC/FITS official scripts. cut_freq = sl/base_T*6 + 1.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh
load_dataset ETTm1

SEED=2026
MODEL=FITS
mkdir -p logs/$MODEL checkpoints results

if [ "${SMOKE:-0}" = "1" ]; then
  SEQ_LENS=(96 720)
  PRED_LENS_OVERRIDE="96 720"
else
  SEQ_LENS=(96 336 512 720)
  PRED_LENS_OVERRIDE=""
fi

for sl in "${SEQ_LENS[@]}"; do
  PL_LIST=$pred_lens
  if [ -n "$PRED_LENS_OVERRIDE" ]; then PL_LIST=$PRED_LENS_OVERRIDE; fi
  for pl in $PL_LIST; do
    apply_fits_overrides $data_key $sl
    log_file=logs/$MODEL/${MODEL}_${data_key}_sl${sl}_pl${pl}_sd${SEED}.log
    echo "[$(date '+%F %T')] $MODEL $data_key sl=$sl pl=$pl cut_freq=$cut_freq lr=$lr"
    python -u run.py \
      --is_training 1 --random_seed $SEED \
      --root_path ./dataset/$subdir/ --data_path $data_path \
      --model_id ${data_key}_${sl}_${pl} \
      --model $MODEL --data $data_name --features M \
      --seq_len $sl --pred_len $pl --enc_in $enc_in \
      --train_epochs $epochs --patience $patience \
      --batch_size $bs --learning_rate $lr --num_workers 4 \
      --itr 1 --cut_freq $cut_freq --individual 0 \
      > "$log_file" 2>&1 || echo "[fail] see $log_file"
  done
done
