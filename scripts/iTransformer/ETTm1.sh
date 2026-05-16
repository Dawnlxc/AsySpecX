#!/bin/bash
# AsySpecX benchmark — iTransformer on ETTm1.
# Hyperparams aligned with thuml/iTransformer/scripts/multivariate_forecasting/<group>/iTransformer*.sh:
#   ETTh1: d_model=256/512 (depends on pl)
#   ETTh2/ETTm1/ETTm2: d_model=128, e_layers=2
#   weather: d_model=512, e_layers=3
#   electricity/PEMS: lr=0.0005, bs=16, d_model=512, e_layers=3
#   traffic: lr=0.001, bs=16, d_model=512, e_layers=4
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh
load_dataset ETTm1

SEED=2026
MODEL=iTransformer
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
    apply_itransformer_overrides $data_key $pl
    log_file=logs/$MODEL/${MODEL}_${data_key}_sl${sl}_pl${pl}_sd${SEED}.log
    echo "[$(date '+%F %T')] $MODEL $data_key sl=$sl pl=$pl seed=$SEED lr=$lr bs=$bs d_model=$d_model e_layers=$e_layers -> $log_file"
    python -u run.py \
      --is_training 1 --random_seed $SEED \
      --root_path ./dataset/$subdir/ --data_path $data_path \
      --model_id ${data_key}_${sl}_${pl} \
      --model $MODEL --data $data_name --features M \
      --seq_len $sl --pred_len $pl --enc_in $enc_in \
      --train_epochs 30 --patience 5 \
      --batch_size $bs --learning_rate $lr --num_workers 4 \
      --itr 1 \
      --d_model $d_model --d_ff $d_ff --e_layers $e_layers --n_heads 8 --dropout 0.1 \
      > "$log_file" 2>&1 || echo "[fail] see $log_file"
  done
done
