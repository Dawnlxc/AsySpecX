#!/bin/bash
# AsySpecX benchmark — PatchTST on ETTh2.
# Hyperparams aligned with yuqinie98/PatchTST/scripts/PatchTST/etth2.sh:
#   ETTh1/h2: d_model=16, n_heads=4, dropout=0.3
#   others:   d_model=128, d_ff=256, n_heads=16, dropout=0.2
#   lr=0.0001 universally; bs per-dataset; lradj=TST; patch_len=16, stride=8
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh
load_dataset ETTh2

SEED=2026
MODEL=PatchTST
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
    apply_patchtst_overrides $data_key
    log_file=logs/$MODEL/${MODEL}_${data_key}_sl${sl}_pl${pl}_sd${SEED}.log
    echo "[$(date '+%F %T')] $MODEL $data_key sl=$sl pl=$pl seed=$SEED lr=$lr bs=$bs d_model=$d_model -> $log_file"
    python -u run.py \
      --is_training 1 --random_seed $SEED \
      --root_path ./dataset/$subdir/ --data_path $data_path \
      --model_id ${data_key}_${sl}_${pl} \
      --model $MODEL --data $data_name --features M \
      --seq_len $sl --pred_len $pl --enc_in $enc_in \
      --train_epochs 30 --patience 10 \
      --batch_size $bs --learning_rate $lr --num_workers 4 \
      --itr 1 \
      --patch_len 16 --stride 8 --revin 1 \
      --d_model $d_model --d_ff $d_ff --n_heads $n_heads --e_layers 3 \
      --dropout $dropout --fc_dropout $fc_dropout --head_dropout $head_dropout \
      --lradj TST \
      > "$log_file" 2>&1 || echo "[fail] see $log_file"
  done
done
