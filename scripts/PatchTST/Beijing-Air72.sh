#!/bin/bash
# AsySpecX benchmark — PatchTST on Beijing-Air72.
# sl=336 only, pl in {96,192,336,720}; hourly Beijing air-quality data.
set -euo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"
source scripts/_common.sh
load_dataset Beijing-Air72

SEED=2026
MODEL=PatchTST
mkdir -p logs/$MODEL checkpoints results

# sl is fixed at 336 for these datasets per paper convention
SEQ_LENS=(336)

for sl in "${SEQ_LENS[@]}"; do
  for pl in $pred_lens; do
    epochs=30; patience=10
    apply_patchtst_overrides $data_key
    
    log_file=logs/$MODEL/${MODEL}_${data_key}_sl${sl}_pl${pl}_sd${SEED}.log
    if [ -f "$log_file" ] && grep -q "mse:" "$log_file"; then
      echo "[skip] already done: $log_file"
      continue
    fi
    echo "[$(date '+%F %T')] $MODEL $data_key sl=$sl pl=$pl seed=$SEED -> $log_file"
    python -u run.py \
      --is_training 1 --random_seed $SEED \
      --root_path ./dataset/$subdir/ --data_path $data_path \
      --model_id ${data_key}_${sl}_${pl} \
      --model $MODEL --data $data_name --features M \
      --seq_len $sl --pred_len $pl --enc_in $enc_in \
      --train_epochs $epochs --patience $patience \
      --batch_size $bs --learning_rate $lr --num_workers 4 \
      --itr 1 \
      --patch_len 16 --stride 8 --revin 1 --d_model $d_model --d_ff $d_ff --n_heads $n_heads --e_layers 3 --dropout $dropout --fc_dropout $fc_dropout --head_dropout 0 --lradj TST \
      > "$log_file" 2>&1 || echo "[fail] see $log_file"
  done
done
