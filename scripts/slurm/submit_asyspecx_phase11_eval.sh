#!/bin/bash
set -Eeuo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
QOS="${QOS:-normal}"
SCREEN_RUN_TAG="${SCREEN_RUN_TAG:-phase11_screen_0715v1}"
SCREEN_OUTROOT="${SCREEN_OUTROOT:-phase11_results/screen}"
EVAL_OUTROOT="${EVAL_OUTROOT:-phase11_results/eval_selected}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
SBATCH_SCRIPT="scripts/slurm/asyspecx_phase11_eval.sbatch"

test -s "$MANIFEST"
mkdir -p "$EVAL_OUTROOT/manifests" logs/slurm
submitted="$EVAL_OUTROOT/manifests/submitted_eval_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\tarm\tdataset\tseq_len\tpred_len\tseed\tcut_freq\n' >"$submitted"
while IFS=$'\t' read -r arm dataset seq_len pred_len seed cut_freq; do
    [ -z "$arm" ] && continue
    [[ "$arm" == \#* ]] && continue
    exports="ALL,SCREEN_RUN_TAG=$SCREEN_RUN_TAG,SCREEN_OUTROOT=$SCREEN_OUTROOT,EVAL_OUTROOT=$EVAL_OUTROOT,ARM=$arm,DATASET=$dataset,SEQ_LEN=$seq_len,PRED_LEN=$pred_len,SEED=$seed,CUT_FREQ=$cut_freq"
    job_id="$(sbatch --parsable --job-name="a11e_${arm}_${dataset}_p${pred_len}" --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" --time=00:30:00 --export="$exports" "$SBATCH_SCRIPT")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$arm" "$dataset" "$seq_len" "$pred_len" "$seed" "$cut_freq" >>"$submitted"
done <"$MANIFEST"
cat "$submitted"
