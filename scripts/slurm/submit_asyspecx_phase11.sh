#!/bin/bash
set -Eeuo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

ACCOUNT="${ACCOUNT:-od-241336}"
PARTITION="${PARTITION:-h24gpu}"
QOS="${QOS:-normal}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
RUN_TAG="${RUN_TAG:-phase11_forecastability_0715v1}"
OUTROOT="${OUTROOT:-phase11_results/screen}"
MANIFEST="${MANIFEST:-configs/phase11_screen.tsv}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
DEFER_TEST="${DEFER_TEST:-1}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_SCRIPT="scripts/slurm/asyspecx_phase11_run.sbatch"

test -s "$MANIFEST"
test -f "$SBATCH_SCRIPT"
mkdir -p "$OUTROOT/manifests" logs/slurm

stamp="$(date +%Y%m%d_%H%M%S)"
submitted_manifest="$OUTROOT/manifests/submitted_${RUN_TAG}_${stamp}.tsv"
printf 'job_id\tarm\tdataset\tseq_len\tpred_len\tseed\tcut_freq\n' >"$submitted_manifest"

submitted=0
while IFS=$'\t' read -r arm dataset seq_len pred_len seed cut_freq; do
    [ -z "$arm" ] && continue
    [[ "$arm" == \#* ]] && continue
    job_name="a11_${arm}_${dataset}_s${seq_len}_p${pred_len}_d${seed}"
    job_name="${job_name:0:120}"
    exports="ALL,RUN_TAG=$RUN_TAG,OUTROOT=$OUTROOT,ARM=$arm,DATASET=$dataset,SEQ_LEN=$seq_len,PRED_LEN=$pred_len,SEED=$seed,CUT_FREQ=$cut_freq,EPOCHS=$EPOCHS,PATIENCE=$PATIENCE,DEFER_TEST=$DEFER_TEST"
    cmd=(sbatch --parsable --job-name="$job_name" --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" --time="$TIME_LIMIT" --export="$exports" "$SBATCH_SCRIPT")
    printf '[submit]'; printf ' %q' "${cmd[@]}"; printf '\n'
    if [ "$DRY_RUN" = "1" ]; then
        job_id="DRY"
    else
        job_id="$("${cmd[@]}")"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$arm" "$dataset" "$seq_len" "$pred_len" "$seed" "$cut_freq" >>"$submitted_manifest"
    submitted=$((submitted + 1))
done <"$MANIFEST"

echo "[phase11-submit] submitted=$submitted dry_run=$DRY_RUN manifest=$submitted_manifest"
