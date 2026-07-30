#!/bin/bash
# Wait for AsySpecX Phase 8 Slurm jobs, aggregate, merge Phase6/7/8, select,
# summarize, (optional) ensemble, glab push, then send ONE done email with
# runtime duration, GPU-job count, and a result summary. No intermediate mails.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

OUTROOT="${1:-phase8_results/hydra}"
MAIL_TO="${2:-yind7@outlook.com}"
NO_PUSH="${3:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_FILE="logs/autopush_asyspecx_phase8.log"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
BASELINE_CSV="${BASELINE_CSV:-}"
PHASE6_CSV="${PHASE6_CSV:-phase6_results/fullfield/results.csv}"
PHASE7_CSV="${PHASE7_CSV:-phase7_results/merged/results.csv}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0709v1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
NO_MAIL="${NO_MAIL:-0}"
MERGED_ROOT="${MERGED_ROOT:-phase8_results/merged}"

mkdir -p logs "$OUTROOT" "$MERGED_ROOT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

activate_env() {
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
    elif command -v conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh"
        if [ -d "$CONDA_ROOT/$CONDA_ENV" ]; then conda activate "$CONDA_ROOT/$CONDA_ENV"; else conda activate "$CONDA_ENV"; fi
    elif [ -x "$CONDA_ROOT/$CONDA_ENV/bin/python" ]; then export PATH="$CONDA_ROOT/$CONDA_ENV/bin:$PATH"; fi
}
count_jobs() { command -v squeue >/dev/null 2>&1 || { echo 0; return; }; squeue -h -u "$USER" -o "%j" | grep -c '^asx8_' || true; }

# read start marker
start_epoch=""; jobs="?"; partition="?"
[ -f "$OUTROOT/.run_meta" ] && . "$OUTROOT/.run_meta" 2>/dev/null && { start_epoch="${start_epoch:-}"; jobs="${jobs:-?}"; partition="${partition:-?}"; }

log "watcher start OUTROOT=$OUTROOT subject=$MAIL_SUBJECT jobs=$jobs"
while true; do n="$(count_jobs)"; log "remaining asx8_ jobs: $n"; [ "$n" = "0" ] && break; sleep "$POLL_SECONDS"; done
activate_env

log "aggregating phase8"
python scripts/slurm/aggregate_asyspecx_phase1.py --root "$OUTROOT" | tee -a "$LOG_FILE"

merge_inputs="$OUTROOT/results.csv"
[ -f "$PHASE7_CSV" ] && merge_inputs="$PHASE7_CSV,$merge_inputs"
[ -f "$PHASE6_CSV" ] && merge_inputs="$PHASE6_CSV,$merge_inputs"
log "merging: $merge_inputs"
python scripts/merge_results.py --csvs "$merge_inputs" --output "$MERGED_ROOT/results.csv" | tee -a "$LOG_FILE"

log "phase8 selection"
ROOT="$MERGED_ROOT" CSV="$MERGED_ROOT/results.csv" PYTHON=python bash scripts/run_phase8_selection.sh >> "$LOG_FILE" 2>&1 || log "some selection failed"

log "selector audit"
python scripts/audit_phase5_selectors.py --csv "$MERGED_ROOT/results.csv" \
    --selected_files "selected_unrestricted_mean.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv" \
    --output_dir "$MERGED_ROOT" >> "$LOG_FILE" 2>&1 || log "audit failed"

ens_arg=()
if [ "$SAVE_PREDICTIONS" = "1" ] && [ -d "$OUTROOT/predictions" ]; then
    log "offline ensemble"
    python scripts/ensemble_predictions.py --pred_dir "$OUTROOT/predictions" --mode simplex_val \
        --output_csv "$MERGED_ROOT/ensemble_results.csv" --summary "$MERGED_ROOT/ensemble_summary.md" >> "$LOG_FILE" 2>&1 || log "ensemble failed"
    [ -f "$MERGED_ROOT/ensemble_results.csv" ] && ens_arg=(--ensemble_csv "$MERGED_ROOT/ensemble_results.csv")
fi

baseline_arg=(); [ -n "$BASELINE_CSV" ] && [ -f "$BASELINE_CSV" ] && baseline_arg=(--baseline_csv "$BASELINE_CSV")
python scripts/summarize_phase8.py --csv "$MERGED_ROOT/results.csv" \
    --selected_csv "$MERGED_ROOT/selected_unrestricted_mean.csv" \
    --anchor_arm phase7_period_multi_auto_acf_patchlinear "${baseline_arg[@]}" "${ens_arg[@]}" \
    --output_dir "$MERGED_ROOT" >> "$LOG_FILE" 2>&1 || log "phase8 summary failed"

push_status="skipped"
if [ "$NO_PUSH" = "1" ]; then log "NO_PUSH=1"; else
    log "glab push"
    if glab push @"$PWD" --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
        --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz'; then
        push_status="ok"; else push_status="failed"; fi
    log "glab push $push_status"
fi

# duration + card usage
end_epoch=$(date +%s)
dur="unknown"
if [ -n "$start_epoch" ]; then
    secs=$((end_epoch - start_epoch)); dur="$((secs/3600))h$(((secs%3600)/60))m"
fi

mail_status="skipped"
if [ "$NO_MAIL" != "1" ] && command -v mail >/dev/null 2>&1; then
    {
        echo "AsySpecX Phase 8-Hydra FINISHED on $(hostname) at $(date)."
        echo "Wall-clock duration: $dur"
        echo "GPU-jobs: $jobs (1 GPU each), partition=$partition"
        echo "glab_push: $push_status"
        echo "Merged CSV: $PWD/$MERGED_ROOT/results.csv"
        echo "Summary: $PWD/$MERGED_ROOT/summary_phase8.md"
        echo "Selector audit: $PWD/$MERGED_ROOT/selector_audit.md"
        echo "Selected(unrestricted_mean): $PWD/$MERGED_ROOT/selected_unrestricted_mean.csv"
        echo
        echo "===== summary_phase8.md (head) ====="
        sed -n '1,120p' "$MERGED_ROOT/summary_phase8.md" 2>/dev/null || true
    } | mail -s "[$MAIL_SUBJECT] AsySpecX Phase8 DONE ($dur, $jobs jobs)" "$MAIL_TO" && mail_status="ok" || mail_status="failed"
    log "mail $mail_status"
fi
log "watcher done push=$push_status mail=$mail_status dur=$dur"
