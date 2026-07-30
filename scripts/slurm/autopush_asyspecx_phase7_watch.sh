#!/bin/bash
# Wait for AsySpecX Phase 7 Slurm jobs, aggregate, merge with Phase 6, select,
# summarize, (optional) ensemble, glab push, email.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

OUTROOT="${1:-phase7_results/breakthrough}"
MAIL_TO="${2:-yind7@outlook.com}"
NO_PUSH="${3:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_FILE="logs/autopush_asyspecx_phase7.log"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
BASELINE_CSV="${BASELINE_CSV:-}"
PHASE6_CSV="${PHASE6_CSV:-phase6_results/fullfield/results.csv}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0708v1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
MERGED_ROOT="${MERGED_ROOT:-phase7_results/merged}"

mkdir -p logs "$OUTROOT" "$MERGED_ROOT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

activate_env() {
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
    elif command -v conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh"
        if [ -d "$CONDA_ROOT/$CONDA_ENV" ]; then conda activate "$CONDA_ROOT/$CONDA_ENV"; else conda activate "$CONDA_ENV"; fi
    elif [ -x "$CONDA_ROOT/$CONDA_ENV/bin/python" ]; then export PATH="$CONDA_ROOT/$CONDA_ENV/bin:$PATH"; fi
}
count_jobs() { command -v squeue >/dev/null 2>&1 || { echo 0; return; }; squeue -h -u "$USER" -o "%j" | grep -c '^asx7_' || true; }

log "watcher start OUTROOT=$OUTROOT subject=$MAIL_SUBJECT"
while true; do n="$(count_jobs)"; log "remaining asx7_ jobs: $n"; [ "$n" = "0" ] && break; sleep "$POLL_SECONDS"; done
activate_env

log "aggregating phase7"
python scripts/slurm/aggregate_asyspecx_phase1.py --root "$OUTROOT" | tee -a "$LOG_FILE"

# merge phase6 + phase7 (phase6 optional)
if [ -f "$PHASE6_CSV" ]; then
    log "merging phase6 + phase7"
    python scripts/merge_results.py --csvs "$PHASE6_CSV,$OUTROOT/results.csv" --output "$MERGED_ROOT/results.csv" | tee -a "$LOG_FILE"
else
    log "phase6 csv absent; using phase7 only as merged"
    cp "$OUTROOT/results.csv" "$MERGED_ROOT/results.csv"
fi

log "phase7 selection"
ROOT="$MERGED_ROOT" CSV="$MERGED_ROOT/results.csv" PYTHON=python bash scripts/run_phase7_selection.sh >> "$LOG_FILE" 2>&1 || log "some selection failed"

log "selector audit"
python scripts/audit_phase5_selectors.py --csv "$MERGED_ROOT/results.csv" \
    --selected_files "selected_unrestricted_mean.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv" \
    --output_dir "$MERGED_ROOT" >> "$LOG_FILE" 2>&1 || log "audit failed"

baseline_arg=(); [ -n "$BASELINE_CSV" ] && [ -f "$BASELINE_CSV" ] && baseline_arg=(--baseline_csv "$BASELINE_CSV")
python scripts/summarize_phase7.py --csv "$MERGED_ROOT/results.csv" \
    --selected_csv "$MERGED_ROOT/selected_unrestricted_mean.csv" \
    --anchor_arm phase6_asx_period_multi "${baseline_arg[@]}" \
    --output_dir "$MERGED_ROOT" >> "$LOG_FILE" 2>&1 || log "phase7 summary failed"

if [ "$SAVE_PREDICTIONS" = "1" ] && [ -d "$OUTROOT/predictions" ]; then
    log "offline ensemble (analysis)"
    python scripts/ensemble_predictions.py --pred_dir "$OUTROOT/predictions" \
        --mode simplex_val --output_csv "$MERGED_ROOT/ensemble_results.csv" \
        --summary "$MERGED_ROOT/ensemble_summary.md" >> "$LOG_FILE" 2>&1 || log "ensemble failed"
fi

push_status="skipped"
if [ "$NO_PUSH" = "1" ]; then log "NO_PUSH=1"; else
    log "glab push"
    if glab push @"$PWD" --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
        --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz'; then
        push_status="ok"; else push_status="failed"; fi
    log "glab push $push_status"
fi

mail_status="skipped"
if command -v mail >/dev/null 2>&1; then
    {
        echo "AsySpecX Phase 7-Breakthrough finished on $(hostname) at $(date)."
        echo "Merged CSV: $PWD/$MERGED_ROOT/results.csv"
        echo "Summary: $PWD/$MERGED_ROOT/summary_phase7.md"
        echo "Selector audit: $PWD/$MERGED_ROOT/selector_audit.md"
        echo "Selected(unrestricted_mean): $PWD/$MERGED_ROOT/selected_unrestricted_mean.csv"
        echo "glab_push: $push_status"
        echo
        sed -n '1,200p' "$MERGED_ROOT/summary_phase7.md" 2>/dev/null || true
    } | mail -s "[$MAIL_SUBJECT] AsySpecX Phase7 Breakthrough done" "$MAIL_TO" && mail_status="ok" || mail_status="failed"
    log "mail $mail_status"
else
    log "mail command missing"
fi
log "watcher done push=$push_status mail=$mail_status"
