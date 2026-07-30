#!/bin/bash
# Wait for AsySpecX Phase 6 Full-Field Slurm jobs, aggregate, run selection +
# audit, summarize, glab push, email.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

OUTROOT="${1:-phase6_results/fullfield}"
MAIL_TO="${2:-yind7@outlook.com}"
NO_PUSH="${3:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_FILE="logs/autopush_asyspecx_phase6.log"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
BASELINE_CSV="${BASELINE_CSV:-}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0707v1}"

mkdir -p logs "$OUTROOT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

activate_env() {
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if [ -d "$CONDA_ROOT/$CONDA_ENV" ]; then conda activate "$CONDA_ROOT/$CONDA_ENV"; else conda activate "$CONDA_ENV"; fi
    elif [ -x "$CONDA_ROOT/$CONDA_ENV/bin/python" ]; then
        export PATH="$CONDA_ROOT/$CONDA_ENV/bin:$PATH"
    fi
}

count_jobs() {
    if ! command -v squeue >/dev/null 2>&1; then echo 0; return; fi
    squeue -h -u "$USER" -o "%j" | grep -c '^asx6_' || true
}

log "watcher start OUTROOT=$OUTROOT MAIL_TO=$MAIL_TO NO_PUSH=$NO_PUSH subject=$MAIL_SUBJECT"
while true; do
    n="$(count_jobs)"; log "remaining asx6_ jobs: $n"
    [ "$n" = "0" ] && break
    sleep "$POLL_SECONDS"
done

activate_env

log "aggregating"
python scripts/slurm/aggregate_asyspecx_phase1.py --root "$OUTROOT" | tee -a "$LOG_FILE"

log "running full-field selection variants"
ROOT="$OUTROOT" CSV="$OUTROOT/results.csv" PYTHON=python \
    bash scripts/run_phase6_fullfield_selection.sh >> "$LOG_FILE" 2>&1 || log "some selection variant failed"

log "selector audit"
python scripts/audit_phase5_selectors.py --csv "$OUTROOT/results.csv" \
    --selected_files "selected_unrestricted_mean.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv" \
    --output_dir "$OUTROOT" >> "$LOG_FILE" 2>&1 || log "audit failed"

baseline_arg=()
[ -n "$BASELINE_CSV" ] && [ -f "$BASELINE_CSV" ] && baseline_arg=(--baseline_csv "$BASELINE_CSV")
python scripts/summarize_phase6_fullfield.py \
    --csv "$OUTROOT/results.csv" \
    --selected_csv "$OUTROOT/selected_unrestricted_mean.csv" \
    --anchor_arm phase6_asx_cross \
    --selected_csvs "$OUTROOT/selected_unrestricted_mean.csv,$OUTROOT/selected_unrestricted_segment_robust.csv,$OUTROOT/selected_unrestricted_margin_prefer_simple.csv,$OUTROOT/selected_policy_family.csv" \
    --selected_names "unrestricted_mean,segment_robust,margin_prefer_simple,policy_family" \
    "${baseline_arg[@]}" \
    --output_dir "$OUTROOT" >> "$LOG_FILE" 2>&1 || log "phase6 summary failed"

python scripts/summarize_cut_freq.py --csv "$OUTROOT/results.csv" \
    --output "$OUTROOT/summary_cut_freq.md" >> "$LOG_FILE" 2>&1 || log "cut_freq summary skipped"

push_status="skipped"
if [ "$NO_PUSH" = "1" ]; then
    log "NO_PUSH=1, skip glab push"
else
    log "glab push start"
    if glab push @"$PWD" \
        --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
        --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'; then
        push_status="ok"; log "glab push ok"
    else
        push_status="failed"; log "glab push failed"
    fi
fi

mail_status="skipped"
if command -v mail >/dev/null 2>&1; then
    {
        echo "AsySpecX Phase 6 Full-Field finished on $(hostname) at $(date)."
        echo "Repo: $PWD"
        echo "CSV: $PWD/$OUTROOT/results.csv"
        echo "Summary: $PWD/$OUTROOT/summary_phase6_fullfield.md"
        echo "Selector audit: $PWD/$OUTROOT/selector_audit.md"
        echo "Selected(unrestricted_mean): $PWD/$OUTROOT/selected_unrestricted_mean.csv"
        echo "glab_push: $push_status"
        echo
        sed -n '1,200p' "$OUTROOT/summary_phase6_fullfield.md" 2>/dev/null || true
    } | mail -s "[$MAIL_SUBJECT] AsySpecX Phase6 FullField done" "$MAIL_TO" && mail_status="ok" || mail_status="failed"
    log "mail status: $mail_status"
else
    log "mail command missing"
fi

log "watcher done push=$push_status mail=$mail_status summary=$OUTROOT/summary_phase6_fullfield.md"
