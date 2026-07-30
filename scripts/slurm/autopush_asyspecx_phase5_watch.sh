#!/bin/bash
# Wait for AsySpecX Phase 5-Lockdown Slurm jobs, aggregate, run the three
# selection variants, summarize, glab push, email.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

OUTROOT="${1:-phase5_results/main}"
MAIL_TO="${2:-yind7@outlook.com}"
NO_PUSH="${3:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_FILE="logs/autopush_asyspecx_phase5.log"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
BASELINE_CSV="${BASELINE_CSV:-}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0704v1}"

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
    squeue -h -u "$USER" -o "%j" | grep -c '^asx5_' || true
}

log "watcher start OUTROOT=$OUTROOT MAIL_TO=$MAIL_TO NO_PUSH=$NO_PUSH subject=$MAIL_SUBJECT"
while true; do
    n="$(count_jobs)"
    log "remaining asx5_ jobs: $n"
    [ "$n" = "0" ] && break
    sleep "$POLL_SECONDS"
done

activate_env

log "aggregating"
python scripts/slurm/aggregate_asyspecx_phase1.py --root "$OUTROOT" | tee -a "$LOG_FILE"

log "running three selection variants"
ROOT="$OUTROOT" CSV="$OUTROOT/results.csv" PYTHON=python \
    bash scripts/run_phase5_selection.sh >> "$LOG_FILE" 2>&1 || log "some selection variant failed"

baseline_arg=()
[ -n "$BASELINE_CSV" ] && [ -f "$BASELINE_CSV" ] && baseline_arg=(--baseline_csv "$BASELINE_CSV")
python scripts/summarize_phase5.py \
    --csv "$OUTROOT/results.csv" \
    --selected_csv "$OUTROOT/selected_unrestricted_mean.csv" \
    --anchor_arm phase5_asx_cross \
    "${baseline_arg[@]}" \
    --output_dir "$OUTROOT" >> "$LOG_FILE" 2>&1 || log "phase5 summary failed"

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
        echo "AsySpecX Phase 5-Lockdown finished on $(hostname) at $(date)."
        echo "Repo: $PWD"
        echo "CSV: $PWD/$OUTROOT/results.csv"
        echo "Summary: $PWD/$OUTROOT/summary_phase5.md"
        echo "Selected (unrestricted_mean): $PWD/$OUTROOT/selected_unrestricted_mean.csv"
        echo "Selected (last_segment): $PWD/$OUTROOT/selected_unrestricted_last_segment.csv"
        echo "Selected (policy_family): $PWD/$OUTROOT/selected_policy_family.csv"
        echo "glab_push: $push_status"
        echo
        sed -n '1,200p' "$OUTROOT/summary_phase5.md" 2>/dev/null || true
    } | mail -s "[$MAIL_SUBJECT] AsySpecX Phase5 Lockdown done" "$MAIL_TO" && mail_status="ok" || mail_status="failed"
    log "mail status: $mail_status"
else
    log "mail command missing"
fi

log "watcher done push=$push_status mail=$mail_status summary=$OUTROOT/summary_phase5.md csv=$OUTROOT/results.csv"
