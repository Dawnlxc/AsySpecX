#!/bin/bash
# Wait for AsySpecX Phase 3-GapClose Slurm jobs, aggregate, push, email.
set -euo pipefail

cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

OUTROOT="${1:-phase3_gapclose_results/main}"
MAIL_TO="${2:-yind7@outlook.com}"
NO_PUSH="${3:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_FILE="logs/autopush_asyspecx_phase3_gapclose.log"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"

mkdir -p logs "$OUTROOT"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

activate_env() {
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if [ -d "$CONDA_ROOT/$CONDA_ENV" ]; then
            conda activate "$CONDA_ROOT/$CONDA_ENV"
        else
            conda activate "$CONDA_ENV"
        fi
    elif [ -x "$CONDA_ROOT/$CONDA_ENV/bin/python" ]; then
        export PATH="$CONDA_ROOT/$CONDA_ENV/bin:$PATH"
    fi
}

count_jobs() {
    if ! command -v squeue >/dev/null 2>&1; then
        echo 0
        return
    fi
    squeue -h -u "$USER" -o "%j" | grep -c '^asx3_' || true
}

log "watcher start OUTROOT=$OUTROOT MAIL_TO=$MAIL_TO NO_PUSH=$NO_PUSH"
while true; do
    n="$(count_jobs)"
    log "remaining asx3_ jobs: $n"
    if [ "$n" = "0" ]; then
        break
    fi
    sleep "$POLL_SECONDS"
done

activate_env

log "aggregating"
python scripts/slurm/aggregate_asyspecx_phase1.py --root "$OUTROOT" | tee -a "$LOG_FILE"
python scripts/summarize_phase3_gapclose.py --root "$OUTROOT" --csv "$OUTROOT/results.csv" | tee -a "$LOG_FILE"

if python scripts/select_by_validation.py \
    --csv "$OUTROOT/results.csv" \
    --output "$OUTROOT/selected_results.csv" \
    --summary "$OUTROOT/selected_summary.md" >> "$LOG_FILE" 2>&1; then
    log "validation selection ok"
else
    log "validation selection skipped/failed (likely missing val_mse)"
fi

push_status="skipped"
if [ "$NO_PUSH" = "1" ]; then
    log "NO_PUSH=1, skip glab push"
else
    log "glab push start"
    if glab push @"$PWD" \
        --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
        --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'; then
        push_status="ok"
        log "glab push ok"
    else
        push_status="failed"
        log "glab push failed"
    fi
fi

mail_status="skipped"
if command -v mail >/dev/null 2>&1; then
    {
        echo "AsySpecX Phase 3-GapClose finished on $(hostname) at $(date)."
        echo "Repo: $PWD"
        echo "Root: $PWD/$OUTROOT"
        echo "Summary: $PWD/$OUTROOT/summary_phase3_gapclose.md"
        echo "CSV: $PWD/$OUTROOT/results.csv"
        echo "Selected: $PWD/$OUTROOT/selected_results.csv"
        echo "glab_push: $push_status"
        echo
        sed -n '1,160p' "$OUTROOT/summary_phase3_gapclose.md" 2>/dev/null || true
    } | mail -s "[AsySpecX Phase3 GapClose] done" "$MAIL_TO" && mail_status="ok" || mail_status="failed"
    log "mail status: $mail_status"
else
    log "mail command missing"
fi

log "watcher done push=$push_status mail=$mail_status summary=$OUTROOT/summary_phase3_gapclose.md csv=$OUTROOT/results.csv"
