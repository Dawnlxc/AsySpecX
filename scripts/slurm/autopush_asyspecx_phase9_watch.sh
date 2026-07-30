#!/bin/bash
# Conditional Phase 9 pipeline. Exactly one DONE mail; no progress mail.
set -Eeuo pipefail
cd "$(cd -- "$(dirname -- "$0")/../.." && pwd)"

OUTROOT="${OUTROOT:-phase9_results/main}"
MAIL_TO="${MAIL_TO:-yind7@outlook.com}"
MAIL_SUBJECT="${MAIL_SUBJECT:-asy2-0711v1}"
NO_PUSH="${NO_PUSH:-0}"
NO_MAIL="${NO_MAIL:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
QUEUE_SETTLE_SECONDS="${QUEUE_SETTLE_SECONDS:-10}"
AUTO_QUICK="${AUTO_QUICK:-1}"
AUTO_OOF="${AUTO_OOF:-1}"
CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
LOG_FILE="logs/autopush_asyspecx_phase9.log"
FINAL_ROOT="$OUTROOT/summary"
pipeline_status=failed
pipeline_note="watcher aborted"
push_status=skipped
total_jobs=0
mkdir -p logs "$FINAL_ROOT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

activate_env() {
  # Virga stores named environments under CONDA_ROOT, but CONDA_ROOT itself is
  # not a Conda installation.  Prefer the requested environment directly so a
  # detached watcher does not depend on login-shell Conda functions/modules.
  if [ -x "$CONDA_ROOT/$CONDA_ENV/bin/python" ]; then
    export PATH="$CONDA_ROOT/$CONDA_ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_ROOT/$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
  elif [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "Phase9 watcher: unable to activate Conda environment $CONDA_ENV" >&2
    return 1
  fi
}
count_jobs() { squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -c "^$1" || true; }
wait_stage() {
  local prefix="$1" n
  sleep "$QUEUE_SETTLE_SECONDS"
  while true; do n="$(count_jobs "$prefix")"; log "remaining ${prefix} jobs: $n"; [ "$n" = 0 ] && break; sleep "$POLL_SECONDS"; done
}
stage_manifest() {
  local stage="$1"
  if [ "$stage" = headroom ] && [ -n "${HEADROOM_MANIFEST:-}" ]; then
    printf '%s\n' "$HEADROOM_MANIFEST"
  else
    ls -t "logs/phase9/slurm_phase9_${stage}_"*.csv 2>/dev/null | head -1 || true
  fi
}
submitted_stage_count() {
  local stage="$1" file
  file="$(stage_manifest "$stage")"
  [ -n "$file" ] && echo $(( $(wc -l < "$file") - 1 )) || echo 0
}
validate_stage() {
  local stage="$1" manifest
  manifest="$(stage_manifest "$stage")"
  [ -n "$manifest" ] || { echo "missing submission manifest for $stage"; return 1; }
  python - "$stage" "$manifest" "$OUTROOT/job_status/$stage" <<'PY'
import csv, json, sys
from pathlib import Path

stage, manifest_path, status_root = sys.argv[1:]
rows = list(csv.DictReader(open(manifest_path, newline="", encoding="utf-8")))
if not rows:
    raise SystemExit(f"{stage}: empty submission manifest")
problems = []
ok = 0
for row in rows:
    cell = f"{row['dataset']}_sl{row['seq_len']}_pl{row['pred_len']}"
    path = Path(status_root) / f"{cell}.json"
    if not path.is_file():
        problems.append(f"{cell}:missing_status")
        continue
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"{cell}:invalid_status:{exc}")
        continue
    if status.get("stage") != stage or status.get("cell") != cell:
        problems.append(f"{cell}:stale_or_mismatched_status")
    elif status.get("status") != "ok":
        problems.append(f"{cell}:exit={status.get('exit_code', 'unknown')}")
    else:
        ok += 1
print(f"{stage}: submitted={len(rows)} ok={ok} problems={len(problems)}")
if problems:
    print("; ".join(problems[:30]))
    raise SystemExit(1)
PY
}
send_done() {
  local rc="$1"
  [ -f "$OUTROOT/.done_mail_sent" ] && return 0
  local duration=unknown start_epoch=""
  [ -f "$OUTROOT/.run_meta" ] && . "$OUTROOT/.run_meta" 2>/dev/null || true
  if [ -n "${start_epoch:-}" ]; then secs=$(( $(date +%s) - start_epoch )); duration="$((secs/3600))h$(((secs%3600)/60))m"; fi
  if [ "$NO_MAIL" != 1 ] && command -v mail >/dev/null 2>&1; then
    {
      echo "AsySpecX Phase 9 SafeRoute FINISHED on $(hostname) at $(date)."
      echo "status: $pipeline_status (exit=$rc)"
      echo "note: $pipeline_note"
      echo "duration: $duration"
      echo "total GPU jobs submitted: $total_jobs"
      echo "glab_push: $push_status"
      echo "summary: $PWD/$FINAL_ROOT/summary_phase9_saferoute.md"
      echo
      sed -n '1,180p' "$FINAL_ROOT/summary_phase9_saferoute.md" 2>/dev/null || true
    } | mail -s "[$MAIL_SUBJECT] Phase9 SafeRoute DONE ($pipeline_status)" "$MAIL_TO" && touch "$OUTROOT/.done_mail_sent"
  fi
}
on_exit() { rc=$?; send_done "$rc" || true; }
trap on_exit EXIT

log "Phase9 watcher start OUTROOT=$OUTROOT"
activate_env
log "Phase9 watcher environment python=$(command -v python)"
wait_stage asx9h_
headroom_jobs="$(submitted_stage_count headroom)"; total_jobs=$((total_jobs + headroom_jobs))
if ! stage_check="$(validate_stage headroom 2>&1)"; then pipeline_note="$stage_check"; exit 1; fi
log "$stage_check"

meta_paths="$(find "$OUTROOT/headroom_cells" -type d -name meta_test | sort | paste -sd, -)"
[ -n "$meta_paths" ] || { pipeline_note="no headroom metadata"; exit 1; }
python scripts/audit_router_headroom.py --meta "$meta_paths" --current_validation_selected 0.333805 --output_dir "$OUTROOT/headroom" >> "$LOG_FILE" 2>&1
python scripts/summarize_phase9.py --headroom_dir "$OUTROOT/headroom" --quick_glob "$OUTROOT/quick/*/routed_results.csv" --oof_glob "$OUTROOT/oof/*/routed_results.csv" --output_dir "$FINAL_ROOT" >> "$LOG_FILE" 2>&1
headroom_gain="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sample_block_gain_vs_best_fixed"])' "$OUTROOT/headroom/headroom_summary.json")"
headroom_go="$(python -c 'import sys; print(int(float(sys.argv[1]) >= 0.004))' "$headroom_gain")"
log "headroom gain=$headroom_gain go=$headroom_go"

if [ "$headroom_go" != 1 ] || [ "$AUTO_QUICK" != 1 ]; then
  pipeline_status=complete
  pipeline_note="stopped at headroom gate (gain=$headroom_gain)"
else
  log "submitting quick router stage"
  STAGE=quick NO_MAIL=1 NO_WATCH=1 DRY_RUN=0 bash scripts/slurm/submit_asyspecx_phase9.sh >> "$LOG_FILE" 2>&1
  wait_stage asx9q_
  quick_jobs="$(submitted_stage_count quick)"; total_jobs=$((total_jobs + quick_jobs))
  if ! stage_check="$(validate_stage quick 2>&1)"; then pipeline_note="$stage_check"; exit 1; fi
  log "$stage_check"
  python scripts/summarize_phase9.py --headroom_dir "$OUTROOT/headroom" --quick_glob "$OUTROOT/quick/*/routed_results.csv" --oof_glob "$OUTROOT/oof/*/routed_results.csv" --output_dir "$FINAL_ROOT" >> "$LOG_FILE" 2>&1
  quick_gain="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["quick_router"]["gain_vs_anchor"])' "$FINAL_ROOT/phase9_summary.json")"
  severe="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("quick_any_dataset_regression_gt_0_003",False)))' "$FINAL_ROOT/phase9_summary.json")"
  quick_go="$(python -c 'import sys; print(int(float(sys.argv[1]) >= 0.002))' "$quick_gain")"
  log "quick gain=$quick_gain go=$quick_go severe_dataset_regression=$severe"
  if [ "$quick_go" != 1 ]; then
    pipeline_status=complete; pipeline_note="stopped at quick gate (gain=$quick_gain)"
  elif [ "$severe" = 1 ]; then
    pipeline_status=complete; pipeline_note="quick improved globally but a dataset regressed >0.003; conservative recalibration required"
  elif [ "$AUTO_OOF" != 1 ]; then
    pipeline_status=complete; pipeline_note="quick gate passed; AUTO_OOF=0"
  else
    log "submitting rolling OOF stage"
    GLOBAL_QUICK_GATE_PASSED=1 STAGE=oof NO_MAIL=1 NO_WATCH=1 DRY_RUN=0 bash scripts/slurm/submit_asyspecx_phase9.sh >> "$LOG_FILE" 2>&1
    wait_stage asx9o_
    oof_jobs="$(submitted_stage_count oof)"; total_jobs=$((total_jobs + oof_jobs))
    if ! stage_check="$(validate_stage oof 2>&1)"; then pipeline_note="$stage_check"; exit 1; fi
    log "$stage_check"
    python scripts/summarize_phase9.py --headroom_dir "$OUTROOT/headroom" --quick_glob "$OUTROOT/quick/*/routed_results.csv" --oof_glob "$OUTROOT/oof/*/routed_results.csv" --output_dir "$FINAL_ROOT" >> "$LOG_FILE" 2>&1
    pipeline_status=complete; pipeline_note="headroom, quick, and rolling OOF completed"
  fi
fi

if [ "$NO_PUSH" = 1 ]; then
  push_status=skipped
else
  log "glab push"
  push_status=failed
  for attempt in 1 2 3; do
    if glab push @"$PWD" --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
      --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
      --exclude='*.npz' --exclude='*.npy' --exclude='phase9_results/*/*/meta_*'; then
      push_status=ok
      break
    fi
    log "glab push attempt $attempt failed"
    [ "$attempt" = 3 ] || sleep 30
  done
  if [ "$push_status" != ok ]; then
    pipeline_status=failed
    pipeline_note="$pipeline_note; glab push failed after 3 attempts"
  fi
fi
log "watcher done status=$pipeline_status note=$pipeline_note push=$push_status total_jobs=$total_jobs"
