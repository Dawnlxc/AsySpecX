# Prompt for Slurm-side Claude: AsySpecX Phase 8-Hydra

你在 Petrichor/Slurm 登录节点。目标：把 `AsySpecX` 从 glab 拉过来，检查并运行 Phase 8 Hydra arms；脚本有问题就修复并 push 回 glab；结果好了聚合、与 Phase 6/7 合并、跑 selection + audit、summarize、（可选）ensemble、push 回 glab。

**邮件规则（本轮改了）**：只发两封。提交时发一封「开始」邮件（submit 脚本自动发）；全部结束时发一封「完成」邮件（watcher 自动发，含**跑了多久、用了多少卡、结果总结**）。中间过程不发邮件。标题前缀 `asy2-0709v1`，收件人 `yind7@outlook.com`。

Phase 8 背景（Phase 7 结论）：
- best fixed single-arm = `phase7_period_multi_auto_acf_patchlinear`（MSE 0.336435）；validation-selected 0.333277；oracle 0.331332。
- Phase 8 组合 auto/union 周期 + patchlinear + clipped cross + DLinear 分支 + Hydra softmax 分支融合。全部 default-off，Phase 1-7 完全兼容。
- 决策：若某 phase8 臂把 best-fixed 提升 >0.002 或 selected >0.001，跑 full Phase 8；若无臂击败 `phase7_period_multi_auto_acf_patchlinear`，停止建模、用 Phase 7 收尾。selection 只用 validation；oracle 仅分析上界。

## 1. 拉代码

```bash
mkdir -p /scratch3/lin250/bldgFM/DUBABA
cd /scratch3/lin250/bldgFM/DUBABA
glab pull AsySpecX          # 或 glab pull @/mnt/scratch/CRUISE/Du/code/AsySpecX
cd AsySpecX
export ACCOUNT="${ACCOUNT:-od-241336}"; export PARTITION="${PARTITION:-h24gpu}"
export CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"; export CONDA_ENV="${CONDA_ENV:-tsfm}"
export PYTHON="${PYTHON:-python}"
ls -lh dataset/weather/weather.csv dataset/electricity/electricity.csv dataset/ETT-small/ETTh1.csv \
       dataset/ETT-small/ETTm1.csv dataset/traffic/traffic.csv dataset/PEMS/PEMS04.npz dataset/PEMS/PEMS08.npz
```
缺失 dataset 从 DATASETS 剔除并在完成邮件里说明。

## 2. 检查

```bash
for s in scripts/run_phase8_hydra_candidates.sh scripts/run_phase8_selection.sh \
         scripts/slurm/submit_asyspecx_phase8.sh scripts/slurm/autopush_asyspecx_phase8_watch.sh \
         scripts/slurm/asyspecx_phase8_run.sbatch scripts/_common.sh; do bash -n "$s" || echo "FAIL $s"; done

"$PYTHON" -m py_compile models/AsySpecX.py exp/exp_main.py run.py \
  scripts/select_by_validation.py scripts/discover_periods.py scripts/ensemble_predictions.py \
  scripts/merge_results.py scripts/summarize_phase8.py scripts/audit_phase5_selectors.py

"$PYTHON" -m unittest tests/test_asyspecx_phase1.py tests/test_asyspecx_phase4.py \
  tests/test_asyspecx_phase5.py tests/test_asyspecx_phase6.py tests/test_asyspecx_phase7.py tests/test_asyspecx_phase8.py
```

**PERIODS 逗号陷阱**：submit 把 `24,168`→`24+168` 再 export；`run.py --periods` 接受 `+`。dry-run 确认 compact **576** / full **1008** jobs、electricity export `PERIODS=24+168`、PEMS seq_len=720 跳过（除非 RUN_PEMS_SEQ720=1）：

```bash
NO_PUSH=1 NO_MAIL=1 DRY_RUN=1 COMPACT=1 bash scripts/slurm/submit_asyspecx_phase8.sh 2>&1 | tail -1   # 576
NO_PUSH=1 NO_MAIL=1 DRY_RUN=1 COMPACT=0 bash scripts/slurm/submit_asyspecx_phase8.sh 2>&1 | tail -1   # 1008
```

本地验证 union 周期发现（train only）：
```bash
"$PYTHON" scripts/discover_periods.py --dataset electricity --data custom --root_path ./dataset/electricity/ \
  --data_path electricity.csv --seq_len 720 --enc_in 321 --cycle 168 --method union_auto \
  --manual_periods 24,168 --max_periods 5 --period_min 4 --period_max 0 --output /tmp/u.json
# 期望 manual(24,168) 在前，去重，<=5 个
```
有问题就修 + push（注意 exclude npz）：
```bash
glab push @"$PWD" --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz'
```

## 3. Canary

```bash
NO_MAIL=1 NO_PUSH=1 COMPACT=1 \
  DATASETS="weather electricity" SEEDS=2024 SEQ_LENS=96 PRED_LENS=96 \
  EPOCHS=1 PATIENCE=1 OUTROOT=phase8_results/canary \
  bash scripts/slurm/submit_asyspecx_phase8.sh
while squeue -u "$USER" -h -o "%120j" | grep -q "^asx8_"; do sleep 60; done
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase8_results/canary
ROOT=phase8_results/canary CSV=phase8_results/canary/results.csv PYTHON="$PYTHON" bash scripts/run_phase8_selection.sh
"$PYTHON" scripts/summarize_phase8.py --csv phase8_results/canary/results.csv \
  --anchor_arm phase7_period_multi_auto_acf_patchlinear --output_dir phase8_results/canary
```
Canary 要求：results.csv 含 `periods`(union 臂显示合并周期)/`val_mse_seg0..3`/`cut_freq`；hydra 臂能跑；summary 标题 `Phase 8-Hydra Summary`；failed=0；无 NaN。`NO_MAIL=1` 时 canary 不发邮件。失败读 `logs/AsySpecX_phase8/*.log`，修复 push 重试。

## 4. 提交（真正跑，会发开始邮件）

先 compact（576 jobs），要 ensemble 就 SAVE_PREDICTIONS=1：
```bash
COMPACT=1 SAVE_PREDICTIONS=1 bash scripts/slurm/submit_asyspecx_phase8.sh
```
submit 会立刻发一封 START 邮件（含 job 数/分区/arm），并记录开始时间到 `phase8_results/hydra/.run_meta`，然后后台启动 watcher。
若某臂改善达阈值再跑 full：`bash scripts/slurm/submit_asyspecx_phase8.sh`（1008 jobs）。

watcher（`autopush_asyspecx_phase8_watch.sh`）在 `asx8_` 清空后自动：aggregate → merge `phase6_results/fullfield/results.csv` + `phase7_results/merged/results.csv` + phase8 → `run_phase8_selection.sh` → `audit_phase5_selectors.py` → （SAVE_PREDICTIONS=1）`ensemble_predictions.py` → `summarize_phase8.py` → glab push → **一封 DONE 邮件**（含 wall-clock 时长、GPU-jobs 数=用了多少卡、push 状态、summary head）。有 baseline 传 `BASELINE_CSV=...`。

## 5. 监控（不发邮件）

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx8_ || true
tail -f logs/autopush_asyspecx_phase8.log
```

## 6. watcher 兜底（若 watcher 挂了，手动补 + 手动发完成邮件）

```bash
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase8_results/hydra
"$PYTHON" scripts/merge_results.py --csvs phase6_results/fullfield/results.csv,phase7_results/merged/results.csv,phase8_results/hydra/results.csv --output phase8_results/merged/results.csv
ROOT=phase8_results/merged CSV=phase8_results/merged/results.csv PYTHON="$PYTHON" bash scripts/run_phase8_selection.sh
"$PYTHON" scripts/audit_phase5_selectors.py --csv phase8_results/merged/results.csv \
  --selected_files "selected_unrestricted_mean.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv" \
  --output_dir phase8_results/merged
"$PYTHON" scripts/summarize_phase8.py --csv phase8_results/merged/results.csv \
  --selected_csv phase8_results/merged/selected_unrestricted_mean.csv \
  --anchor_arm phase7_period_multi_auto_acf_patchlinear --output_dir phase8_results/merged
glab push @"$PWD" --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz'
# 手动完成邮件（含时长与卡数；start_epoch/jobs 在 .run_meta 里）
. phase8_results/hydra/.run_meta 2>/dev/null || true
{ echo "AsySpecX Phase 8 DONE $(date)"; echo "GPU-jobs: ${jobs:-?} (1 GPU each)";
  echo "Summary: $PWD/phase8_results/merged/summary_phase8.md";
  sed -n '1,120p' phase8_results/merged/summary_phase8.md 2>/dev/null; } \
  | mail -s "[asy2-0709v1] AsySpecX Phase8 DONE" yind7@outlook.com
```

## 7. 完成邮件/回报要点

- 跑了多久（wall-clock）、用了多少卡（GPU-jobs 数，1 卡/job）。
- 提交/失败 jobs 数、缺失 dataset。
- best fixed single-arm 是否被某 phase8 臂超过（summary "Phase8 Arms Improving Phase6/7 Best Cells" 段）。
- unrestricted_mean selected MSE/MAE，vs best-single、vs oracle（只用 val 选）。
- union 周期是否合理；hydra 分支权重/linear/patch gate（若 summary 有诊断段）。
- ensemble MSE vs best-single（若 SAVE_PREDICTIONS=1）。
- 决策：是否达阈值（→full）或用 Phase 7 收尾。
- 是否已 push；两封邮件（START/DONE）是否都发出，标题前缀 `asy2-0709v1`。
