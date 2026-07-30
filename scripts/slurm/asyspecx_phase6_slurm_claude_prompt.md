# Prompt for Slurm-side Claude: AsySpecX Phase 6-Protocol (Full-Field)

你在 Petrichor/Slurm 登录节点。目标：把 `AsySpecX` 从 glab 拉过来，检查并运行 Phase 6 full-field；脚本有问题就修复并 push 回 glab；结果好了聚合、跑 selection variants + selector audit、summarize、push 回 glab，邮件通知 `yind7@outlook.com`，**邮件标题 `asy2-0707v1`**。

Phase 6 背景（Phase 5 结论）：
- Phase 5 只跑了 weather/electricity seq_len=720（144 runs），不是 full-field。Phase 6 要真正跑全部 dataset。
- Fixed single-arm 最好：`phase6_asx_individual_period`（robust）；novelty-preserving 最好：`phase6_asx_period_multi`。
- Phase 5 selector 在 weather 336/720 误选 period_multi（full-val vs last-segment mismatch 4/4）。Phase 6 加了 `segment_mean_plus_std` 稳健模式 + margin/prefer trace + selector audit + test oracle（analysis only）。
- 最终表用 validation selection，绝不用 test 选 arm。Oracle 只做分析上界，不能当成模型结果报告。

## 1. 拉代码

```bash
mkdir -p /scratch3/lin250/bldgFM/DUBABA
cd /scratch3/lin250/bldgFM/DUBABA
glab pull AsySpecX          # 或 glab pull @/mnt/scratch/CRUISE/Du/code/AsySpecX
cd AsySpecX

export ACCOUNT="${ACCOUNT:-od-241336}"
export PARTITION="${PARTITION:-h24gpu}"
export CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
export CONDA_ENV="${CONDA_ENV:-tsfm}"
export PYTHON="${PYTHON:-python}"

ls -lh dataset/weather/weather.csv dataset/electricity/electricity.csv \
       dataset/ETT-small/ETTh1.csv dataset/ETT-small/ETTm1.csv \
       dataset/traffic/traffic.csv dataset/PEMS/PEMS04.npz dataset/PEMS/PEMS08.npz
```

如果某些 dataset 文件缺失（traffic 走 Git LFS；PEMS npz），先确认能读；缺的 dataset 从 DATASETS 里剔除再跑，回报里说明。

## 2. 检查脚本和 Python

```bash
for s in scripts/run_phase6_fullfield_candidates.sh scripts/run_phase6_fullfield_selection.sh \
         scripts/run_phase6_selector_audit.sh scripts/slurm/submit_asyspecx_phase6.sh \
         scripts/slurm/autopush_asyspecx_phase6_watch.sh scripts/slurm/asyspecx_phase6_run.sbatch \
         scripts/_common.sh; do bash -n "$s" || echo "BASH FAIL $s"; done

"$PYTHON" -m py_compile models/AsySpecX.py exp/exp_main.py run.py \
  scripts/select_by_validation.py scripts/audit_phase5_selectors.py \
  scripts/summarize_phase6_fullfield.py scripts/summarize_cut_freq.py \
  scripts/slurm/aggregate_asyspecx_phase1.py

"$PYTHON" -m unittest tests/test_asyspecx_phase1.py tests/test_asyspecx_phase4.py \
  tests/test_asyspecx_phase5.py tests/test_asyspecx_phase6.py
```

**PERIODS 逗号陷阱**：submit 把 `24,168`→`24+168` 再 export；`run.py --periods` 接受 `+`。dry-run 确认 electricity export 是 `PERIODS=24+168`、PEMS seq_len=720 被跳过（除非 RUN_PEMS_SEQ720=1）、带 `VAL_NUM_SEGMENTS=4`；默认总数应是 **864**：

```bash
NO_PUSH=1 DRY_RUN=1 bash scripts/slurm/submit_asyspecx_phase6.sh 2>&1 | tail -3
NO_PUSH=1 DRY_RUN=1 DATASETS=PEMS04 SEQ_LENS="96 720" bash scripts/slurm/submit_asyspecx_phase6.sh 2>&1 | grep -c s720   # 期望 0
```

先跑一次 Phase 6 selector audit（在已有 Phase 5 结果上，不需要 GPU）确认 audit 链路 OK：

```bash
ROOT=phase5_results/main PYTHON="$PYTHON" bash scripts/run_phase6_selector_audit.sh
sed -n '1,40p' phase5_results/main/selector_audit.md
```

有问题就修脚本/code，重复检查，然后 push：

```bash
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
```

## 3. Canary

```bash
NO_PUSH=1 \
  ARMS="phase6_asx_cross phase6_asx_individual phase6_asx_period_multi" \
  DATASETS="weather electricity" SEEDS=2024 SEQ_LENS=96 PRED_LENS=96 \
  EPOCHS=1 PATIENCE=1 VAL_NUM_SEGMENTS=4 OUTROOT=phase6_results/canary \
  bash scripts/slurm/submit_asyspecx_phase6.sh

while squeue -u "$USER" -h -o "%120j" | grep -q "^asx6_"; do
  squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx6_ || true
  sleep 60
done

"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase6_results/canary
ROOT=phase6_results/canary CSV=phase6_results/canary/results.csv PYTHON="$PYTHON" bash scripts/run_phase6_fullfield_selection.sh
"$PYTHON" scripts/summarize_phase6_fullfield.py --csv phase6_results/canary/results.csv \
  --selected_csv phase6_results/canary/selected_unrestricted_mean.csv \
  --anchor_arm phase6_asx_cross --output_dir phase6_results/canary
```

Canary 要求：
- `results.csv` 非空，含 `cut_freq`、`periods`、`val_mse_seg0..3` 列。
- electricity 的 `periods` 是 `24+168`（没被截断）。
- `summary_phase6_fullfield.md` 标题 `Phase 6 Full-Field Summary`。
- `failed_runs: 0`，无 NaN/Inf。

失败读 `logs/slurm/*.err`、`logs/AsySpecX_phase6/*.log`，修复 push 重试。

## 4. 提交完整 Phase 6 full-field

```bash
bash scripts/slurm/submit_asyspecx_phase6.sh
```

默认 ~864 jobs：6 arms × {ETTh1,ETTm1,weather,electricity,traffic,PEMS04,PEMS08} × seq_len × pred_len × 3 seeds；PEMS 只 seq_len=96（`RUN_PEMS_SEQ720=1` 才加 720）；`VAL_NUM_SEGMENTS=4`。有 baseline 就 `BASELINE_CSV=...`（列 `dataset,seq_len,pred_len,model,mse,mae`）。

watcher（`autopush_asyspecx_phase6_watch.sh`）在 `asx6_` 清空后自动：
- 聚合 `phase6_results/fullfield/results.csv`
- `run_phase6_fullfield_selection.sh`（unrestricted_mean / segment_robust / margin_prefer_simple / policy_family）
- `audit_phase5_selectors.py` → `selector_audit.md`
- `summarize_phase6_fullfield.py` → `summary_phase6_fullfield.md`
- `glab push` + `mail -s "[asy2-0707v1] AsySpecX Phase6 FullField done" yind7@outlook.com`

## 5. 监控

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx6_ || true
tail -f logs/autopush_asyspecx_phase6.log
```

## 6. watcher 兜底

```bash
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase6_results/fullfield
ROOT=phase6_results/fullfield CSV=phase6_results/fullfield/results.csv PYTHON="$PYTHON" bash scripts/run_phase6_fullfield_selection.sh
"$PYTHON" scripts/audit_phase5_selectors.py --csv phase6_results/fullfield/results.csv \
  --selected_files "selected_unrestricted_mean.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv" \
  --output_dir phase6_results/fullfield
"$PYTHON" scripts/summarize_phase6_fullfield.py --csv phase6_results/fullfield/results.csv \
  --selected_csv phase6_results/fullfield/selected_unrestricted_mean.csv \
  --anchor_arm phase6_asx_cross --output_dir phase6_results/fullfield

glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'

{
  echo "AsySpecX Phase 6 Full-Field finished on $(hostname) at $(date)."
  echo "CSV: $PWD/phase6_results/fullfield/results.csv"
  echo "Summary: $PWD/phase6_results/fullfield/summary_phase6_fullfield.md"
  echo "Selector audit: $PWD/phase6_results/fullfield/selector_audit.md"
  sed -n '1,160p' phase6_results/fullfield/summary_phase6_fullfield.md 2>/dev/null || true
} | mail -s "[asy2-0707v1] AsySpecX Phase6 FullField done" yind7@outlook.com
```

## 7. 最终回报

- 提交/失败 jobs 数（失败 arm/dataset）；缺失/剔除的 dataset。
- raw CSV + 4 个 selected CSV + selector_audit.md + summary_phase6_fullfield.md 路径。
- fixed single-arm 最好是谁（应为 `phase6_asx_individual_period` 或 `phase6_asx_period_multi`）。
- unrestricted_mean 选出的 arm（per dataset/pred_len），确认只用 val_mse。
- segment_robust / margin_prefer_simple 是否在 weather 336/720 把 period_multi 换回 individual/individual_revin。
- selector audit：各 selector 的 mse_mean、delta_vs_best_single、delta_vs_oracle。
- period_multi 的 periods 是否 `24+168`（没被截断）；PEMS 是否只跑 seq_len=96。
- 是否已 push 回 glab；邮件是否已发 `yind7@outlook.com`，标题 `asy2-0707v1`。
```
