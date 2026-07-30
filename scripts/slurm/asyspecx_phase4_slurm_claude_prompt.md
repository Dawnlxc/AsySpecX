# Prompt for Slurm-side Claude: AsySpecX Phase 4-Finalize

你在 Petrichor/Slurm 登录节点上操作。目标：把 `AsySpecX` 从 glab 拉过来，检查并运行 Phase 4-Finalize（final candidates + 可选 weather/electricity tuning），脚本有问题就修复并 push 回 glab；结果好了聚合、validation selection、summarize、push 回 glab，并邮件通知 `yind7@outlook.com`。

关键背景（Phase 3 结论，用来解读 Phase 4）：
- weather 最优是 `fits_individual`（channel-specific spectral backbone），cross-transfer 默认不该开。
- electricity 最优是 `anchor_sparse_period`（sparse-period adapter 有效）。
- `individual + cross`（individual_hier_split）不好，不要默认组合。
- 最终表用 **validation selection**，绝不用 test 选 arm。

## 1. 拉代码

```bash
mkdir -p /scratch3/lin250/bldgFM/DUBABA
cd /scratch3/lin250/bldgFM/DUBABA
glab pull AsySpecX          # 或 glab pull @/mnt/scratch/CRUISE/Du/code/AsySpecX
cd AsySpecX
```

默认 Slurm/conda：

```bash
export ACCOUNT="${ACCOUNT:-od-241336}"
export PARTITION="${PARTITION:-h24gpu}"
export CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
export CONDA_ENV="${CONDA_ENV:-tsfm}"
export PYTHON="${PYTHON:-python}"
```

确认数据存在：

```bash
ls -lh dataset/weather/weather.csv dataset/electricity/electricity.csv
```

## 2. 检查脚本和 Python

```bash
bash -n scripts/run_phase4_final_candidates.sh
bash -n scripts/run_phase4_weather_tuning.sh
bash -n scripts/run_phase4_electricity_tuning.sh
bash -n scripts/slurm/submit_asyspecx_phase4.sh
bash -n scripts/slurm/autopush_asyspecx_phase4_watch.sh
bash -n scripts/slurm/asyspecx_phase4_run.sbatch

"$PYTHON" -m py_compile \
  models/AsySpecX.py exp/exp_main.py run.py \
  scripts/select_by_validation.py \
  scripts/summarize_phase4.py \
  scripts/summarize_cut_freq.py \
  scripts/slurm/aggregate_asyspecx_phase1.py

"$PYTHON" -m unittest tests/test_asyspecx_phase1.py tests/test_asyspecx_phase4.py
```

**注意 PERIODS 逗号陷阱**：sbatch `--export` 用逗号分隔，会把 `24,168` 截断。submit 脚本已把逗号转成 `+`（`24+168`），`run.py --periods` 同时接受 `+` 和 `,`。检查 dry-run 的 `--export` 里 period 是 `24+168` 而不是被截断成 `24`：

```bash
NO_PUSH=1 DRY_RUN=1 ARMS="phase4_asx_period_multi" \
  DATASETS=electricity SEEDS=2024 SEQ_LENS=720 PRED_LENS=96 EPOCHS=1 \
  bash scripts/slurm/submit_asyspecx_phase4.sh
```

Dry-run 预期：1 job，且 export 含 `PERIODS=24+168`。

如果检查失败，直接修复脚本/code，重复检查。修复后 push：

```bash
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
```

## 3. Canary（最小冒烟）

```bash
NO_PUSH=1 \
  ARMS="phase4_asx_individual phase4_asx_period_single phase4_asx_period_multi" \
  DATASETS="weather electricity" SEEDS=2024 SEQ_LENS=720 PRED_LENS=96 \
  EPOCHS=1 PATIENCE=1 OUTROOT=phase4_results/canary \
  bash scripts/slurm/submit_asyspecx_phase4.sh

while squeue -u "$USER" -h -o "%120j" | grep -q "^asx4_"; do
  squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx4_ || true
  sleep 60
done

"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase4_results/canary
"$PYTHON" scripts/select_by_validation.py \
  --csv phase4_results/canary/results.csv \
  --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
  --output phase4_results/canary/selected_results.csv \
  --summary phase4_results/canary/selected_summary.md
"$PYTHON" scripts/summarize_phase4.py \
  --csv phase4_results/canary/results.csv \
  --selected_csv phase4_results/canary/selected_results.csv \
  --output phase4_results/canary/summary_phase4.md
```

Canary 要求：
- `phase4_results/canary/results.csv` 非空，含 `cut_freq`、`periods` 列。
- period_multi arm 的 `periods` 列是 `24+168`（electricity），不是 `24`。
- `summary_phase4.md` 标题是 `Phase 4-Finalize Summary`（不是 Phase 1/2/3）。
- `selected_results.csv` 生成，且每个 selected arm 输出**所有 seed** 的行。
- `failed_runs: 0`，无 NaN/Inf/backward failure。

失败就读 `logs/slurm/*.err`、`*.out`、`logs/AsySpecX_phase4/*.log`，修复后 push，重试 canary。

## 4. 提交完整 Phase 4 final candidates

```bash
bash scripts/slurm/submit_asyspecx_phase4.sh
```

默认矩阵：
- arms: 7 个（`phase4_asx_cross` `phase4_asx_individual` `phase4_asx_period_single` `phase4_asx_period_multi` `phase4_asx_individual_period` `phase4_asx_individual_revin` `phase4_asx_cross_revin`）
- datasets: `weather electricity`
- seq_len: `720`
- pred_len: dataset 默认 4 horizon（96 192 336 720）
- seeds: `2024 2025 2026`
- periods: weather=`144`，electricity=`24+168`
- total ≈ 7 × 2 × 4 × 3 = 168 jobs

watcher（`autopush_asyspecx_phase4_watch.sh`）会在 `asx4_` 作业清空后自动：
- 聚合 `phase4_results/main/results.csv`
- validation selection → `selected_results.csv` + `selected_summary.md`
- `summarize_phase4.py` → `summary_phase4.md`
- `summarize_cut_freq.py` → `summary_cut_freq.md`
- `glab push @"$PWD"`
- `mail -s "[AsySpecX Phase4 Finalize] done" yind7@outlook.com`

有外部 baseline CSV（列：`dataset,pred_len,model,mse,mae`）时，传 `BASELINE_CSV=...` 给 submit/watcher，summary 会输出 vs FITS/PatchTST/SparseTSF 的 gap 和 win/loss。

## 5. 可选 tuning（补 gap，非默认）

weather（compact 默认；`FULL_SWEEP=1` 打开更大 sweep + sparse_period）：
```bash
GPU=0 SEED=2024 bash scripts/run_phase4_weather_tuning.sh
```

electricity（compact 默认；`FULL_SWEEP=1` 打开更大 sweep）：
```bash
GPU=0 SEED=2024 bash scripts/run_phase4_electricity_tuning.sh
```

这两个是**本地循环脚本**（不走 sbatch 矩阵），适合单卡串跑或包进 sbatch。跑完用 validation selection + `summarize_cut_freq.py` 分析，best cut_freq 只能按 validation 选。

## 6. 监控

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx4_ || true
tail -f logs/autopush_asyspecx_phase4.log
```

## 7. watcher 没自动完成时兜底

```bash
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase4_results/main
"$PYTHON" scripts/select_by_validation.py \
  --csv phase4_results/main/results.csv \
  --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
  --output phase4_results/main/selected_results.csv \
  --summary phase4_results/main/selected_summary.md
"$PYTHON" scripts/summarize_phase4.py \
  --csv phase4_results/main/results.csv \
  --selected_csv phase4_results/main/selected_results.csv \
  --output phase4_results/main/summary_phase4.md
"$PYTHON" scripts/summarize_cut_freq.py \
  --csv phase4_results/main/results.csv \
  --output phase4_results/main/summary_cut_freq.md || true

glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'

{
  echo "AsySpecX Phase 4-Finalize finished on $(hostname) at $(date)."
  echo "Repo: $PWD"
  echo "CSV: $PWD/phase4_results/main/results.csv"
  echo "Summary: $PWD/phase4_results/main/summary_phase4.md"
  echo "Selected: $PWD/phase4_results/main/selected_results.csv"
  sed -n '1,160p' phase4_results/main/summary_phase4.md 2>/dev/null || true
} | mail -s "[AsySpecX Phase4 Finalize] done" yind7@outlook.com
```

## 8. 最终回报

- 提交了多少 jobs，失败多少（失败 arm/tag）。
- raw / summary / selected CSV 路径。
- validation selection 选出的 arm（per dataset / pred_len），确认只用 `val_mse`。
- weather: `phase4_asx_individual` 是否仍最好；cut_freq tuning 后与 published FITS 的 gap（96=0.145 192=0.188 336=0.236 720=0.308）。若仍小输 0.001–0.005，报告为 close second。
- electricity: multi-period 是否改善 192/336；`period_single` 与 `period_multi` 谁赢。
- period_multi arm 的 `periods` 是否正确是 `24+168`（没被 sbatch 逗号截断）。
- 是否已 push 回 glab；邮件是否已发 `yind7@outlook.com`。
```
