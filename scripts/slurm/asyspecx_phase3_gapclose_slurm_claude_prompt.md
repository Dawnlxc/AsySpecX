# Prompt for Slurm-side Claude: AsySpecX Phase 3-GapClose

你在 Petrichor/Slurm 登录节点上操作。目标：把 `AsySpecX` 拉到 Slurm scratch，检查并运行 Phase 3-GapClose；脚本有问题就修复并 push 回 glab；结果好了聚合、push 回 glab，并邮件通知 `yind7@outlook.com`。

## 1. 拉代码

```bash
mkdir -p /scratch3/lin250/bldgFM/DUBABA
cd /scratch3/lin250/bldgFM/DUBABA
glab pull AsySpecX
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
bash -n scripts/run_phase3_gapclose.sh
bash -n scripts/run_phase3_gapclose_sweep.sh
bash -n scripts/slurm/submit_asyspecx_phase3_gapclose.sh
bash -n scripts/slurm/autopush_asyspecx_phase3_gapclose_watch.sh
bash -n scripts/slurm/asyspecx_phase3_run.sbatch

"$PYTHON" -m py_compile \
  models/AsySpecX.py exp/exp_main.py run.py utils/tools.py \
  scripts/select_by_validation.py \
  scripts/summarize_phase3_gapclose.py \
  scripts/slurm/aggregate_asyspecx_phase1.py

"$PYTHON" -m unittest tests/test_asyspecx_phase1.py

grep -RIn "qsub\|#PBS\|/srv/scratch/cruise/du/code/tf-optimizer\|katana" \
  scripts/run_phase3_gapclose.sh \
  scripts/run_phase3_gapclose_sweep.sh \
  scripts/slurm/submit_asyspecx_phase3_gapclose.sh \
  scripts/slurm/autopush_asyspecx_phase3_gapclose_watch.sh \
  scripts/slurm/asyspecx_phase3_run.sbatch || true

NO_PUSH=1 DRY_RUN=1 ARMS="phase3_fits_shared phase3_fits_individual" \
  DATASETS=weather SEEDS=2026 SEQ_LENS=720 PRED_LENS=96 EPOCHS=1 \
  bash scripts/slurm/submit_asyspecx_phase3_gapclose.sh
```

Dry-run 预期：2 jobs。

如果检查失败，直接修复脚本/code，然后重复检查。修复后 push：

```bash
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
```

## 3. Canary

先跑 weather 的最小 canary：

```bash
NO_PUSH=1 ARMS="phase3_fits_shared phase3_fits_individual phase3_fits_shared_revin_affine phase3_anchor_sparse_period" \
  DATASETS=weather SEEDS=2026 SEQ_LENS=720 PRED_LENS=96 EPOCHS=1 PATIENCE=1 \
  OUTROOT=phase3_gapclose_results/canary \
  bash scripts/slurm/submit_asyspecx_phase3_gapclose.sh
```

等待完成：

```bash
while squeue -u "$USER" -h -o "%120j" | grep -q "^asx3_"; do
  squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx3_ || true
  sleep 60
done

"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase3_gapclose_results/canary
"$PYTHON" scripts/summarize_phase3_gapclose.py --root phase3_gapclose_results/canary --csv phase3_gapclose_results/canary/results.csv
"$PYTHON" scripts/select_by_validation.py \
  --csv phase3_gapclose_results/canary/results.csv \
  --output phase3_gapclose_results/canary/selected_results.csv \
  --summary phase3_gapclose_results/canary/selected_summary.md

find phase3_gapclose_results/canary -maxdepth 2 -type f -name '*.csv' -print
tail -160 logs/slurm/asx3_* logs/AsySpecX_phase3/*.log 2>/dev/null || true
```

Canary 要求：

- `phase3_gapclose_results/canary/results.csv` 非空。
- `phase3_gapclose_results/canary/summary_phase3_gapclose.md` 标题是 `Phase 3-GapClose`。
- `selected_results.csv` 生成，且使用 `val_mse` 选择。
- `failed_runs: 0`。
- 无 NaN/Inf/backward failure。

失败则读 `logs/slurm/*.err`、`*.out`、`logs/AsySpecX_phase3/*.log`，修复后 push，再重试 canary。

## 4. 提交完整 Phase 3-GapClose

```bash
bash scripts/slurm/submit_asyspecx_phase3_gapclose.sh
```

默认矩阵：

- datasets: `weather electricity`
- seq_len: `720`
- pred_len: dataset 默认 4 个 horizon
- seeds: `2026 2027`
- arms: 12 个 Phase3 arms
- total: 192 jobs
- period: weather 默认 `144`，electricity 默认 `24`

watcher：`scripts/slurm/autopush_asyspecx_phase3_gapclose_watch.sh`。它会等所有 `asx3_` 作业清空后自动：

- 聚合 `phase3_gapclose_results/main/results.csv`
- 生成 `phase3_gapclose_results/main/summary_phase3_gapclose.md`
- 运行 validation selection，生成 `selected_results.csv` 和 `selected_summary.md`
- `glab push @"$PWD"`
- `mail -s ... yind7@outlook.com`

## 5. 监控

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx3_ || true
tail -f logs/autopush_asyspecx_phase3_gapclose.log
```

## 6. watcher 没自动完成时兜底

```bash
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase3_gapclose_results/main
"$PYTHON" scripts/summarize_phase3_gapclose.py --root phase3_gapclose_results/main --csv phase3_gapclose_results/main/results.csv
"$PYTHON" scripts/select_by_validation.py \
  --csv phase3_gapclose_results/main/results.csv \
  --output phase3_gapclose_results/main/selected_results.csv \
  --summary phase3_gapclose_results/main/selected_summary.md

glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'

{
  echo "AsySpecX Phase 3-GapClose finished on $(hostname) at $(date)."
  echo "Repo: $PWD"
  echo "CSV: $PWD/phase3_gapclose_results/main/results.csv"
  echo "Summary: $PWD/phase3_gapclose_results/main/summary_phase3_gapclose.md"
  echo "Selected: $PWD/phase3_gapclose_results/main/selected_results.csv"
  sed -n '1,120p' phase3_gapclose_results/main/summary_phase3_gapclose.md 2>/dev/null || true
} | mail -s "[AsySpecX Phase3 GapClose] done" yind7@outlook.com
```

## 7. 最终回报

最终请回报：

- 提交了多少 jobs。
- 失败多少 jobs，失败 arm/tag 是什么。
- raw/summary/selected CSV 路径。
- weather 哪个 arm 最好。
- electricity 哪个 arm 最好。
- individual lift 是否缩小 gap。
- norm ablation 是否改善 weather。
- sparse-period 是否改善 electricity 96/192/336。
- validation selection 是否成功且只用 `val_mse`。
- 是否已经 push 回 glab。
- 邮件是否已发给 `yind7@outlook.com`。
