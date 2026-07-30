# Prompt for Slurm-side Claude

你在 Petrichor/Slurm 登录节点上操作。目标：把 `AsySpecX` 拉到 Slurm 侧，检查并运行 AsySpecX Phase 2；脚本有问题就直接修复并 push 回 glab；作业结束后聚合结果、push 结果回 glab，并邮件通知 `yind7@outlook.com`。

## 1. 拉代码

```bash
mkdir -p /scratch3/lin250/bldgFM/DUBABA
cd /scratch3/lin250/bldgFM/DUBABA
glab pull AsySpecX
cd AsySpecX
```

默认配置：

```bash
export ACCOUNT="${ACCOUNT:-od-241336}"
export PARTITION="${PARTITION:-h24gpu}"
export CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
export CONDA_ENV="${CONDA_ENV:-tsfm}"
```

## 2. 检查脚本和 Python

```bash
bash -n scripts/slurm/submit_asyspecx_phase2.sh
bash -n scripts/slurm/autopush_asyspecx_phase2_watch.sh
bash -n scripts/slurm/asyspecx_phase2_run.sbatch
bash -n scripts/run_phase2_asyspecx.sh
bash -n scripts/run_phase2_sweep.sh
python -m py_compile models/AsySpecX.py exp/exp_main.py run.py utils/tools.py scripts/slurm/aggregate_asyspecx_phase1.py
python -m unittest tests/test_asyspecx_phase1.py

grep -RIn "qsub\|#PBS\|/srv/scratch/cruise/du/code/tf-optimizer\|katana" \
  scripts/slurm/submit_asyspecx_phase2.sh \
  scripts/slurm/autopush_asyspecx_phase2_watch.sh \
  scripts/slurm/asyspecx_phase2_run.sbatch \
  scripts/run_phase2_asyspecx.sh scripts/run_phase2_sweep.sh || true

NO_PUSH=1 DRY_RUN=1 ONLY=phase2_global_all DATASETS=ETTh1 SEEDS=2026 SEQ_LENS=96 PRED_LENS=96 EPOCHS=1 \
  bash scripts/slurm/submit_asyspecx_phase2.sh
```

要求：

- 不能调用 `qsub`。
- 不能硬编码 katana 路径。
- 默认使用 `sbatch`。
- 默认 `ACCOUNT=od-241336`。
- 默认 `PARTITION=h24gpu`。
- 默认 `CONDA_ROOT=/scratch3/lin250/conda_envs`。
- 默认 `CONDA_ENV=tsfm`。
- Phase 2 arms 必须存在：
  - `phase2_global_all`
  - `phase2_global_diag_only`
  - `phase2_global_offdiag_only`
  - `phase2_global_split`
  - `phase2_hier_all`
  - `phase2_hier_split`
  - `phase2_self_band_gain_global`
  - `phase2_global_all_clip05`
  - `phase2_hier_all_clip05`

如果检查失败，直接修复，再重复检查。修复后 push 回 glab：

```bash
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
```

## 3. Canary

先跑 PeMS/ETT 都不需要。只跑 ETTh1、一个 horizon、两个关键 arm、一个 seed：

```bash
NO_PUSH=1 ARMS="phase2_global_all phase2_global_diag_only phase2_global_offdiag_only phase2_self_band_gain_global" \
  DATASETS=ETTh1 SEEDS=2026 SEQ_LENS=96 PRED_LENS=96 EPOCHS=1 \
  OUTROOT=phase2_results/canary \
  bash scripts/slurm/submit_asyspecx_phase2.sh
```

等待 canary 完成：

```bash
while squeue -u "$USER" -h -o "%100j" | grep -q "^asx2_"; do
  squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx2_ || true
  sleep 60
done

python scripts/slurm/aggregate_asyspecx_phase1.py --root phase2_results/canary
find phase2_results/canary -name run_summary.json -print
tail -120 logs/slurm/asx2_* logs/AsySpecX_phase2/*.log 2>/dev/null || true
```

Canary 要求：

- `phase2_results/canary/results.csv` 存在。
- `phase2_results/canary/summary.md` 存在。
- `failed_runs: 0`。

失败则读：

```bash
tail -200 logs/slurm/*.err logs/slurm/*.out logs/AsySpecX_phase2/*.log 2>/dev/null || true
```

修复脚本或环境调用，push 后重试 canary。

## 4. 提交完整 Phase 2

```bash
bash scripts/slurm/submit_asyspecx_phase2.sh
```

默认矩阵：

- arms: 9 个 Phase2 arms
- datasets: `ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08`
- seeds: `2026 2027`
- non-PEMS seq_len: `96 720`
- PEMS seq_len: `96`
- pred_len: 每个 dataset 的默认 pred_lens

默认总 jobs：864。

提交脚本会启动 detached watcher：`scripts/slurm/autopush_asyspecx_phase2_watch.sh`。watcher 等所有 `asx2_` Slurm 作业清空后自动：

- 运行 `python scripts/slurm/aggregate_asyspecx_phase1.py --root phase2_results/main`
- 生成 `phase2_results/main/results.csv`
- 生成 `phase2_results/main/summary.md`
- `glab push @"$PWD"` 把结果和日志推回 glab
- `mail -s ... yind7@outlook.com`

## 5. 监控

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx2_ || true
tail -f logs/autopush_asyspecx_phase2.log
```

## 6. watcher 没自动完成时手动兜底

```bash
python scripts/slurm/aggregate_asyspecx_phase1.py --root phase2_results/main
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
{
  echo "AsySpecX phase2 finished on $(hostname) at $(date)."
  echo "Repo: $PWD"
  echo "Summary: $PWD/phase2_results/main/summary.md"
  echo "CSV: $PWD/phase2_results/main/results.csv"
  sed -n '1,140p' phase2_results/main/summary.md 2>/dev/null || true
} | mail -s "[AsySpecX phase2] done" yind7@outlook.com
```

## 7. 最终回报

最终请回报：

- 提交了多少 jobs。
- 成功多少、失败多少。
- 最佳 arm 和均值 MSE/MAE。
- `summary.md` 路径。
- `results.csv` 路径。
- 是否已经 push 回 glab。
- 邮件是否已发给 `yind7@outlook.com`。
