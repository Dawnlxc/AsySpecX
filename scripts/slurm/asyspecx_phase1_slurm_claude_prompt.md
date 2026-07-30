# Prompt for Slurm-side Claude

你在 Petrichor/Slurm 登录节点上操作。目标：把 `AsySpecX` 拉到 Slurm 侧，检查并运行 AsySpecX phase-1 修复实验；脚本有问题就直接修复并 push 回 glab；作业结束后聚合结果、push 结果回 glab，并邮件通知 `yind7`。

请按这个流程执行。

## 1. 拉代码到 scratch 目录

```bash
mkdir -p /scratch3/lin250/bldgFM/DUBABA
cd /scratch3/lin250/bldgFM/DUBABA
glab pull AsySpecX
cd AsySpecX
```

## 2. 先检查提交脚本是否真是 Slurm 版

```bash
bash -n scripts/slurm/submit_asyspecx_phase1.sh
bash -n scripts/slurm/autopush_asyspecx_phase1_watch.sh
bash -n scripts/slurm/asyspecx_phase1_run.sbatch
python -m py_compile scripts/slurm/aggregate_asyspecx_phase1.py
grep -RIn "qsub\|#PBS\|/srv/scratch/cruise/du/code/tf-optimizer\|katana" \
  scripts/slurm/submit_asyspecx_phase1.sh \
  scripts/slurm/autopush_asyspecx_phase1_watch.sh \
  scripts/slurm/asyspecx_phase1_run.sbatch \
  scripts/slurm/aggregate_asyspecx_phase1.py || true
NO_PUSH=1 DRY_RUN=1 bash scripts/slurm/submit_asyspecx_phase1.sh
```

要求：

- 不能调用 `qsub`。
- 不能硬编码 katana 路径。
- 默认必须使用 `sbatch`。
- 默认 `ACCOUNT=od-241336`。
- 默认 `PARTITION=h24gpu`。
- 默认 `CONDA_ROOT=/scratch3/lin250/conda_envs`。
- 默认 `CONDA_ENV=tsfm`。
- phase1 四组 arm 必须存在：
  - `phase1_fits_only`
  - `phase1_cross_zero_global`
  - `phase1_safe_cross`
  - `phase1_safe_cross_backcast`

如果检查不通过，直接修复脚本，再执行上面检查。修复后 push 回 glab：

```bash
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
```

## 3. 如果 Slurm/conda 可用，先提交一个小 canary

```bash
NO_PUSH=1 ONLY=phase1_fits_only DATASETS=ETTh1 SEEDS=2026 \
  SEQ_LENS=96 PRED_LENS=96 EPOCHS=1 \
  OUTROOT=phase1_results/canary \
  bash scripts/slurm/submit_asyspecx_phase1.sh
```

等 canary 完成，检查 log 和 summary：

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx1_ || true
find phase1_results/canary -name run_summary.json -print
python scripts/slurm/aggregate_asyspecx_phase1.py --root phase1_results/canary
tail -120 logs/slurm/asx1_* logs/AsySpecX_phase1/*.log 2>/dev/null || true
```

如果 canary 失败，读：

```bash
tail -200 logs/slurm/*.err logs/slurm/*.out logs/AsySpecX_phase1/*.log 2>/dev/null || true
```

然后修复脚本或环境调用方式，push 回 glab 后再重试 canary。

## 4. Canary OK 后，提交完整 phase1 实验

```bash
bash scripts/slurm/submit_asyspecx_phase1.sh
```

默认矩阵：

- arms: 4 个 phase1 arms
- datasets: `ETTh1 ETTm1 weather electricity traffic PEMS04 PEMS08`
- seeds: `2026 2027`
- non-PEMS seq_len: `96 720`
- PEMS seq_len: `96`
- pred_len: 使用 `scripts/_common.sh` 中每个 dataset 的默认 pred_lens

提交脚本会启动 detached watcher：`scripts/slurm/autopush_asyspecx_phase1_watch.sh`。watcher 应在所有 `asx1_` Slurm 作业清空后自动：

- 运行 `python scripts/slurm/aggregate_asyspecx_phase1.py --root phase1_results/main`
- 生成 `phase1_results/main/results.csv`
- 生成 `phase1_results/main/summary.md`
- `glab push @"$PWD"` 把结果和日志推回 glab
- `mail -s ... yind7@outlook.com`

## 5. 监控

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx1_ || true
tail -f logs/autopush_asyspecx_phase1.log
```

## 6. 如果 watcher 没有自动完成，手动兜底

```bash
python scripts/slurm/aggregate_asyspecx_phase1.py --root phase1_results/main
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
{
  echo "AsySpecX phase1 finished on $(hostname) at $(date)."
  echo "Repo: $PWD"
  echo "Summary: $PWD/phase1_results/main/summary.md"
  echo "CSV: $PWD/phase1_results/main/results.csv"
  sed -n '1,120p' phase1_results/main/summary.md 2>/dev/null || true
} | mail -s "[AsySpecX phase1] done" yind7@outlook.com
```

最终回报：

- 提交了多少 jobs
- 成功多少、失败多少
- `summary.md` 路径
- `results.csv` 路径
- 是否已经 push 回 glab
- 邮件是否已发
