# Prompt for Slurm-side Claude: AsySpecX Phase 5-Lockdown

你在 Petrichor/Slurm 登录节点。目标：把 `AsySpecX` 从 glab 拉过来，检查并运行 Phase 5-Lockdown；脚本有问题就修复并 push 回 glab；结果好了聚合、跑三种 validation selection、summarize、push 回 glab，邮件通知 `yind7@outlook.com`，**邮件标题 `asy2-0704v1`**。

Phase 5 背景（Phase 4 结论）：
- electricity 最优 `phase5_asx_period_multi`（4 horizon 全赢候选池）。
- weather 最优 `phase5_asx_individual_revin` / `phase5_asx_individual`。
- Phase 4 selector 在 weather 336/720 误选了 period_multi。Phase 5 selector 已加稳健化：val segmented metrics、mean_plus_std、margin+prefer_arm_order、per-dataset allowlist。
- 候选池已冻结，不新增结构。最终表用 validation selection，绝不用 test 选 arm。

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

ls -lh dataset/weather/weather.csv dataset/electricity/electricity.csv
```

## 2. 检查脚本和 Python

```bash
for s in scripts/run_phase5_fullfield_candidates.sh scripts/run_phase5_confirm_weather_electricity.sh \
         scripts/run_phase5_selection.sh scripts/slurm/submit_asyspecx_phase5.sh \
         scripts/slurm/autopush_asyspecx_phase5_watch.sh scripts/slurm/asyspecx_phase5_run.sbatch \
         scripts/_common.sh; do bash -n "$s" || echo "BASH FAIL $s"; done

"$PYTHON" -m py_compile models/AsySpecX.py exp/exp_main.py run.py \
  scripts/select_by_validation.py scripts/summarize_phase5.py \
  scripts/summarize_cut_freq.py scripts/slurm/aggregate_asyspecx_phase1.py

"$PYTHON" -m unittest tests/test_asyspecx_phase1.py tests/test_asyspecx_phase4.py tests/test_asyspecx_phase5.py
```

**PERIODS 逗号陷阱**：submit 脚本把 `24,168` 转成 `24+168` 再 export；`run.py --periods` 接受 `+`。dry-run 确认 electricity 的 export 是 `PERIODS=24+168`，weather 是 `PERIODS=144`，且带 `VAL_NUM_SEGMENTS=4`：

```bash
NO_PUSH=1 DRY_RUN=1 ARMS="phase5_asx_period_multi" \
  DATASETS=electricity SEEDS=2024 SEQ_LENS=720 PRED_LENS=96 EPOCHS=1 \
  bash scripts/slurm/submit_asyspecx_phase5.sh
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
  ARMS="phase5_asx_cross phase5_asx_individual phase5_asx_individual_revin phase5_asx_period_multi phase5_asx_individual_period" \
  DATASETS="weather electricity" SEEDS=2024 SEQ_LENS=720 PRED_LENS=96 \
  EPOCHS=1 PATIENCE=1 VAL_NUM_SEGMENTS=4 OUTROOT=phase5_results/canary \
  bash scripts/slurm/submit_asyspecx_phase5.sh

while squeue -u "$USER" -h -o "%120j" | grep -q "^asx5_"; do
  squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx5_ || true
  sleep 60
done

"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase5_results/canary
ROOT=phase5_results/canary CSV=phase5_results/canary/results.csv PYTHON="$PYTHON" bash scripts/run_phase5_selection.sh
"$PYTHON" scripts/summarize_phase5.py --csv phase5_results/canary/results.csv \
  --selected_csv phase5_results/canary/selected_unrestricted_mean.csv \
  --anchor_arm phase5_asx_cross --output_dir phase5_results/canary
```

Canary 要求：
- `results.csv` 非空，含 `cut_freq`、`periods`、`val_mse_seg0..3`、`val_mae` 列。
- electricity period_multi 的 `periods` 是 `24+168`（没被逗号截断）。
- 三个 selection 变体都生成（unrestricted_mean / unrestricted_last_segment / policy_family）。last_segment 用到 `val_mse_seg3`。
- `summary_phase5.md` 标题是 `Phase 5-Lockdown Summary`。
- `failed_runs: 0`，无 NaN/Inf/backward failure。

失败读 `logs/slurm/*.err`、`*.out`、`logs/AsySpecX_phase5/*.log`，修复后 push，重试。

## 4. 提交完整 Phase 5

```bash
bash scripts/slurm/submit_asyspecx_phase5.sh
```

默认矩阵：6 arms × {weather,electricity} × 4 horizon × 3 seeds ≈ 144 jobs；`VAL_NUM_SEGMENTS=4`；weather periods=144，electricity periods=24+168。
- 想加 gate-l1 arm：`ENABLE_PERIOD_REG=1`。
- 想跑更多 seed 确认小差异：先跑 `bash scripts/run_phase5_confirm_weather_electricity.sh`（本地串跑，5 seeds）或用 submit 加 `SEEDS="2021 2022 2023 2024 2025"`。
- 有外部 baseline：`BASELINE_CSV=...`（列 `dataset,seq_len,pred_len,model,mse,mae`）传给 submit。

watcher（`autopush_asyspecx_phase5_watch.sh`）在 `asx5_` 清空后自动：
- 聚合 `phase5_results/main/results.csv`
- 跑 `run_phase5_selection.sh`（三个变体）
- `summarize_phase5.py` → `summary_phase5.md`
- `summarize_cut_freq.py` → `summary_cut_freq.md`
- `glab push @"$PWD"`
- `mail -s "[asy2-0704v1] AsySpecX Phase5 Lockdown done" yind7@outlook.com`

## 5. 监控

```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx5_ || true
tail -f logs/autopush_asyspecx_phase5.log
```

## 6. watcher 兜底

```bash
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase5_results/main
ROOT=phase5_results/main CSV=phase5_results/main/results.csv PYTHON="$PYTHON" bash scripts/run_phase5_selection.sh
"$PYTHON" scripts/summarize_phase5.py --csv phase5_results/main/results.csv \
  --selected_csv phase5_results/main/selected_unrestricted_mean.csv \
  --anchor_arm phase5_asx_cross --output_dir phase5_results/main

glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'

{
  echo "AsySpecX Phase 5-Lockdown finished on $(hostname) at $(date)."
  echo "Repo: $PWD"
  echo "CSV: $PWD/phase5_results/main/results.csv"
  echo "Summary: $PWD/phase5_results/main/summary_phase5.md"
  echo "Selected(unrestricted_mean): $PWD/phase5_results/main/selected_unrestricted_mean.csv"
  sed -n '1,160p' phase5_results/main/summary_phase5.md 2>/dev/null || true
} | mail -s "[asy2-0704v1] AsySpecX Phase5 Lockdown done" yind7@outlook.com
```

## 7. 最终回报

- 提交/失败 jobs 数（失败 arm/tag）。
- raw/summary/三个 selected CSV 路径。
- **unrestricted_mean** 选出的 arm（per dataset/pred_len），确认只用 val_mse。
- weather 336/720：unrestricted_mean 是否还误选 period_multi；last_segment 与 policy_family 是否修正为 individual/individual_revin。
- electricity：period_multi 是否仍最优。
- paired statistics：period_multi vs cross 的 win/loss。
- period_multi 的 `periods` 是否 `24+168`（没被截断）。
- 是否已 push 回 glab；邮件是否已发 `yind7@outlook.com`，标题 `asy2-0704v1`。
