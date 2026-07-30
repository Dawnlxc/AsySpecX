# Prompt for Slurm-side Claude: AsySpecX Phase 7-Breakthrough

你在 Petrichor/Slurm 登录节点。目标：把 `AsySpecX` 从 glab 拉过来，检查并运行 Phase 7 breakthrough arms；脚本有问题就修复并 push 回 glab；结果好了聚合、与 Phase 6 合并、跑 selection + audit、summarize、（可选）offline ensemble、push 回 glab，邮件通知 `yind7@outlook.com`，**邮件标题 `asy2-0708v1`**。

Phase 7 背景（Phase 6 结论）：
- Phase 6 selector 已接近 test oracle（selected 0.334771 vs oracle 0.333085），大提升需要新的互补候选臂，而不是换 selector。
- best fixed single-arm = `phase6_asx_period_multi`（MSE 0.338568）。
- Phase 7 加 8 个新臂：clipped/learned-clip period_multi、auto-period（acf/fft）、patch-linear、组合。全部 default-off，Phase 1-6 完全兼容。
- 决策规则：若某新臂把 full-field mean 提升 >0.002 就跑 full Phase 7；若没有任何臂击败 `phase6_asx_period_multi`，停止建模、用 Phase 6 写论文。**不做 asym-vs-sym directionality。**
- Oracle 只是分析上界，绝不当模型报告；selection 只用 validation。

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
缺失 dataset（traffic 走 LFS、PEMS npz）从 DATASETS 剔除，回报里说明。

## 2. 检查

```bash
for s in scripts/run_phase7_breakthrough_candidates.sh scripts/run_phase7_selection.sh \
         scripts/slurm/submit_asyspecx_phase7.sh scripts/slurm/autopush_asyspecx_phase7_watch.sh \
         scripts/slurm/asyspecx_phase7_run.sbatch scripts/_common.sh; do bash -n "$s" || echo "FAIL $s"; done

"$PYTHON" -m py_compile models/AsySpecX.py exp/exp_main.py run.py \
  scripts/select_by_validation.py scripts/discover_periods.py scripts/ensemble_predictions.py \
  scripts/merge_results.py scripts/summarize_phase7.py scripts/audit_phase5_selectors.py \
  scripts/slurm/aggregate_asyspecx_phase1.py

"$PYTHON" -m unittest tests/test_asyspecx_phase1.py tests/test_asyspecx_phase4.py \
  tests/test_asyspecx_phase5.py tests/test_asyspecx_phase6.py tests/test_asyspecx_phase7.py
```

**PERIODS 逗号陷阱**：submit 把 `24,168`→`24+168` 再 export；`run.py --periods` 接受 `+`。dry-run 确认默认 **1152** jobs（COMPACT=720）、electricity export `PERIODS=24+168`、`VAL_NUM_SEGMENTS=4`、PEMS seq_len=720 被跳过（除非 RUN_PEMS_SEQ720=1）：

```bash
NO_PUSH=1 DRY_RUN=1 bash scripts/slurm/submit_asyspecx_phase7.sh 2>&1 | tail -2
NO_PUSH=1 DRY_RUN=1 COMPACT=1 bash scripts/slurm/submit_asyspecx_phase7.sh 2>&1 | tail -1   # 720
```

先本地验证 auto-period discovery（train split only，不需 GPU）：
```bash
"$PYTHON" scripts/discover_periods.py --dataset electricity --data custom \
  --root_path ./dataset/electricity/ --data_path electricity.csv --seq_len 720 --enc_in 321 \
  --cycle 168 --method auto_acf --topk 3 --period_min 4 --period_max 0 --output /tmp/el.json
# 期望类似 [168, 24, 336]
```

有问题就修 + push：
```bash
glab push @"$PWD" --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz'
```

## 3. Canary

```bash
NO_PUSH=1 COMPACT=1 \
  DATASETS="weather electricity" SEEDS=2024 SEQ_LENS=96 PRED_LENS=96 \
  EPOCHS=1 PATIENCE=1 VAL_NUM_SEGMENTS=4 OUTROOT=phase7_results/canary \
  bash scripts/slurm/submit_asyspecx_phase7.sh

while squeue -u "$USER" -h -o "%120j" | grep -q "^asx7_"; do sleep 60; done
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase7_results/canary
ROOT=phase7_results/canary CSV=phase7_results/canary/results.csv PYTHON="$PYTHON" bash scripts/run_phase7_selection.sh
"$PYTHON" scripts/summarize_phase7.py --csv phase7_results/canary/results.csv \
  --anchor_arm phase6_asx_period_multi --output_dir phase7_results/canary
```
Canary 要求：results.csv 含 `periods`(auto 臂显示发现的周期)/`val_mse_seg0..3`/`cut_freq`；auto_acf 臂 periods 非默认；learned_clip 臂有 `eta_mean` 诊断；summary 标题 `Phase 7-Breakthrough Summary`；failed=0；无 NaN。失败读 `logs/AsySpecX_phase7/*.log`，修复 push 重试。

## 4. 提交

推荐先 COMPACT（720 jobs），要存预测做 ensemble 就 `SAVE_PREDICTIONS=1`：
```bash
COMPACT=1 SAVE_PREDICTIONS=1 bash scripts/slurm/submit_asyspecx_phase7.sh
```
若某臂改善 >0.002，再跑 full：
```bash
bash scripts/slurm/submit_asyspecx_phase7.sh          # 1152 jobs
```
watcher（`autopush_asyspecx_phase7_watch.sh`）在 `asx7_` 清空后自动：aggregate → 与 `phase6_results/fullfield/results.csv` merge → `run_phase7_selection.sh` → `audit_phase5_selectors.py` → `summarize_phase7.py` →（SAVE_PREDICTIONS=1 时）`ensemble_predictions.py` → glab push → `mail -s "[asy2-0708v1] AsySpecX Phase7 Breakthrough done" yind7@outlook.com`。有 baseline 传 `BASELINE_CSV=...`；Phase6 csv 路径用 `PHASE6_CSV=...`（缺省则仅用 Phase7）。

## 5. 监控
```bash
squeue -u "$USER" -o "%.18i %.9P %.120j %.8T %.10M %.9l %.6D %R" | grep asx7_ || true
tail -f logs/autopush_asyspecx_phase7.log
```

## 6. watcher 兜底
```bash
"$PYTHON" scripts/slurm/aggregate_asyspecx_phase1.py --root phase7_results/breakthrough
"$PYTHON" scripts/merge_results.py --csvs phase6_results/fullfield/results.csv,phase7_results/breakthrough/results.csv --output phase7_results/merged/results.csv
ROOT=phase7_results/merged CSV=phase7_results/merged/results.csv PYTHON="$PYTHON" bash scripts/run_phase7_selection.sh
"$PYTHON" scripts/audit_phase5_selectors.py --csv phase7_results/merged/results.csv \
  --selected_files "selected_unrestricted_mean.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv" \
  --output_dir phase7_results/merged
"$PYTHON" scripts/summarize_phase7.py --csv phase7_results/merged/results.csv \
  --selected_csv phase7_results/merged/selected_unrestricted_mean.csv \
  --anchor_arm phase6_asx_period_multi --output_dir phase7_results/merged
# 若存了预测:
"$PYTHON" scripts/ensemble_predictions.py --pred_dir phase7_results/breakthrough/predictions --mode simplex_val \
  --output_csv phase7_results/merged/ensemble_results.csv --summary phase7_results/merged/ensemble_summary.md

glab push @"$PWD" --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz'
{ echo "AsySpecX Phase 7 done $(date)"; echo "Summary: $PWD/phase7_results/merged/summary_phase7.md";
  sed -n '1,160p' phase7_results/merged/summary_phase7.md 2>/dev/null; } | mail -s "[asy2-0708v1] AsySpecX Phase7 Breakthrough done" yind7@outlook.com
```

## 7. 回报
- 提交/失败 jobs 数、缺失 dataset。
- merged CSV + selected + selector_audit + summary_phase7.md 路径。
- best fixed single-arm 是否仍 `phase6_asx_period_multi`，还是被某 phase7 臂超过（"New arms improving old best cells" 段）。
- unrestricted_mean selected MSE/MAE，vs best-single delta，vs oracle delta（只用 val 选）。
- auto_acf/auto_fft 发现的周期是否合理（electricity 期望含 168/24）。
- learned_clip 的 eta_mean / clip_active_fraction；patchlinear 是否帮到短 horizon。
- 若存预测：ensemble MSE vs best-single。
- 决策：是否有臂 >0.002 改善（→ full run）；若无，建议用 Phase 6 收尾。
- 是否已 push；邮件是否已发 `yind7@outlook.com`，标题 `asy2-0708v1`。
