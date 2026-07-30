# Prompt for Slurm-side Codex: AsySpecX Phase 9 SafeRoute

你在 Petrichor/Slurm 登录节点上操作。目标：从 glab 拉取 `AsySpecX`，严格检查 Phase 9 SafeRoute 实现、checkpoint manifest、提交脚本和数据；先跑无邮件 canary，再启动完整的条件流水线。脚本有问题就修复并 push 回 glab。结果完成后汇总并 push 回 glab，最后邮件通知 `yind7@outlook.com`。

邮件标题前缀固定为 `asy2-0711v1`。本轮只能有两封邮件：

1. 正式 full pipeline 提交时一封 START。
2. 所有实际运行阶段结束或 gate 停止后一封 DONE。

Canary 必须 `NO_MAIL=1`。不得发送中间进度邮件，不得为 quick/OOF 子阶段重复发送 START，不得在 watcher 已写 `.done_mail_sent` 后手工再发 DONE。

## 1. 拉代码与环境

```bash
mkdir -p /scratch3/lin250/bldgFM/DUBABA
cd /scratch3/lin250/bldgFM/DUBABA
glab pull @/mnt/scratch/CRUISE/Du/code/AsySpecX
cd AsySpecX

export ACCOUNT="${ACCOUNT:-od-241336}"
export PARTITION="${PARTITION:-h24gpu}"
export CONDA_ROOT="${CONDA_ROOT:-/scratch3/lin250/conda_envs}"
export CONDA_ENV="${CONDA_ENV:-tsfm}"
export PATH="${CONDA_ROOT}/${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ROOT}/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export MAIL_TO="yind7@outlook.com"
export MAIL_SUBJECT="asy2-0711v1"

command -v glab sbatch squeue mail
```

正式提交前 `glab`、Slurm 命令和 `mail` 必须全部存在；缺 `mail` 时提交脚本会在提交任何正式 job 前失败，不能静默跳过 START 邮件。

确认数据、旧结果中的 auto-period cache、冻结 checkpoint 都还在。Phase 9 不重新训练 Phase 6/7/8 full experts，缺 checkpoint 不允许静默跳过：

```bash
ls -lh dataset/weather/weather.csv dataset/electricity/electricity.csv \
  dataset/traffic/traffic.csv dataset/ETT-small/ETTh1.csv dataset/ETT-small/ETTm1.csv \
  dataset/PEMS/PEMS04.npz dataset/PEMS/PEMS08.npz

find phase7_results/breakthrough/auto_periods phase8_results/hydra/auto_periods \
  -type f -name '*.json' | sort | head -40
find checkpoints -type f -name checkpoint.pth | wc -l
```

如果数据路径不同，只修复 Phase 9 manifest 生成时的 `--data_root`/脚本路径，不改数据切分和 benchmark 逻辑。若 checkpoint 路径模式与实际 setting 不同，先核对旧 Slurm log 中的完整 setting，再修 `scripts/build_phase9_manifest.py`；不要用错误 checkpoint 顶替。

## 2. 静态检查与测试

```bash
for s in \
  scripts/run_phase9_headroom.sh \
  scripts/run_phase9_router_quick.sh \
  scripts/run_phase9_router_oof.sh \
  scripts/run_phase9_oof_experts.sh \
  scripts/slurm/asyspecx_phase9_run.sbatch \
  scripts/slurm/submit_asyspecx_phase9.sh \
  scripts/slurm/autopush_asyspecx_phase9_watch.sh; do
  bash -n "$s" || exit 1
done

grep -RIn "qsub\|#PBS\|/srv/scratch/cruise/du/code/tf-optimizer" \
  router scripts/build_phase9_manifest.py scripts/build_router_meta.py \
  scripts/audit_router_headroom.py scripts/train_safe_router.py \
  scripts/evaluate_safe_router.py scripts/build_router_oof_meta.py \
  scripts/run_phase9* scripts/slurm/*phase9* || true

python -m py_compile \
  router/*.py \
  scripts/build_phase9_manifest.py \
  scripts/build_router_meta.py \
  scripts/audit_router_headroom.py \
  scripts/train_safe_router.py \
  scripts/evaluate_safe_router.py \
  scripts/build_router_oof_meta.py \
  scripts/summarize_phase9.py

python -m unittest tests/test_asyspecx_phase9.py
python -m unittest \
  tests/test_asyspecx_phase1.py \
  tests/test_asyspecx_phase4.py \
  tests/test_asyspecx_phase5.py \
  tests/test_asyspecx_phase6.py \
  tests/test_asyspecx_phase7.py \
  tests/test_asyspecx_phase8.py
```

检查 router backend。默认是 xgboost；若 tsfm 环境没有 xgboost，不要让 48 个 quick jobs 一起失败，可设置受支持的 fallback 并在最终报告说明：

```bash
if python -c 'import xgboost' 2>/dev/null; then
  export ROUTER_BACKEND=xgboost
else
  export ROUTER_BACKEND=hist_gradient_boosting
  echo "[warn] xgboost missing; using $ROUTER_BACKEND"
fi
```

Slurm 默认 `EXPERT_DEVICE_POLICY=resident`，避免 traffic 的大 individual experts 每个 batch 反复搬运。若 canary 明确 CUDA OOM，再设置 `EXPERT_DEVICE_POLICY=one_at_a_time`，并在最终报告说明性能折衷。

确认邮件只存在于正式 submit 的 START 和 watcher 的 DONE 路径：

```bash
grep -RIn "mail -s" scripts/slurm/*phase9* scripts/run_phase9* || true
```

Dry-run 默认应为 48 个 headroom jobs：

```bash
NO_MAIL=1 NO_WATCH=1 DRY_RUN=1 STAGE=headroom \
  bash scripts/slurm/submit_asyspecx_phase9.sh | tail -2
```

## 3. 严格 manifest canary

先验证一个真实 cell 的五专家、三 seeds checkpoint：

```bash
mkdir -p phase9_results/preflight
python scripts/build_phase9_manifest.py \
  --dataset weather --seq_len 96 --pred_len 96 \
  --expert_seeds 2024,2025,2026 \
  --experts anchor,dlinear,split_clip,individual_revin,individual_period \
  --checkpoint_root checkpoints --data_root dataset --repo_root . \
  --output phase9_results/preflight/weather_sl96_pl96.json --strict 1
```

要求：缺任意请求 checkpoint 都失败；manifest 中所有 expert 的 dataset/seq_len/pred_len/enc_in 一致；anchor 存在；加载 checkpoint 时 state dict 必须 strict compatible。

若 auto-ACF cache 缺失，只能用 TRAIN split 重建：

```bash
python scripts/discover_periods.py \
  --dataset weather --data custom --root_path ./dataset/weather/ --data_path weather.csv \
  --seq_len 96 --enc_in 21 --cycle 144 --method auto_acf --topk 3 \
  --period_min 4 --period_max 0 --fallback_periods 144 \
  --output phase8_results/hydra/auto_periods/weather_sl96_auto_acf.json
```

不要用 val/test 发现周期。

## 4. 无邮件 canary

先跑一个 weather 96/96 headroom job：

```bash
NO_MAIL=1 NO_WATCH=1 NO_PUSH=1 STAGE=headroom \
  DATASETS=weather SEQ_LENS=96 PRED_LENS=96 \
  EXPERT_SEEDS=2024,2025,2026 \
  OUTROOT=phase9_results/canary \
  bash scripts/slurm/submit_asyspecx_phase9.sh

while squeue -u "$USER" -h -o "%120j" | grep -q '^asx9h_'; do
  squeue -u "$USER" -o "%.18i %.9P %.100j %.8T %.10M %.9l %.6D %R" | grep asx9h_ || true
  sleep 60
done

cat phase9_results/canary/job_status/headroom/weather_sl96_pl96.json
sed -n '1,120p' phase9_results/canary/headroom_cells/weather_sl96_pl96/audit/headroom_audit.md
```

Canary 要求：

- job status 为 `ok`。
- compact meta 只有 `part-*.npz` + `manifest.json`；不得生成 `[N,K,H,C]` 全量预测文件。
- `manifest.json` 记录 `split=test`、`full_predictions_saved=false`、feature names、expert names。
- headroom audit 明确包含 `ANALYSIS ONLY -- test labels used -- not a valid model result`。
- sample/cell/block/sample-block oracle 均有限数。
- `headroom_by_horizon.csv` 已生成，oracle expert choice counts 完整。

再跑一个 quick mechanics canary，仍然不发邮件：

```bash
NO_MAIL=1 NO_WATCH=1 NO_PUSH=1 STAGE=quick \
  DATASETS=weather SEQ_LENS=96 PRED_LENS=96 \
  EXPERT_SEEDS=2024,2025,2026 ROUTER_BACKEND="$ROUTER_BACKEND" \
  ROUTER_MIN_SAMPLES=32 OUTROOT=phase9_results/canary \
  bash scripts/slurm/submit_asyspecx_phase9.sh

while squeue -u "$USER" -h -o "%120j" | grep -q '^asx9q_'; do sleep 60; done
cat phase9_results/canary/job_status/quick/weather_sl96_pl96.json
cat phase9_results/canary/quick/weather_sl96_pl96/routed_results.csv
cat phase9_results/canary/quick/weather_sl96_pl96/router/training_summary.json
```

要求 `test_labels_used=false` / `test_labels_used_for_decision=false`；alpha 在 `[0,1]`；不确定样本 exact anchor fallback；无 NaN/Inf。

若检查或 canary 失败，读 `logs/slurm/asx9*.err/.out`，修复后重跑测试/canary。修复代码后立即 push 回 glab：

```bash
glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='*.npz' --exclude='*.npy'
```

## 5. 正式条件流水线

清楚区分三阶段 gate：

1. Headroom：48 jobs。若 sample-block oracle 相对 best fixed 改善 `<0.004`，停止，不提交 quick。
2. Quick：最多再 48 jobs。若相对 anchor 改善 `<0.002`，停止，不提交 rolling OOF。
3. 若 quick 任一数据集退化 `>0.003`，停止并报告需要更保守 calibration，不自动进入 OOF。
4. Rolling OOF：仅上述 gate 通过后最多再 48 jobs。使用一个 OOF seed，60/20 与 80/20 expanding folds，purge 默认 pred_len。

OOF 每个 fold 的 scaler 与 auto-ACF periods 必须只用该 fold 可见训练前缀拟合；`oof_folds.json` 中必须满足 `scaler_fit_observations <= first_validation_origin`，随后预测统一转换回 official-train standardized space 再构造标签。

正式启动只执行一次下面命令。它发送唯一 START 邮件，并启动 detached watcher；watcher 内部提交 quick/OOF 时强制 `NO_MAIL=1 NO_WATCH=1`：

```bash
MAIL_TO=yind7@outlook.com MAIL_SUBJECT=asy2-0711v1 \
  ROUTER_BACKEND="$ROUTER_BACKEND" AUTO_QUICK=1 AUTO_OOF=1 \
  OUTROOT=phase9_results/main \
  bash scripts/slurm/submit_asyspecx_phase9.sh
```

START 标题应类似：

```text
[asy2-0711v1] Phase9 SafeRoute START (48 jobs)
```

不得手工再发 START。

## 6. 监控，不发邮件

```bash
squeue -u "$USER" -o "%.18i %.9P %.100j %.8T %.10M %.9l %.6D %R" \
  | grep -E 'asx9[hoq]_' || true
tail -f logs/autopush_asyspecx_phase9.log
```

Watcher 自动执行：

- 等 `asx9h_` 清空。
- 聚合全局 `headroom_audit.md/headroom_by_cell.csv/headroom_by_dataset.csv`。
- 按 headroom gate 决定是否提交 `asx9q_`。
- quick 通过且无严重数据集退化时决定是否提交 `asx9o_`。
- 生成 `phase9_results/main/summary/summary_phase9_saferoute.md` 和 `phase9_summary.json`。
- 生成 `headroom_by_cell.csv`、`headroom_by_dataset.csv`、`headroom_by_horizon.csv`，以及 routed 的 dataset/seq_len/pred_len、activation 与 paired 统计。
- 对每个提交 manifest 严格核对 matching `job_status`；缺失、失败或陈旧 status 都令流水线失败。
- push summaries、CSV、manifest、router metadata 和 logs 回 glab；compact meta/full arrays 不 push。
- 最后只发一封 DONE，并创建 `phase9_results/main/.done_mail_sent`。

## 7. Watcher 失败时兜底

先确认没有 Phase 9 jobs：

```bash
squeue -u "$USER" -h -o "%120j" | grep -E '^asx9[hoq]_' || true
```

重新汇总已有结果：

```bash
meta_paths="$(find phase9_results/main/headroom_cells -type d -name meta_test | sort | paste -sd, -)"
python scripts/audit_router_headroom.py \
  --meta "$meta_paths" --current_validation_selected 0.333805 \
  --output_dir phase9_results/main/headroom

python scripts/summarize_phase9.py \
  --headroom_dir phase9_results/main/headroom \
  --quick_glob 'phase9_results/main/quick/*/routed_results.csv' \
  --oof_glob 'phase9_results/main/oof/*/routed_results.csv' \
  --output_dir phase9_results/main/summary

glab push @"$PWD" \
  --exclude='.git' --exclude='dataset' --exclude='datasets' --exclude='checkpoints' \
  --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='*.npz' --exclude='*.npy' --exclude='phase9_results/*/*/meta_*'
```

只有 watcher 没有发 DONE、且 `.done_mail_sent` 不存在时，才手工发送一次：

```bash
if [ ! -f phase9_results/main/.done_mail_sent ]; then
  {
    echo "AsySpecX Phase 9 SafeRoute finished on $(hostname) at $(date)."
    echo "Repo: $PWD"
    echo "Summary: $PWD/phase9_results/main/summary/summary_phase9_saferoute.md"
    sed -n '1,180p' phase9_results/main/summary/summary_phase9_saferoute.md
  } | mail -s "[asy2-0711v1] Phase9 SafeRoute DONE (manual fallback)" yind7@outlook.com \
    && touch phase9_results/main/.done_mail_sent
fi
```

## 8. 最终回报

最终请报告：

- Headroom/quick/OOF 分别提交多少 jobs、失败多少及失败 cell。
- 总 wall-clock 时间与总 GPU-job 数。
- best fixed、cell/sample/horizon-block/sample-block oracle；明确 oracle 仅分析。
- headroom gate 是否通过。
- quick routed MSE/MAE、相对 anchor 改善、fallback、mean alpha、false/catastrophic activation rate。
- 各数据集是否有退化超过 `0.003`。
- OOF 是否按 gate 运行；若运行，报告同样指标及是否达到 distillation gate。
- `summary_phase9_saferoute.md`、headroom CSV、routed CSV 路径。
- backend 是 xgboost 还是 fallback。
- 是否 push 回 glab。
- START/DONE 是否各发送一次给 `yind7@outlook.com`，标题前缀是否为 `asy2-0711v1`。

不得把 test oracle 写成有效模型结果；不得因为 oracle 好看绕过 validation/OOF gate。
