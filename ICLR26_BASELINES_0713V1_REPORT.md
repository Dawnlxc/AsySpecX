# MixLinear / PhaseFormer ICLR 2026 复现与 asy1 对比（0713v1）

## 结论

本次已把两个官方实现放入 AsySpecX，完成 Weather、Electricity 上
`seq_len=720`、`pred_len in {96,192,336,720}` 的单卡 H100 正式运行。

- 24/24 个正式运行记录成功，相关 Slurm 作业均为 `COMPLETED 0:0`；MixLinear
  Weather 每个 horizon 跑 3 个 alpha，并且只按 validation loss 选择。
- 在 8 个 dataset-horizon 单元上，asy1 对 MixLinear 和 PhaseFormer 都取得更低
  MSE，即 16/16 次逐模型比较获胜。
- MixLinear 是最极端的小参数方案，实际仅 95--759 个 PyTorch 参数元素，但
  MSE 相对 asy1 平均高 0.022865。
- PhaseFormer 的分数更接近 asy1，MSE 平均差 0.007407；不过当前官方逐 horizon
  配置并非始终约 1K 参数，Electricity H192/H720 实际达到 273,160/275,998。
- Electricity H192/H720 上 asy1 同时拥有更低 MSE 和更少参数，严格支配当前
  PhaseFormer 官方配置；其他 PhaseFormer 单元则以更少参数换取小幅精度损失。

## 最终分数

asy1 是既有严格评测的两种子均值；两个 ICLR 2026 baseline 是本次官方代码的
单种子运行。表内 baseline 项为 `MSE / MAE`。

| Dataset | Horizon | asy1 MSE | MixLinear MSE / MAE | PhaseFormer MSE / MAE |
|---|---:|---:|---:|---:|
| Weather | 96 | **0.139562** | 0.179305 / 0.237158 | 0.148620 / 0.193560 |
| Weather | 192 | **0.181457** | 0.221808 / 0.274112 | 0.192744 / 0.236468 |
| Weather | 336 | **0.231628** | 0.267432 / 0.307729 | 0.245097 / 0.280126 |
| Weather | 720 | **0.304462** | 0.329212 / 0.350868 | 0.320802 / 0.335032 |
| Electricity | 96 | **0.128007** | 0.138606 / 0.233400 | 0.128465 / 0.220317 |
| Electricity | 192 | **0.145293** | 0.154282 / 0.248337 | 0.146529 / 0.236189 |
| Electricity | 336 | **0.160480** | 0.170718 / 0.264574 | 0.166116 / 0.258199 |
| Electricity | 720 | **0.197044** | 0.209489 / 0.297907 | 0.198819 / 0.284560 |

## 实际参数量

MixLinear 的频域权重为 complex tensor。`numel` 是 PyTorch 通常的参数元素计数；
括号内是把一个 complex 元素折算成两个 real scalar 后的数量，更适合与实值模型
比较存储自由度。PhaseFormer 和 asy1 均为实值参数。

| Dataset | Horizon | asy1 | MixLinear numel (real-equiv) | PhaseFormer actual numel |
|---|---:|---:|---:|---:|
| Weather | 96 | 13,722 | 195 (245) | 5,616 |
| Weather | 192 | 15,258 | 299 (397) | 3,702 |
| Weather | 336 | 17,562 | 455 (625) | 3,756 |
| Weather | 720 | 23,706 | 759 (1,121) | 3,900 |
| Electricity | 96 | 80,592 | 95 (141) | 3,666 |
| Electricity | 192 | 86,736 | 107 (153) | **273,160** |
| Electricity | 336 | 95,952 | 131 (189) | 3,756 |
| Electricity | 720 | 120,528 | 187 (277) | **275,998** |

参数判断：

- MixLinear 始终远小于 asy1，使用约 0.12%--3.20% 的 PyTorch numel，代价是所有
  8 个单元 MSE 更高。
- PhaseFormer Weather 使用 asy1 的约 16.45%--40.93% 参数；Electricity H96/H336
  使用约 4.55%/3.91%。
- PhaseFormer Electricity H192/H720 因官方配置采用 `latent_dim=128`，参数达到
  asy1 的 3.15x/2.29x，同时 MSE 仍略高，因此这两个点不是小模型 Pareto 点。

## 与论文数字的核对

下列列表均按 H96/H192/H336/H720 排列。

| Model / Dataset | 本次实际 MSE | 论文 MSE |
|---|---|---|
| MixLinear / Weather | 0.179305, 0.221808, 0.267432, 0.329212 | 0.170, 0.212, 0.257, 0.321 |
| MixLinear / Electricity | 0.138606, 0.154282, 0.170718, 0.209489 | 0.138, 0.154, 0.170, 0.209 |
| PhaseFormer / Weather | 0.148620, 0.192744, 0.245097, 0.320802 | 0.148, 0.193, 0.242, 0.309 |
| PhaseFormer / Electricity | 0.128465, 0.146529, 0.166116, 0.198819 | 0.129, 0.148, 0.165, 0.201 |

- PhaseFormer 论文结果是三种子平均；本次官方 TSL 入口固定 seed 2021，因此是单种子
  复现。MixLinear 使用官方 seed 2023。
- MixLinear 论文在 H720 报告 0.176K 参数；当前官方代码在 Electricity H720
  实例化为 187 numel（277 real-equiv），Weather H720 因官方 Weather 设置的
  period/lpf 不同而为 759 numel。
- PhaseFormer 论文效率表在 H96 报告 Weather 308、Electricity 1.156K 参数；当前
  官方 TSL commit 加上公开运行配置实际实例化为 5,616/3,666。本文对项目比较采用
  实际 `sum(p.numel() for p in model.parameters() if p.requires_grad)`，论文数只作为
  引用值，不能替代运行配置的审计值。

## 协议与选择规则

- 数据：现有 AsySpecX `dataset/weather/weather.csv` 和
  `dataset/electricity/electricity.csv`。
- 输入长度 720，预测长度 96/192/336/720，standard custom 7:1:2 split，features M。
- 每个 Slurm 作业一张 H100；MixLinear seed 2023，PhaseFormer seed 2021。
- MixLinear Weather 按官方 alpha 网格 `{0.01, 0.5, 0.99}` 运行；最终选择依次为
  H96 `0.99`、H192 `0.99`、H336 `0.99`、H720 `0.01`，只看 validation loss，
  不看 test MSE。
- PhaseFormer 采用公开逐 horizon 推荐配置。Electricity H192/H720 的高参数量来自
  该配置本身，并非参数统计把 channel 数重复乘入。
- asy1 数字来自既有严格 held-out test 两种子均值，不是本轮重新选择。

## 官方代码、兼容性修补与审计

- MixLinear：`aitianma/MixLinear`，commit
  `71b5db62e38b1cb108faa1e1d4687287b4568f3b`。
- PhaseFormer TSL：`neumyor/PhaseFormer_TSL`，commit
  `ed1db61c6abfa9326d5ca2a56c6c4ba53ea592ab`；逐 horizon 配置来源仓库
  `neumyor/PhaseFormer` commit `ff14c16fab136aef60048b20f2ef0d3bc6d86fc7`。
- MixLinear 仅做 pandas 新版本兼容修补，并移除每个 forward 的 shape debug print；
  PhaseFormer 将 NumPy 2 中已删除的 `np.Inf` 改为 `np.inf`。
- PhaseFormer DataLoader 改为 `num_workers=0`，规避同节点多作业时
  `Shared memory manager connection has timed out`；样本、shuffle、模型和优化器设置
  不变。
- 成功作业：28725280--28725295、28725620--28725627，全部 `COMPLETED 0:0`。

## 产物位置

远程根目录：

```text
/scratch3/lin250/bldgFM/DUBABA/AsySpecX/baselines_2026/
```

结果目录：

```text
/scratch3/lin250/bldgFM/DUBABA/AsySpecX/baseline_2026_results/iclr26_baselines_0713v1/
```

关键文件：

```text
selected_results.csv
all_runs.csv
summary.md
submissions.tsv
phase_retry_submissions.tsv
```

复现脚本：

```text
baselines_2026/scripts/audit_model_params.py
baselines_2026/scripts/write_run_summary.py
baselines_2026/scripts/aggregate_baseline_2026.py
baselines_2026/scripts/slurm/baseline_2026_cell.sbatch
baselines_2026/scripts/slurm/submit_baseline_2026.sh
```

官方资料：

- MixLinear ICLR 2026: https://openreview.net/forum?id=QUj0KuCumD
- MixLinear code: https://github.com/aitianma/MixLinear
- PhaseFormer ICLR 2026: https://openreview.net/forum?id=Lk9SqMQzhX
- PhaseFormer code: https://github.com/neumyor/PhaseFormer_TSL
