# AsySpecX — Asymmetric Spectral Transfer for Time Series Forecasting

Reference implementation of **AsySpecX** (paper in preparation), a long-term multivariate time series forecaster built around asymmetric spectral transfer.

The training/evaluation harness started as a fork of [TQNet](https://github.com/ACAT-SCUT/TQNet) (ICML 2025) and has been trimmed to a single-model layout. Baselines are intentionally **not** shipped in this repo — point a baseline's own repository at the same datasets to reproduce comparison numbers.

## Layout

```
models/AsySpecX.py         the model — defines class Model(nn.Module)
exp/exp_main.py            train / eval loop
data_provider/             ETT, custom CSV, Solar, PEMS dataset loaders
layers/                    shared layers (RevIN, attention families, embeddings, …)
utils/                     metrics, masking, time features, tools
run.py                     argparse entry point
scripts/
  _common.sh               shared dataset metadata + apply_asyspecx_overrides
  AsySpecX/<Dataset>.sh    per-dataset sweep scripts
  AsySpecX/_template.sh    shared template sourced by each per-dataset script
  slurm/
    baseline.sbatch        generic per-job slurm launcher (1 GPU, 12h, account=OD-241336, env=tsfm)
    submit_all.sh          wrapper for --smoke / --full / --dataset Y
analysis_exp/              post-hoc analysis & visualization scripts
acf_plot.ipynb             exploratory notebook (autocorrelation diagnostics)
```

The following directories are produced at runtime and are gitignored: `logs/`, `results/`, `checkpoints/`, `dataset/`, `figures/`.

## Benchmark protocol

- **Datasets**: ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic, PEMS03, PEMS04, PEMS07, PEMS08 (plus several optional air-quality datasets defined in `scripts/_common.sh`)
- **Seeds**: `{2026, 2027}` per run
- **Lookback sweep**: `seq_len ∈ {96, 720}` (PEMS uses fixed `seq_len = 96`)
- **Pred-len sweep**: `{96, 192, 336, 720}` for non-PEMS datasets, `{12, 24, 48, 96}` for PEMS
- **Hyperparameters**: probe-driven AsySpecX defaults (`gate_init=0`, `gate_max=1.0`; `rank=2` for high-channel datasets, `rank=8` for small-channel) — see `apply_asyspecx_overrides` in `scripts/_common.sh`

## Environment

The slurm template assumes the `tsfm` conda env (Python 3.10, torch 2.5.1+cu124, pandas 2.0.3). To recreate elsewhere:

```bash
conda create -n tsfm python=3.10 -y
conda activate tsfm
pip install -r requirements.txt
```

## Data

The standard LTSF datasets (ETTh1/2, ETTm1/2, weather, electricity, traffic, PEMS03/04/07/08) come from the [Autoformer / SCINet Google Drive bundle](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view). Place the CSVs / npz files under `dataset/<subdir>/`, e.g. `dataset/ETT-small/ETTh1.csv`. The `subdir` for each dataset key is defined in `scripts/_common.sh::load_dataset`.

## Running

### Local single run
```bash
conda activate tsfm
bash scripts/AsySpecX/ETTh1.sh           # full sweep for ETTh1
SMOKE=1 bash scripts/AsySpecX/ETTh1.sh   # restrict to sl=96, pl=96
```

### Slurm
```bash
bash scripts/slurm/submit_all.sh --smoke                 # 2 representative datasets
bash scripts/slurm/submit_all.sh --full                  # all datasets
bash scripts/slurm/submit_all.sh --dataset ETTh1         # one specific dataset
```

Each slurm job runs the full `seed × sl × pl` sweep for one dataset sequentially within a 12h, 1-GPU, 64GB allocation. The slurm template resolves the repo root relative to the script's own location, so it's portable across users / paths.

## Acknowledgement

Training harness adapted from [TQNet](https://github.com/ACAT-SCUT/TQNet) (Lin et al., ICML 2025).
