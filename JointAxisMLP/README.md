# JointAxisMLP — JointMLP with JA v4 cross-channel

Reference implementation of **JointMLP** with the **JointAxisTWCM v4** cross-channel backend (per-bin per-frame gain `g_{k, t'}`). Long-term multivariate time series forecasting.

JointMLP combines the TQNet MLP backbone with a frequency-conditioned JA cross-channel mixer, replacing both `temporalQuery[cycle_index]` and the flat channel `MultiheadAttention` from TQNet. This subproject ships only the **v4** JA backend; the dispatcher that switched between v3–v7 during upstream development has been trimmed.

## Layout

```
models/
  JointMLP.py                MLP backbone + RIN + JA-v4 cross-channel (single Model)
  JointAxisTWCMv4.py         the JA v4 implementation imported by JointMLP
exp/exp_main.py              train / eval loop
data_provider/               ETT, custom CSV, Solar, PEMS dataset loaders
layers/                      shared layers (RevIN, attention families, embeddings, …)
utils/                       metrics, masking, time features, tools
run.py                       argparse entry point (carries the --jmlp_* flags)
scripts/
  _common.sh                 shared dataset metadata (load_dataset)
  JointMLP/<Dataset>.sh      per-dataset sweep scripts
  JointMLP/_template.sh      shared template sourced by each per-dataset script
  slurm/
    baseline.sbatch          generic per-job slurm launcher (1 GPU, 12h, account=OD-241336, env=tsfm)
    submit_all.sh            wrapper for --smoke / --full / --dataset Y
acf_plot.ipynb               exploratory notebook (autocorrelation diagnostics)
```

Runtime-generated directories `logs/`, `results/`, `checkpoints/`, `dataset/` are gitignored.

## Benchmark protocol

- **Datasets**: ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic, PEMS03, PEMS04, PEMS07, PEMS08 (additional air-quality datasets defined in `scripts/_common.sh`)
- **Seeds**: `{2026, 2027}` per run
- **Lookback sweep**: `seq_len ∈ {96, 336, 720}` (PEMS uses fixed `seq_len = 96`)
- **Pred-len sweep**: `{96, 192, 336, 720}` for non-PEMS, `{12, 24, 48, 96}` for PEMS

## Key hyperparameters (`--jmlp_*`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--jmlp_window` | `-1` (auto: `max(8, min(seq_len/4, 64))`) | STFT window length `W` |
| `--jmlp_stride` | `-1` (auto: `W/2`) | STFT hop |
| `--jmlp_rank` | `8` | rank `R` of `H = A diag(g) Bᵀ` |
| `--jmlp_delta_hidden` | `64` | δ-MLP hidden dim |
| `--jmlp_gate_init` | `-1.0` | JA gate logit init |
| `--jmlp_use_entropy_gate` | `1` | per-channel entropy gate on/off |
| `--jmlp_innovation_only` | `0` | apply JA only on `x − moving_avg(x)` |

## Environment

The slurm template assumes the `tsfm` conda env (Python 3.10, torch 2.5.1+cu124, pandas 2.0.3). To recreate elsewhere:

```bash
conda create -n tsfm python=3.10 -y
conda activate tsfm
pip install -r requirements.txt
```

## Data

Standard LTSF datasets (ETTh1/2, ETTm1/2, weather, electricity, traffic, PEMS03/04/07/08) from the [Autoformer / SCINet Google Drive bundle](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view). Place files under `dataset/<subdir>/`, e.g. `dataset/ETT-small/ETTh1.csv`. Per-dataset `subdir` defined in `scripts/_common.sh::load_dataset`.

## Running

### Local single run
```bash
conda activate tsfm
bash scripts/JointMLP/ETTh1.sh           # full sweep for ETTh1
SMOKE=1 bash scripts/JointMLP/ETTh1.sh   # restrict to sl=96, pl=96
```

### Slurm
```bash
bash scripts/slurm/submit_all.sh --smoke                 # 2 representative datasets
bash scripts/slurm/submit_all.sh --full                  # all datasets
bash scripts/slurm/submit_all.sh --dataset ETTh1         # one specific dataset
```

Each slurm job runs the full `seed × sl × pl` sweep for one dataset sequentially within a 12h, 1-GPU, 64GB allocation. The slurm template resolves the repo root relative to the script's own location, so it's portable across users / paths.

## Acknowledgement

Training harness adapted from [TQNet](https://github.com/ACAT-SCUT/TQNet) (Lin et al., ICML 2025). The JointMLP / JointAxisTWCM family was developed alongside [AsySpecX](../AsySpecX/).
