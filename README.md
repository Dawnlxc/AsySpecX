# AsySpecX & JointMLP

Two related models for multivariate long-term time series forecasting, sharing a single TQNet-derived training / evaluation harness.

| Model | File | Idea |
| --- | --- | --- |
| **AsySpecX** | [`models/AsySpecX.py`](models/AsySpecX.py) | Asymmetric Spectral Transfer — low-rank `H = A diag(g_m) Bᵀ` with per-band gates, applied in the frequency domain. Paper in preparation. |
| **JointMLP (JA v4)** | [`models/JointMLP.py`](models/JointMLP.py) + [`models/JointAxisTWCMv4.py`](models/JointAxisTWCMv4.py) | TQNet MLP backbone + frequency-conditioned JointAxisTWCM **v4** (per-bin per-frame gain `g_{k, t'}`) replacing TQNet's `temporalQuery[cycle_index]` and channel `MultiheadAttention`. |

Pick a model via `--model AsySpecX` or `--model JointMLP`.

## Layout

```
models/
  AsySpecX.py          ← model A (single Model class)
  JointMLP.py          ← model B (MLP backbone + RIN + JA-v4 cross-channel)
  JointAxisTWCMv4.py   ← JA v4 backend imported by JointMLP
exp/exp_main.py        ← shared train/eval loop; model_dict registers both
data_provider/         ← ETT / custom CSV / Solar / PEMS loaders
layers/                ← shared layers (RevIN, attention families, embeddings, …)
utils/                 ← metrics, masking, time features, tools
run.py                 ← argparse entry point (carries flags for both models)
requirements.txt       ← shared dependency pins

scripts/
  _common.sh                       ← load_dataset + apply_asyspecx_overrides
  AsySpecX/<Dataset>.sh            ← per-dataset launcher (sources _template.sh)
  AsySpecX/_template.sh            ← shared template for the AsySpecX sweep
  JointMLP/<Dataset>.sh            ← per-dataset launcher
  JointMLP/_template.sh            ← shared template for the JointMLP sweep
  slurm/baseline.sbatch            ← `sbatch ... baseline.sbatch <MODEL> <DATASET>`
  slurm/submit_all.sh              ← `--model X --smoke | --full | --dataset Y`

analysis_exp/          ← post-hoc analysis & visualization scripts (AsySpecX-focused)
Figures/               ← published figures (carried over from TQNet)
acf_plot.ipynb         ← exploratory autocorrelation notebook
```

Runtime-generated dirs `logs/`, `results/`, `checkpoints/`, `dataset/`, `figures/`, `probe/` are gitignored.

## Environment

Python 3.10 + PyTorch 2.5.1+cu124. On the HPC node we use the `tsfm` conda env at `/scratch3/lin250/conda_envs/tsfm`. To recreate elsewhere:

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
bash scripts/AsySpecX/ETTh1.sh           # full AsySpecX sweep for ETTh1
SMOKE=1 bash scripts/AsySpecX/ETTh1.sh   # AsySpecX, sl=96 pl=96 only
bash scripts/JointMLP/ETTh1.sh           # full JointMLP (JA v4) sweep for ETTh1
```

### Slurm
```bash
bash scripts/slurm/submit_all.sh --model AsySpecX --smoke
bash scripts/slurm/submit_all.sh --model JointMLP --full
bash scripts/slurm/submit_all.sh --model AsySpecX --dataset ETTh1
```

Each slurm job runs the full `seed × sl × pl` sweep for one (model, dataset) sequentially within a 12h, 1-GPU, 64GB allocation. The slurm template resolves the repo root relative to the script's own location, so it's portable across users / paths.

## Benchmark protocol

- **Seeds**: `{2026, 2027}` per configuration
- **Lookback sweep**: `seq_len ∈ {96, 720}` (AsySpecX) / `{96, 336, 720}` (JointMLP); PEMS fixed at `seq_len = 96`
- **Pred-len sweep**: `{96, 192, 336, 720}` for non-PEMS, `{12, 24, 48, 96}` for PEMS
- **AsySpecX hyperparameters**: probe-driven defaults — `gate_init=0`, `gate_max=1.0`, `rank=2` for high-channel datasets, `rank=8` for small-channel. See `apply_asyspecx_overrides` in `scripts/_common.sh`.
- **JointMLP hyperparameters**: `--jmlp_window` (auto), `--jmlp_stride` (auto), `--jmlp_rank=8`, `--jmlp_delta_hidden=64`, `--jmlp_gate_init=-1.0`, entropy gate on, innovation-only off. See `scripts/JointMLP/_template.sh`.

## License & attribution

Apache 2.0 — see [`LICENSE`](LICENSE). The training/evaluation harness is adapted from [TQNet](https://github.com/ACAT-SCUT/TQNet) (Lin et al., ICML 2025).
