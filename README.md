# AsySpecX & JointMLP

Two promising research models for long-term multivariate time series forecasting, plus the 10 published baselines they compete against, sharing a single TQNet-derived training/evaluation harness.

→ **See [`RESULTS.md`](RESULTS.md) for the full comparison vs baselines.**

## The two promising lines

### 1. JointMLP (JA v4)  —  `models/JointMLP.py` + `models/JointAxisTWCMv4.py`

TQNet MLP backbone + frequency-conditioned JA cross-channel mixer (v4: per-bin per-frame gain `g_{k, t'}`). Replaces TQNet's `temporalQuery[cycle_index]` and the flat channel `MultiheadAttention`.

**Wins on standard MTSF**: 8 / 44 settings vs the 10 baselines (tied with TQNet and CycleNet for #2 baseline-or-active winner; iTransformer leads with 12). Strongest on `ETTm1` (sweeps pl ∈ {96, 192, 336}), `weather pl={96, 336}`, `PEMS03 pl={12, 24}`, `ETTh1 pl=192`. See [`RESULTS.md`](RESULTS.md#jointmlp-line--8-wins--44-settings).

### 2. AsySpecX  —  `models/AsySpecX.py`

Asymmetric Spectral Transfer: low-rank `H = A diag(g_m) Bᵀ` with per-band gates, applied in the frequency domain. Paper in preparation.

Best at long lookback (sl=720): wins **19 / 28** head-to-head cells against JointMLP on the standard MTSF suite — ETTh2 sweep (pl ∈ {96, 192, 336, 720}), traffic sweep, electricity sweep. At sl=96 it doesn't yet beat the strongest baselines.

## All models in the repo

| Category | File | Status |
| --- | --- | --- |
| **Active — AsySpecX** | [`models/AsySpecX.py`](models/AsySpecX.py) | Frequency-domain asymmetric spectral transfer; strongest at long lookback |
| **Active — JointMLP** | [`models/JointMLP.py`](models/JointMLP.py) | TQNet MLP backbone + JA cross-channel |
| | [`models/JointAxisTWCMv4.py`](models/JointAxisTWCMv4.py) | JA v4 backend imported by JointMLP |
| **Baselines (10)** | `TQNet`, `CycleNet`, `DLinear`, `iTransformer`, `PatchTST`, `FITS`, `FreTS`, `FilterNet`, `SparseTSF`, `MixLinear` | Comparison reference |

Pick a model via `--model <Name>` (any key in `exp/exp_main.py::model_dict`).

## Layout

```
models/                     13 model files (3 active + 10 baselines)
exp/exp_main.py             shared train/eval loop; model_dict registers every model
data_provider/              ETT / custom CSV / Solar / PEMS dataset loaders
layers/                     shared layers (RevIN, attention families, embeddings, …)
utils/                      metrics, masking, time features, tools
run.py                      argparse entry point (carries flags for all models)
requirements.txt            shared dependency pins

scripts/
  _common.sh                load_dataset + per-model apply_<name>_overrides
  AsySpecX/<Dataset>.sh     per-dataset sweep (sources _template.sh)
  JointMLP/<Dataset>.sh
  TQNet/<Dataset>.sh        ... (one subdir per baseline)
  slurm/baseline.sbatch     sbatch ... baseline.sbatch <MODEL> <DATASET>
  slurm/submit_all.sh       --model X / --all-baselines / --smoke / --full / --dataset Y

analysis_exp/               post-hoc analysis & visualization scripts
Figures/                    published figures (carried over from TQNet upstream)
acf_plot.ipynb              exploratory autocorrelation notebook
RESULTS.md                  comparison vs baselines — see top of this README
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
bash scripts/JointMLP/ETTm1.sh           # full sweep for one (model, dataset)
SMOKE=1 bash scripts/JointMLP/ETTm1.sh   # restrict to sl=96, pl=96
bash scripts/TQNet/ETTh1.sh              # baselines work the same way
```

### Slurm
```bash
bash scripts/slurm/submit_all.sh --model JointMLP --smoke              # one model, smoke
bash scripts/slurm/submit_all.sh --model AsySpecX --full               # one model, full sweep
bash scripts/slurm/submit_all.sh --model TQNet --dataset ETTh1         # one (model, dataset)
bash scripts/slurm/submit_all.sh --all-baselines --full                # 10 baselines × 11 datasets
bash scripts/slurm/submit_all.sh --all-baselines --smoke               # 5 reps × 2 datasets
```

Each slurm job runs the full `seed × sl × pl` sweep for one (model, dataset) sequentially within a 12 h, 1-GPU, 64 GB allocation. The slurm template resolves the repo root from the script's own location, so it's portable.

## Benchmark protocol

- **Seeds**: `{2026, 2027}` for the active lines; baseline sweep used seed `2026` only
- **Lookback sweep**: `seq_len ∈ {96, 336, 720}` (PEMS fixed at `seq_len = 96`)
- **Pred-len sweep**: `{96, 192, 336, 720}` for non-PEMS, `{12, 24, 48, 96}` for PEMS

## License & attribution

Apache 2.0 — see [`LICENSE`](LICENSE). The training/evaluation harness and baseline implementations are adapted from [TQNet](https://github.com/ACAT-SCUT/TQNet) (Lin et al., ICML 2025); per-baseline upstream sources are credited in `models/<Name>.py` docstrings.
