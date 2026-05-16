# AsySpecX & JointMLP

Long-term multivariate time series forecasting research repo. Two active model lines plus a curated set of published baselines for head-to-head benchmarking. Shared TQNet-derived training/evaluation harness.

→ **See [`RESULTS.md`](RESULTS.md) for the comparison numbers.**

## Models

| Category | File | Notes |
| --- | --- | --- |
| **Active line — AsySpecX** | [`models/AsySpecX.py`](models/AsySpecX.py) | Asymmetric Spectral Transfer (paper in preparation). Lite/deployment variant. |
| | [`models/AsySpecXResid.py`](models/AsySpecXResid.py) | Paper-flagship variant: FITS-style self-predictor + Hermitian residual cross block. Strongest on long-horizon ETTh2 (−49 % vs DLinear) and PEMS_BAY sensor outage (−62 %). |
| | [`models/FreqHerm.py`](models/FreqHerm.py) | Foundational symmetric cross-block; used as a building block in AsySpecXResid. |
| | [`models/FreqHermCycle.py`](models/FreqHermCycle.py) | FreqHerm + CycleNet-style cycle decomposition. Strongest on PEMS_BAY traffic recovery (−65 % vs DLinear). |
| | [`models/FreqHermCycleAttn.py`](models/FreqHermCycleAttn.py) | + attention readout. Strongest on Beijing-Air132 multimodal (−16 % vs TQNet). |
| **Active line — JointMLP** | [`models/JointMLP.py`](models/JointMLP.py) | TQNet MLP backbone + JA cross-channel mixer. |
| | [`models/JointAxisTWCMv4.py`](models/JointAxisTWCMv4.py) | JA v4 backend: per-bin per-frame gain `g_{k, t'}`. Imported by `JointMLP`. |
| **Published baselines (10)** | TQNet, CycleNet, DLinear, iTransformer, PatchTST, FITS, FreTS, FilterNet, SparseTSF, MixLinear | One `models/<Name>.py` each. Per-baseline scripts under `scripts/<Name>/`. |

Pick a model via `--model <Name>` (any key in `exp/exp_main.py::model_dict`).

## Layout

```
models/                     all model files; each exports `class Model(nn.Module)`
exp/exp_main.py             shared train/eval loop; model_dict registers every model
data_provider/              ETT / custom CSV / Solar / PEMS dataset loaders
layers/                     shared layers (RevIN, attention families, embeddings, …)
utils/                      metrics, masking, time features, tools
run.py                      argparse entry point (carries flags for all models)
requirements.txt            shared dependency pins

scripts/
  _common.sh                load_dataset + per-model apply_<name>_overrides helpers
                            (apply_asyspecx_overrides, apply_cyclenet_overrides, …)
  AsySpecX/<Dataset>.sh     per-(model, dataset) sweep scripts
  JointMLP/<Dataset>.sh     (one subdir per model that has a sweep)
  FreqHerm/<Dataset>.sh
  FreqHermCycle/<Dataset>.sh
  TQNet/<Dataset>.sh        (one subdir per baseline)
  ...
  slurm/baseline.sbatch     sbatch ... baseline.sbatch <MODEL> <DATASET>
  slurm/submit_all.sh       --model X / --all-baselines / --smoke / --full / --dataset Y

analysis_exp/               post-hoc analysis & visualization scripts
Figures/                    published figures (carried over from TQNet upstream)
acf_plot.ipynb              exploratory autocorrelation notebook
RESULTS.md                  comparison numbers vs baselines (see top of this README)
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
bash scripts/AsySpecX/ETTh1.sh           # full sweep for one (model, dataset)
SMOKE=1 bash scripts/AsySpecX/ETTh1.sh   # restrict to sl=96, pl=96
bash scripts/TQNet/ETTh1.sh              # baselines work the same way
```

### Slurm
```bash
bash scripts/slurm/submit_all.sh --model AsySpecX --smoke      # one model, smoke
bash scripts/slurm/submit_all.sh --model JointMLP --full       # one model, full sweep
bash scripts/slurm/submit_all.sh --model TQNet --dataset ETTh1 # one (model, dataset)
bash scripts/slurm/submit_all.sh --all-baselines --full        # all 10 baselines × 11 datasets = 110 jobs
bash scripts/slurm/submit_all.sh --all-baselines --smoke       # 5 reps × 2 datasets = 10 jobs
```

Each slurm job runs the full `seed × sl × pl` sweep for one (model, dataset) sequentially within a 12 h, 1-GPU, 64 GB allocation. The slurm template resolves the repo root from the script's own location, so it's portable.

## Benchmark protocol

- **Seeds**: `{2026, 2027}` for the active lines; baseline sweep used seed `2026` only
- **Lookback sweep**: `seq_len ∈ {96, 720}` for AsySpecX / `{96, 336, 720}` for JointMLP; PEMS fixed at `seq_len = 96`
- **Pred-len sweep**: `{96, 192, 336, 720}` for non-PEMS, `{12, 24, 48, 96}` for PEMS

## License & attribution

Apache 2.0 — see [`LICENSE`](LICENSE). The training/evaluation harness and baseline implementations are adapted from [TQNet](https://github.com/ACAT-SCUT/TQNet) (Lin et al., ICML 2025); per-baseline upstream sources are credited in `models/<Name>.py` docstrings.
