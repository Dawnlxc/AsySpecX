# AsySpecX & JointAxisMLP

Collaborative workspace for two related lines of research on multivariate long-term time series forecasting.

Both subprojects share the same training/evaluation harness, descended from [ACAT-SCUT/TQNet](https://github.com/ACAT-SCUT/TQNet) (ICML 2025). They are kept side-by-side rather than merged so each can evolve independently while sharing datasets, conventions, and tooling.

## Subprojects

| Folder | Model | Idea | Status |
| --- | --- | --- | --- |
| [`AsySpecX/`](AsySpecX/) | `AsySpecX` (`models/AsySpecX.py`) | Asymmetric Spectral Transfer — low-rank `H = A diag(g_m) Bᵀ` with per-band gates, applied in the frequency domain. | Active (paper in preparation) |
| [`JointAxisMLP/`](JointAxisMLP/) | `JointMLP` (`models/JointMLP.py`) + JA v4 backend (`models/JointAxisTWCMv4.py`) | TQNet MLP backbone + frequency-conditioned JointAxisTWCM **v4** (per-bin per-frame gain `g_{k, t'}`). | Active |

See each subproject's own `README.md` for model details, hyperparameters, datasets, and run scripts.

## Repository layout

```
.
├── AsySpecX/           # asymmetric spectral transfer line
│   ├── models/AsySpecX.py
│   ├── exp/            # shared train/eval loop
│   ├── data_provider/  # ETT / custom CSV / Solar / PEMS loaders
│   ├── layers/         # shared layers (RevIN, attention families, embeddings, …)
│   ├── utils/          # metrics, masking, time features, tools
│   ├── scripts/        # per-dataset shell scripts + slurm launcher
│   ├── Figures/        # published figures
│   ├── acf_plot.ipynb
│   └── run.py
├── JointAxisMLP/       # JointMLP + JA v4 cross-channel line (mirror layout)
│   └── models/         # JointMLP.py + JointAxisTWCMv4.py
└── README.md           # (this file)
```

## Environment

Both subprojects target Python 3.10 with PyTorch 2.5.1+cu124. On the HPC node we use the `tsfm` conda env at `/scratch3/lin250/conda_envs/tsfm`. To recreate elsewhere:

```bash
conda create -n tsfm python=3.10 -y
conda activate tsfm
pip install -r AsySpecX/requirements.txt
```

The two subprojects currently pin the same dependencies; if they diverge, each subdirectory's `requirements.txt` is the source of truth for that subproject.

## Data

Datasets are **not** tracked in this repo (see `.gitignore`). Get the standard LTSF datasets (ETTh1/2, ETTm1/2, weather, electricity, traffic, PEMS03/04/07/08) from the [Autoformer / SCINet Google Drive bundle](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view) and place the CSVs under `AsySpecX/dataset/` and/or `JointAxisMLP/dataset/`, e.g. `AsySpecX/dataset/ETT-small/ETTh1.csv`.

## Conventions

- **Excluded from git**: `checkpoints/`, `results/`, `logs/`, `probe/`, `dataset/`, `figures/` (lowercase, working drafts), `__pycache__/`, `*.pth`, `slurm-*.out`. See [`.gitignore`](.gitignore).
- **Kept in git**: `Figures/` (capitalized, published figures only), code, scripts, notebooks, small reference outputs.
- **Seeds**: both subprojects run seeds `{2026, 2027}` per configuration.
- **License**: Apache 2.0 (inherited from upstream TQNet) — see [`LICENSE`](LICENSE).

## Attribution

The training/eval harness is adapted from the TQNet repository (Lin et al., ICML 2025). Per-subproject details in each `README.md`.
