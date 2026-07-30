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

### AsySpecX phase-1 repair experiment

AsySpecX supports the phase-1 flags from the FITS-style repair study:
`--spectral_lift {complex_mlp,fits_linear}`, `--cross_mode {none,asym_lowrank}`,
`--gate_type {global,channel_band}`, `--gate_init_logit`, `--mask_self_transfer`,
`--residual_clip_eta`, `--backcast_loss_weight`, `--force_cross_off`, and
`--skip_dc_cross`.

Run the four prepared ablations on Slurm:

```bash
# canary
NO_PUSH=1 ONLY=phase1_fits_only DATASETS=ETTh1 SEEDS=2026 SEQ_LENS=96 PRED_LENS=96 EPOCHS=1 \
  OUTROOT=phase1_results/canary bash scripts/slurm/submit_asyspecx_phase1.sh

# full default matrix with watcher, aggregation, glab push, and email
bash scripts/slurm/submit_asyspecx_phase1.sh
```

The Slurm-side handoff prompt is in
`scripts/slurm/asyspecx_phase1_slurm_claude_prompt.md`.

### AsySpecX phase-2 attribution experiment

Phase 2 keeps the phase-1 backbone and adds mechanism attribution controls:
`--residual_part {all,diag_only,offdiag_only,split}`, `--cross_mode self_band_gain`,
`--gate_type hier_channel_band`, `--gate_lr_mult`, `--eval_residual_part`, and
`--self_gain_init_std`.

Defaults:

- `residual_part`: unset; internally maps to `all`, or `offdiag_only` when old `mask_self_transfer=1` is used
- `cross_mode`: existing parser default remains `hybrid`, mapped to `asym_lowrank` for AsySpecX
- `gate_type`: `global`
- `gate_lr_mult`: `1.0`
- `eval_residual_part`: `default`
- `self_gain_init_std`: `1e-3`
- `backcast_loss_weight`: `0.0`

Single dataset/length run:

```bash
DATASET=ETTh1 SEQ_LEN=96 PRED_LEN=96 SEED=2026 bash scripts/run_phase2_asyspecx.sh
```

Recommended order:

1. Mechanism attribution: `phase2_global_all`, `phase2_global_diag_only`, `phase2_global_offdiag_only`, `phase2_global_split`, `phase2_self_band_gain_global`
2. Safe gate unlock: `phase2_hier_all`, `phase2_hier_split`, `phase2_hier_all_clip05`
3. Small sweep: `gate_lr_mult 5 vs 10`, `residual_clip_eta -1 vs 0.5 vs 1.0`, `gate_init_logit -6 vs -4`

Slurm handoff prompt:

```bash
scripts/slurm/asyspecx_phase2_slurm_claude_prompt.md
```

### AsySpecX phase-3 GapClose

Phase 3 targets the weather/electricity gap against strong baselines. New controls:
`--lift_sharing {shared,individual}`, `--norm_mode {rin_noaffine,revin_affine,subtract_last,none}`,
`--temporal_adapter {none,sparse_period}`, `--period`, `--periodic_init`,
`--periodic_sharing`, `--temporal_fusion`, `--temporal_gate_type`, and
`--temporal_gate_init_logit`.

Single run:

```bash
DATASET=weather SEQ_LEN=720 PRED_LEN=96 PERIOD=144 SEED=2026 bash scripts/run_phase3_gapclose.sh
DATASET=electricity SEQ_LEN=720 PRED_LEN=96 PERIOD=24 SEED=2026 bash scripts/run_phase3_gapclose.sh
```

Slurm handoff prompt:

```bash
scripts/slurm/asyspecx_phase3_gapclose_slurm_claude_prompt.md
```

Validation selection:

```bash
python scripts/select_by_validation.py \
  --csv phase3_gapclose_results/main/results.csv \
  --output phase3_gapclose_results/main/selected_results.csv \
  --summary phase3_gapclose_results/main/selected_summary.md
```

Interpretation rules:

- If `phase3_fits_shared` still loses published FITS, check FITS parity: cut_freq, normalization, scale, training config.
- If `phase3_fits_individual` beats shared, weather/electricity need channel-specific spectral lift more than stronger cross-transfer.
- If `revin_affine` or `subtract_last` helps weather, gap is normalization/distribution shift.
- If sparse-period helps electricity 96/192/336, explicit cross-period temporal bias matters.
- If sparse-period hurts weather, keep it dataset-specific and use validation selection.
- Never select by test metric; only `val_mse` selection is allowed.

### AsySpecX phase-4 Finalize

Phase 4 finalizes the table with a small candidate pool + fair validation
selection. It extends the sparse-period adapter to **multi-period** and the
temporal gate to **horizon/channel-aware**. New controls (all backward-compatible;
Phase 1-3 configs run unchanged):

- `--periods` — multi-period list, e.g. `24,168` or `24+168`. Empty falls back to
  `--period` (old single-period behavior is preserved exactly). Use `+` on Slurm:
  `sbatch --export` splits on comma and would truncate a `24,168` list; `run.py`
  parses both `+` and `,`.
- `--period_fusion {sum_gated,softmax}` (default `sum_gated`) — per-period sigmoid
  gates vs softmax over periods. Single period ⇒ weight 1 (no fusion change).
- `--period_gate_type {global,period,period_horizon,period_channel,period_horizon_channel}`
  (default `period`), `--period_gate_init_logit` (default `0.0`).
- `--temporal_gate_type` now also accepts `horizon` and `horizon_channel`
  (default still `global`), `--temporal_gate_init_logit` (default `-4.0`).
- `--periodic_l1_weight` / `--periodic_l2_weight` (default `0.0`) — train-only L1/L2
  on in-mask periodic weights (`model.extra_loss()`); never enters val/test metrics.

Final candidates (≤7 arms so validation selection stays fair): `phase4_asx_cross`,
`phase4_asx_individual`, `phase4_asx_period_single`, `phase4_asx_period_multi`,
`phase4_asx_individual_period`, `phase4_asx_individual_revin`, `phase4_asx_cross_revin`.

```bash
# local single-arm-set run (per-dataset default PERIODS; overridable)
DATASET=weather     SEQ_LEN=720 PRED_LEN=96 SEED=2024 bash scripts/run_phase4_final_candidates.sh
DATASET=electricity SEQ_LEN=720 PRED_LEN=96 SEED=2024 PERIODS=24,168 bash scripts/run_phase4_final_candidates.sh

# targeted tuning (compact by default; FULL_SWEEP=1 to expand)
bash scripts/run_phase4_weather_tuning.sh
bash scripts/run_phase4_electricity_tuning.sh

# Slurm full matrix (7 arms x 2 datasets x 4 horizons x 3 seeds ~ 168 jobs)
bash scripts/slurm/submit_asyspecx_phase4.sh   # handoff: scripts/slurm/asyspecx_phase4_slurm_claude_prompt.md

# fair validation selection (aggregate over seeds; NEVER select by test)
python scripts/select_by_validation.py \
  --csv phase4_results/main/results.csv \
  --selection_keys dataset,seq_len,pred_len --replicate_key seed --arm_key arm \
  --output phase4_results/main/selected_results.csv \
  --summary phase4_results/main/selected_summary.md

# Phase 4 summary + cut_freq diagnostics
python scripts/summarize_phase4.py --csv phase4_results/main/results.csv \
  --selected_csv phase4_results/main/selected_results.csv \
  --output phase4_results/main/summary_phase4.md            # optional: --baseline_csv baselines.csv
python scripts/summarize_cut_freq.py --csv phase4_results/main/results.csv \
  --output phase4_results/main/summary_cut_freq.md
```

Recommended `--selection_keys` is `dataset,seq_len,pred_len` (aggregate `val_mse`
over replicate seeds). Do **not** put `seed` in the keys unless you deliberately
want per-seed selection.

Interpretation rules:

- **Weather**: if `phase4_asx_individual` + cut_freq tuning matches or beats FITS,
  the gap is closed. If it still loses by 0.001-0.005, report as a close second and
  note weather prefers a pure channel-specific spectral backbone.
- **Electricity**: if `phase4_asx_period_multi` improves 192/336, it becomes the
  main candidate; if `phase4_asx_period_single` still wins, keep the simpler adapter.
- **Individual + cross**: Phase 3 showed `individual_hier_split` is bad — do not
  default-combine an individual backbone with the cross block.
- **Validation selection**: the AsySpecX-family final number comes from the ≤5
  candidate pool via `val_mse` selection; test metrics are used only after selection.
- **Directionality**: the asym-vs-sym directionality ablation is deferred to an
  appendix until the final performance table is stable.

### AsySpecX phase-5 Lockdown

Phase 5 freezes a small candidate pool and hardens validation selection (Phase 4
mis-picked `period_multi` on weather 336/720). No new structure. New controls
(all default to legacy behavior):

- `--val_num_segments K` (default `1`) — split the chronologically-ordered
  validation set into K contiguous segments and log `val_mse_seg*`/`val_mae_seg*`
  on the best model (ordered loader, shuffle off). K=1 keeps old logs.
- `--temporal_gate_l1_weight` (default `0.0`) — train-only L1 on the temporal
  fusion gate mean (weather periodic-branch guard); enters `model.extra_loss()`
  only, never val/test metrics.

Locked arms: `phase5_asx_cross`, `phase5_asx_cross_clip05`, `phase5_asx_individual`,
`phase5_asx_individual_revin`, `phase5_asx_period_multi`, `phase5_asx_individual_period`
(+ `phase5_asx_period_multi_gate_l1` only with `ENABLE_PERIOD_REG=1`). Flags are
defined once in `scripts/_common.sh:phase5_arm_flags`.

Enhanced selector (`scripts/select_by_validation.py`) — `--metric_mode
{mean,mean_plus_std,last_segment}`, `--std_weight`, `--selection_margin_abs/pct`,
`--prefer_arm_order`, `--arm_allowlist_json`. Test metrics never drive selection
(guarded by `--allow_test_selection`, off by default).

```bash
# full-field locked candidates (local matrix; PEMS seq_len=96 only unless RUN_PEMS_SEQ720=1)
bash scripts/run_phase5_fullfield_candidates.sh
# weather/electricity confirmation with 5 seeds
bash scripts/run_phase5_confirm_weather_electricity.sh
# Slurm full matrix (~144 jobs, VAL_NUM_SEGMENTS=4); handoff prompt:
bash scripts/slurm/submit_asyspecx_phase5.sh   # scripts/slurm/asyspecx_phase5_slurm_claude_prompt.md

# three selection variants (unrestricted_mean is the main result)
bash scripts/run_phase5_selection.sh           # ROOT=phase5_results/main CSV=.../results.csv
# summary + paired statistics + baseline gaps
python scripts/summarize_phase5.py --csv phase5_results/main/results.csv \
  --selected_csv phase5_results/main/selected_unrestricted_mean.csv \
  --anchor_arm phase5_asx_cross --output_dir phase5_results/main   # optional --baseline_csv
```

Interpretation rules:

- Do not pick arms by test metric. Report at least one fixed single-arm result
  (`phase5_asx_period_multi` for the cross/period story, or
  `phase5_asx_individual_period` for robust one-arm performance) AND the
  validation-selected result separately.
- Weather: `individual` / `individual_revin` are expected to win; do not force the
  periodic adapter unless validation strongly supports it.
- Electricity: `period_multi` is expected to win.
- Full-field: `cross` / `cross_clip05` matter where off-diagonal cross-transfer helps.
- `configs/selection/policy_family.json` (per-dataset pools + prefer order) is an
  **analysis policy**; the unrestricted validation selection
  (`configs/selection/unrestricted.json`) is also reported and is the cleanest result.
- Directionality (asym-vs-sym) ablation stays deferred until the final table is stable.

### AsySpecX phase-6 Protocol

Phase 6 adds no model structure and no core training change. It (a) audits the
Phase 5 selectors, (b) hardens the selector, and (c) provides a TRUE full-field
runner (Phase 5 only covered weather/electricity seq_len=720).

New selector mode: `--metric_mode segment_mean_plus_std` — pools every
`val_mse_seg*` across every seed and scores `mean + std_weight·std`, penalizing
arms with high validation-segment variance (weather's failure mode). Errors if
seg columns are absent; never falls back to test. The selection summary now
emits a **Margin / Prefer-Order Trace** (raw best arm, raw best score, near-best
arms, final selected arm) so prefer-order flips are auditable.

```bash
# selector audit on EXISTING Phase 5 results (5 variants + oracle upper bound)
bash scripts/run_phase6_selector_audit.sh                 # ROOT=phase5_results/main
python scripts/audit_phase5_selectors.py --csv phase5_results/main/results.csv \
  --selected_files selected_unrestricted_mean.csv,selected_unrestricted_last_segment.csv,selected_unrestricted_segment_robust.csv,selected_unrestricted_margin_prefer_simple.csv,selected_policy_family.csv \
  --output_dir phase5_results/main            # -> selector_audit.md / .csv / group_details.csv

# TRUE full-field candidates (7 datasets x 2 seq_len x preds x 3 seeds x 6 arms = 864 runs)
DRY_RUN=1 bash scripts/run_phase6_fullfield_candidates.sh  # prints estimate + commands
bash scripts/run_phase6_fullfield_candidates.sh            # PEMS seq_len=96 only unless RUN_PEMS_SEQ720=1
# Slurm full matrix + watcher (selection + audit + summary + push + mail):
bash scripts/slurm/submit_asyspecx_phase6.sh               # scripts/slurm/asyspecx_phase6_slurm_claude_prompt.md

# full-field selection (4 variants) + summary
bash scripts/run_phase6_fullfield_selection.sh             # ROOT=phase6_results/fullfield
python scripts/summarize_phase6_fullfield.py --csv phase6_results/fullfield/results.csv \
  --selected_csv phase6_results/fullfield/selected_unrestricted_mean.csv \
  --anchor_arm phase6_asx_cross --output_dir phase6_results/fullfield   # optional --baseline_csv / --selected_csvs
```

Reporting policy:

- Report **fixed single-arm** and **validation-selected** separately. Fixed
  candidates: `phase6_asx_period_multi` (novelty-preserving) and
  `phase6_asx_individual_period` (robust metric-focused).
- Validation-selected uses validation metrics aggregated across seeds per
  dataset/seq_len/pred_len; test metrics reported only after selection.
- `configs/selection/phase6_policy_family.json` is an **analysis policy**, not a
  replacement for the cleaner unrestricted selection
  (`configs/selection/phase6_unrestricted.json`).
- **Oracle** (per-cell test-best) is an analysis upper bound only and must NOT be
  reported as a valid selected model.
- Directionality (asym-vs-sym) ablation stays deferred until the full-field table
  is stable.

### AsySpecX phase-7 Breakthrough Arms

Phase 6's selector is already close to the test oracle, so Phase 7 adds
*complementary candidate arms* (no Transformer/attention/GNN, no directionality
ablation). New model features (all default-off, Phase 1-6 unchanged):

- **Learned energy-controlled clip** — `--energy_control learned_clip`
  (`--learned_clip_scope component_channel_band`, `--learned_clip_eta_init 1.0`,
  `--learned_clip_eta_max 2.0`, `--clip_lr_mult`). Learned per-component/band/
  channel eta; the applied scale is still clamped to ≤1, so it can only shrink
  the cross residual, never amplify it.
- **Auto-period discovery** — `--period_mode {manual,auto_acf,auto_fft}` resolved
  at runner level by `scripts/discover_periods.py` (TRAIN split only, cached to
  `auto_periods/*.json`, falls back to manual on failure). Model still takes
  `--periods`.
- **Patch-linear temporal residual** — `--patch_adapter patch_linear`
  (`--patch_len`, `--patch_stride`, `--patch_basis_dim`, `--patch_fusion`,
  `--patch_gate_type`, `--patch_gate_init_logit`, `--patch_l1/l2_weight`). Tiny
  channel-independent Linear over patch features; fused into the horizon after
  the sparse-period adapter; gate starts near-off. Train-only L1/L2 in
  `extra_loss`.
- **Offline convex ensemble** — `--save_predictions 1 --pred_save_dir ...
  --pred_tag ...` saves val/test preds; `scripts/ensemble_predictions.py` learns
  convex/ridge weights on VALIDATION only (test targets never used). Reported
  separately as AsySpecX-Ensemble (analysis).

8 Phase-7 arms (COMPACT=1 → 5): `phase7_period_multi[_split_clip05|_all_clip05|
_learned_clip|_auto_acf|_auto_fft|_patchlinear|_auto_acf_patchlinear]`.

```bash
# recommended: compact Phase 7 across all datasets first
COMPACT=1 SEEDS="2024 2025 2026" bash scripts/run_phase7_breakthrough_candidates.sh   # DRY_RUN=1 to preview
# Slurm: full=1152 runs (COMPACT=720); auto-period resolved in-job
bash scripts/slurm/submit_asyspecx_phase7.sh          # scripts/slurm/asyspecx_phase7_slurm_claude_prompt.md
# merge with Phase 6, select, summarize
python scripts/merge_results.py --csvs phase6_results/fullfield/results.csv,phase7_results/breakthrough/results.csv --output phase7_results/merged/results.csv
ROOT=phase7_results/merged bash scripts/run_phase7_selection.sh
python scripts/summarize_phase7.py --csv phase7_results/merged/results.csv \
  --selected_csv phase7_results/merged/selected_unrestricted_mean.csv \
  --anchor_arm phase6_asx_period_multi --output_dir phase7_results/merged
# offline ensemble (needs SAVE_PREDICTIONS=1 runs)
python scripts/ensemble_predictions.py --pred_dir phase7_results/breakthrough/predictions --mode simplex_val \
  --output_csv phase7_results/merged/ensemble_results.csv --summary phase7_results/merged/ensemble_summary.md
```

Run strategy: (1) compact Phase 7 everywhere; (2) if one arm improves the
full-field mean by >0.002, run full Phase 7; (3) keep patchlinear if it helps
short-horizon electricity/traffic/PEMS; (4) adopt auto-period if it helps
PEMS/traffic/ETT; (5) if clipped period_multi improves ETTh1 without hurting
PEMS/electricity, make it the new AsySpecX-Single; (6) if nothing beats
`phase6_asx_period_multi`, stop model development and write the paper with
Phase 6. Oracle stays analysis-only.

### AsySpecX phase-8 Hydra

Phase 8 combines the winning Phase-7 ingredients and adds a linear/DLinear branch
and a parallel branch mixture (no attention/Transformer/GNN). All default-off;
Phase 1-7 unchanged.

- **Period union** — `--period_mode union_auto --max_periods 5`: manual periods
  first, then auto_acf, then auto_fft, deduped/capped; discovered on the TRAIN
  split only (`scripts/discover_periods.py --method union_auto`).
- **Linear branch** — `--linear_adapter {direct_linear,dlinear_decomp,
  multiscale_dlinear}` on normalized time input, with `--linear_sharing`
  (individual guarded by `--individual_linear_max_channels`), `--linear_init`,
  `--moving_avg_kernel` (replicate-padded, length-preserving), `--multiscale_factors`,
  `--multiscale_fusion/_gate_type`. Fused via `--linear_fusion`/`--linear_gate_type`/
  `--linear_gate_init_logit`; `--linear_l1/l2_weight` train-only in `extra_loss`.
- **Hydra branch fusion** — `--branch_fusion softmax_static`
  (`--branch_fusion_scope`, `--branch_init_main_logit 4.0`,
  `--branch_init_aux_logit -4.0`): softmax mixture over available branches
  (spec/period/patch/linear); weights sum to 1 (bounded), spec starts dominant.
  `sequential` (default) reproduces Phase 7 exactly.
- **Robust prediction export + ensemble** — `--save_predictions 1` writes float32
  val/test preds+targets+metadata; `scripts/ensemble_predictions.py` fits
  convex/ridge weights on VALIDATION only (test never used), reported separately.

7 arms (COMPACT=1 → 4): `phase8_auto_acf_patchlinear[_split_clip05]`,
`phase8_union_auto_patchlinear_{split_clip05,dlinear}`,
`phase8_auto_acf_patchlinear_dlinear`, `phase8_hydra_{softmax,multiscale}_dlinear`.

```bash
# 1. compact Phase 8 (576 runs; full=1008). DRY_RUN=1 previews the count.
COMPACT=1 SEEDS="2024 2025 2026" bash scripts/run_phase8_hydra_candidates.sh
# Slurm: submit sends ONE start email; watcher sends ONE done email (duration+GPU-jobs+summary)
bash scripts/slurm/submit_asyspecx_phase8.sh          # scripts/slurm/asyspecx_phase8_slurm_claude_prompt.md
# 2. merge Phase 6/7/8       3. select        4. summarize
python scripts/merge_results.py --csvs phase6_results/fullfield/results.csv,phase7_results/merged/results.csv,phase8_results/hydra/results.csv --output phase8_results/merged/results.csv
ROOT=phase8_results/merged bash scripts/run_phase8_selection.sh
python scripts/summarize_phase8.py --csv phase8_results/merged/results.csv \
  --selected_csv phase8_results/merged/selected_unrestricted_mean.csv --output_dir phase8_results/merged
# 5. if compact improves best-fixed by >0.002 or selected by >0.001, run full Phase 8.
# 6. if no Phase 8 arm beats phase7_period_multi_auto_acf_patchlinear, stop and write the paper with Phase 7.
```

### AsySpecX phase-9 SafeRoute

Phase 9 freezes the Phase 6-8 experts and addresses validation-test alignment
with an explicit, small advantage router. It adds no forecasting branch,
Transformer, attention, or GNN. Phase 1-8 model defaults and forward behavior
remain unchanged.

Locked experts are manifest driven: `anchor` (Phase-7 auto-ACF+patchlinear),
`dlinear`, `split_clip`, `individual_revin`, and `individual_period`. A manifest
is generated per `(dataset, seq_len, pred_len)` and contains exact seed checkpoint
paths plus the full architecture/data config. Loading is strict: missing seeds,
state-dict/config mismatches, cell mismatches, or sample-order mismatches are
errors. A smaller pool is allowed with `--experts`, but `anchor` is mandatory.

The streaming evaluator keeps only one batch of `[experts, horizon, channels]`
predictions in memory. Its default output is chunked compact NPZ metadata with
context features, forecast summaries, seed uncertainty, and per-sample/block
losses. It never writes the full multi-expert prediction tensor. Full export is
opt-in through `--save_full_predictions 1` and uses separate `.npy` memmaps per
expert.

Important Phase-9 CLI defaults:

- `--expert_manifest ''`, `--anchor_expert anchor`,
  `--expert_seeds 2024,2025,2026`
- `--router_num_horizon_blocks 4`, `--router_target advantage`
- `--router_backend xgboost`, `--router_scope cell`,
  `--router_min_samples 256`
- `--router_cv_folds 4`, `--router_purge_steps 0` (`0` means `pred_len`)
- `--router_meta_source val`, `--router_oof_seed 2024`
- `--router_confidence_alpha 0.1`, `--router_min_gain 0.0`
- `--router_decision safe_top1_blend`, `--router_full_gain 0.02`,
  `--router_uncertainty_beta 0.1`, `--router_temperature 0.1`
- `--router_channel_groups 1`, `--save_full_predictions 0`

`router_channel_groups > 1` is an audit-only opt-in. Channel assignments are
fit by deterministic KMeans from train-only variance/trend/period/entropy/ACF
descriptors and persisted in compact-meta JSON. Channel-group routing must stay
disabled unless its oracle improves the sample-block oracle by at least `0.002`.

```bash
# One strict cell manifest (fails if any requested checkpoint is missing).
python scripts/build_phase9_manifest.py \
  --dataset weather --seq_len 720 --pred_len 96 \
  --expert_seeds 2024,2025,2026 \
  --output phase9_results/manifests/weather_sl720_pl96.json

# Fine-grained test oracle/headroom audit. Every oracle is ANALYSIS ONLY.
EXPERT_MANIFEST=phase9_results/manifests/weather_sl720_pl96.json \
  OUTROOT=phase9_results/headroom_weather bash scripts/run_phase9_headroom.sh

# Optional channel-group oracle audit; this still does not enable group routing.
EXPERT_MANIFEST=phase9_results/manifests/weather_sl720_pl96.json \
  ROUTER_CHANNEL_GROUPS=4 OUTROOT=phase9_results/headroom_weather_groups \
  bash scripts/run_phase9_headroom.sh

# Quick validation-adapted router: val meta -> purged OOF calibration -> streaming test.
EXPERT_MANIFEST=phase9_results/manifests/weather_sl720_pl96.json \
  ROUTER_BACKEND=xgboost OUTROOT=phase9_results/quick_weather \
  bash scripts/run_phase9_router_quick.sh

# Rolling OOF is hard-gated by quick gain >= 0.002.
EXPERT_MANIFEST=phase9_results/manifests/weather_sl720_pl96.json \
  QUICK_RESULT=phase9_results/quick_weather/routed_results.csv \
  OUTROOT=phase9_results/oof_weather bash scripts/run_phase9_router_oof.sh

# Full conditional Slurm pipeline: one START email and one DONE email only.
bash scripts/slurm/submit_asyspecx_phase9.sh
# Handoff instructions: scripts/slurm/asyspecx_phase9_slurm_codex_prompt.md

# Unit and backward-compatibility tests.
python -m unittest tests/test_asyspecx_phase9.py
python -m unittest tests/test_asyspecx_phase1.py tests/test_asyspecx_phase4.py \
  tests/test_asyspecx_phase5.py tests/test_asyspecx_phase6.py \
  tests/test_asyspecx_phase7.py tests/test_asyspecx_phase8.py
```

Each rolling-OOF fold fits its data scaler and auto-ACF periods only on the
observations visible to that fold. Fold predictions are converted back to the
common official-train standardized space before router labels are assembled.

The headroom stage writes `headroom_by_cell.csv`, `headroom_by_dataset.csv`,
and `headroom_by_horizon.csv`, including oracle choice counts and gains against
both best-fixed and validation-selected baselines. Routed summaries include
dataset/sequence-length/prediction-length breakdowns, expert activation counts,
coverage, fallback, and paired cell statistics versus anchor. The Slurm watcher
validates that every submitted cell produced a matching successful status file;
missing or stale status is a pipeline failure rather than a silent partial result.

Locked stop/go rules:

1. If sample-block oracle gain over best fixed is below `0.004`, stop internal
   routing work.
2. If the quick router does not improve anchor by at least `0.002`, do not run
   rolling OOF.
3. If quick improves overall but any dataset degrades by more than `0.003`, stop
   and use more conservative confidence or dataset calibration.
4. Only consider distillation if rolling OOF improves by at least `0.003` and
   catastrophic activation remains low.
5. Test-label oracles and actual-advantage diagnostics are analysis only and
   are never valid selected-model results.

The main leakage/storage risks are overlapping rolling windows, sample-ID
misalignment, test labels entering fit/calibration, an inverted LCB residual,
unbounded blend alpha, full prediction export, and seed variance in inconsistent
normalization spaces. Phase-9 tests cover the corresponding invariants; the
manifest records `normalization_space=dataset_standardized`.

## Benchmark protocol

- **Seeds**: `{2026, 2027}` for the active lines; baseline sweep used seed `2026` only
- **Lookback sweep**: `seq_len ∈ {96, 336, 720}` (PEMS fixed at `seq_len = 96`)
- **Pred-len sweep**: `{96, 192, 336, 720}` for non-PEMS, `{12, 24, 48, 96}` for PEMS

## License & attribution

Apache 2.0 — see [`LICENSE`](LICENSE). The training/evaluation harness and baseline implementations are adapted from [TQNet](https://github.com/ACAT-SCUT/TQNet) (Lin et al., ICML 2025); per-baseline upstream sources are credited in `models/<Name>.py` docstrings.
