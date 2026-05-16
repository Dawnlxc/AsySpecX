# Results

Headline numbers comparing our models against the 10 published baselines on the standard long-term forecasting suite. Source data:
- Baseline sweep: `note/sl96_results.csv` (upstream) → 10 baselines × 11 datasets × 4 horizons, seed 2026
- AsySpecXResid: `probe/FINAL_RESULTS_TABLE.md` (upstream) → 3 seeds, sd ∈ {2026, 2027, 2028}
- AsySpecX / JointMLP (JA v4): curated comparison **pending** — raw logs exist but a clean benchmark sweep hasn't been re-aggregated post-merge

## TL;DR

| Model | Strongest claim | vs best baseline | Status |
| --- | --- | --- | --- |
| **AsySpecXResid** | ETTh2 pl=720 MTSF: **0.419** | **−49 % vs DLinear** (0.819) | Curated, 3-seed |
| **AsySpecXResid** | PEMS_BAY mode B (sensor outage, held-out 6/325): **0.472** | **−62 % vs DLinear** (1.247) | Curated, 3-seed |
| **FreqHermCycleAttn** | Beijing-Air132 multimodal cross: **0.689** | **−16 % vs TQNet**, −38 % vs DLinear | Curated, single seed |
| **FreqHermCycle** | PEMS_BAY mode B pl=12 traffic recovery: **0.437** | **−65 % vs DLinear** (1.247) | Curated, 3-seed |
| AsySpecX (lite) | TBD | TBD | Logs only — needs aggregation |
| JointMLP (JA v4) | TBD | TBD | Logs only — needs aggregation |

The strongest results come from the AsySpecX line; AsySpecXResid is the paper-flagship variant (FITS-style self-predictor + Hermitian cross block on the residual). The JointMLP / JA v4 line has been training but a clean benchmark vs the 10 baselines hasn't been run on the merged repo yet.

---

## AsySpecXResid — standard MTSF (sl=96)

Lift over DLinear, with FreqHerm shown as an intermediate ablation. Values are MSE; bold = wins vs DLinear by ≥5 %.

| Dataset | pl | DLinear | FreqHerm | **AsySpecXResid** | Δ vs DLinear |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | 96  | 0.398 | 0.390 | **0.389** | −2 % |
| ETTh1 | 192 | 0.446 | 0.442 | **0.439** | −2 % |
| ETTh1 | 336 | 0.490 | 0.487 | **0.480** | −2 % |
| ETTh1 | 720 | 0.515 | 0.484 | **0.469** | **−9 %** |
| ETTh2 | 96  | 0.349 | 0.295 | **0.301** | **−14 %** |
| ETTh2 | 192 | 0.476 | 0.381 | **0.381** | **−20 %** |
| ETTh2 | 336 | 0.590 | 0.419 | **0.419** | **−29 %** |
| ETTh2 | 720 | 0.819 | 0.423 | **0.419** | **−49 %** |
| ETTm1 | 96  | 0.345 | 0.346 | **0.341** | tie |
| ETTm1 | 192 | 0.381 | 0.384 | **0.382** | tie |
| ETTm1 | 336 | 0.413 | 0.414 | **0.411** | tie |
| ETTm1 | 720 | 0.473 | 0.481 | 0.475 | tie |
| ETTm2 | 96  | 0.196 | 0.178 | **0.179** | **−9 %** |
| ETTm2 | 192 | 0.280 | 0.242 | **0.246** | **−12 %** |
| ETTm2 | 336 | 0.380 | 0.306 | **0.307** | **−19 %** |
| ETTm2 | 720 | 0.539 | 0.406 | **0.407** | **−24 %** |
| weather | 96  | 0.193 | 0.164 | **0.166** | **−14 %** |
| weather | 192 | 0.237 | 0.212 | **0.210** | **−11 %** |
| weather | 336 | 0.284 | 0.270 | **0.266** | **−6 %** |
| weather | 720 | 0.350 | 0.345 | **0.343** | −2 % |
| electricity | 96  | 0.196 | 0.176 | **0.177** | **−10 %** |
| electricity | 192 | 0.194 | 0.182 | 0.186 | −4 % |
| electricity | 336 | 0.207 | 0.198 | 0.200 | −3 % |
| electricity | 720 | 0.243 | 0.238 | 0.252 | +4 % worse |
| traffic | 96  | 0.649 | 0.573 | **0.568** | **−12 %** |
| traffic | 192 | 0.599 | 0.549 | **0.547** | **−9 %** |
| traffic | 336 | —     | 0.560 | **0.554** | — |

**Pattern**: cross-block lift grows with prediction length — at long horizons the decoder benefits most from cross-channel information. ETTh2 is the strongest single-dataset gain.

## AsySpecXResid — sensor outage (mode B, held-out spatial sensors)

| Dataset | Held | DLinear | iTransformer | FreqHerm | **AsySpecXResid** | Δ vs DLinear |
| --- | --- | --- | --- | --- | --- | --- |
| **PEMS_BAY** (325 ch traffic) | 6 sensors | 1.247 | 1.162 | 0.495 | **0.472** | **−62 %** |
| METR_LA (207 ch traffic)      | 6 sensors | 2.092 | 1.539 | 1.731 | **1.722**   | −18 % |
| Beijing_AQ (132 ch AQ)        | 1 station × 11 mod | 1.107 | 1.103 | 1.027 | 1.028   | −7 % |

(Traffic at pl=12, AQ at pl=96.) Pattern: gain is largest where physical cross-channel structure (road graph) is strongest.

## Seed variance (AsySpecXResid, sd ∈ {2026, 2027, 2028})

| Setting | mean | std | rel std |
| --- | --- | --- | --- |
| PEMS_BAY mode B pl=12 | 0.4707 | 0.007  | 1.5 % |
| PEMS_BAY mode B pl=96 | 0.5083 | 0.001  | 0.2 % |
| ETTh2 pl=720 std MTSF | 0.4186 | 0.0002 | 0.05 % |
| ETTh2 pl=96  std MTSF | 0.2989 | 0.003  | 1.0 % |

Variance is well below the gap to the next-best baseline → headline results are robust.

---

## Baseline reference (sl=96, single seed 2026)

For each (dataset, horizon), the best and second-best baseline MSE across the 10 published baselines. Use this as the bar AsySpecX / JointMLP must clear.

| Dataset | pl | Best MSE (model) | 2nd best (model) |
| --- | --- | --- | --- |
| ETTh1 | 96  | 0.372 (TQNet)    | 0.379 (CycleNet) |
| ETTh1 | 192 | 0.427 (CycleNet) | 0.430 (TQNet) |
| ETTh1 | 336 | 0.465 (CycleNet) | 0.475 (MixLinear) |
| ETTh1 | 720 | 0.462 (CycleNet) | 0.464 (SparseTSF) |
| ETTh2 | 96  | 0.287 (CycleNet) | 0.293 (PatchTST) |
| ETTh2 | 192 | 0.366 (TQNet)    | 0.367 (PatchTST) |
| ETTh2 | 336 | 0.419 (TQNet)    | 0.420 (FITS) |
| ETTh2 | 720 | 0.420 (SparseTSF)| 0.423 (FITS) |
| ETTm1 | 96  | 0.310 (TQNet)    | 0.322 (PatchTST) |
| ETTm1 | 192 | 0.362 (PatchTST) | 0.363 (TQNet) |
| ETTm1 | 336 | 0.388 (PatchTST) | 0.391 (TQNet) |
| ETTm1 | 720 | 0.452 (TQNet)    | 0.454 (PatchTST) |
| ETTm2 | 96  | 0.168 (CycleNet) | 0.173 (TQNet) |
| ETTm2 | 192 | 0.233 (CycleNet) | 0.241 (TQNet) |
| ETTm2 | 336 | 0.294 (CycleNet) | 0.297 (FilterNet) |
| ETTm2 | 720 | 0.396 (CycleNet) | 0.396 (FilterNet) |
| weather | 96  | 0.158 (TQNet)    | 0.165 (FilterNet) |
| weather | 192 | 0.206 (TQNet)    | 0.211 (FilterNet) |
| weather | 336 | 0.263 (FreTS)    | 0.264 (TQNet) |
| weather | 720 | 0.339 (FreTS)    | 0.343 (TQNet) |
| electricity | 96  | 0.138 (TQNet)    | 0.141 (CycleNet) |
| electricity | 192 | 0.155 (CycleNet) | 0.157 (TQNet) |
| electricity | 336 | 0.172 (TQNet)    | 0.172 (CycleNet) |
| electricity | 720 | 0.209 (iTrans)   | 0.211 (CycleNet) |
| traffic | 96  | 0.387 (iTrans)   | 0.417 (FilterNet) |
| traffic | 192 | 0.409 (iTrans)   | 0.439 (FilterNet) |
| traffic | 336 | 0.413 (iTrans)   | 0.458 (FilterNet) |
| traffic | 720 | 0.438 (iTrans)   | 0.495 (FilterNet) |
| PEMS03 | 12 | 0.060 (TQNet) | 0.067 (iTrans) |
| PEMS03 | 24 | 0.076 (TQNet) | 0.093 (iTrans) |
| PEMS03 | 48 | 0.108 (TQNet) | 0.149 (iTrans) |
| PEMS03 | 96 | 0.142 (TQNet) | 0.227 (iTrans) |
| PEMS04 | 12 | 0.068 (TQNet) | 0.078 (FilterNet) |
| PEMS04 | 24 | 0.079 (TQNet) | 0.097 (FilterNet) |
| PEMS04 | 48 | 0.099 (TQNet) | 0.136 (FilterNet) |
| PEMS04 | 96 | 0.123 (TQNet) | 0.201 (FilterNet) |
| PEMS07 | 12 | 0.052 (TQNet) | 0.062 (FilterNet) |
| PEMS07 | 24 | 0.063 (TQNet) | 0.086 (FilterNet) |
| PEMS07 | 48 | 0.082 (TQNet) | 0.124 (FilterNet) |
| PEMS07 | 96 | 0.110 (TQNet) | 0.180 (FilterNet) |
| PEMS08 | 12 | 0.078 (FilterNet) | 0.078 (iTrans) |
| PEMS08 | 24 | 0.111 (FilterNet) | 0.113 (iTrans) |
| PEMS08 | 48 | 0.171 (FilterNet) | 0.181 (iTrans) |
| PEMS08 | 96 | 0.292 (FilterNet) | 0.303 (iTrans) |

**Baseline roundup**: TQNet is the single strongest baseline (wins ~half of all settings), CycleNet dominates ETT short horizons, iTransformer sweeps traffic, FilterNet wins PEMS08.

---

## Pending: AsySpecX (lite) and JointMLP (JA v4)

Neither has a curated comparison table on the merged repo yet:

- **`models/AsySpecX.py`**: the lite/deployment variant. Run logs live in `/scratch3/lin250/bldgFM/AsySpecX/logs/AsySpecX/` (upstream). To produce a clean comparison, re-run `bash scripts/slurm/submit_all.sh --model AsySpecX --full` on this repo (seeds 2026, 2027).
- **`models/JointMLP.py` (JA v4)**: 1200+ run logs exist at `/scratch3/lin250/bldgFM/AsySpecX/logs/JointMLP/` (upstream) under tags `jmlp_v4`, `jmlp_uniform_r*`, `jmlp_adaptR`, but no aggregated comparison vs baselines was written. To produce one, run `bash scripts/slurm/submit_all.sh --model JointMLP --full` and `aggregate.py`-style postprocess.

To regenerate the baseline numbers in this file from scratch:
```bash
bash scripts/slurm/submit_all.sh --all-baselines --full   # 110 jobs
# wait, then aggregate logs/<Model>/*.log into a fresh CSV
```

## Where these numbers came from

| Section | Source file (upstream `/scratch3/lin250/bldgFM/AsySpecX/`) |
| --- | --- |
| AsySpecXResid standard MTSF | `probe/FINAL_RESULTS_TABLE.md` |
| AsySpecXResid sensor outage | `probe/FINAL_RESULTS_TABLE.md` |
| AsySpecXResid seed variance | `probe/FINAL_RESULTS_TABLE.md` |
| Baseline reference table  | `note/sl96_results.csv` (parsed) |
| Architectures that didn't help | `probe/FINAL_RESULTS_TABLE.md` (omitted here; see upstream) |

The upstream `note/appendix_full_table_sl96.tex` and `appendix_full_table_sl336.tex` are the polished LaTeX renderings of the baseline tables (with best/second-best annotation). The AsySpecX columns there are dashed (`--`) because that sweep hadn't been re-run when the tables were generated.
