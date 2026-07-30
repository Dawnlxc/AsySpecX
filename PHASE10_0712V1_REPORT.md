# AsySpecX Phase 10 overnight report (`0712v1`)

Generated on 2026-07-12 (Australia/Sydney). The authoritative run artifacts live under
`/scratch3/lin250/bldgFM/DUBABA/AsySpecX/phase10_results/`.

## Bottom line

- The old `asy2-0711v1` concern was real: the last valid Phase 8 forecast had 0/28 wins against the published baseline table, and Phase 9's router did not repair it.
- The Phase 10 cycle-aware Asy line now has reproducible Weather wins at H96 and H336, a statistical tie at H192, and a remaining H720 gap.
- Electricity has a strict resource/accuracy trade-off. With `seq_len=96`, the tiny factorized model becomes 11--16x smaller than the channel-individual model but does not beat the baseline. With explicit weekly context, the compact periodic model beats the published baseline at all four horizons.
- The recommended balanced Electricity point is `seq_len=504`: H96/H336 MSE `0.134789/0.165176` with `101,440/122,800` parameters. The highest-accuracy compact line uses `seq_len=720`.

## Why `asy2-0711v1` looked weak

- Phase 9 itself was operationally healthy: 98/98 jobs succeeded.
- The oracle showed headroom (`+0.019904`), but the quick router made MSE worse by `0.004715`; the out-of-fold gate correctly stopped it.
- The previous Weather `seq_len=96` cutoff calculation was too aggressive because shell integer division happened before multiplication.
- Electricity's main missing signal was weekly context, not more cross-channel rank. A full `168 x 321` phase table was valuable; rank-8 compression was not.

## Implemented ideas

### 1. CycleResidual-Asy

A learned phase/channel template is subtracted before residual forecasting and added at the future phases. Weather uses a 144-step daily cycle; Electricity uses a 168-step weekly cycle. This adds only 3,024 parameters on Weather and 53,928 on Electricity.

### 2. Low-rank channel-conditioned spectral lift

The channel lift is parameterized as a shared complex matrix plus rank-2/4 channel-conditioned deltas. It starts from the shared model exactly and is evaluated without a Python loop over 321 channels.

### 3. Compact same-phase period adapter

The old adapter allocated dense `[period, horizon, lookback]` matrices even though only same-phase taps were legal. The compact adapter stores only legal taps and evaluates contiguous phase runs with strided views. It preserves the FP32 function exactly while reducing the periodic branch by roughly 41x at `seq_len=720`.

### 4. Energy-knee cutoff search

Electricity cutoff sweeps showed that total retained energy alone is insufficient: cf61/cf85 retained about 89--90% but lost too much accuracy. The first successful H96 cutoff was cf151; H336 crossed the baseline at cf121.

## Weather (`seq_len=96`, three-seed confirmation where shown)

| Horizon | Recommended arm | Cutoff | Test MSE mean | Pop. std | Published baseline | Seeds below baseline | Params | Peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | individual + full cycle + direct residual | 13 | 0.157466 | 0.000263 | 0.1583 | 3/3 | 27,762 | 81.25 |
| 192 | individual + full cycle + direct residual | 25 | 0.206055 | 0.000103 | 0.2060 | 1/3 | 103,782 | 91.61 |
| 336 | individual + full cycle + direct residual | 13 | 0.263200 | 0.000148 | 0.2634 | 3/3 | 70,098 | 107.62 |
| 720 | individual + full cycle + direct residual | 13 | 0.342456 | single seed | 0.3387 | 0/1 | 138,306 | 143.99 |

Interpretation: H96 and H336 are stable wins. H192 is a practical tie, not a stable win. H720 still needs a different long-horizon residual mechanism.

## Electricity resource/accuracy profiles

### H96 lookback Pareto

| Profile | Seq | Test MSE | Published baseline | Params | Peak MiB | Train s | Full-test inference s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ultra-light factorized lift | 96 | 0.142908 | 0.1375 | 77,862 | 326.26 | 278.3 | 3.23 |
| one-week compact period | 168 | 0.142071 | 0.1375 | 67,144 | 438.41 | 583.6 | 4.54 |
| two-week compact period | 336 | 0.138065 | 0.1375 | 80,764 | 634.68 | 706.1 | 5.39 |
| **three-week balanced** | **504** | **0.134789** | **0.1375** | **101,440** | **837.61** | **821.5** | **6.41** |
| four-week-plus accurate | 720 | 0.134186 | 0.1375 | 138,364 | 1,086.61 | 952.4 | 7.22 |

The `seq_len=720` compact H96 configuration was confirmed across three seeds: mean `0.134627`, population std `0.000312`, and 3/3 seeds beat the baseline.

### H336 lookback Pareto

| Profile | Seq | Test MSE | Published baseline | Params | Peak MiB | Train s | Full-test inference s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ultra-light factorized lift | 96 | 0.174018 | 0.1723 | 117,006 | 661.14 | 588.1 | 6.45 |
| one-week compact period | 168 | 0.174198 | 0.1723 | 74,760 | 867.04 | 876.8 | 7.50 |
| two-week compact period | 336 | 0.167584 | 0.1723 | 95,336 | 1,045.56 | 939.6 | 7.81 |
| **three-week balanced** | **504** | **0.165176** | **0.1723** | **122,800** | **1,245.30** | **1,106.3** | **9.23** |
| four-week-plus accurate | 720 | 0.164353 | 0.1723 | 168,700 | 1,498.69 | 1,234.5 | 9.66 |

The `seq_len=720` compact H336 configuration was confirmed across three seeds: mean `0.164533`, population std `0.000199`, and 3/3 seeds beat both the published baseline and the previous Phase 8 `seq_len=720` result.

### Compact full four-horizon line (`seq_len=720`)

| Horizon | Test MSE | Published sl96 baseline | Params | Dense params | Parameter reduction | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.134186 | 0.1375 | 138,364 | 273,340 | 49.38% | 1,086.61 |
| 192 | 0.149402 | 0.1555 | 150,508 | 420,412 | 64.20% | 1,253.63 |
| 336 | 0.164353 | 0.1723 | 168,700 | 641,020 | 73.68% | 1,498.69 |
| 720 | 0.201989 | 0.2089 | 217,544 | 1,229,672 | 82.31% | 2,165.08 |

The long-lookback comparison against the published `seq_len=96` table is not an apples-to-apples input-budget comparison. Against the previous Phase 8 `seq_len=720` results, Phase 10 improves H96/H192/H336; H720 is approximately flat/slightly worse.

The compact adapter is primarily a parameter/optimizer-state optimization. At H96/H192/H336 it saved a small amount of peak memory, but at H720 activation memory dominated and peak CUDA memory was 36.75 MiB (1.73%) higher than the dense form.

## What did not work

- Phase 9 quick routing: oracle headroom existed, but learned routing degraded the forecast.
- Rank-8 cycle table: too much information loss; the learned full table's top 8 singular directions explain only about 66% of its energy.
- Aggressive Electricity cutoffs cf61/cf85: small models, but both missed the baseline.
- Huge channel-individual Electricity lifts: up to 1.92M parameters, with almost no practical gain over the 78--117k factorized lift.
- Weather H720: cycle + direct residual helps but does not close the final gap.

## Validation and operational checks

- 146 legacy + Phase 10 unit tests passed.
- Two-stage canaries checked exact output equivalence, gradient flow, GPU training, parameter counts, peak CUDA memory, and dense/compact parity.
- 92 formal H100 jobs plus 6 canary/comparison jobs completed successfully; no job failed.
- Every formal job requested one GPU (`gres/gpu=1`), four CPUs, 48 GiB host RAM, normal QOS, and a two-hour limit.
- No code push, email, or destructive repository cleanup was performed.

Re-run manifests are available as `configs/phase10_recommended_balanced.tsv` and
`configs/phase10_recommended_accurate.tsv`.

## Next ideas worth pursuing

1. Fixed Haar/wavelet residual only for Weather H720, with 2--3 scales and zero-initialized additive fusion.
2. Distill the full Electricity cycle table to rank 32/64 from its trained SVD, then fine-tune; do not train rank 8 from scratch.
3. Horizon-chunked training/inference for H720 to reduce activation memory without changing the small parameter count.
4. Band-specific low-rank channel deltas only above the daily/weekly harmonic knee, rather than increasing rank everywhere.
