# AsySpecX Phase 11: Horizon-Conditioned Forecastability Kernel

## Verdict

The time-series-specific kernel is viable, but the useful asymmetry is
past-to-future, not variable-to-variable lead--lag.

The winning Stage-A model is a real, rank-8, channel-separable temporal kernel
with a learned per-channel/per-mode amplitude (`fk_r8_cs`).  Across three
seed-matched Weather H720 runs it beats the equal-input dense direct residual
on test MSE in 3/3 seeds and also has lower mean validation MSE.  It therefore
passes the predeclared stable-win rule.

The gain is real but small.  It does not beat the published Weather H720
reference of 0.3387, so this is a compact residual improvement rather than a
new state of the art.

## Model

For every channel independently,

\[
\widehat Y_{b,h,c}=\sum_{r=1}^{R}U_{h,r}s_{c,r}
\sum_{t=1}^{T}V_{r,t}X_{b,t,c}.
\]

`U` and `V` are shared temporal factors and `s[c,r]` is an optional
channel-specific mode amplitude.  The implementation never contracts over the
channel dimension: changing channel `c1` cannot alter another channel's
forecast.

## Protocol

- Primary cell: Weather, `seq_len=96`, `pred_len=720`, `cut_freq=13`.
- Existing individual spectral lift + full cycle residual was held fixed.
- Stage-A screen: anchor, dense direct, ranks 4/8, channel scaling, and
  train-only ridge/SVD ranks 8/16.
- Screen training set `eval_test_during_train=0` and `defer_test=1`.
- A selector rejected incomplete runs or any run that had already opened test.
- Resource ratios were measured with synthetic tensors on one H100 to remove
  cross-node scheduler noise.
- After validation selection, frozen checkpoints were evaluated once on test.
- Final confirmation reran dense, `fk_r8_cs`, and plain `fk_r8` with the final
  optimized implementation for seeds 2024/2025/2026.
- All jobs requested one H100 each.  The final code passes 159/159 repository
  tests.

## Three-seed result

| Arm | Validation MSE mean | Test MSE mean ± pop. std | Paired wins | Test MAE mean | Parameters | Fixed train | Fixed inference | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense direct | 0.6853554 | 0.3425529 ± 0.0002046 | reference | 0.3386665 | 138,306 | 1.000x | 1.000x | reference |
| **rank-8 + channel scale** | **0.6853062** | **0.3418097 ± 0.0001664** | **3/3** | **0.3384249** | **75,882** | **0.947x** | **0.958x** | **stable win** |
| rank-8 shared | 0.6855741 | 0.3426983 ± 0.0000321 | 1/3 | 0.3387700 | 75,714 | 0.915x | 0.939x | resource-only / accuracy tie |

The winning kernel reduces mean test MSE by 0.0007431 (0.2169%) and parameters
by 45.13% relative to dense direct.  Controlled fixed-work training and
inference are 5.3% and 4.2% faster.  Peak CUDA allocation is nearly unchanged
because activations dominate: 143.44 MiB versus 143.99 MiB in training.

### Seed-matched test MSE

| Seed | Dense direct | Rank-8 + channel scale | Delta |
| ---: | ---: | ---: | ---: |
| 2024 | 0.3423641 | 0.3420451 | -0.0003190 |
| 2025 | 0.3424574 | 0.3416929 | -0.0007645 |
| 2026 | 0.3428371 | 0.3416911 | -0.0011460 |

The winning mean remains 0.0031097 (0.918%) above the published 0.3387
reference.  None of the three seeds beats that reference.

## Ablation decisions

- Keep `fk_r8_cs`: channel-specific mode amplitudes matter even though channel
  values are never mixed.  Its learned fusion gate remains open
  (`mean=0.0461`) rather than collapsing.
- Do not promote plain `fk_r8`: it saves resources but loses 2/3 paired tests
  and has worse mean validation MSE.
- Do not promote ridge/SVD initialization: rank-8 and rank-16 validation MSE
  were 0.6879948 and 0.6879045.  The train-only artifact was valid and finite,
  but the normalized least-squares objective was dominated by hard/outlier
  future windows and did not improve the trained model.
- The initial generic `einsum` plus per-batch matrix-rank SVD erased the
  theoretical compute benefit.  Replacing it with two batched matrix
  multiplications and an O(R) mode-activity diagnostic restored the expected
  speedup without changing the mathematical operator.

## Next stage

Stage A passes, so a Stage-B smooth spectral-mixture envelope is justified.
It should be added only within each variable and compared against `fk_r8_cs`.
A complex within-variable phase kernel should be attempted only if the real SM
version first wins.  Cross-variable lead--lag remains pruned.
