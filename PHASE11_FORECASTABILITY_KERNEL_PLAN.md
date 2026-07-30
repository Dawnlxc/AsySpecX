# AsySpecX Phase 11: Horizon-Conditioned Forecastability Kernel

## Decision being tested

Cross-variable lead--lag is not a Phase 11 hypothesis. Existing evidence is
treated as a hard pruning signal: after conditioning on each variable's own
history and periodic structure, cross-variable directional mixing has not shown
stable incremental forecast value.

Phase 11 moves the asymmetry to the correct axis: past time coordinates and
future horizon coordinates. The Stage-A kernel is a channel-independent
past-to-future operator

\[
    \widehat Y_{b,h,c}=\sum_{r=1}^{R}U_{h,r}s_{c,r}
    \sum_{t=1}^{T}V_{r,t}X_{b,t,c},
\]

where `U` and `V` are shared across channels and `s` is an optional
channel-by-mode scale. No value from channel `c1` is ever used to predict
channel `c2`.

This is a standalone normalized-scale forecast branch. It is fused with the
spectral backbone either convexly or additively through a bounded gate.

## Stage A: real low-rank forecastability kernel

### Implementation

1. Add `ForecastabilityAdapter` to `models/AsySpecX.py`.
2. Add a `forecast_kernel=lowrank_time` model option. Legacy behavior with
   `forecast_kernel=none` must be bit-for-bit unchanged.
3. Support ranks 4, 8, and 16; optional channel-by-mode scaling is allowed,
   but cross-channel contractions are forbidden.
4. Add training-only ridge/SVD initialization:
   - construct normalized train windows only (including the current
     identity-initialized affine RevIN convention);
   - accumulate `Cxx = X^T X` and `Cyx = Y^T X` without materializing all
     windows;
   - solve `W = Cyx (Cxx + lambda I)^-1`;
   - truncate the SVD of `W` and split each singular value symmetrically into
     the future and past factors;
   - store an auditable artifact with data split, rank, ridge, shapes, sample
     count, and retained Frobenius energy.
5. Add diagnostics for gate statistics, raw forecast RMS, fused delta RMS,
   effective rank, and whether channel scaling is active.

### Correctness gates

- Legacy models load and produce identical predictions when the kernel is off.
- Output shape is `[B, H, C]` for arbitrary `T`, `H`, and `C`.
- A manual dense `U @ diag(s_c) @ V` calculation matches the module.
- Changing one input channel cannot change another output channel.
- Gradients reach both factors, channel scales, and the fusion gate.
- Initializer rejects wrong dataset/rank/shape metadata.
- The initializer reads the train split only.

## Stage-A experiment matrix

Primary cell: Weather, `seq_len=96`, `pred_len=720`, `cut_freq=13`.

| Arm | Kernel | Rank | Init | Channel scale | Fusion |
| --- | --- | ---: | --- | --- | --- |
| `dense_direct` | existing direct linear | dense | zeros | shared | additive |
| `fk_r4` | low-rank time | 4 | small random | off | convex |
| `fk_r8` | low-rank time | 8 | small random | off | convex |
| `fk_r8_cs` | low-rank time | 8 | small random | on | convex |
| `fk_r8_svd` | low-rank time | 8 | train ridge/SVD | on | convex |
| `fk_r16_svd` | low-rank time | 16 | train ridge/SVD | on | convex |

All arms keep the same spectral/cycle backbone. `dense_direct` is the capacity
control. A no-kernel anchor is retained from the same current code line or
rerun when scheduler budget permits.

### Execution ladder

1. Unit tests and CPU synthetic smoke.
2. One H100 canary per implementation family with 2--3 epochs.
3. One-seed full screen (`seed=2026`) for all valid Stage-A arms.
4. Set `eval_test_during_train=0` and `defer_test=1` for the full screen.
   Select by validation MSE only, then open test metrics from frozen selected
   checkpoints in a separate evaluation command.
5. Promote at most two arms to seeds 2024/2025/2026.

### Promotion rule

An arm advances from the one-seed screen only when all conditions hold:

- successful completion with finite metrics;
- validation MSE is no worse than 0.5% above the strongest equal-input-budget
  anchor, or it strictly improves validation MSE;
- parameter count is below the dense direct branch;
- no material fixed-work train/inference regression beyond 10%; train runtime
  is compared per completed epoch so a candidate is not penalized merely for
  continuing to improve after an anchor has early-stopped;
- the learned fusion gate is not uniformly collapsed unless validation improves.

A three-seed result is called a win only when at least two of three seeds beat
the equal-budget anchor and mean validation MSE improves. Otherwise it is a tie
or a resource-only result.

## Stage B: only if Stage A passes

Stage B replaces the free past basis with a smooth spectral-mixture envelope
and then, in a separate ablation, adds a bounded within-variable horizon phase.
Neither cross-channel mixing nor a dynamic router is allowed.

The order is mandatory:

1. real low-rank kernel;
2. real kernel plus smooth spectral-mixture envelope;
3. complex within-variable horizon phase.

If Stage A fails, Stage B is cancelled. If the real spectral-mixture version
works but complex phase does not, the complex version is discarded.

## Reported measurements

Every retained run reports validation/test MSE and MAE, parameter count, train
time, full-test inference time, peak CUDA memory, fusion-gate statistics, and
the initializer's retained singular-value energy. Accuracy, resource-only
improvements, ties, and failures are reported separately.
