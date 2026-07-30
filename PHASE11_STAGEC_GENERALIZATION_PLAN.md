# AsySpecX Phase 11 Stage C: Generalization and Falsification

## Decision

Stage B established a small, replicated gain on one frozen Weather
`seq_len=96, pred_len=720` cell. Stage C asks whether the gain belongs to a
time-series kernel family or is specific to that cell. No new kernel
parameters, channel mixing, router, or lead--lag mechanism may be added.

The only retained kernel arms are:

| Arm | Role |
| --- | --- |
| `anchor` | same dataset-specific backbone with no forecastability kernel |
| `fk_r8_cs` | real rank-8 Stage-A past-to-future kernel |
| `fk_sm2_mode` | smallest stable real spectral-mixture extension |
| `fk_sm4_ph4_h` | validation-selected bounded within-variable phase extension |

## Wave 1: established backbone transfer

Wave 1 uses cells whose backbone families already have audited Phase 10
evidence. Each arm changes only the forecastability-kernel option.

| Dataset | Base profile | Seq | Horizons | Cut frequency |
| --- | --- | ---: | --- | ---: |
| Weather | `ind_cycle_full` | 96 | 96, 192, 336 | 13 |
| Electricity | `compact_period_cycle_full` | 504 | 96, 336 | 127 |

The existing Weather H720 Stage-B result is frozen evidence and is not rerun
in the one-seed screen. Wave 1 therefore contains 5 new cells and 20 full
screen jobs.

### Execution ladder

1. Unit tests and shell/config validation.
2. Two-epoch canary for all four arms on one Weather and one Electricity cell
   (8 single-H100 jobs).
3. Seed-2026 full screen for all 20 rows with
   `eval_test_during_train=0` and `defer_test=1`.
4. Same-H100 fixed-work audits on Weather H336 and Electricity H336.
5. Aggregate validation-only selection. Test metrics remain unopened.

## Frozen Wave-1 gate

For each candidate (`fk_sm2_mode`, `fk_sm4_ph4_h`), compare the validation MSE
against both `fk_r8_cs` and `anchor` in the same cell. A candidate is eligible
for Wave 2 only when all conditions hold:

- strict validation win over Stage A in at least 4 of 5 new cells;
- strict validation win over the no-kernel anchor in at least 4 of 5 cells;
- macro-median relative validation delta versus Stage A is negative;
- worst relative validation regression versus Stage A is no more than 0.05%;
- all runs are finite, gates are active, and SM/phase diagnostics are bounded;
- parameter ratio versus Stage A is at most 1.15 on both profiles;
- on both fixed-work profiles, train ratio versus Stage A is at most 1.15,
  inference ratio is at most 1.10, and peak-memory ratio is at most 1.02.

If both candidates are eligible, complex phase is retained only if
`fk_sm4_ph4_h` beats `fk_sm2_mode` head-to-head in at least 3 of 5 validation
cells and has a lower macro-median relative validation delta versus Stage A.
Otherwise `fk_sm2_mode` is selected. No test metric may break a tie.

If neither candidate passes, Stage C stops: Stage B is reported as a
Weather-H720-specific result and complex phase is not promoted.

## Wave 2: cross-domain transfer, only if Wave 1 passes

The intended locked Wave-2 cells are:

| Dataset | Base profile | Seq | Horizons |
| --- | --- | ---: | --- |
| Traffic | dataset-audited profile, frozen before launch | 96 | 96, 336 |
| ETTm1 | dataset-audited profile, frozen before launch | 96 | 96, 720 |

Wave 2 repeats the validation-only protocol. The selected Stage-C candidate is
then confirmed against `fk_r8_cs` on four predeclared representative cells:
Weather H96, Electricity H336, Traffic H96, and ETTm1 H720, using seeds
2024/2025/2026. Test is opened only after validation selection.

The final generalization claim requires at least 8/12 paired test wins,
negative equal-cell macro-average relative test delta, and a positive aggregate
result in at least 3 of 4 datasets. Otherwise the result is dataset-specific,
a tie, or a resource-only outcome.

## Reporting

Every cell reports validation/test MSE and MAE when permitted, parameters,
training time, inference time, peak CUDA memory, kernel/SM/phase diagnostics,
and fixed-work ratios. Aggregate metrics weight cells equally so high-channel
datasets do not dominate by element count.
