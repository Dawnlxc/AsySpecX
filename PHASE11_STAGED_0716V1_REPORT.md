# AsySpecX Phase 11 Stage D result

Date: 2026-07-16

## Verdict

Reject `fk_sm2_tail2` as a transferable time-series kernel upgrade. Keep
`fk_r8_cs` as the validated asymmetric past-to-future kernel.

Stage D stopped before confirmation and before test evaluation. The frozen
screen became mathematically impossible after the ETTm1 rows completed, so
the remaining Weather/Traffic jobs were cancelled rather than spending more
GPU time on a candidate that could no longer pass.

## What Stage D tested

The candidate was a channel-separable, real spectral-mixture correction with
no cross-variable mixing and no complex phase. It introduced no new schedule
parameter:

`rho(L,H) = max(0, 1 - 2L/H)`.

For `H <= 2L`, it was required to recover Stage A exactly. For longer
horizons, the same schedule gradually activated the real SM correction.

## Integrity checks

- The authoritative remote repository regression passed 203/203 tests.
- All six two-epoch canaries completed with `test_deferred=true` and null test
  MSE/MAE.
- ETTm1 H96 and H192 recovered Stage A exactly: identical validation MSE,
  extension scale zero, SM factor one, and SM gate zero.
- Active H336/H720 rows reported the predeclared scales and non-collapsed,
  finite SM diagnostics.
- No cross-variable block was enabled in any Stage-D row.

## Formal validation evidence

| ETTm1 horizon | Stage A `fk_r8_cs` | unshrunk SM | tail2 | tail2 vs Stage A | tail2 vs unshrunk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.3810896 | 0.3810896 | 0.3810896 | +0.0000% | +0.0000% |
| 192 | 0.5036939 | 0.5036939 | 0.5036939 | +0.0000% | +0.0000% |
| 336 | 0.6522649 | 0.6534351 | 0.6529983 | **+0.1124%** | -0.0668% |
| 720 | 0.9712688 | 0.9719722 | 0.9718922 | **+0.0642%** | -0.0082% |

The shrink schedule softened the damage of the unshrunk SM correction, but it
did not beat Stage A in either completed active cell.

The frozen gate required at least 3/4 active-cell wins and a worst regression
no larger than +0.05%. After two active losses:

- wins so far were 0/2;
- even winning both remaining Traffic cells could produce only 2/4;
- both observed regressions exceeded +0.05%;
- the worst observed regression was +0.1124%.

Thus the accuracy gate was irrecoverably failed before any test metric was
opened.

## Formal fixed-work evidence

On the same H100 with ETTm1 L96/H720, tail2 relative to Stage A was:

| train forward/backward | inference | peak CUDA memory | parameters |
| ---: | ---: | ---: | ---: |
| **1.2549x** | 1.0322x | 1.0004x | 1.0037x |

The frozen active-resource limits were 1.10x, 1.05x, 1.02x, and 1.01x. The
training-time gate therefore failed independently of validation accuracy.

For context, the Traffic H720 canary resource smoke was inexpensive relative
to Stage A (train 1.0129x, inference 0.9839x, memory 0.9998x). The discrepancy
shows that SM overhead is hidden by the high-channel Traffic workload but is
material on low-channel ETTm1; it is not a uniformly cheap extension.

## Decision and interpretation

The Stage-D hypothesis was only half supported:

- supported: the deterministic schedule is a correct horizon-safe wrapper;
  it exactly preserves Stage A at short horizons and activates stably at long
  horizons;
- rejected: stable activation does not translate into transferable forecast
  gain, and its low-channel training overhead exceeds the budget.

This result does not rely on inter-variable lead-lag information: the tested
backbone was strictly channel-separable. Together with Stages B/C, the result
narrows the useful contribution to the learned asymmetric past-to-future
kernel itself (`fk_r8_cs`), not spectral-mixture or complex-phase decoration.

## Execution record

- Formal manifest: 27 planned validation-only rows.
- Completed before the decisive stop: all 12 ETTm1 rows, forming four complete
  three-arm cells.
- Cancelled after the gate became impossible: incomplete Weather/Traffic rows
  and their dependent resource jobs.
- Confirmation submitted: no.
- Test opened: no.
- Final decision: `advance_to_confirmation=0`, `open_test=0`.

Machine-readable evidence is under
`remote_phase11_staged_results/staged_screen/early_stop_selection/`.
