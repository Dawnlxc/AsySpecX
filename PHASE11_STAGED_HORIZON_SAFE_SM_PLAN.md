# AsySpecX Phase 11 Stage D: Horizon-Safe Spectral Mixture

## Decision boundary

Stage C rejected both unshrunk Stage-B extensions as general upgrades. Stage D
does not reopen that gate or tune a new threshold on the five Stage-C cells.
It tests one new, predeclared hypothesis:

> A real spectral-mixture correction is useful only when the forecast extends
> materially beyond the observed context. Before that point it should recover
> Stage A exactly rather than asking training to learn an approximate identity.

Complex phase remains pruned. Cross-variable mixing is also excluded from the
Stage-D backbones, so the entire tested path is channel-separable.

## Kernel

Let `L` be the input length, `H` the prediction length, and let the unshrunk
real-SM response of temporal mode `r` be

\[
A_r(\omega)=\exp\left(\gamma_r\,\widetilde m_r(\omega)\right),
\]

where `m_tilde` is the centered bounded mixture log-shape. Stage D introduces
no learned parameters. It fixes

\[
\rho(L,H)=\max\left(0,1-\frac{2L}{H}\right)
\]

and uses

\[
A_r^{safe}(\omega)=
\exp\left(\rho(L,H)\gamma_r\,\widetilde m_r(\omega)\right).
\]

The arm is named `fk_sm2_tail2`.

- `H <= 2L`: `rho=0`; the effective basis is exactly the Stage-A basis, the
  FFT path is bypassed, and SM parameters receive no gradient.
- `L=96, H=336`: `rho=3/7 = 0.428571...`.
- `L=96, H=720`: `rho=11/15 = 0.733333...`.
- the schedule is deterministic, monotone, shared across datasets/channels,
  and is not tunable inside Stage D.

The comparison arms are:

| Arm | Role |
| --- | --- |
| `fk_r8_cs` | Stage-A learned asymmetric past-to-future kernel |
| `fk_sm2_mode` | unshrunk real-SM control from Stage B |
| `fk_sm2_tail2` | zero-parameter horizon-safe real SM |

## Data and backbone lock

Every Stage-D row uses `ind_cycle_full`: individual spectral lift, ReVIN
affine normalization, no cross block, no temporal adapter, and full cycle
residual. This is deliberately stricter than the best historical Traffic
profile because it removes cross-variable mixing as a confound.

Dataset files were present and non-empty before implementation:

| Dataset | File SHA-256 |
| --- | --- |
| ETTm1 | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` |
| Traffic | `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16` |
| Weather | `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63` |

The formal seed-2026 screen is validation-only:

| Role | Dataset | Seq | Horizons | Cut frequency | Count |
| --- | --- | ---: | --- | ---: | ---: |
| seen bridge | Weather | 96 | 720 | 13 | 1 cell x 3 arms |
| unseen identity | ETTm1 | 96 | 96, 192 | 7 | 2 cells x 3 arms |
| unseen active | ETTm1 | 96 | 336, 720 | 7 | 2 cells x 3 arms |
| unseen identity | Traffic | 96 | 96, 192 | 25 | 2 cells x 3 arms |
| unseen active | Traffic | 96 | 336, 720 | 25 | 2 cells x 3 arms |

Total: 9 cells and 27 full training jobs. Weather is a bridge/audit only and
cannot select the arm. ETTm1 and Traffic were not run in Stage C and form the
new generalization screen.

All training uses `eval_test_during_train=0` and `defer_test=1`. A selector
must reject the entire matrix if any input row has non-null test MSE/MAE or
`test_deferred != true`.

## Execution ladder

1. Model/CLI/config unit tests, including exact output, weight, and gradient
   identity at `rho=0`.
2. Full local Phase-11 regression and full remote repository regression.
3. Two-epoch H100 canaries spanning one identity and two active profiles.
4. Same-H100 synthetic fixed-work smoke audits.
5. Formal seed-2026 validation-only screen (27 jobs).
6. Fixed-work audits on Traffic H192, ETTm1 H720, and Traffic H720.
7. Frozen aggregate selection.
8. Confirmation/test opening only if every Stage-D gate passes.

## Frozen screen gate

### Integrity and identity

- all 27 rows are finite and `status=ok`;
- no test metric is present or used;
- `fk_sm2_tail2` reports the exact predeclared `rho` for every cell;
- on all four identity cells, its validation MSE differs from `fk_r8_cs` by
  at most `1e-7`, SM factor min/max are exactly one within `1e-7`, and the SM
  gate remains zero within `1e-8`;
- active rows have finite, non-collapsed SM diagnostics and bounded factors;
- parameter ratio versus Stage A is at most `1.01`.

### Unseen active-cell accuracy

Across ETTm1/Traffic H336/H720 only, `fk_sm2_tail2` must:

- strictly beat `fk_r8_cs` in at least 3/4 cells;
- strictly beat unshrunk `fk_sm2_mode` in at least 3/4 cells;
- have negative equal-cell median and mean relative validation delta versus
  Stage A;
- have worst relative regression versus Stage A no larger than `0.05%`.

The already-seen Weather H720 bridge must not regress more than `0.02%`
relative to Stage A. It is not counted among the 3/4 wins.

### Fixed-work resources

On both active H720 audits, the maximum `fk_sm2_tail2` ratio versus Stage A
must be:

- train forward/backward <= `1.10`;
- inference <= `1.05`;
- peak CUDA memory <= `1.02`;
- parameters <= `1.01`.

On the Traffic H192 identity audit, train, inference, and peak-memory ratios
must each be <= `1.02`, demonstrating that the zero-scale fast path is real.

If any condition fails, Stage D stops without test evaluation. The correct
conclusion is that `fk_r8_cs` remains the transferable kernel and horizon-safe
SM is either an identity wrapper or another cell-specific refinement.

## Confirmation gate, only after screen pass

Train validation-only seed-2024 and seed-2025 rows for `fk_r8_cs` and
`fk_sm2_tail2` on ETTm1/Traffic H336/H720, then combine them with the frozen
seed-2026 screen checkpoints. The selected checkpoints may be evaluated on
test only after mean validation remains lower for `fk_sm2_tail2`.

The final generalization claim requires:

- at least 8/12 paired test-MSE wins;
- negative equal-cell macro-mean relative test-MSE delta;
- a positive aggregate test result in both ETTm1 and Traffic;
- all resource gates retained.

Otherwise Stage D is reported as a tie, dataset-specific result, or
resource-only outcome. Test metrics may never be used to choose a schedule,
threshold, dataset, or horizon.
