# Phase 3-GapClose Summary

- total_ok_runs: 4
- csv: phase3_gapclose_results/canary/results.csv

## Arm Means

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase3_anchor_sparse_period | 1 | 0.358616 | 0.38114 | 0.832193 |
| phase3_fits_individual | 1 | 0.38447 | 0.398524 | 0.981029 |
| phase3_fits_shared | 1 | 0.36483 | 0.385073 | 0.851692 |
| phase3_fits_shared_revin_affine | 1 | 0.364666 | 0.384916 | 0.85126 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| weather | 720 | 96 | phase3_anchor_sparse_period | 0.35861560702323914 | 0.38113999366760254 |

## Best Cell Count

| arm | n |
| --- | ---: |
| phase3_anchor_sparse_period | 1 |

## Delta Versus Anchor

| arm | cells | delta_mse_mean | delta_mae_mean |
| --- | ---: | ---: | ---: |
| phase3_anchor_sparse_period | 0 |  |  |
| phase3_fits_individual | 0 |  |  |
| phase3_fits_shared | 0 |  |  |
| phase3_fits_shared_revin_affine | 0 |  |  |

## Dataset Notes

### weather

- runs: 4
- mse_mean: 0.368146
- mae_mean: 0.387413

