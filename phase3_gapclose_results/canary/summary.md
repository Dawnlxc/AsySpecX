# AsySpecX Phase 3-GapClose Summary

- total_runs: 4
- ok_runs: 4
- failed_runs: 0
- results_csv: phase3_gapclose_results/canary/results.csv

## Arm Means

| arm | n | val_mse_mean | mse_mean | mae_mean |
| --- | ---: | ---: | ---: | ---: |
| phase3_anchor_sparse_period | 1 | 0.832193 | 0.358616 | 0.38114 |
| phase3_fits_individual | 1 | 0.981029 | 0.38447 | 0.398524 |
| phase3_fits_shared | 1 | 0.851692 | 0.36483 | 0.385073 |
| phase3_fits_shared_revin_affine | 1 | 0.85126 | 0.364666 | 0.384916 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| weather | 720 | 96 | phase3_anchor_sparse_period | 0.358616 | 0.38114 |

