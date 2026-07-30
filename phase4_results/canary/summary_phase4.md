# Phase 4-Finalize Summary

Validation selection is performed using val_mse averaged over replicate seeds for each dataset/seq_len/pred_len group. Test metrics are used only after selection.

- total_ok_runs: 6
- csv: phase4_results/canary/results.csv

## Arm Means

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase4_asx_individual | 2 | 0.830125 | 0.640558 | 1.0455 |
| phase4_asx_period_multi | 2 | 0.711086 | 0.593305 | 0.899706 |
| phase4_asx_period_single | 2 | 0.710887 | 0.593227 | 0.899641 |

## Best Arm Per Dataset/Seq_len/Pred_len BY TEST (analysis only -- not for selection)

| dataset | seq_len | pred_len | best_arm | mse_mean | mae_mean |
| --- | ---: | ---: | --- | ---: | ---: |
| electricity | 720 | 96 | phase4_asx_period_single | 1.09272 | 0.824383 |
| weather | 720 | 96 | phase4_asx_period_single | 0.32905 | 0.362071 |

## Validation-Selected Summary

- selected_test_mse_mean: 0.710887
- selected_test_mae_mean: 0.593227

### Selected Arm Counts (per group)

| arm | groups |
| --- | ---: |
| phase4_asx_period_single | 2 |

### Selected Arm Per Dataset/Pred_len

| dataset | pred_len | selected_arm | test_mse_mean | test_mae_mean |
| --- | ---: | --- | ---: | ---: |
| electricity | 96 | phase4_asx_period_single | 1.09272 | 0.824383 |
| weather | 96 | phase4_asx_period_single | 0.32905 | 0.362071 |

### Per Dataset Selected

| dataset | test_mse_mean | test_mae_mean |
| --- | ---: | ---: |
| electricity | 1.09272 | 0.824383 |
| weather | 0.32905 | 0.362071 |

## Single-Arm Candidate Summary

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase4_asx_cross | 0 | | | |
| phase4_asx_individual | 2 | 0.830125 | 0.640558 | 1.0455 |
| phase4_asx_period_single | 2 | 0.710887 | 0.593227 | 0.899641 |
| phase4_asx_period_multi | 2 | 0.711086 | 0.593305 | 0.899706 |
| phase4_asx_individual_period | 0 | | | |

## Fairness Note

Validation selection is performed using val_mse averaged over replicate seeds for each dataset/seq_len/pred_len group. Test metrics are used only after selection.

