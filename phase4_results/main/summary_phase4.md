# Phase 4-Finalize Summary

Validation selection is performed using val_mse averaged over replicate seeds for each dataset/seq_len/pred_len group. Test metrics are used only after selection.

- total_ok_runs: 168
- csv: phase4_results/main/results.csv

## Arm Means

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase4_asx_cross | 24 | 0.19893 | 0.270275 | 0.311559 |
| phase4_asx_cross_revin | 24 | 0.198835 | 0.26885 | 0.311667 |
| phase4_asx_individual | 24 | 0.196715 | 0.267222 | 0.312439 |
| phase4_asx_individual_period | 24 | 0.19578 | 0.265862 | 0.311565 |
| phase4_asx_individual_revin | 24 | 0.196733 | 0.265817 | 0.31199 |
| phase4_asx_period_multi | 24 | 0.196581 | 0.266393 | 0.309633 |
| phase4_asx_period_single | 24 | 0.196743 | 0.266716 | 0.309782 |

## Best Arm Per Dataset/Seq_len/Pred_len BY TEST (analysis only -- not for selection)

| dataset | seq_len | pred_len | best_arm | mse_mean | mae_mean |
| --- | ---: | ---: | --- | ---: | ---: |
| electricity | 720 | 192 | phase4_asx_period_multi | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase4_asx_period_multi | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase4_asx_period_multi | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase4_asx_period_multi | 0.138265 | 0.238301 |
| weather | 720 | 192 | phase4_asx_individual_revin | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase4_asx_individual_revin | 0.240016 | 0.279124 |
| weather | 720 | 720 | phase4_asx_individual | 0.309688 | 0.331898 |
| weather | 720 | 96 | phase4_asx_individual_revin | 0.147851 | 0.200624 |

## Validation-Selected Summary

- selected_test_mse_mean: 0.195153
- selected_test_mae_mean: 0.264011

### Selected Arm Counts (per group)

| arm | groups |
| --- | ---: |
| phase4_asx_individual_revin | 2 |
| phase4_asx_period_multi | 6 |

### Selected Arm Per Dataset/Pred_len

| dataset | pred_len | selected_arm | test_mse_mean | test_mae_mean |
| --- | ---: | --- | ---: | ---: |
| electricity | 192 | phase4_asx_period_multi | 0.151979 | 0.249248 |
| electricity | 336 | phase4_asx_period_multi | 0.166828 | 0.263942 |
| electricity | 720 | phase4_asx_period_multi | 0.203556 | 0.293765 |
| electricity | 96 | phase4_asx_period_multi | 0.138265 | 0.238301 |
| weather | 192 | phase4_asx_individual_revin | 0.190203 | 0.240335 |
| weather | 336 | phase4_asx_period_multi | 0.249379 | 0.28983 |
| weather | 720 | phase4_asx_period_multi | 0.313159 | 0.336044 |
| weather | 96 | phase4_asx_individual_revin | 0.147851 | 0.200624 |

### Per Dataset Selected

| dataset | test_mse_mean | test_mae_mean |
| --- | ---: | ---: |
| electricity | 0.165157 | 0.261314 |
| weather | 0.225148 | 0.266709 |

## Single-Arm Candidate Summary

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase4_asx_cross | 24 | 0.19893 | 0.270275 | 0.311559 |
| phase4_asx_individual | 24 | 0.196715 | 0.267222 | 0.312439 |
| phase4_asx_period_single | 24 | 0.196743 | 0.266716 | 0.309782 |
| phase4_asx_period_multi | 24 | 0.196581 | 0.266393 | 0.309633 |
| phase4_asx_individual_period | 24 | 0.19578 | 0.265862 | 0.311565 |

## Fairness Note

Validation selection is performed using val_mse averaged over replicate seeds for each dataset/seq_len/pred_len group. Test metrics are used only after selection.

