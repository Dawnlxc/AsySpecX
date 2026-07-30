# AsySpecX Phase 1 Summary

- total_runs: 168
- ok_runs: 168
- failed_runs: 0
- results_csv: phase4_results/main/results.csv

## Arm Means

| arm | n | val_mse_mean | mse_mean | mae_mean |
| --- | ---: | ---: | ---: | ---: |
| phase4_asx_cross | 24 | 0.311559 | 0.19893 | 0.270275 |
| phase4_asx_cross_revin | 24 | 0.311667 | 0.198835 | 0.26885 |
| phase4_asx_individual | 24 | 0.312439 | 0.196715 | 0.267222 |
| phase4_asx_individual_period | 24 | 0.311565 | 0.19578 | 0.265862 |
| phase4_asx_individual_revin | 24 | 0.31199 | 0.196733 | 0.265817 |
| phase4_asx_period_multi | 24 | 0.309633 | 0.196581 | 0.266393 |
| phase4_asx_period_single | 24 | 0.309782 | 0.196743 | 0.266716 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| electricity | 720 | 96 | phase4_asx_period_multi | 0.138265 | 0.238301 |
| electricity | 720 | 192 | phase4_asx_period_multi | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase4_asx_period_multi | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase4_asx_period_multi | 0.203556 | 0.293765 |
| weather | 720 | 96 | phase4_asx_individual_revin | 0.147851 | 0.200624 |
| weather | 720 | 192 | phase4_asx_individual_revin | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase4_asx_individual_revin | 0.240016 | 0.279124 |
| weather | 720 | 720 | phase4_asx_individual | 0.309688 | 0.331898 |

