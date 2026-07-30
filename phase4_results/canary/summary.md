# AsySpecX Phase 1 Summary

- total_runs: 6
- ok_runs: 6
- failed_runs: 0
- results_csv: phase4_results/canary/results.csv

## Arm Means

| arm | n | val_mse_mean | mse_mean | mae_mean |
| --- | ---: | ---: | ---: | ---: |
| phase4_asx_individual | 2 | 1.0455 | 0.830125 | 0.640558 |
| phase4_asx_period_multi | 2 | 0.899706 | 0.711086 | 0.593305 |
| phase4_asx_period_single | 2 | 0.899641 | 0.710887 | 0.593227 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| electricity | 720 | 96 | phase4_asx_period_single | 1.09272 | 0.824383 |
| weather | 720 | 96 | phase4_asx_period_single | 0.32905 | 0.362071 |

