# AsySpecX Phase 1 Summary

- total_runs: 4
- ok_runs: 4
- failed_runs: 0
- results_csv: phase2_results/canary/results.csv

## Arm Means

| arm | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| phase2_global_all | 1 | 1.46048 | 0.806169 |
| phase2_global_diag_only | 1 | 1.46049 | 0.806169 |
| phase2_global_offdiag_only | 1 | 1.46048 | 0.806168 |
| phase2_self_band_gain_global | 1 | 1.46048 | 0.806173 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| ETTh1 | 96 | 96 | phase2_global_offdiag_only | 1.46048 | 0.806168 |

