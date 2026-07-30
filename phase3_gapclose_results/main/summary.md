# AsySpecX Phase 3-GapClose Summary

- total_runs: 192
- ok_runs: 192
- failed_runs: 0
- results_csv: phase3_gapclose_results/main/results.csv

## Arm Means

| arm | n | val_mse_mean | mse_mean | mae_mean |
| --- | ---: | ---: | ---: | ---: |
| phase3_anchor_hier_split | 16 | 0.311361 | 0.199457 | 0.270836 |
| phase3_anchor_revin_affine | 16 | 0.311697 | 0.199286 | 0.269433 |
| phase3_anchor_sparse_period | 16 | 0.309613 | 0.197288 | 0.267364 |
| phase3_diag_only_weather_guard | 16 | 0.317302 | 0.198111 | 0.268881 |
| phase3_fits_individual | 16 | 0.312407 | 0.196689 | 0.267174 |
| phase3_fits_shared | 16 | 0.328353 | 0.206272 | 0.275327 |
| phase3_fits_shared_revin_affine | 16 | 0.327438 | 0.206277 | 0.274475 |
| phase3_fits_shared_subtract_last | 16 | 0.328255 | 0.206306 | 0.275206 |
| phase3_fits_sparse_period | 16 | 0.327393 | 0.205226 | 0.273765 |
| phase3_individual_hier_split | 16 | 0.313554 | 0.203263 | 0.275062 |
| phase3_individual_sparse_period | 16 | 0.311411 | 0.195611 | 0.265607 |
| phase3_offdiag_only_anchor | 16 | 0.320764 | 0.205365 | 0.274802 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| electricity | 720 | 96 | phase3_anchor_sparse_period | 0.138135 | 0.23821 |
| electricity | 720 | 192 | phase3_anchor_sparse_period | 0.152705 | 0.250575 |
| electricity | 720 | 336 | phase3_anchor_sparse_period | 0.167234 | 0.264617 |
| electricity | 720 | 720 | phase3_anchor_sparse_period | 0.203856 | 0.29472 |
| weather | 720 | 96 | phase3_fits_individual | 0.147974 | 0.202081 |
| weather | 720 | 192 | phase3_fits_individual | 0.190554 | 0.242698 |
| weather | 720 | 336 | phase3_fits_individual | 0.240546 | 0.282079 |
| weather | 720 | 720 | phase3_fits_individual | 0.309721 | 0.331902 |

