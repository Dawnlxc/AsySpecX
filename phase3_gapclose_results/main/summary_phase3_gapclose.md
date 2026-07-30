# Phase 3-GapClose Summary

- total_ok_runs: 192
- csv: phase3_gapclose_results/main/results.csv

## Arm Means

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase3_anchor_hier_split | 16 | 0.199457 | 0.270836 | 0.311361 |
| phase3_anchor_revin_affine | 16 | 0.199286 | 0.269433 | 0.311697 |
| phase3_anchor_sparse_period | 16 | 0.197288 | 0.267364 | 0.309613 |
| phase3_diag_only_weather_guard | 16 | 0.198111 | 0.268881 | 0.317302 |
| phase3_fits_individual | 16 | 0.196689 | 0.267174 | 0.312407 |
| phase3_fits_shared | 16 | 0.206272 | 0.275327 | 0.328353 |
| phase3_fits_shared_revin_affine | 16 | 0.206277 | 0.274475 | 0.327438 |
| phase3_fits_shared_subtract_last | 16 | 0.206306 | 0.275206 | 0.328255 |
| phase3_fits_sparse_period | 16 | 0.205226 | 0.273765 | 0.327393 |
| phase3_individual_hier_split | 16 | 0.203263 | 0.275062 | 0.313554 |
| phase3_individual_sparse_period | 16 | 0.195611 | 0.265607 | 0.311411 |
| phase3_offdiag_only_anchor | 16 | 0.205365 | 0.274802 | 0.320764 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| electricity | 720 | 192 | phase3_anchor_sparse_period | 0.15264832973480225 | 0.2504502236843109 |
| electricity | 720 | 336 | phase3_anchor_sparse_period | 0.16691282391548157 | 0.2643808424472809 |
| electricity | 720 | 720 | phase3_anchor_sparse_period | 0.20349211990833282 | 0.2941526174545288 |
| electricity | 720 | 96 | phase3_anchor_sparse_period | 0.1373506337404251 | 0.2373502552509308 |
| weather | 720 | 192 | phase3_fits_individual | 0.19044892489910126 | 0.24258245527744293 |
| weather | 720 | 336 | phase3_fits_individual | 0.2405073642730713 | 0.28201237320899963 |
| weather | 720 | 720 | phase3_fits_individual | 0.3096681237220764 | 0.3318565785884857 |
| weather | 720 | 96 | phase3_fits_individual | 0.14791370928287506 | 0.20193377137184143 |

## Best Cell Count

| arm | n |
| --- | ---: |
| phase3_anchor_sparse_period | 4 |
| phase3_fits_individual | 4 |

## Delta Versus Anchor

| arm | cells | delta_mse_mean | delta_mae_mean |
| --- | ---: | ---: | ---: |
| phase3_anchor_revin_affine | 16 | -0.000171037 | -0.00140257 |
| phase3_anchor_sparse_period | 16 | -0.00216851 | -0.00347198 |
| phase3_diag_only_weather_guard | 16 | -0.00134627 | -0.00195497 |
| phase3_fits_individual | 16 | -0.0027675 | -0.00366194 |
| phase3_fits_shared | 16 | 0.00681536 | 0.00449099 |
| phase3_fits_shared_revin_affine | 16 | 0.00682016 | 0.00363935 |
| phase3_fits_shared_subtract_last | 16 | 0.0068493 | 0.00437051 |
| phase3_fits_sparse_period | 16 | 0.00576867 | 0.00292877 |
| phase3_individual_hier_split | 16 | 0.00380648 | 0.00422606 |
| phase3_individual_sparse_period | 16 | -0.00384623 | -0.00522921 |
| phase3_offdiag_only_anchor | 16 | 0.00590812 | 0.00396588 |

## Dataset Notes

### weather

- runs: 96
- mse_mean: 0.233341
- mae_mean: 0.274789

### electricity

- runs: 96
- mse_mean: 0.169851
- mae_mean: 0.2682

