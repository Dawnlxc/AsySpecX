# AsySpecX Phase 1 Summary

- total_runs: 864
- ok_runs: 864
- failed_runs: 0
- results_csv: phase2_results/main/results.csv

## Arm Means

| arm | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| phase2_global_all | 96 | 0.345011 | 0.350067 |
| phase2_global_all_clip05 | 96 | 0.353282 | 0.354132 |
| phase2_global_diag_only | 96 | 0.381957 | 0.369351 |
| phase2_global_offdiag_only | 96 | 0.344918 | 0.349984 |
| phase2_global_split | 96 | 0.344553 | 0.349757 |
| phase2_hier_all | 96 | 0.344668 | 0.349154 |
| phase2_hier_all_clip05 | 96 | 0.347917 | 0.352805 |
| phase2_hier_split | 96 | 0.34339 | 0.348216 |
| phase2_self_band_gain_global | 96 | 0.383633 | 0.370266 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| ETTh1 | 96 | 96 | phase2_global_offdiag_only | 0.388656 | 0.397451 |
| ETTh1 | 96 | 192 | phase2_hier_all_clip05 | 0.434544 | 0.424038 |
| ETTh1 | 96 | 336 | phase2_hier_all_clip05 | 0.476338 | 0.448984 |
| ETTh1 | 96 | 720 | phase2_hier_all_clip05 | 0.462827 | 0.463423 |
| ETTh1 | 720 | 96 | phase2_global_diag_only | 0.379211 | 0.402406 |
| ETTh1 | 720 | 192 | phase2_global_diag_only | 0.41367 | 0.422651 |
| ETTh1 | 720 | 336 | phase2_hier_all_clip05 | 0.432953 | 0.437562 |
| ETTh1 | 720 | 720 | phase2_global_all | 0.432241 | 0.45799 |
| ETTm1 | 96 | 96 | phase2_hier_all_clip05 | 0.351375 | 0.37879 |
| ETTm1 | 96 | 192 | phase2_hier_all_clip05 | 0.38752 | 0.393861 |
| ETTm1 | 96 | 336 | phase2_hier_all_clip05 | 0.41739 | 0.412666 |
| ETTm1 | 96 | 720 | phase2_hier_all_clip05 | 0.479091 | 0.444227 |
| ETTm1 | 720 | 96 | phase2_global_diag_only | 0.311967 | 0.355447 |
| ETTm1 | 720 | 192 | phase2_global_diag_only | 0.344648 | 0.373797 |
| ETTm1 | 720 | 336 | phase2_global_all_clip05 | 0.372374 | 0.390214 |
| ETTm1 | 720 | 720 | phase2_global_all_clip05 | 0.420347 | 0.415759 |
| PEMS04 | 96 | 12 | phase2_hier_split | 0.093382 | 0.204917 |
| PEMS04 | 96 | 24 | phase2_hier_split | 0.139701 | 0.256739 |
| PEMS04 | 96 | 48 | phase2_hier_split | 0.251006 | 0.358373 |
| PEMS04 | 96 | 96 | phase2_hier_split | 0.391775 | 0.461033 |
| PEMS08 | 96 | 12 | phase2_hier_split | 0.0919475 | 0.199896 |
| PEMS08 | 96 | 24 | phase2_hier_split | 0.150822 | 0.258565 |
| PEMS08 | 96 | 48 | phase2_hier_split | 0.291008 | 0.37986 |
| PEMS08 | 96 | 96 | phase2_hier_split | 0.578502 | 0.553007 |
| electricity | 96 | 96 | phase2_hier_all_clip05 | 0.200682 | 0.286465 |
| electricity | 96 | 192 | phase2_hier_all_clip05 | 0.200776 | 0.288331 |
| electricity | 96 | 336 | phase2_hier_all_clip05 | 0.215392 | 0.302677 |
| electricity | 96 | 720 | phase2_hier_all_clip05 | 0.254075 | 0.332995 |
| electricity | 720 | 96 | phase2_global_diag_only | 0.141434 | 0.243053 |
| electricity | 720 | 192 | phase2_hier_all_clip05 | 0.155242 | 0.256887 |
| electricity | 720 | 336 | phase2_hier_all_clip05 | 0.169876 | 0.271712 |
| electricity | 720 | 720 | phase2_hier_all_clip05 | 0.207247 | 0.302379 |
| traffic | 96 | 96 | phase2_hier_all_clip05 | 0.621731 | 0.390616 |
| traffic | 96 | 192 | phase2_hier_all_clip05 | 0.583513 | 0.374748 |
| traffic | 96 | 336 | phase2_hier_all_clip05 | 0.592421 | 0.372266 |
| traffic | 96 | 720 | phase2_hier_all_clip05 | 0.627568 | 0.391903 |
| traffic | 720 | 96 | phase2_hier_all_clip05 | 0.389636 | 0.279721 |
| traffic | 720 | 192 | phase2_global_split | 0.402306 | 0.283253 |
| traffic | 720 | 336 | phase2_hier_all_clip05 | 0.414726 | 0.290014 |
| traffic | 720 | 720 | phase2_global_all_clip05 | 0.454384 | 0.307868 |
| weather | 96 | 96 | phase2_hier_split | 0.19256 | 0.247705 |
| weather | 96 | 192 | phase2_hier_split | 0.241855 | 0.284618 |
| weather | 96 | 336 | phase2_hier_split | 0.293175 | 0.318162 |
| weather | 96 | 720 | phase2_hier_all | 0.365184 | 0.362315 |
| weather | 720 | 96 | phase2_hier_split | 0.152591 | 0.211253 |
| weather | 720 | 192 | phase2_global_diag_only | 0.194515 | 0.247364 |
| weather | 720 | 336 | phase2_global_diag_only | 0.244673 | 0.286226 |
| weather | 720 | 720 | phase2_global_diag_only | 0.313946 | 0.335363 |

