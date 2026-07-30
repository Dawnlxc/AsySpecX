# AsySpecX Phase 1 Summary

- total_runs: 864
- ok_runs: 864
- failed_runs: 0
- results_csv: phase6_results/fullfield/results.csv

## Arm Means

| arm | n | val_mse_mean | mse_mean | mae_mean |
| --- | ---: | ---: | ---: | ---: |
| phase6_asx_cross | 144 | 0.522756 | 0.342646 | 0.34807 |
| phase6_asx_cross_clip05 | 144 | 0.534708 | 0.346954 | 0.352163 |
| phase6_asx_individual | 144 | 0.576115 | 0.380614 | 0.368741 |
| phase6_asx_individual_period | 144 | 0.57468 | 0.379273 | 0.367375 |
| phase6_asx_individual_revin | 144 | 0.573946 | 0.381989 | 0.368918 |
| phase6_asx_period_multi | 144 | 0.520418 | 0.338568 | 0.344618 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| ETTh1 | 96 | 96 | phase6_asx_cross_clip05 | 0.388208 | 0.397494 |
| ETTh1 | 96 | 192 | phase6_asx_cross_clip05 | 0.435523 | 0.425184 |
| ETTh1 | 96 | 336 | phase6_asx_cross_clip05 | 0.474925 | 0.445853 |
| ETTh1 | 96 | 720 | phase6_asx_cross_clip05 | 0.462566 | 0.462729 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.379997 | 0.403904 |
| ETTh1 | 720 | 192 | phase6_asx_cross_clip05 | 0.414458 | 0.423751 |
| ETTh1 | 720 | 336 | phase6_asx_cross_clip05 | 0.433757 | 0.438276 |
| ETTh1 | 720 | 720 | phase6_asx_cross_clip05 | 0.431043 | 0.455236 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.351353 | 0.379215 |
| ETTm1 | 96 | 192 | phase6_asx_cross_clip05 | 0.388281 | 0.394429 |
| ETTm1 | 96 | 336 | phase6_asx_cross_clip05 | 0.416828 | 0.412478 |
| ETTm1 | 96 | 720 | phase6_asx_cross_clip05 | 0.478628 | 0.444328 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.307015 | 0.351471 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.343985 | 0.370746 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.375191 | 0.388813 |
| ETTm1 | 720 | 720 | phase6_asx_period_multi | 0.423044 | 0.419741 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.0931334 | 0.204459 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.138738 | 0.255873 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.249868 | 0.355835 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.381494 | 0.455779 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.0914837 | 0.199324 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.149582 | 0.257745 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.29149 | 0.37999 |
| PEMS08 | 96 | 96 | phase6_asx_period_multi | 0.561115 | 0.545434 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.194684 | 0.279893 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.194564 | 0.281049 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.208904 | 0.295086 |
| electricity | 96 | 720 | phase6_asx_individual_period | 0.249786 | 0.327229 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.138265 | 0.238301 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.203556 | 0.293765 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.621603 | 0.392252 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.584051 | 0.370466 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.590953 | 0.37248 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.624815 | 0.388458 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.388673 | 0.278225 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.400557 | 0.281291 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.411927 | 0.285796 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.448077 | 0.304166 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.187501 | 0.239589 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.231881 | 0.274434 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.283318 | 0.308607 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.3567 | 0.353665 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.147851 | 0.200624 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase6_asx_individual_revin | 0.240016 | 0.279124 |
| weather | 720 | 720 | phase6_asx_individual | 0.309688 | 0.331898 |

