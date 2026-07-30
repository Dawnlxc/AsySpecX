# Phase 6 Full-Field Summary

Report fixed single-arm and validation-selected separately. Oracle is analysis only and must not be reported as a valid selected model.

- total_runs: 864
- ok_runs: 864
- failed_runs: 0
- anchor_arm: phase6_asx_cross
- best_fixed_single_arm: phase6_asx_period_multi (mse_mean=0.338568)
- test_oracle_mse_mean (ANALYSIS ONLY): 0.333085

## Arm Means

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase6_asx_cross | 144 | 0.342646 | 0.34807 | 0.522756 |
| phase6_asx_cross_clip05 | 144 | 0.346954 | 0.352163 | 0.534708 |
| phase6_asx_individual | 144 | 0.380614 | 0.368741 | 0.576115 |
| phase6_asx_individual_period | 144 | 0.379273 | 0.367375 | 0.57468 |
| phase6_asx_individual_revin | 144 | 0.381989 | 0.368918 | 0.573946 |
| phase6_asx_period_multi | 144 | 0.338568 | 0.344618 | 0.520418 |

## Best Arm Per Dataset/Seq_len/Pred_len BY TEST (analysis only -- not for selection)

| dataset | seq_len | pred_len | best_arm | mse_mean | mae_mean |
| --- | ---: | ---: | --- | ---: | ---: |
| ETTh1 | 720 | 192 | phase6_asx_cross_clip05 | 0.414458 | 0.423751 |
| ETTh1 | 720 | 336 | phase6_asx_cross_clip05 | 0.433757 | 0.438276 |
| ETTh1 | 720 | 720 | phase6_asx_cross_clip05 | 0.431043 | 0.455236 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.379997 | 0.403904 |
| ETTh1 | 96 | 192 | phase6_asx_cross_clip05 | 0.435523 | 0.425184 |
| ETTh1 | 96 | 336 | phase6_asx_cross_clip05 | 0.474925 | 0.445853 |
| ETTh1 | 96 | 720 | phase6_asx_cross_clip05 | 0.462566 | 0.462729 |
| ETTh1 | 96 | 96 | phase6_asx_cross_clip05 | 0.388208 | 0.397494 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.343985 | 0.370746 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.375191 | 0.388813 |
| ETTm1 | 720 | 720 | phase6_asx_period_multi | 0.423044 | 0.419741 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.307015 | 0.351471 |
| ETTm1 | 96 | 192 | phase6_asx_cross_clip05 | 0.388281 | 0.394429 |
| ETTm1 | 96 | 336 | phase6_asx_cross_clip05 | 0.416828 | 0.412478 |
| ETTm1 | 96 | 720 | phase6_asx_cross_clip05 | 0.478628 | 0.444328 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.351353 | 0.379215 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.0931334 | 0.204459 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.138738 | 0.255873 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.249868 | 0.355835 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.381494 | 0.455779 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.0914837 | 0.199324 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.149582 | 0.257745 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.29149 | 0.37999 |
| PEMS08 | 96 | 96 | phase6_asx_period_multi | 0.561115 | 0.545434 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.138265 | 0.238301 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.194564 | 0.281049 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.208904 | 0.295086 |
| electricity | 96 | 720 | phase6_asx_individual_period | 0.249786 | 0.327229 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.194684 | 0.279893 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.400557 | 0.281291 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.411927 | 0.285796 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.448077 | 0.304166 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.388673 | 0.278225 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.584051 | 0.370466 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.590953 | 0.37248 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.624815 | 0.388458 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.621603 | 0.392252 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase6_asx_individual_revin | 0.240016 | 0.279124 |
| weather | 720 | 720 | phase6_asx_individual | 0.309688 | 0.331898 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.147851 | 0.200624 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.231881 | 0.274434 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.283318 | 0.308607 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.3567 | 0.353665 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.187501 | 0.239589 |

## Best-Cell Count (by test, analysis only)

| arm | cells |
| --- | ---: |
| phase6_asx_cross_clip05 | 16 |
| phase6_asx_individual | 1 |
| phase6_asx_individual_period | 7 |
| phase6_asx_individual_revin | 7 |
| phase6_asx_period_multi | 17 |

## Paired Statistics vs Anchor

Paired by dataset/seq_len/pred_len/seed.

| arm | pairs | dMSE_mean | dMSE_std | dMSE_2sd | win/loss/tie | dMAE_mean |
| --- | ---: | ---: | ---: | ---: | :--- | ---: |
| phase6_asx_cross_clip05 | 144 | 0.00430873 | 0.0403852 | 0.0807703 | 79/65/0 | 0.00409361 |
| phase6_asx_individual | 144 | 0.037968 | 0.123976 | 0.247953 | 71/73/0 | 0.0206714 |
| phase6_asx_individual_period | 144 | 0.0366278 | 0.123003 | 0.246006 | 80/64/0 | 0.0193057 |
| phase6_asx_individual_revin | 144 | 0.039344 | 0.123272 | 0.246544 | 72/72/0 | 0.020848 |
| phase6_asx_period_multi | 144 | -0.00407738 | 0.00613627 | 0.0122725 | 136/8/0 | -0.00345135 |

## Per Dataset

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| ETTh1 | 144 | 0.437705 | 0.438393 |
| ETTm1 | 144 | 0.388374 | 0.397376 |
| PEMS04 | 72 | 0.353237 | 0.393899 |
| PEMS08 | 72 | 0.39343 | 0.408689 |
| electricity | 144 | 0.192272 | 0.283454 |
| traffic | 144 | 0.529868 | 0.344926 |
| weather | 144 | 0.248492 | 0.284443 |

## Per Seq_len

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 720 | 360 | 0.324015 | 0.333069 |
| 96 | 504 | 0.388574 | 0.376346 |

## Per Pred_len

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 12 | 36 | 0.10635 | 0.216415 |
| 192 | 180 | 0.340952 | 0.337067 |
| 24 | 36 | 0.186112 | 0.290357 |
| 336 | 180 | 0.368471 | 0.355455 |
| 48 | 36 | 0.399049 | 0.442686 |
| 720 | 180 | 0.40911 | 0.384646 |
| 96 | 216 | 0.399333 | 0.377374 |

## Per Dataset Family

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| ETT | 288 | 0.413039 | 0.417884 |
| LargeC | 288 | 0.36107 | 0.31419 |
| PEMS | 144 | 0.373334 | 0.401294 |
| Weather | 144 | 0.248492 | 0.284443 |

## Validation-Selected Summary

- selected_test_mse_mean: 0.334771
- selected_test_mae_mean: 0.342178
- selected vs best_fixed_single (phase6_asx_period_multi): delta_mse=-0.00379665
- selected vs test_oracle (ANALYSIS ONLY): delta_mse=0.0016864

### Selected Arm Counts (per group)

| arm | groups |
| --- | ---: |
| phase6_asx_cross | 2 |
| phase6_asx_cross_clip05 | 6 |
| phase6_asx_individual_period | 8 |
| phase6_asx_individual_revin | 7 |
| phase6_asx_period_multi | 25 |

## Selected Per Dataset

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| ETTh1 | 24 | 0.432684 | 0.43161 |
| ETTm1 | 24 | 0.38652 | 0.395209 |
| PEMS04 | 12 | 0.215808 | 0.317986 |
| PEMS08 | 12 | 0.27799 | 0.348205 |
| electricity | 24 | 0.188695 | 0.278621 |
| traffic | 24 | 0.508832 | 0.334142 |
| weather | 24 | 0.244999 | 0.280391 |

## Selected Per Pred_len

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 12 | 6 | 0.0923086 | 0.201892 |
| 192 | 30 | 0.334173 | 0.331199 |
| 24 | 6 | 0.14416 | 0.256809 |
| 336 | 30 | 0.362527 | 0.350335 |
| 48 | 6 | 0.270679 | 0.367912 |
| 720 | 30 | 0.402007 | 0.378232 |
| 96 | 36 | 0.338973 | 0.347806 |

Selector fairness note: selection uses validation metrics aggregated over seeds; test metrics reported only after selection.

## Selector Comparison

| selector | groups | mse_mean | mae_mean | delta_vs_best_single |
| --- | ---: | ---: | ---: | ---: |
| unrestricted_mean | 48 | 0.334771 | 0.342178 | -0.00379665 |
| segment_robust | 48 | 0.33493 | 0.342405 | -0.00363768 |
| margin_prefer_simple | 48 | 0.33493 | 0.342333 | -0.00363801 |
| policy_family | 48 | 0.334646 | 0.341964 | -0.00392204 |

## Validation Segment Mismatch (full val_mse vs last segment)

Last segment column: val_mse_seg3.

| dataset | groups | mismatches |
| --- | ---: | ---: |
| ETTh1 | 8 | 7 |
| ETTm1 | 8 | 2 |
| PEMS04 | 4 | 0 |
| PEMS08 | 4 | 0 |
| electricity | 8 | 1 |
| traffic | 8 | 0 |
| weather | 8 | 6 |

