# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=mean_plus_std, col=val_mse)
- std_weight: 0.5  margin_abs: 0.0  margin_pct: 0.002
- prefer_arm_order: phase6_asx_individual_revin,phase6_asx_individual,phase6_asx_cross,phase6_asx_cross_clip05,phase6_asx_period_multi,phase6_asx_individual_period
- arm_allowlist_json: configs/selection/phase6_policy_family.json
- selection_groups: 48
- selected_test_mse_mean: 0.334646
- selected_test_mae_mean: 0.341964

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase6_asx_cross | 6 |
| phase6_asx_cross_clip05 | 6 |
| phase6_asx_individual | 3 |
| phase6_asx_individual_period | 6 |
| phase6_asx_individual_revin | 8 |
| phase6_asx_period_multi | 19 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| ETTh1 | 8 | 0.433162 | 0.432175 |
| ETTm1 | 8 | 0.386955 | 0.395595 |
| PEMS04 | 4 | 0.215828 | 0.318078 |
| PEMS08 | 4 | 0.27799 | 0.348205 |
| electricity | 8 | 0.188571 | 0.278564 |
| traffic | 8 | 0.508832 | 0.334142 |
| weather | 8 | 0.243447 | 0.278169 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 12 | 2 | 0.0923473 | 0.202075 |
| 192 | 10 | 0.334215 | 0.331233 |
| 24 | 2 | 0.14416 | 0.256809 |
| 336 | 10 | 0.362087 | 0.349813 |
| 48 | 2 | 0.270679 | 0.367912 |
| 720 | 10 | 0.401795 | 0.377657 |
| 96 | 12 | 0.338973 | 0.347806 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 0.960417 | 0.416658 | 0.425533 |
| ETTh1 | 720 | 336 | phase6_asx_cross | 1.15749 | 0.44444 | 0.443511 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.39668 | 0.448751 | 0.460177 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.702437 | 0.379997 | 0.403904 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.00689 | 0.437965 | 0.425315 |
| ETTh1 | 96 | 336 | phase6_asx_cross | 1.29243 | 0.480089 | 0.445358 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.56381 | 0.467565 | 0.454944 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.707768 | 0.389832 | 0.398654 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.504655 | 0.343985 | 0.370746 |
| ETTm1 | 720 | 336 | phase6_asx_individual | 0.645224 | 0.376329 | 0.389791 |
| ETTm1 | 720 | 720 | phase6_asx_individual | 0.932215 | 0.426381 | 0.417995 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.391028 | 0.307015 | 0.351471 |
| ETTm1 | 96 | 192 | phase6_asx_individual | 0.531557 | 0.390304 | 0.393907 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.677844 | 0.418064 | 0.414438 |
| ETTm1 | 96 | 720 | phase6_asx_cross | 0.994243 | 0.482212 | 0.447196 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.406619 | 0.351353 | 0.379215 |
| PEMS04 | 96 | 12 | phase6_asx_cross | 0.0976583 | 0.093211 | 0.204826 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.143671 | 0.138738 | 0.255873 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.261489 | 0.249868 | 0.355835 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.408216 | 0.381494 | 0.455779 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.101058 | 0.0914837 | 0.199324 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.161303 | 0.149582 | 0.257745 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.286605 | 0.29149 | 0.37999 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.526843 | 0.579403 | 0.555761 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.127913 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.142687 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.174421 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.116671 | 0.138265 | 0.238301 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.169819 | 0.194564 | 0.281049 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.183896 | 0.208904 | 0.295086 |
| electricity | 96 | 720 | phase6_asx_individual_period | 0.217552 | 0.249786 | 0.327229 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.171562 | 0.194684 | 0.279893 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.332518 | 0.400557 | 0.281291 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.343802 | 0.411927 | 0.285796 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.387122 | 0.448077 | 0.304166 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.328462 | 0.388673 | 0.278225 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.477731 | 0.584051 | 0.370466 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.48048 | 0.590953 | 0.37248 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.531819 | 0.624815 | 0.388458 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.521147 | 0.621603 | 0.392252 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.437818 | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase6_asx_individual_revin | 0.502234 | 0.240016 | 0.279124 |
| weather | 720 | 720 | phase6_asx_individual_revin | 0.596892 | 0.31011 | 0.328973 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.381546 | 0.147851 | 0.200624 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.528437 | 0.231881 | 0.274434 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.605618 | 0.283318 | 0.308607 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.724113 | 0.3567 | 0.353665 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.459113 | 0.187501 | 0.239589 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 0.960417 | phase6_asx_period_multi | phase6_asx_period_multi | 0.960417 |
| ETTh1 | 720 | 336 | phase6_asx_period_multi | 1.15709 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 1.15749 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.39668 | phase6_asx_period_multi | phase6_asx_period_multi | 1.39668 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.702437 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.702437 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.00689 | phase6_asx_period_multi | phase6_asx_period_multi | 1.00689 |
| ETTh1 | 96 | 336 | phase6_asx_period_multi | 1.28985 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 1.29243 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.56381 | phase6_asx_period_multi | phase6_asx_period_multi | 1.56381 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.707768 | phase6_asx_period_multi | phase6_asx_period_multi | 0.707768 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.504655 | phase6_asx_individual_period | phase6_asx_individual_period | 0.504655 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.644057 | phase6_asx_individual_period; phase6_asx_individual | phase6_asx_individual | 0.645224 |
| ETTm1 | 720 | 720 | phase6_asx_individual_period | 0.93097 | phase6_asx_individual_period; phase6_asx_individual | phase6_asx_individual | 0.932215 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.391028 | phase6_asx_individual_period | phase6_asx_individual_period | 0.391028 |
| ETTm1 | 96 | 192 | phase6_asx_individual_period | 0.531157 | phase6_asx_individual_period; phase6_asx_individual | phase6_asx_individual | 0.531557 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.677844 | phase6_asx_cross; phase6_asx_period_multi | phase6_asx_cross | 0.677844 |
| ETTm1 | 96 | 720 | phase6_asx_cross | 0.994243 | phase6_asx_cross; phase6_asx_period_multi; phase6_asx_cross_clip05 | phase6_asx_cross | 0.994243 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.406619 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.406619 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.0975638 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 0.0976583 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.143671 | phase6_asx_period_multi | phase6_asx_period_multi | 0.143671 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.261489 | phase6_asx_period_multi | phase6_asx_period_multi | 0.261489 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.408216 | phase6_asx_period_multi | phase6_asx_period_multi | 0.408216 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.101058 | phase6_asx_period_multi | phase6_asx_period_multi | 0.101058 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.161303 | phase6_asx_period_multi | phase6_asx_period_multi | 0.161303 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.286605 | phase6_asx_period_multi | phase6_asx_period_multi | 0.286605 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.526843 | phase6_asx_cross | phase6_asx_cross | 0.526843 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.127913 | phase6_asx_period_multi | phase6_asx_period_multi | 0.127913 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.142687 | phase6_asx_period_multi | phase6_asx_period_multi | 0.142687 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.174421 | phase6_asx_period_multi | phase6_asx_period_multi | 0.174421 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.116671 | phase6_asx_period_multi | phase6_asx_period_multi | 0.116671 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.169819 | phase6_asx_individual_period | phase6_asx_individual_period | 0.169819 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.183896 | phase6_asx_individual_period | phase6_asx_individual_period | 0.183896 |
| electricity | 96 | 720 | phase6_asx_individual_period | 0.217552 | phase6_asx_individual_period | phase6_asx_individual_period | 0.217552 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.171562 | phase6_asx_individual_period | phase6_asx_individual_period | 0.171562 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.332518 | phase6_asx_period_multi | phase6_asx_period_multi | 0.332518 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.343802 | phase6_asx_period_multi | phase6_asx_period_multi | 0.343802 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.387122 | phase6_asx_period_multi | phase6_asx_period_multi | 0.387122 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.328462 | phase6_asx_period_multi | phase6_asx_period_multi | 0.328462 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.477731 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.477731 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.48048 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.48048 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.531819 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.531819 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.521147 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.521147 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.437818 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.437818 |
| weather | 720 | 336 | phase6_asx_individual_revin | 0.502234 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.502234 |
| weather | 720 | 720 | phase6_asx_individual | 0.595796 | phase6_asx_individual; phase6_asx_individual_revin | phase6_asx_individual_revin | 0.596892 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.381546 | phase6_asx_individual_revin; phase6_asx_individual | phase6_asx_individual_revin | 0.381546 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.528437 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.528437 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.605618 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.605618 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.724113 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.724113 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.459113 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.459113 |
