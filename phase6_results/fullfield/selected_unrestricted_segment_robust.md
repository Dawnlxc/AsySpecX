# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=segment_mean_plus_std, col=val_mse)
- std_weight: 0.5  margin_abs: 0.0  margin_pct: 0.0
- prefer_arm_order: (none)
- arm_allowlist_json: configs/selection/phase6_unrestricted.json
- selection_groups: 48
- selected_test_mse_mean: 0.33493
- selected_test_mae_mean: 0.342405

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase6_asx_cross | 4 |
| phase6_asx_cross_clip05 | 6 |
| phase6_asx_individual_period | 8 |
| phase6_asx_individual_revin | 6 |
| phase6_asx_period_multi | 24 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| ETTh1 | 8 | 0.432684 | 0.43161 |
| ETTm1 | 8 | 0.38652 | 0.395209 |
| PEMS04 | 4 | 0.215808 | 0.317986 |
| PEMS08 | 4 | 0.27799 | 0.348205 |
| electricity | 8 | 0.188695 | 0.278621 |
| traffic | 8 | 0.508832 | 0.334142 |
| weather | 8 | 0.245953 | 0.281751 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 12 | 2 | 0.0923086 | 0.201892 |
| 192 | 10 | 0.334916 | 0.332261 |
| 24 | 2 | 0.14416 | 0.256809 |
| 336 | 10 | 0.362547 | 0.35036 |
| 48 | 2 | 0.270679 | 0.367912 |
| 720 | 10 | 0.402007 | 0.378232 |
| 96 | 12 | 0.338973 | 0.347806 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 1.02366 | 0.416658 | 0.425533 |
| ETTh1 | 720 | 336 | phase6_asx_period_multi | 1.25111 | 0.442249 | 0.440963 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.60388 | 0.448751 | 0.460177 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.753557 | 0.379997 | 0.403904 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.06829 | 0.437965 | 0.425315 |
| ETTh1 | 96 | 336 | phase6_asx_period_multi | 1.36312 | 0.478458 | 0.443393 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.77412 | 0.467565 | 0.454944 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.760748 | 0.389832 | 0.398654 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.552368 | 0.343985 | 0.370746 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.699029 | 0.375191 | 0.388813 |
| ETTm1 | 720 | 720 | phase6_asx_individual_period | 0.994413 | 0.425106 | 0.416921 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.432371 | 0.307015 | 0.351471 |
| ETTm1 | 96 | 192 | phase6_asx_individual_period | 0.584736 | 0.389886 | 0.393571 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.738954 | 0.418064 | 0.414438 |
| ETTm1 | 96 | 720 | phase6_asx_period_multi | 1.06602 | 0.481556 | 0.446495 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.448504 | 0.351353 | 0.379215 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.100391 | 0.0931334 | 0.204459 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.149081 | 0.138738 | 0.255873 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.271045 | 0.249868 | 0.355835 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.418006 | 0.381494 | 0.455779 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.103824 | 0.0914837 | 0.199324 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.167745 | 0.149582 | 0.257745 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.301782 | 0.29149 | 0.37999 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.560491 | 0.579403 | 0.555761 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.131799 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.146345 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.182984 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.119775 | 0.138265 | 0.238301 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.174073 | 0.194564 | 0.281049 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.188036 | 0.208904 | 0.295086 |
| electricity | 96 | 720 | phase6_asx_individual_revin | 0.225318 | 0.25078 | 0.327686 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.176284 | 0.194684 | 0.279893 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.362855 | 0.400557 | 0.281291 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.375552 | 0.411927 | 0.285796 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.425865 | 0.448077 | 0.304166 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.357087 | 0.388673 | 0.278225 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.510018 | 0.584051 | 0.370466 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.506342 | 0.590953 | 0.37248 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.563542 | 0.624815 | 0.388458 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.553484 | 0.621603 | 0.392252 |
| weather | 720 | 192 | phase6_asx_cross | 0.559162 | 0.197634 | 0.250958 |
| weather | 720 | 336 | phase6_asx_cross | 0.625027 | 0.249578 | 0.290083 |
| weather | 720 | 720 | phase6_asx_period_multi | 0.702121 | 0.313159 | 0.336044 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.503265 | 0.147851 | 0.200624 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.6457 | 0.231881 | 0.274434 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.730007 | 0.283318 | 0.308607 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.84025 | 0.3567 | 0.353665 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.580059 | 0.187501 | 0.239589 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 1.02366 | phase6_asx_period_multi | phase6_asx_period_multi | 1.02366 |
| ETTh1 | 720 | 336 | phase6_asx_period_multi | 1.25111 | phase6_asx_period_multi | phase6_asx_period_multi | 1.25111 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.60388 | phase6_asx_period_multi | phase6_asx_period_multi | 1.60388 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.753557 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.753557 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.06829 | phase6_asx_period_multi | phase6_asx_period_multi | 1.06829 |
| ETTh1 | 96 | 336 | phase6_asx_period_multi | 1.36312 | phase6_asx_period_multi | phase6_asx_period_multi | 1.36312 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.77412 | phase6_asx_period_multi | phase6_asx_period_multi | 1.77412 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.760748 | phase6_asx_period_multi | phase6_asx_period_multi | 0.760748 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.552368 | phase6_asx_individual_period | phase6_asx_individual_period | 0.552368 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.699029 | phase6_asx_individual_period | phase6_asx_individual_period | 0.699029 |
| ETTm1 | 720 | 720 | phase6_asx_individual_period | 0.994413 | phase6_asx_individual_period | phase6_asx_individual_period | 0.994413 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.432371 | phase6_asx_individual_period | phase6_asx_individual_period | 0.432371 |
| ETTm1 | 96 | 192 | phase6_asx_individual_period | 0.584736 | phase6_asx_individual_period | phase6_asx_individual_period | 0.584736 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.738954 | phase6_asx_cross | phase6_asx_cross | 0.738954 |
| ETTm1 | 96 | 720 | phase6_asx_period_multi | 1.06602 | phase6_asx_period_multi | phase6_asx_period_multi | 1.06602 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.448504 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.448504 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.100391 | phase6_asx_period_multi | phase6_asx_period_multi | 0.100391 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.149081 | phase6_asx_period_multi | phase6_asx_period_multi | 0.149081 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.271045 | phase6_asx_period_multi | phase6_asx_period_multi | 0.271045 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.418006 | phase6_asx_period_multi | phase6_asx_period_multi | 0.418006 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.103824 | phase6_asx_period_multi | phase6_asx_period_multi | 0.103824 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.167745 | phase6_asx_period_multi | phase6_asx_period_multi | 0.167745 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.301782 | phase6_asx_period_multi | phase6_asx_period_multi | 0.301782 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.560491 | phase6_asx_cross | phase6_asx_cross | 0.560491 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.131799 | phase6_asx_period_multi | phase6_asx_period_multi | 0.131799 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.146345 | phase6_asx_period_multi | phase6_asx_period_multi | 0.146345 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.182984 | phase6_asx_period_multi | phase6_asx_period_multi | 0.182984 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.119775 | phase6_asx_period_multi | phase6_asx_period_multi | 0.119775 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.174073 | phase6_asx_individual_period | phase6_asx_individual_period | 0.174073 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.188036 | phase6_asx_individual_period | phase6_asx_individual_period | 0.188036 |
| electricity | 96 | 720 | phase6_asx_individual_revin | 0.225318 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.225318 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.176284 | phase6_asx_individual_period | phase6_asx_individual_period | 0.176284 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.362855 | phase6_asx_period_multi | phase6_asx_period_multi | 0.362855 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.375552 | phase6_asx_period_multi | phase6_asx_period_multi | 0.375552 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.425865 | phase6_asx_period_multi | phase6_asx_period_multi | 0.425865 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.357087 | phase6_asx_period_multi | phase6_asx_period_multi | 0.357087 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.510018 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.510018 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.506342 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.506342 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.563542 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.563542 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.553484 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.553484 |
| weather | 720 | 192 | phase6_asx_cross | 0.559162 | phase6_asx_cross | phase6_asx_cross | 0.559162 |
| weather | 720 | 336 | phase6_asx_cross | 0.625027 | phase6_asx_cross | phase6_asx_cross | 0.625027 |
| weather | 720 | 720 | phase6_asx_period_multi | 0.702121 | phase6_asx_period_multi | phase6_asx_period_multi | 0.702121 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.503265 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.503265 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.6457 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.6457 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.730007 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.730007 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.84025 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.84025 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.580059 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.580059 |
