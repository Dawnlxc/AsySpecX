# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=mean, col=val_mse)
- std_weight: 0.0  margin_abs: 0.0  margin_pct: 0.0
- prefer_arm_order: (none)
- arm_allowlist_json: configs/selection/phase6_unrestricted.json
- selection_groups: 48
- selected_test_mse_mean: 0.334771
- selected_test_mae_mean: 0.342178

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase6_asx_cross | 2 |
| phase6_asx_cross_clip05 | 6 |
| phase6_asx_individual_period | 8 |
| phase6_asx_individual_revin | 7 |
| phase6_asx_period_multi | 25 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| ETTh1 | 8 | 0.432684 | 0.43161 |
| ETTm1 | 8 | 0.38652 | 0.395209 |
| PEMS04 | 4 | 0.215808 | 0.317986 |
| PEMS08 | 4 | 0.27799 | 0.348205 |
| electricity | 8 | 0.188695 | 0.278621 |
| traffic | 8 | 0.508832 | 0.334142 |
| weather | 8 | 0.244999 | 0.280391 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 12 | 2 | 0.0923086 | 0.201892 |
| 192 | 10 | 0.334173 | 0.331199 |
| 24 | 2 | 0.14416 | 0.256809 |
| 336 | 10 | 0.362527 | 0.350335 |
| 48 | 2 | 0.270679 | 0.367912 |
| 720 | 10 | 0.402007 | 0.378232 |
| 96 | 12 | 0.338973 | 0.347806 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 0.956738 | 0.416658 | 0.425533 |
| ETTh1 | 720 | 336 | phase6_asx_period_multi | 1.15617 | 0.442249 | 0.440963 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.39234 | 0.448751 | 0.460177 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.700948 | 0.379997 | 0.403904 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.00668 | 0.437965 | 0.425315 |
| ETTh1 | 96 | 336 | phase6_asx_period_multi | 1.28815 | 0.478458 | 0.443393 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.5633 | 0.467565 | 0.454944 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.707168 | 0.389832 | 0.398654 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.504519 | 0.343985 | 0.370746 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.643925 | 0.375191 | 0.388813 |
| ETTm1 | 720 | 720 | phase6_asx_individual_period | 0.930672 | 0.425106 | 0.416921 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.390972 | 0.307015 | 0.351471 |
| ETTm1 | 96 | 192 | phase6_asx_individual_period | 0.53115 | 0.389886 | 0.393571 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.677555 | 0.418064 | 0.414438 |
| ETTm1 | 96 | 720 | phase6_asx_period_multi | 0.993906 | 0.481556 | 0.446495 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.405896 | 0.351353 | 0.379215 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.0974851 | 0.0931334 | 0.204459 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.143597 | 0.138738 | 0.255873 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.258745 | 0.249868 | 0.355835 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.405152 | 0.381494 | 0.455779 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.100835 | 0.0914837 | 0.199324 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.160876 | 0.149582 | 0.257745 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.286323 | 0.29149 | 0.37999 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.523604 | 0.579403 | 0.555761 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.127752 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.142607 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.174302 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.116401 | 0.138265 | 0.238301 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.169799 | 0.194564 | 0.281049 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.183885 | 0.208904 | 0.295086 |
| electricity | 96 | 720 | phase6_asx_individual_revin | 0.216667 | 0.25078 | 0.327686 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.1715 | 0.194684 | 0.279893 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.332126 | 0.400557 | 0.281291 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.343343 | 0.411927 | 0.285796 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.386709 | 0.448077 | 0.304166 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.328253 | 0.388673 | 0.278225 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.47703 | 0.584051 | 0.370466 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.478681 | 0.590953 | 0.37248 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.529383 | 0.624815 | 0.388458 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.519353 | 0.621603 | 0.392252 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.437773 | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase6_asx_period_multi | 0.499061 | 0.249379 | 0.28983 |
| weather | 720 | 720 | phase6_asx_period_multi | 0.591314 | 0.313159 | 0.336044 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.381475 | 0.147851 | 0.200624 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.52831 | 0.231881 | 0.274434 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.605493 | 0.283318 | 0.308607 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.724068 | 0.3567 | 0.353665 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.459084 | 0.187501 | 0.239589 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 0.956738 | phase6_asx_period_multi | phase6_asx_period_multi | 0.956738 |
| ETTh1 | 720 | 336 | phase6_asx_period_multi | 1.15617 | phase6_asx_period_multi | phase6_asx_period_multi | 1.15617 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.39234 | phase6_asx_period_multi | phase6_asx_period_multi | 1.39234 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.700948 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.700948 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.00668 | phase6_asx_period_multi | phase6_asx_period_multi | 1.00668 |
| ETTh1 | 96 | 336 | phase6_asx_period_multi | 1.28815 | phase6_asx_period_multi | phase6_asx_period_multi | 1.28815 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.5633 | phase6_asx_period_multi | phase6_asx_period_multi | 1.5633 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.707168 | phase6_asx_period_multi | phase6_asx_period_multi | 0.707168 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.504519 | phase6_asx_individual_period | phase6_asx_individual_period | 0.504519 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.643925 | phase6_asx_individual_period | phase6_asx_individual_period | 0.643925 |
| ETTm1 | 720 | 720 | phase6_asx_individual_period | 0.930672 | phase6_asx_individual_period | phase6_asx_individual_period | 0.930672 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.390972 | phase6_asx_individual_period | phase6_asx_individual_period | 0.390972 |
| ETTm1 | 96 | 192 | phase6_asx_individual_period | 0.53115 | phase6_asx_individual_period | phase6_asx_individual_period | 0.53115 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.677555 | phase6_asx_cross | phase6_asx_cross | 0.677555 |
| ETTm1 | 96 | 720 | phase6_asx_period_multi | 0.993906 | phase6_asx_period_multi | phase6_asx_period_multi | 0.993906 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.405896 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.405896 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.0974851 | phase6_asx_period_multi | phase6_asx_period_multi | 0.0974851 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.143597 | phase6_asx_period_multi | phase6_asx_period_multi | 0.143597 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.258745 | phase6_asx_period_multi | phase6_asx_period_multi | 0.258745 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.405152 | phase6_asx_period_multi | phase6_asx_period_multi | 0.405152 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.100835 | phase6_asx_period_multi | phase6_asx_period_multi | 0.100835 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.160876 | phase6_asx_period_multi | phase6_asx_period_multi | 0.160876 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.286323 | phase6_asx_period_multi | phase6_asx_period_multi | 0.286323 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.523604 | phase6_asx_cross | phase6_asx_cross | 0.523604 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.127752 | phase6_asx_period_multi | phase6_asx_period_multi | 0.127752 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.142607 | phase6_asx_period_multi | phase6_asx_period_multi | 0.142607 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.174302 | phase6_asx_period_multi | phase6_asx_period_multi | 0.174302 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.116401 | phase6_asx_period_multi | phase6_asx_period_multi | 0.116401 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.169799 | phase6_asx_individual_period | phase6_asx_individual_period | 0.169799 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.183885 | phase6_asx_individual_period | phase6_asx_individual_period | 0.183885 |
| electricity | 96 | 720 | phase6_asx_individual_revin | 0.216667 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.216667 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.1715 | phase6_asx_individual_period | phase6_asx_individual_period | 0.1715 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.332126 | phase6_asx_period_multi | phase6_asx_period_multi | 0.332126 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.343343 | phase6_asx_period_multi | phase6_asx_period_multi | 0.343343 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.386709 | phase6_asx_period_multi | phase6_asx_period_multi | 0.386709 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.328253 | phase6_asx_period_multi | phase6_asx_period_multi | 0.328253 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.47703 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.47703 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.478681 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.478681 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.529383 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.529383 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.519353 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.519353 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.437773 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.437773 |
| weather | 720 | 336 | phase6_asx_period_multi | 0.499061 | phase6_asx_period_multi | phase6_asx_period_multi | 0.499061 |
| weather | 720 | 720 | phase6_asx_period_multi | 0.591314 | phase6_asx_period_multi | phase6_asx_period_multi | 0.591314 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.381475 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.381475 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.52831 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.52831 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.605493 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.605493 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.724068 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.724068 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.459084 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.459084 |
