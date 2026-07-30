# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=mean_plus_std, col=val_mse)
- std_weight: 0.25  margin_abs: 0.0  margin_pct: 0.002
- prefer_arm_order: phase6_asx_individual_revin,phase6_asx_individual,phase6_asx_cross,phase6_asx_cross_clip05,phase6_asx_period_multi,phase6_asx_individual_period
- arm_allowlist_json: configs/selection/phase6_unrestricted.json
- selection_groups: 48
- selected_test_mse_mean: 0.33493
- selected_test_mae_mean: 0.342333

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase6_asx_cross | 8 |
| phase6_asx_cross_clip05 | 6 |
| phase6_asx_individual | 2 |
| phase6_asx_individual_period | 4 |
| phase6_asx_individual_revin | 9 |
| phase6_asx_period_multi | 19 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| ETTh1 | 8 | 0.433162 | 0.432175 |
| ETTm1 | 8 | 0.386859 | 0.395392 |
| PEMS04 | 4 | 0.215828 | 0.318078 |
| PEMS08 | 4 | 0.27799 | 0.348205 |
| electricity | 8 | 0.188817 | 0.278734 |
| traffic | 8 | 0.508832 | 0.334142 |
| weather | 8 | 0.245002 | 0.280412 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 12 | 2 | 0.0923473 | 0.202075 |
| 192 | 10 | 0.334137 | 0.33107 |
| 24 | 2 | 0.14416 | 0.256809 |
| 336 | 10 | 0.36314 | 0.351 |
| 48 | 2 | 0.270679 | 0.367912 |
| 720 | 10 | 0.402182 | 0.378401 |
| 96 | 12 | 0.338973 | 0.347806 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 0.958578 | 0.416658 | 0.425533 |
| ETTh1 | 720 | 336 | phase6_asx_cross | 1.15713 | 0.44444 | 0.443511 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.39451 | 0.448751 | 0.460177 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.701693 | 0.379997 | 0.403904 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.00678 | 0.437965 | 0.425315 |
| ETTh1 | 96 | 336 | phase6_asx_cross | 1.29144 | 0.480089 | 0.445358 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.56356 | 0.467565 | 0.454944 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.707468 | 0.389832 | 0.398654 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.504587 | 0.343985 | 0.370746 |
| ETTm1 | 720 | 336 | phase6_asx_individual | 0.645169 | 0.376329 | 0.389791 |
| ETTm1 | 720 | 720 | phase6_asx_individual | 0.932069 | 0.426381 | 0.417995 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.391 | 0.307015 | 0.351471 |
| ETTm1 | 96 | 192 | phase6_asx_individual_revin | 0.531848 | 0.389531 | 0.392282 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.6777 | 0.418064 | 0.414438 |
| ETTm1 | 96 | 720 | phase6_asx_cross | 0.994136 | 0.482212 | 0.447196 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.406257 | 0.351353 | 0.379215 |
| PEMS04 | 96 | 12 | phase6_asx_cross | 0.0976304 | 0.093211 | 0.204826 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.143634 | 0.138738 | 0.255873 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.260117 | 0.249868 | 0.355835 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.406684 | 0.381494 | 0.455779 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.100946 | 0.0914837 | 0.199324 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.16109 | 0.149582 | 0.257745 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.286464 | 0.29149 | 0.37999 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.525224 | 0.579403 | 0.555761 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.127832 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.142647 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.174361 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.116536 | 0.138265 | 0.238301 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.169809 | 0.194564 | 0.281049 |
| electricity | 96 | 336 | phase6_asx_individual_revin | 0.184215 | 0.209877 | 0.29599 |
| electricity | 96 | 720 | phase6_asx_individual_revin | 0.21668 | 0.25078 | 0.327686 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.171531 | 0.194684 | 0.279893 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.332322 | 0.400557 | 0.281291 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.343573 | 0.411927 | 0.285796 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.386916 | 0.448077 | 0.304166 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.328357 | 0.388673 | 0.278225 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.477381 | 0.584051 | 0.370466 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.47958 | 0.590953 | 0.37248 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.530601 | 0.624815 | 0.388458 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.52025 | 0.621603 | 0.392252 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.437795 | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase6_asx_cross | 0.49944 | 0.249578 | 0.290083 |
| weather | 720 | 720 | phase6_asx_cross | 0.592143 | 0.312987 | 0.335959 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.381511 | 0.147851 | 0.200624 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.528374 | 0.231881 | 0.274434 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.605556 | 0.283318 | 0.308607 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.724091 | 0.3567 | 0.353665 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.459099 | 0.187501 | 0.239589 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| ETTh1 | 720 | 192 | phase6_asx_period_multi | 0.958578 | phase6_asx_period_multi | phase6_asx_period_multi | 0.958578 |
| ETTh1 | 720 | 336 | phase6_asx_period_multi | 1.15663 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 1.15713 |
| ETTh1 | 720 | 720 | phase6_asx_period_multi | 1.39451 | phase6_asx_period_multi | phase6_asx_period_multi | 1.39451 |
| ETTh1 | 720 | 96 | phase6_asx_cross_clip05 | 0.701693 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.701693 |
| ETTh1 | 96 | 192 | phase6_asx_period_multi | 1.00678 | phase6_asx_period_multi | phase6_asx_period_multi | 1.00678 |
| ETTh1 | 96 | 336 | phase6_asx_period_multi | 1.289 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 1.29144 |
| ETTh1 | 96 | 720 | phase6_asx_period_multi | 1.56356 | phase6_asx_period_multi | phase6_asx_period_multi | 1.56356 |
| ETTh1 | 96 | 96 | phase6_asx_period_multi | 0.707468 | phase6_asx_period_multi | phase6_asx_period_multi | 0.707468 |
| ETTm1 | 720 | 192 | phase6_asx_individual_period | 0.504587 | phase6_asx_individual_period | phase6_asx_individual_period | 0.504587 |
| ETTm1 | 720 | 336 | phase6_asx_individual_period | 0.643991 | phase6_asx_individual_period; phase6_asx_individual | phase6_asx_individual | 0.645169 |
| ETTm1 | 720 | 720 | phase6_asx_individual_period | 0.930821 | phase6_asx_individual_period; phase6_asx_individual | phase6_asx_individual | 0.932069 |
| ETTm1 | 720 | 96 | phase6_asx_individual_period | 0.391 | phase6_asx_individual_period | phase6_asx_individual_period | 0.391 |
| ETTm1 | 96 | 192 | phase6_asx_individual_period | 0.531154 | phase6_asx_individual_period; phase6_asx_individual; phase6_asx_individual_revin | phase6_asx_individual_revin | 0.531848 |
| ETTm1 | 96 | 336 | phase6_asx_cross | 0.6777 | phase6_asx_cross; phase6_asx_period_multi | phase6_asx_cross | 0.6777 |
| ETTm1 | 96 | 720 | phase6_asx_period_multi | 0.994093 | phase6_asx_period_multi; phase6_asx_cross; phase6_asx_cross_clip05 | phase6_asx_cross | 0.994136 |
| ETTm1 | 96 | 96 | phase6_asx_cross_clip05 | 0.406257 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.406257 |
| PEMS04 | 96 | 12 | phase6_asx_period_multi | 0.0975244 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 0.0976304 |
| PEMS04 | 96 | 24 | phase6_asx_period_multi | 0.143634 | phase6_asx_period_multi | phase6_asx_period_multi | 0.143634 |
| PEMS04 | 96 | 48 | phase6_asx_period_multi | 0.260117 | phase6_asx_period_multi | phase6_asx_period_multi | 0.260117 |
| PEMS04 | 96 | 96 | phase6_asx_period_multi | 0.406684 | phase6_asx_period_multi | phase6_asx_period_multi | 0.406684 |
| PEMS08 | 96 | 12 | phase6_asx_period_multi | 0.100946 | phase6_asx_period_multi | phase6_asx_period_multi | 0.100946 |
| PEMS08 | 96 | 24 | phase6_asx_period_multi | 0.16109 | phase6_asx_period_multi | phase6_asx_period_multi | 0.16109 |
| PEMS08 | 96 | 48 | phase6_asx_period_multi | 0.286464 | phase6_asx_period_multi | phase6_asx_period_multi | 0.286464 |
| PEMS08 | 96 | 96 | phase6_asx_cross | 0.525224 | phase6_asx_cross | phase6_asx_cross | 0.525224 |
| electricity | 720 | 192 | phase6_asx_period_multi | 0.127832 | phase6_asx_period_multi | phase6_asx_period_multi | 0.127832 |
| electricity | 720 | 336 | phase6_asx_period_multi | 0.142647 | phase6_asx_period_multi | phase6_asx_period_multi | 0.142647 |
| electricity | 720 | 720 | phase6_asx_period_multi | 0.174361 | phase6_asx_period_multi | phase6_asx_period_multi | 0.174361 |
| electricity | 720 | 96 | phase6_asx_period_multi | 0.116536 | phase6_asx_period_multi | phase6_asx_period_multi | 0.116536 |
| electricity | 96 | 192 | phase6_asx_individual_period | 0.169809 | phase6_asx_individual_period | phase6_asx_individual_period | 0.169809 |
| electricity | 96 | 336 | phase6_asx_individual_period | 0.183891 | phase6_asx_individual_period; phase6_asx_individual_revin | phase6_asx_individual_revin | 0.184215 |
| electricity | 96 | 720 | phase6_asx_individual_revin | 0.21668 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.21668 |
| electricity | 96 | 96 | phase6_asx_individual_period | 0.171531 | phase6_asx_individual_period | phase6_asx_individual_period | 0.171531 |
| traffic | 720 | 192 | phase6_asx_period_multi | 0.332322 | phase6_asx_period_multi | phase6_asx_period_multi | 0.332322 |
| traffic | 720 | 336 | phase6_asx_period_multi | 0.343573 | phase6_asx_period_multi | phase6_asx_period_multi | 0.343573 |
| traffic | 720 | 720 | phase6_asx_period_multi | 0.386916 | phase6_asx_period_multi | phase6_asx_period_multi | 0.386916 |
| traffic | 720 | 96 | phase6_asx_period_multi | 0.328357 | phase6_asx_period_multi | phase6_asx_period_multi | 0.328357 |
| traffic | 96 | 192 | phase6_asx_cross_clip05 | 0.477381 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.477381 |
| traffic | 96 | 336 | phase6_asx_cross_clip05 | 0.47958 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.47958 |
| traffic | 96 | 720 | phase6_asx_cross_clip05 | 0.530601 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.530601 |
| traffic | 96 | 96 | phase6_asx_cross_clip05 | 0.52025 | phase6_asx_cross_clip05 | phase6_asx_cross_clip05 | 0.52025 |
| weather | 720 | 192 | phase6_asx_individual_revin | 0.437795 | phase6_asx_individual_revin; phase6_asx_individual_period | phase6_asx_individual_revin | 0.437795 |
| weather | 720 | 336 | phase6_asx_period_multi | 0.499419 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 0.49944 |
| weather | 720 | 720 | phase6_asx_period_multi | 0.592058 | phase6_asx_period_multi; phase6_asx_cross | phase6_asx_cross | 0.592143 |
| weather | 720 | 96 | phase6_asx_individual_revin | 0.381511 | phase6_asx_individual_revin; phase6_asx_individual_period; phase6_asx_individual | phase6_asx_individual_revin | 0.381511 |
| weather | 96 | 192 | phase6_asx_individual_revin | 0.528374 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.528374 |
| weather | 96 | 336 | phase6_asx_individual_revin | 0.605556 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.605556 |
| weather | 96 | 720 | phase6_asx_individual_revin | 0.724091 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.724091 |
| weather | 96 | 96 | phase6_asx_individual_revin | 0.459099 | phase6_asx_individual_revin | phase6_asx_individual_revin | 0.459099 |
