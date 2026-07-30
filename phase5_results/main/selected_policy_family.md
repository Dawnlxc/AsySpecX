# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=mean_plus_std, col=val_mse)
- std_weight: 0.5  margin_abs: 0.0  margin_pct: 0.002
- prefer_arm_order: phase5_asx_individual_revin,phase5_asx_individual,phase5_asx_cross,phase5_asx_cross_clip05,phase5_asx_period_multi,phase5_asx_individual_period
- arm_allowlist_json: configs/selection/policy_family.json
- selection_groups: 8
- selected_test_mse_mean: 0.193601
- selected_test_mae_mean: 0.261789

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase5_asx_individual_revin | 4 |
| phase5_asx_period_multi | 4 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| electricity | 4 | 0.165157 | 0.261314 |
| weather | 4 | 0.222045 | 0.262264 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 192 | 2 | 0.171091 | 0.244792 |
| 336 | 2 | 0.203422 | 0.271533 |
| 720 | 2 | 0.256833 | 0.311369 |
| 96 | 2 | 0.143058 | 0.219463 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.127913 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.142687 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.174421 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.116671 | 0.138265 | 0.238301 |
| weather | 720 | 192 | phase5_asx_individual_revin | 0.437818 | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase5_asx_individual_revin | 0.502234 | 0.240016 | 0.279124 |
| weather | 720 | 720 | phase5_asx_individual_revin | 0.596892 | 0.31011 | 0.328973 |
| weather | 720 | 96 | phase5_asx_individual_revin | 0.381546 | 0.147851 | 0.200624 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.127913 | phase5_asx_period_multi | phase5_asx_period_multi | 0.127913 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.142687 | phase5_asx_period_multi | phase5_asx_period_multi | 0.142687 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.174421 | phase5_asx_period_multi | phase5_asx_period_multi | 0.174421 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.116671 | phase5_asx_period_multi | phase5_asx_period_multi | 0.116671 |
| weather | 720 | 192 | phase5_asx_individual_revin | 0.437818 | phase5_asx_individual_revin | phase5_asx_individual_revin | 0.437818 |
| weather | 720 | 336 | phase5_asx_individual_revin | 0.502234 | phase5_asx_individual_revin | phase5_asx_individual_revin | 0.502234 |
| weather | 720 | 720 | phase5_asx_individual | 0.595796 | phase5_asx_individual; phase5_asx_individual_revin | phase5_asx_individual_revin | 0.596892 |
| weather | 720 | 96 | phase5_asx_individual_revin | 0.381546 | phase5_asx_individual_revin; phase5_asx_individual | phase5_asx_individual_revin | 0.381546 |
