# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=mean, col=val_mse)
- std_weight: 0.0  margin_abs: 0.0  margin_pct: 0.0
- prefer_arm_order: (none)
- arm_allowlist_json: configs/selection/unrestricted.json
- selection_groups: 8
- selected_test_mse_mean: 0.195153
- selected_test_mae_mean: 0.264011

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase5_asx_individual_revin | 2 |
| phase5_asx_period_multi | 6 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| electricity | 4 | 0.165157 | 0.261314 |
| weather | 4 | 0.225148 | 0.266709 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 192 | 2 | 0.171091 | 0.244792 |
| 336 | 2 | 0.208104 | 0.276886 |
| 720 | 2 | 0.258358 | 0.314905 |
| 96 | 2 | 0.143058 | 0.219463 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.127752 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.142607 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.174302 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.116401 | 0.138265 | 0.238301 |
| weather | 720 | 192 | phase5_asx_individual_revin | 0.437773 | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase5_asx_period_multi | 0.499061 | 0.249379 | 0.28983 |
| weather | 720 | 720 | phase5_asx_period_multi | 0.591314 | 0.313159 | 0.336044 |
| weather | 720 | 96 | phase5_asx_individual_revin | 0.381475 | 0.147851 | 0.200624 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.127752 | phase5_asx_period_multi | phase5_asx_period_multi | 0.127752 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.142607 | phase5_asx_period_multi | phase5_asx_period_multi | 0.142607 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.174302 | phase5_asx_period_multi | phase5_asx_period_multi | 0.174302 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.116401 | phase5_asx_period_multi | phase5_asx_period_multi | 0.116401 |
| weather | 720 | 192 | phase5_asx_individual_revin | 0.437773 | phase5_asx_individual_revin | phase5_asx_individual_revin | 0.437773 |
| weather | 720 | 336 | phase5_asx_period_multi | 0.499061 | phase5_asx_period_multi | phase5_asx_period_multi | 0.499061 |
| weather | 720 | 720 | phase5_asx_period_multi | 0.591314 | phase5_asx_period_multi | phase5_asx_period_multi | 0.591314 |
| weather | 720 | 96 | phase5_asx_individual_revin | 0.381475 | phase5_asx_individual_revin | phase5_asx_individual_revin | 0.381475 |
