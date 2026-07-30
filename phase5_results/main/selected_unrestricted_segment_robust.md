# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=segment_mean_plus_std, col=val_mse)
- std_weight: 0.5  margin_abs: 0.0  margin_pct: 0.0
- prefer_arm_order: (none)
- arm_allowlist_json: configs/selection/unrestricted.json
- selection_groups: 8
- selected_test_mse_mean: 0.196106
- selected_test_mae_mean: 0.265371

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase5_asx_cross | 2 |
| phase5_asx_individual_revin | 1 |
| phase5_asx_period_multi | 5 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| electricity | 4 | 0.165157 | 0.261314 |
| weather | 4 | 0.227056 | 0.269428 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 192 | 2 | 0.174807 | 0.250103 |
| 336 | 2 | 0.208203 | 0.277013 |
| 720 | 2 | 0.258358 | 0.314905 |
| 96 | 2 | 0.143058 | 0.219463 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.131799 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.146345 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.182984 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.119775 | 0.138265 | 0.238301 |
| weather | 720 | 192 | phase5_asx_cross | 0.559162 | 0.197634 | 0.250958 |
| weather | 720 | 336 | phase5_asx_cross | 0.625027 | 0.249578 | 0.290083 |
| weather | 720 | 720 | phase5_asx_period_multi | 0.702121 | 0.313159 | 0.336044 |
| weather | 720 | 96 | phase5_asx_individual_revin | 0.503265 | 0.147851 | 0.200624 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.131799 | phase5_asx_period_multi | phase5_asx_period_multi | 0.131799 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.146345 | phase5_asx_period_multi | phase5_asx_period_multi | 0.146345 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.182984 | phase5_asx_period_multi | phase5_asx_period_multi | 0.182984 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.119775 | phase5_asx_period_multi | phase5_asx_period_multi | 0.119775 |
| weather | 720 | 192 | phase5_asx_cross | 0.559162 | phase5_asx_cross | phase5_asx_cross | 0.559162 |
| weather | 720 | 336 | phase5_asx_cross | 0.625027 | phase5_asx_cross | phase5_asx_cross | 0.625027 |
| weather | 720 | 720 | phase5_asx_period_multi | 0.702121 | phase5_asx_period_multi | phase5_asx_period_multi | 0.702121 |
| weather | 720 | 96 | phase5_asx_individual_revin | 0.503265 | phase5_asx_individual_revin | phase5_asx_individual_revin | 0.503265 |
