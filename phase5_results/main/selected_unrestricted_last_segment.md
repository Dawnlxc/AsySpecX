# Phase 5-Lockdown Validation Selection Summary

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse (metric_mode=last_segment, col=val_mse_seg3)
- std_weight: 0.0  margin_abs: 0.0  margin_pct: 0.0
- prefer_arm_order: (none)
- arm_allowlist_json: configs/selection/unrestricted.json
- selection_groups: 8
- selected_test_mse_mean: 0.197953
- selected_test_mae_mean: 0.267799

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase5_asx_cross_clip05 | 3 |
| phase5_asx_individual_period | 1 |
| phase5_asx_period_multi | 4 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| electricity | 4 | 0.165157 | 0.261314 |
| weather | 4 | 0.230749 | 0.274284 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 192 | 2 | 0.177539 | 0.252706 |
| 336 | 2 | 0.209546 | 0.277909 |
| 720 | 2 | 0.256632 | 0.312822 |
| 96 | 2 | 0.148095 | 0.227759 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_score | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.131255 | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.140816 | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.163265 | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.120384 | 0.138265 | 0.238301 |
| weather | 720 | 192 | phase5_asx_cross_clip05 | 0.422628 | 0.203098 | 0.256164 |
| weather | 720 | 336 | phase5_asx_cross_clip05 | 0.497376 | 0.252263 | 0.291876 |
| weather | 720 | 720 | phase5_asx_individual_period | 0.580211 | 0.309708 | 0.331879 |
| weather | 720 | 96 | phase5_asx_cross_clip05 | 0.362837 | 0.157925 | 0.217216 |

## Margin / Prefer-Order Trace

| dataset | seq_len | pred_len | raw_best_arm | raw_best_score | near_best_arms | final_selected_arm | selected_score |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.131255 | phase5_asx_period_multi | phase5_asx_period_multi | 0.131255 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.140816 | phase5_asx_period_multi | phase5_asx_period_multi | 0.140816 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.163265 | phase5_asx_period_multi | phase5_asx_period_multi | 0.163265 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.120384 | phase5_asx_period_multi | phase5_asx_period_multi | 0.120384 |
| weather | 720 | 192 | phase5_asx_cross_clip05 | 0.422628 | phase5_asx_cross_clip05 | phase5_asx_cross_clip05 | 0.422628 |
| weather | 720 | 336 | phase5_asx_cross_clip05 | 0.497376 | phase5_asx_cross_clip05 | phase5_asx_cross_clip05 | 0.497376 |
| weather | 720 | 720 | phase5_asx_individual_period | 0.580211 | phase5_asx_individual_period | phase5_asx_individual_period | 0.580211 |
| weather | 720 | 96 | phase5_asx_cross_clip05 | 0.362837 | phase5_asx_cross_clip05 | phase5_asx_cross_clip05 | 0.362837 |
