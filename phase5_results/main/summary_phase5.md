# Phase 5-Lockdown Summary

Do not pick arms by test metric. Report a fixed single-arm result AND the validation-selected result separately.

- total_runs: 144
- ok_runs: 144
- failed_runs: 0
- anchor_arm: phase5_asx_cross

## Arm Means

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase5_asx_cross | 24 | 0.19893 | 0.270275 | 0.311559 |
| phase5_asx_cross_clip05 | 24 | 0.200887 | 0.27247 | 0.31365 |
| phase5_asx_individual | 24 | 0.196715 | 0.267222 | 0.312439 |
| phase5_asx_individual_period | 24 | 0.19578 | 0.265862 | 0.311565 |
| phase5_asx_individual_revin | 24 | 0.196733 | 0.265817 | 0.31199 |
| phase5_asx_period_multi | 24 | 0.196581 | 0.266393 | 0.309633 |

## Best Arm Per Dataset/Seq_len/Pred_len BY TEST (analysis only -- not for selection)

| dataset | seq_len | pred_len | best_arm | mse_mean | mae_mean |
| --- | ---: | ---: | --- | ---: | ---: |
| electricity | 720 | 192 | phase5_asx_period_multi | 0.151979 | 0.249248 |
| electricity | 720 | 336 | phase5_asx_period_multi | 0.166828 | 0.263942 |
| electricity | 720 | 720 | phase5_asx_period_multi | 0.203556 | 0.293765 |
| electricity | 720 | 96 | phase5_asx_period_multi | 0.138265 | 0.238301 |
| weather | 720 | 192 | phase5_asx_individual_revin | 0.190203 | 0.240335 |
| weather | 720 | 336 | phase5_asx_individual_revin | 0.240016 | 0.279124 |
| weather | 720 | 720 | phase5_asx_individual | 0.309688 | 0.331898 |
| weather | 720 | 96 | phase5_asx_individual_revin | 0.147851 | 0.200624 |

## Best-Cell Count (by test, analysis only)

| arm | cells |
| --- | ---: |
| phase5_asx_individual | 1 |
| phase5_asx_individual_revin | 3 |
| phase5_asx_period_multi | 4 |

## Delta Versus Anchor (cell-mean)

| arm | cells | delta_mse_mean | delta_mae_mean |
| --- | ---: | ---: | ---: |
| phase5_asx_cross_clip05 | 8 | 0.00195666 | 0.00219474 |
| phase5_asx_individual | 8 | -0.00221563 | -0.00305264 |
| phase5_asx_individual_period | 8 | -0.00315039 | -0.00441258 |
| phase5_asx_individual_revin | 8 | -0.00219675 | -0.00445824 |
| phase5_asx_period_multi | 8 | -0.00234942 | -0.00388137 |

## Single-Arm Candidate Summary

| arm | n | mse_mean | mae_mean | val_mse_mean |
| --- | ---: | ---: | ---: | ---: |
| phase5_asx_cross | 24 | 0.19893 | 0.270275 | 0.311559 |
| phase5_asx_cross_clip05 | 24 | 0.200887 | 0.27247 | 0.31365 |
| phase5_asx_individual | 24 | 0.196715 | 0.267222 | 0.312439 |
| phase5_asx_individual_revin | 24 | 0.196733 | 0.265817 | 0.31199 |
| phase5_asx_period_multi | 24 | 0.196581 | 0.266393 | 0.309633 |
| phase5_asx_individual_period | 24 | 0.19578 | 0.265862 | 0.311565 |

## Paired Statistics vs Anchor

Paired by dataset/seq_len/pred_len/seed.

| arm | pairs | dMSE_mean | dMSE_std | dMSE_2sd | win/loss/tie | dMAE_mean | dMAE_std |
| --- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| phase5_asx_cross_clip05 | 24 | 0.00195666 | 0.00328266 | 0.00656532 | 7/17/0 | 0.00219474 | 0.00295545 |
| phase5_asx_individual | 24 | -0.00221563 | 0.00415509 | 0.00831017 | 14/10/0 | -0.00305264 | 0.0041238 |
| phase5_asx_individual_period | 24 | -0.00315039 | 0.00350288 | 0.00700575 | 17/7/0 | -0.00441258 | 0.00317744 |
| phase5_asx_individual_revin | 24 | -0.00219675 | 0.00444621 | 0.00889242 | 14/10/0 | -0.00445824 | 0.00520259 |
| phase5_asx_period_multi | 24 | -0.00234942 | 0.00241706 | 0.00483412 | 21/3/0 | -0.00388137 | 0.00395631 |

## Validation-Selected Summary

- selected_test_mse_mean: 0.195153
- selected_test_mae_mean: 0.264011

### Selected Arm Counts (per group)

| arm | groups |
| --- | ---: |
| phase5_asx_individual_revin | 2 |
| phase5_asx_period_multi | 6 |

### Selected vs Best Single-Arm

- selected_mse_mean: 0.195153
- best_single_arm: phase5_asx_individual_period (mse_mean=0.19578)
- delta(selected - best_single): -0.000627302

## Per Dataset

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| electricity | 12 | 0.165157 | 0.261314 |
| weather | 12 | 0.225148 | 0.266709 |

## Per Pred_len

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 192 | 6 | 0.171091 | 0.244792 |
| 336 | 6 | 0.208104 | 0.276886 |
| 720 | 6 | 0.258358 | 0.314905 |
| 96 | 6 | 0.143058 | 0.219463 |

## Per Seq_len

| key | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 720 | 24 | 0.195153 | 0.264011 |

## Validation Segment Mismatch (full val_mse vs last segment)

Last segment column: val_mse_seg3. How often the arm chosen by mean full val_mse differs from the arm chosen by mean last-segment val_mse.

| dataset | groups | mismatches |
| --- | ---: | ---: |
| electricity | 4 | 0 |
| weather | 4 | 4 |

## Fairness Note

Selection uses validation metrics aggregated over seeds. Test metrics are reported only after arm selection.

