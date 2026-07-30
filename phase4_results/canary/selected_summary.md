# Phase 4-Finalize Validation Selection Summary

Validation selection is performed using val_mse averaged over replicate seeds for each dataset/seq_len/pred_len group. Test metrics are used only after selection.

- selection_keys: dataset,seq_len,pred_len
- replicate_key: seed
- select_metric: val_mse
- selection_groups: 2
- selected_test_mse_mean: 0.710887
- selected_test_mae_mean: 0.593227

## Selected Arm Counts

| arm | groups |
| --- | ---: |
| phase4_asx_period_single | 2 |

## Selected Per Dataset

| dataset | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| electricity | 1 | 1.09272 | 0.824383 |
| weather | 1 | 0.32905 | 0.362071 |

## Selected Per Pred_len

| pred_len | groups | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| 96 | 2 | 0.710887 | 0.593227 |

## Per Group Selection

| dataset | seq_len | pred_len | selected_arm | mean_val_mse | mean_test_mse | mean_test_mae |
| --- | --- | --- | --- | ---: | ---: | ---: |
| electricity | 720 | 96 | phase4_asx_period_single | 0.966936 | 1.09272 | 0.824383 |
| weather | 720 | 96 | phase4_asx_period_single | 0.832347 | 0.32905 | 0.362071 |
