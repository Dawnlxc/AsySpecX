# AsySpecX Phase 1 Summary

- total_runs: 384
- ok_runs: 384
- failed_runs: 0
- results_csv: phase1_results/main/results.csv

## Arm Means

| arm | n | mse_mean | mae_mean |
| --- | ---: | ---: | ---: |
| phase1_cross_zero_global | 96 | 0.345011 | 0.350067 |
| phase1_fits_only | 96 | 0.387106 | 0.371494 |
| phase1_safe_cross | 96 | 0.363808 | 0.359248 |
| phase1_safe_cross_backcast | 96 | 0.381788 | 0.374015 |

## Best Arm Per Dataset/Length

| dataset | seq_len | pred_len | best_arm | mse | mae |
| --- | ---: | ---: | --- | ---: | ---: |
| ETTh1 | 96 | 96 | phase1_safe_cross_backcast | 0.38842 | 0.399232 |
| ETTh1 | 96 | 192 | phase1_cross_zero_global | 0.436082 | 0.425842 |
| ETTh1 | 96 | 336 | phase1_cross_zero_global | 0.479734 | 0.452613 |
| ETTh1 | 96 | 720 | phase1_cross_zero_global | 0.466145 | 0.467193 |
| ETTh1 | 720 | 96 | phase1_fits_only | 0.379291 | 0.402379 |
| ETTh1 | 720 | 192 | phase1_safe_cross_backcast | 0.413502 | 0.422816 |
| ETTh1 | 720 | 336 | phase1_fits_only | 0.43325 | 0.438469 |
| ETTh1 | 720 | 720 | phase1_cross_zero_global | 0.432241 | 0.45799 |
| ETTm1 | 96 | 96 | phase1_cross_zero_global | 0.358461 | 0.383594 |
| ETTm1 | 96 | 192 | phase1_cross_zero_global | 0.391943 | 0.396297 |
| ETTm1 | 96 | 336 | phase1_cross_zero_global | 0.421332 | 0.415279 |
| ETTm1 | 96 | 720 | phase1_cross_zero_global | 0.485439 | 0.447837 |
| ETTm1 | 720 | 96 | phase1_safe_cross | 0.318921 | 0.359933 |
| ETTm1 | 720 | 192 | phase1_safe_cross | 0.345453 | 0.374501 |
| ETTm1 | 720 | 336 | phase1_safe_cross | 0.372433 | 0.390177 |
| ETTm1 | 720 | 720 | phase1_safe_cross | 0.420326 | 0.415798 |
| PEMS04 | 96 | 12 | phase1_cross_zero_global | 0.0965735 | 0.206863 |
| PEMS04 | 96 | 24 | phase1_cross_zero_global | 0.146541 | 0.260054 |
| PEMS04 | 96 | 48 | phase1_cross_zero_global | 0.258832 | 0.362168 |
| PEMS04 | 96 | 96 | phase1_cross_zero_global | 0.428283 | 0.48493 |
| PEMS08 | 96 | 12 | phase1_cross_zero_global | 0.095272 | 0.20212 |
| PEMS08 | 96 | 24 | phase1_cross_zero_global | 0.155553 | 0.26186 |
| PEMS08 | 96 | 48 | phase1_cross_zero_global | 0.32175 | 0.397219 |
| PEMS08 | 96 | 96 | phase1_cross_zero_global | 0.658282 | 0.599444 |
| electricity | 96 | 96 | phase1_cross_zero_global | 0.203824 | 0.287434 |
| electricity | 96 | 192 | phase1_cross_zero_global | 0.204644 | 0.290886 |
| electricity | 96 | 336 | phase1_cross_zero_global | 0.217681 | 0.304588 |
| electricity | 96 | 720 | phase1_cross_zero_global | 0.257344 | 0.33455 |
| electricity | 720 | 96 | phase1_cross_zero_global | 0.141633 | 0.243415 |
| electricity | 720 | 192 | phase1_cross_zero_global | 0.156213 | 0.255978 |
| electricity | 720 | 336 | phase1_cross_zero_global | 0.171654 | 0.271269 |
| electricity | 720 | 720 | phase1_cross_zero_global | 0.21012 | 0.30373 |
| traffic | 96 | 96 | phase1_cross_zero_global | 0.654832 | 0.3983 |
| traffic | 96 | 192 | phase1_safe_cross | 0.607958 | 0.374375 |
| traffic | 96 | 336 | phase1_safe_cross | 0.616716 | 0.383272 |
| traffic | 96 | 720 | phase1_safe_cross | 0.654128 | 0.400969 |
| traffic | 720 | 96 | phase1_cross_zero_global | 0.389922 | 0.279136 |
| traffic | 720 | 192 | phase1_cross_zero_global | 0.402306 | 0.283253 |
| traffic | 720 | 336 | phase1_cross_zero_global | 0.415363 | 0.289417 |
| traffic | 720 | 720 | phase1_safe_cross | 0.454483 | 0.307936 |
| weather | 96 | 96 | phase1_cross_zero_global | 0.197387 | 0.251477 |
| weather | 96 | 192 | phase1_cross_zero_global | 0.24592 | 0.287425 |
| weather | 96 | 336 | phase1_cross_zero_global | 0.295873 | 0.319655 |
| weather | 96 | 720 | phase1_cross_zero_global | 0.366251 | 0.362259 |
| weather | 720 | 96 | phase1_cross_zero_global | 0.168013 | 0.224256 |
| weather | 720 | 192 | phase1_cross_zero_global | 0.209678 | 0.259953 |
| weather | 720 | 336 | phase1_cross_zero_global | 0.255372 | 0.293679 |
| weather | 720 | 720 | phase1_cross_zero_global | 0.318774 | 0.339947 |

