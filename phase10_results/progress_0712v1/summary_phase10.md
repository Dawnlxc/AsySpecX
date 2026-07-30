# AsySpecX Phase 10 Summary

- discovered summaries: 34
- ok: 34
- failed_or_incomplete: 0

Selection below uses validation MSE only. Test MSE is read after selection.

## Validation-selected cells

| dataset | sl | pl | arm | cf | seeds | val | test | vs Phase8 | vs baseline | params | peak MiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| electricity | 96 | 96 | cross_cycle_full_direct_res | 25 | 1 | 0.124474 | 0.148310 | 0.044144 | 0.010810 | 72390 | 330.6 |
| electricity | 96 | 336 | cross_cycle_full_direct_res | 25 | 1 | 0.155004 | 0.180310 | 0.026972 | 0.008010 | 99134 | 669.3 |
| electricity | 720 | 96 | period_cycle_full | 181 | 1 | 0.111525 | 0.134186 | 0.002348 |  | 273340 | 1127.6 |
| electricity | 720 | 336 | period_cycle_full | 181 | 1 | 0.138857 | 0.164353 | 0.002267 |  | 641020 | 1511.7 |
| weather | 96 | 96 | ind_cycle_full_direct_res | 13 | 1 | 0.392233 | 0.157346 | 0.020051 | -0.000954 | 27762 | 81.2 |
| weather | 96 | 192 | ind_cycle_full | 13 | 3 | 0.470454 | 0.207777 | 0.022789 | 0.001777 | 25998 | 88.4 |
| weather | 96 | 336 | ind_cycle_full | 13 | 3 | 0.556942 | 0.264387 | 0.020054 | 0.000987 | 37170 | 104.6 |
| weather | 96 | 720 | ind_cycle_full | 13 | 3 | 0.685754 | 0.343646 | 0.013054 | 0.004946 | 67746 | 138.0 |

## Fixed-config ranking (validation metric)

Only the common screening horizons 96 and 336 are used for ranking; all-horizon means are descriptive.

| dataset | sl | arm | cf | screen complete | screen runs | seeds | screen val | screen test | all horizons | all test | params max | peak MiB |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| electricity | 96 | cross_cycle_full_direct_res | 25 | 1 | 2 | 2026 | 0.139739 | 0.164310 | 96+336 | 0.164310 | 99134 | 669.3 |
| electricity | 96 | cross_cycle_full | 25 | 1 | 2 | 2026 | 0.141483 | 0.167449 | 96+336 | 0.167449 | 66206 | 627.1 |
| electricity | 96 | cross | 25 | 1 | 2 | 2026 | 0.185851 | 0.210777 | 96+336 | 0.210777 | 12278 | 572.6 |
| electricity | 96 | cross_cycle_full_dlinear_res | 25 | 0 | 1 | 2026 | 0.124530 | 0.148370 | 96 | 0.148370 | 81702 | 365.4 |
| electricity | 720 | period_cycle_full | 181 | 1 | 2 | 2026 | 0.125191 | 0.149269 | 96+336 | 0.149269 | 641020 | 1511.7 |
| electricity | 720 | period | 181 | 1 | 2 | 2026 | 0.129152 | 0.151932 | 96+336 | 0.151932 | 587092 | 1343.8 |
| weather | 96 | ind_cycle_full | 13 | 1 | 6 | 2024+2025+2026 | 0.476955 | 0.211608 | 96+192+336+720 | 0.243660 | 67746 | 138.0 |
| weather | 96 | ind_dlinear_res | 13 | 1 | 2 | 2026 | 0.487974 | 0.216738 | 96+336 | 0.216738 | 99666 | 106.5 |
| weather | 96 | ind_direct_res | 13 | 1 | 2 | 2026 | 0.488102 | 0.216792 | 96+336 | 0.216792 | 67074 | 104.0 |
| weather | 96 | ind | 25 | 1 | 2 | 2026 | 0.490732 | 0.217289 | 96+336 | 0.217289 | 122346 | 102.1 |
| weather | 96 | ind | 13 | 1 | 2 | 2026 | 0.493508 | 0.218288 | 96+336 | 0.218288 | 34146 | 99.3 |
| weather | 96 | ind | 5 | 1 | 2 | 2026 | 0.500251 | 0.221403 | 96+336 | 0.221403 | 5586 | 99.8 |
| weather | 96 | ind_cycle_full_direct_res | 13 | 0 | 1 | 2026 | 0.392233 | 0.157346 | 96 | 0.157346 | 27762 | 81.2 |
