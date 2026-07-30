# AsySpecX Phase 10 Summary

- discovered summaries: 23
- ok: 23
- failed_or_incomplete: 0

Selection below uses validation MSE only. Test MSE is read after selection.

## Validation-selected cells

| dataset | sl | pl | arm | cf | seeds | val | test | vs Phase8 | vs baseline | params | peak MiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| electricity | 720 | 96 | period_cycle_full | 181 | 1 | 0.111525 | 0.134186 | 0.002348 |  | 273340 | 1127.6 |
| weather | 96 | 96 | ind_cycle_full | 13 | 3 | 0.396968 | 0.158829 | 0.018568 | 0.000529 | 18354 | 79.7 |
| weather | 96 | 192 | ind_cycle_full | 13 | 3 | 0.470454 | 0.207777 | 0.022789 | 0.001777 | 25998 | 88.4 |
| weather | 96 | 336 | ind_cycle_full | 13 | 3 | 0.556942 | 0.264387 | 0.020054 | 0.000987 | 37170 | 104.6 |
| weather | 96 | 720 | ind_cycle_full | 13 | 2 | 0.685768 | 0.343665 | 0.013035 | 0.004965 | 67746 | 138.0 |

## Fixed-config ranking (validation metric)

Only the common screening horizons 96 and 336 are used for ranking; all-horizon means are descriptive.

| dataset | sl | arm | cf | screen complete | screen runs | seeds | screen val | screen test | all horizons | all test | params max | peak MiB |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| electricity | 720 | period_cycle_full | 181 | 0 | 1 | 2026 | 0.111525 | 0.134186 | 96 | 0.134186 | 273340 | 1127.6 |
| electricity | 720 | period | 181 | 0 | 1 | 2026 | 0.115639 | 0.137182 | 96 | 0.137182 | 219412 | 997.6 |
| weather | 96 | ind_cycle_full | 13 | 1 | 6 | 2024+2025+2026 | 0.476955 | 0.211608 | 96+192+336+720 | 0.234573 | 67746 | 138.0 |
| weather | 96 | ind_dlinear_res | 13 | 1 | 2 | 2026 | 0.487974 | 0.216738 | 96+336 | 0.216738 | 99666 | 106.5 |
| weather | 96 | ind_direct_res | 13 | 1 | 2 | 2026 | 0.488102 | 0.216792 | 96+336 | 0.216792 | 67074 | 104.0 |
| weather | 96 | ind | 25 | 1 | 2 | 2026 | 0.490732 | 0.217289 | 96+336 | 0.217289 | 122346 | 102.1 |
| weather | 96 | ind | 13 | 1 | 2 | 2026 | 0.493508 | 0.218288 | 96+336 | 0.218288 | 34146 | 99.3 |
| weather | 96 | ind | 5 | 1 | 2 | 2026 | 0.500251 | 0.221403 | 96+336 | 0.221403 | 5586 | 99.8 |
