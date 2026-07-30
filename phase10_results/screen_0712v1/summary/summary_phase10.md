# AsySpecX Phase 10 Summary

- discovered summaries: 12
- ok: 12
- failed_or_incomplete: 0

Selection below uses validation MSE only. Test MSE is read after selection.

## Validation-selected cells

| dataset | sl | pl | arm | cf | seeds | val | test | vs Phase8 | vs baseline | params | peak MiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weather | 96 | 96 | ind_cycle_full | 13 | 1 | 0.396414 | 0.158833 | 0.018564 | 0.000533 | 18354 | 79.7 |
| weather | 96 | 336 | ind_cycle_full | 13 | 1 | 0.556536 | 0.264347 | 0.020094 | 0.000947 | 37170 | 104.6 |

## Fixed-config ranking (validation metric)

| dataset | sl | arm | cf | runs | horizons | seeds | val mean | test mean | params max | peak MiB |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| weather | 96 | ind_cycle_full | 13 | 2 | 96+336 | 2026 | 0.476475 | 0.211590 | 37170 | 104.6 |
| weather | 96 | ind_dlinear_res | 13 | 2 | 96+336 | 2026 | 0.487974 | 0.216738 | 99666 | 106.5 |
| weather | 96 | ind_direct_res | 13 | 2 | 96+336 | 2026 | 0.488102 | 0.216792 | 67074 | 104.0 |
| weather | 96 | ind | 25 | 2 | 96+336 | 2026 | 0.490732 | 0.217289 | 122346 | 102.1 |
| weather | 96 | ind | 13 | 2 | 96+336 | 2026 | 0.493508 | 0.218288 | 34146 | 99.3 |
| weather | 96 | ind | 5 | 2 | 96+336 | 2026 | 0.500251 | 0.221403 | 5586 | 99.8 |
