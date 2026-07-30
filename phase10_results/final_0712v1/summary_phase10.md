# AsySpecX Phase 10 Summary

- discovered summaries: 92
- ok: 92
- failed_or_incomplete: 0

Selection first prefers the largest replicated seed count in each cell, then uses mean validation MSE only. Test MSE is read after selection.

## Validation-selected cells

| dataset | sl | pl | arm | cf | seeds | wins | val | test mean | test std | vs Phase8 | vs published sl96 baseline | params | peak MiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| electricity | 96 | 96 | ind_cycle_full_direct_res | 25 | 1 | 0 | 0.117298 | 0.141660 | 0.000000 | 0.050794 | 0.004160 | 898578 | 342.1 |
| electricity | 96 | 336 | ind_cycle_full | 25 | 1 | 0 | 0.146601 | 0.174035 | 0.000000 | 0.033247 | 0.001735 | 1924074 | 651.1 |
| electricity | 168 | 96 | compact_period_cycle_full | 43 | 1 | 0 | 0.119130 | 0.142071 | 0.000000 |  | 0.004571 | 67144 | 438.4 |
| electricity | 168 | 336 | compact_period_cycle_full | 43 | 1 | 0 | 0.148934 | 0.174198 | 0.000000 |  | 0.001898 | 74760 | 867.0 |
| electricity | 336 | 96 | compact_period_cycle_full | 85 | 1 | 0 | 0.116099 | 0.138065 | 0.000000 |  | 0.000565 | 80764 | 634.7 |
| electricity | 336 | 336 | compact_period_cycle_full | 85 | 1 | 1 | 0.142584 | 0.167584 | 0.000000 |  | -0.004716 | 95336 | 1045.6 |
| electricity | 504 | 96 | compact_period_cycle_full | 127 | 1 | 1 | 0.112048 | 0.134789 | 0.000000 |  | -0.002711 | 101440 | 837.6 |
| electricity | 504 | 336 | compact_period_cycle_full | 127 | 1 | 1 | 0.140118 | 0.165176 | 0.000000 |  | -0.007124 | 122800 | 1245.3 |
| electricity | 720 | 96 | compact_period_cycle_full | 181 | 3 | 3 | 0.111917 | 0.134627 | 0.000312 | 0.001907 | -0.002873 | 138364 | 1086.6 |
| electricity | 720 | 192 | compact_period_cycle_full | 181 | 1 | 1 | 0.124178 | 0.149402 | 0.000000 | 0.001905 | -0.006098 | 150508 | 1253.6 |
| electricity | 720 | 336 | compact_period_cycle_full | 181 | 3 | 3 | 0.138862 | 0.164533 | 0.000199 | 0.002087 | -0.007767 | 168700 | 1498.7 |
| electricity | 720 | 720 | compact_period_cycle_full | 181 | 1 | 1 | 0.170002 | 0.201989 | 0.000000 | -0.001082 | -0.006911 | 217544 | 2165.1 |
| weather | 96 | 96 | ind_cycle_full_direct_res | 13 | 3 | 3 | 0.393130 | 0.157466 | 0.000263 | 0.019931 | -0.000834 | 27762 | 81.2 |
| weather | 96 | 192 | ind_cycle_full_direct_res | 25 | 3 | 1 | 0.467176 | 0.206055 | 0.000103 | 0.024511 | 0.000055 | 103782 | 91.6 |
| weather | 96 | 336 | ind_cycle_full_direct_res | 13 | 3 | 3 | 0.555131 | 0.263200 | 0.000148 | 0.021241 | -0.000200 | 70098 | 107.6 |
| weather | 96 | 720 | ind_cycle_full | 13 | 3 | 0 | 0.685754 | 0.343646 | 0.000153 | 0.013054 | 0.004946 | 67746 | 138.0 |

## Fixed-config ranking (validation metric)

Only the common screening horizons 96 and 336 are used for ranking; all-horizon means are descriptive.

| dataset | sl | arm | cf | screen complete | screen runs | seeds | screen val | screen test | all horizons | all test | params max | peak MiB |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| electricity | 96 | ind_cycle_full_direct_res | 25 | 1 | 2 | 2026 | 0.132118 | 0.157509 | 96+336 | 0.157509 | 1957002 | 693.0 |
| electricity | 96 | ind_cycle_full | 25 | 1 | 2 | 2026 | 0.132341 | 0.158649 | 96+336 | 0.158649 | 1924074 | 651.1 |
| electricity | 96 | factor_r4_cycle_direct_res | 25 | 1 | 2 | 2026 | 0.133731 | 0.158463 | 96+336 | 0.158463 | 117006 | 661.1 |
| electricity | 96 | factor_r2_cycle_direct_res | 25 | 1 | 2 | 2026 | 0.133817 | 0.158634 | 96+336 | 0.158634 | 105164 | 661.0 |
| electricity | 96 | factor_r2_cycle | 25 | 1 | 2 | 2026 | 0.134167 | 0.160176 | 96+336 | 0.160176 | 72236 | 619.0 |
| electricity | 96 | factor_r4_cycle | 25 | 1 | 2 | 2026 | 0.134530 | 0.160029 | 96+336 | 0.160029 | 84078 | 619.2 |
| electricity | 96 | cross_cycle_full_direct_res | 25 | 1 | 2 | 2026 | 0.139739 | 0.164310 | 96+336 | 0.164310 | 99134 | 669.3 |
| electricity | 96 | cross_cycle_full_dlinear_res | 25 | 1 | 2 | 2026 | 0.139752 | 0.164337 | 96+336 | 0.164337 | 131726 | 703.6 |
| electricity | 96 | cross_cycle_full | 25 | 1 | 2 | 2026 | 0.141483 | 0.167449 | 96+336 | 0.167449 | 66206 | 627.1 |
| electricity | 96 | ind_cycle_r8 | 25 | 1 | 2 | 2026 | 0.146896 | 0.172024 | 96+336 | 0.172024 | 1874058 | 650.6 |
| electricity | 96 | ind_direct_res | 25 | 1 | 2 | 2026 | 0.172052 | 0.195843 | 96+336 | 0.195843 | 1903074 | 650.8 |
| electricity | 96 | ind | 25 | 1 | 2 | 2026 | 0.178041 | 0.202722 | 96+336 | 0.202722 | 1870146 | 582.6 |
| electricity | 96 | cross | 25 | 1 | 2 | 2026 | 0.185851 | 0.210777 | 96+336 | 0.210777 | 12278 | 572.6 |
| electricity | 168 | compact_period_cycle_full | 43 | 1 | 2 | 2026 | 0.134032 | 0.158135 | 96+336 | 0.158135 | 74760 | 867.0 |
| electricity | 336 | compact_period_cycle_full | 85 | 1 | 2 | 2026 | 0.129342 | 0.152825 | 96+336 | 0.152825 | 95336 | 1045.6 |
| electricity | 504 | compact_period_cycle_full | 127 | 1 | 2 | 2026 | 0.126083 | 0.149982 | 96+336 | 0.149982 | 122800 | 1245.3 |
| electricity | 720 | period_cycle_full | 181 | 1 | 2 | 2026 | 0.125191 | 0.149269 | 96+192+336+720 | 0.162483 | 1229672 | 2128.3 |
| electricity | 720 | compact_period_cycle_full | 181 | 1 | 6 | 2024+2025+2026 | 0.125389 | 0.149580 | 96+192+336+720 | 0.156109 | 217544 | 2165.1 |
| electricity | 720 | cross_cycle_full | 181 | 1 | 2 | 2026 | 0.126233 | 0.150853 | 96+336 | 0.150853 | 156842 | 1396.5 |
| electricity | 720 | cross_cycle_full | 151 | 1 | 2 | 2026 | 0.127375 | 0.152624 | 96+336 | 0.152624 | 127566 | 1353.9 |
| electricity | 720 | cross_cycle_full | 121 | 1 | 2 | 2026 | 0.129012 | 0.154370 | 96+336 | 0.154370 | 103570 | 1311.4 |
| electricity | 720 | period | 181 | 1 | 2 | 2026 | 0.129152 | 0.151932 | 96+336 | 0.151932 | 587092 | 1343.8 |
| electricity | 720 | cross_cycle_full | 85 | 1 | 2 | 2026 | 0.134347 | 0.162273 | 96+336 | 0.162273 | 81710 | 1264.3 |
| electricity | 720 | cross_cycle_full | 61 | 1 | 2 | 2026 | 0.136289 | 0.164179 | 96+336 | 0.164179 | 71418 | 1229.6 |
| weather | 96 | ind_cycle_full_direct_res | 13 | 1 | 6 | 2024+2025+2026 | 0.474130 | 0.210333 | 96+192+336+720 | 0.222433 | 138306 | 144.0 |
| weather | 96 | ind_cycle_full | 13 | 1 | 6 | 2024+2025+2026 | 0.476955 | 0.211608 | 96+192+336+720 | 0.243660 | 67746 | 138.0 |
| weather | 96 | ind_dlinear_res | 13 | 1 | 2 | 2026 | 0.487974 | 0.216738 | 96+336 | 0.216738 | 99666 | 106.5 |
| weather | 96 | ind_direct_res | 13 | 1 | 2 | 2026 | 0.488102 | 0.216792 | 96+336 | 0.216792 | 67074 | 104.0 |
| weather | 96 | ind | 25 | 1 | 2 | 2026 | 0.490732 | 0.217289 | 96+336 | 0.217289 | 122346 | 102.1 |
| weather | 96 | ind | 13 | 1 | 2 | 2026 | 0.493508 | 0.218288 | 96+336 | 0.218288 | 34146 | 99.3 |
| weather | 96 | ind | 5 | 1 | 2 | 2026 | 0.500251 | 0.221403 | 96+336 | 0.221403 | 5586 | 99.8 |
| weather | 96 | ind_cycle_full_direct_res | 25 | 0 | 1 | 2024+2025+2026 | 0.554727 | 0.263048 | 192+336 | 0.220303 | 158298 | 109.8 |
