# Phase 11 Stage B real-SM confirmation

Stable win = at least 2/3 paired test wins over Stage A and lower mean validation MSE.

| arm | val mean | test MSE mean +/- std | wins | MAE | params | fixed train | fixed infer | stable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fk_r8_cs | 0.6853062 | 0.3418097 +/- 0.0001664 | 0/3 | 0.3384249 | 75882 | 1.000x | 1.000x | 0 |
| fk_sm2_mode | 0.6852830 | 0.3416446 +/- 0.0001653 | 3/3 | 0.3383220 | 75938 | 1.094x | 1.025x | 1 |
