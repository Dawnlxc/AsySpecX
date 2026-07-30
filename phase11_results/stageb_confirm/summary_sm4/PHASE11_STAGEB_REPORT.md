# Phase 11 Stage B real-SM confirmation

Stable win = at least 2/3 paired test wins over Stage A and lower mean validation MSE.

| arm | val mean | test MSE mean +/- std | wins | MAE | params | fixed train | fixed infer | stable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fk_r8_cs | 0.6853062 | 0.3418097 +/- 0.0001664 | 0/3 | 0.3384249 | 75882 | 1.000x | 1.000x | 0 |
| fk_sm4_mode | 0.6852741 | 0.3417113 +/- 0.0001672 | 3/3 | 0.3383718 | 75986 | 1.097x | 1.026x | 1 |
