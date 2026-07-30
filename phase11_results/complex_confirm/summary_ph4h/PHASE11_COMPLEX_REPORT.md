# Complex phase three-seed confirmation

Stable win = at least 2/3 paired test wins and lower mean validation MSE than real SM.

| arm | val mean | test MSE mean +/- std | wins | MAE | params | train | infer | stable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fk_sm4_mode | 0.6852741 | 0.3417113 +/- 0.0001672 | 0/3 | 0.3383718 | 75986 | 1.000x | 1.000x | 0 |
| fk_sm4_ph4_h | 0.6852451 | 0.3416851 +/- 0.0001661 | 3/3 | 0.3383472 | 76018 | 1.044x | 1.062x | 1 |
