# AsySpecX Phase 11 final confirmation

A stable win requires at least two seed-matched test wins over dense and lower mean validation MSE.

| arm | val mean | test MSE mean ± std | wins vs dense | MAE mean | params | fixed train | fixed infer | stable win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_direct | 0.6853554 | 0.3425529 ± 0.0002046 | 0/3 | 0.3386665 | 138306 | 1.000x | 1.000x | 0 |
| fk_r8_cs | 0.6853062 | 0.3418097 ± 0.0001664 | 3/3 | 0.3384249 | 75882 | 0.947x | 0.958x | 1 |
| fk_r8 | 0.6855741 | 0.3426983 ± 0.0000321 | 1/3 | 0.3387700 | 75714 | 0.915x | 0.939x | 0 |

Published Weather H720 reference: 0.3387.
