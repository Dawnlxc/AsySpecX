# Phase 11 Stage B validation-only selection

No test metric was read or used.

| arm | val MSE | vs Stage A | params | fixed train | fixed infer | SM gate | envelope | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| fk_sm4_mode | 0.6851097 | -0.0045% | 75986 | 1.034x | 1.231x | 0.9463 | 0.670..2.362 | 0 |
| fk_sm2_mode | 0.6851393 | -0.0002% | 75938 | 1.068x | 1.230x | 0.8891 | 0.088..5.359 | 0 |
| fk_r8_cs | 0.6851406 | +0.0000% | 75882 |  |  |  |  | 0 |
| fk_sm2_shared | 0.6851518 | +0.0016% | 75889 | 1.031x | 1.248x | 0.7947 | 0.143..3.838 | 0 |
| fk_sm4_frozen | 0.6852767 | +0.0199% | 75218 | 1.029x | 1.232x | 0.9920 | 0.584..3.949 | 0 |
| dense_direct | 0.6853235 | +0.0267% | 138306 |  |  |  |  | 0 |

Promoted: none
