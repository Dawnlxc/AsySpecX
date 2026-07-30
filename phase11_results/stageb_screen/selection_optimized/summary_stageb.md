# Phase 11 Stage B validation-only selection

No test metric was read or used.

| arm | val MSE | vs Stage A | params | fixed train | fixed infer | SM gate | envelope | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| fk_sm4_mode | 0.6851097 | -0.0045% | 75986 | 1.097x | 1.026x | 0.9463 | 0.670..2.362 | 1 |
| fk_sm2_mode | 0.6851393 | -0.0002% | 75938 | 1.094x | 1.025x | 0.8891 | 0.088..5.359 | 1 |
| fk_r8_cs | 0.6851406 | +0.0000% | 75882 |  |  |  |  | 0 |
| fk_sm2_shared | 0.6851518 | +0.0016% | 75889 | 1.097x | 1.029x | 0.7947 | 0.143..3.838 | 0 |
| fk_sm4_frozen | 0.6852767 | +0.0199% | 75218 | 1.093x | 1.027x | 0.9920 | 0.584..3.949 | 0 |
| dense_direct | 0.6853235 | +0.0267% | 138306 |  |  |  |  | 0 |

Promoted: fk_sm4_mode, fk_sm2_mode
