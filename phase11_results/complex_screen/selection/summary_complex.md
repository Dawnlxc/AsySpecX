# Complex-phase validation-only selection

No test metric was read or used.

| arm | val MSE | vs real SM | params | train | infer | phase max | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fk_sm4_ph4_h | 0.6850928 | -0.0025% | 76018 | 1.044x | 1.062x | 0.5967 | 1 |
| fk_sm4_ph4_q | 0.6850969 | -0.0019% | 76018 | 1.044x | 1.126x | 0.4143 | 0 |
| fk_sm4_ph2_q | 0.6851062 | -0.0005% | 76002 | 1.048x | 1.051x | 0.2793 | 1 |
| fk_sm4_mode | 0.6851097 | +0.0000% | 75986 | 1.000x | 1.000x |  | 0 |
| dense_direct | 0.6853235 | +0.0312% | 138306 | 0.909x | 0.942x |  | 0 |

Promoted: fk_sm4_ph4_h, fk_sm4_ph2_q
