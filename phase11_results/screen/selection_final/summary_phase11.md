# AsySpecX Phase 11 validation-only screen

No test metric was read or used by this selector.

| arm | val MSE | vs reference | params | dense ratio | fixed train | fixed infer | peak MiB | gate | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fk_r8_cs | 0.6851406 | -0.027% | 75882 | 0.549 | 0.947x | 0.958x | 143.4 | 0.0469 | 1 |
| dense_direct | 0.6853235 | +0.000% | 138306 | 1.000 | 1.000x | 1.000x | 144.0 |  | 0 |
| fk_r8 | 0.6855323 | +0.030% | 75714 | 0.547 | 0.915x | 0.939x | 143.4 | 0.0430 | 1 |
| fk_r4 | 0.6855788 | +0.037% | 72450 | 0.524 | 0.939x | 0.975x | 143.3 | 0.0468 | 1 |
| anchor | 0.6865768 | +0.183% | 67746 | 0.490 |  |  | 138.0 |  | 0 |
| fk_r16_svd | 0.6879045 | +0.377% | 82578 | 0.597 |  |  | 143.7 | 0.0302 | 0 |
| fk_r8_svd | 0.6879948 | +0.390% | 75882 | 0.549 |  |  | 143.4 | 0.0316 | 0 |

Promoted: fk_r8_cs, fk_r8
