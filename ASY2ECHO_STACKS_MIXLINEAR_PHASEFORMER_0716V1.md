# Asy2+Echo-Micro stacked on MixLinear and PhaseFormer — 0716v1

Generated: 2026-07-16T14:47:54+10:00

## Verdict

Completed and audited: **32/32** cells; lower-test-MSE wins: **31/32** among completed cells.

A cell is counted only when the validation-selected checkpoint, first-and-only held-out test marker, expected sample count, parameter counts, and Slurm artifacts all agree. Negative `MSE change` is better.

## Protocol

- Datasets: Weather (21 channels, cycle 144) and Electricity (321 channels, cycle 168).
- Inputs `L`: 96 and 720; horizons `H`: 96, 192, 336, 720.
- Overlay: Asy2+Echo-Micro with cycle rank 1, time rank 4, PCKD rank 1 for H>cycle, Echo on, Cross off, RevIN on, plus one zero-initialized signed fusion gate.
- MixLinear uses seed 2023, MSE loss, official LR/batch/alpha/lpf settings. PhaseFormer uses seed 2021, Huber loss, official compact per-horizon settings.
- Backbone and overlay are jointly trained; backbone keeps its official LR, overlay LR is 0.005 with the same decay factor.
- The archived official baseline metrics are compared to new strict validation-selected, deferred-test overlay runs. Test is never read during overlay training.

## MixLinear + Asy2+Echo-Micro

Audited 16/16; lower-MSE cells 16/16.

### Weather, L=96

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.199544 | 0.177350 | -11.12% | 0.241247 | 0.224674 | 105 | 945 | 1,050 |
| 192 | 0.244444 | 0.228289 | -6.61% | 0.277257 | 0.265782 | 173 | 1,145 | 1,318 |
| 336 | 0.296596 | 0.282548 | -4.74% | 0.313341 | 0.303614 | 275 | 1,149 | 1,424 |
| 720 | 0.370333 | 0.357685 | -3.42% | 0.360759 | 0.352029 | 507 | 1,157 | 1,664 |

### Weather, L=720

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.179305 | 0.149724 | -16.50% | 0.237158 | 0.205627 | 195 | 3,441 | 3,636 |
| 192 | 0.221808 | 0.193820 | -12.62% | 0.274112 | 0.247864 | 299 | 3,641 | 3,940 |
| 336 | 0.267432 | 0.253080 | -5.37% | 0.307729 | 0.292618 | 455 | 3,645 | 4,100 |
| 720 | 0.329212 | 0.313600 | -4.74% | 0.350868 | 0.337419 | 759 | 3,653 | 4,412 |

### Electricity, L=96

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.209898 | 0.165275 | -21.26% | 0.281040 | 0.257386 | 47 | 1,269 | 1,316 |
| 192 | 0.206786 | 0.172086 | -16.78% | 0.282648 | 0.263074 | 57 | 1,565 | 1,622 |
| 336 | 0.220822 | 0.188619 | -14.58% | 0.297409 | 0.279554 | 73 | 1,565 | 1,638 |
| 720 | 0.262312 | 0.228063 | -13.06% | 0.330765 | 0.311059 | 113 | 1,577 | 1,690 |

### Electricity, L=720

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.138606 | 0.131073 | -5.43% | 0.233400 | 0.227552 | 95 | 3,765 | 3,860 |
| 192 | 0.154282 | 0.147622 | -4.32% | 0.248337 | 0.242443 | 107 | 4,061 | 4,168 |
| 336 | 0.170718 | 0.163407 | -4.28% | 0.264574 | 0.258832 | 131 | 4,061 | 4,192 |
| 720 | 0.209489 | 0.202955 | -3.12% | 0.297907 | 0.292060 | 187 | 4,073 | 4,260 |

## PhaseFormer + Asy2+Echo-Micro

Audited 16/16; lower-MSE cells 15/16.

### Weather, L=96

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.179010 | 0.168157 | -6.06% | 0.216349 | 0.209570 | 4,524 | 945 | 5,469 |
| 192 | 0.230423 | 0.219306 | -4.82% | 0.259385 | 0.253582 | 3,052 | 1,145 | 4,197 |
| 336 | 0.282912 | 0.272174 | -3.80% | 0.296787 | 0.294264 | 3,106 | 1,149 | 4,255 |
| 720 | 0.359476 | 0.348864 | -2.95% | 0.345573 | 0.344119 | 3,250 | 1,157 | 4,407 |

### Weather, L=720

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.148620 | 0.149070 | +0.30% | 0.193560 | 0.195949 | 5,616 | 3,441 | 9,057 |
| 192 | 0.192744 | 0.191044 | -0.88% | 0.236468 | 0.237544 | 3,702 | 3,641 | 7,343 |
| 336 | 0.245097 | 0.241525 | -1.46% | 0.280126 | 0.277517 | 3,756 | 3,645 | 7,401 |
| 720 | 0.320802 | 0.311720 | -2.83% | 0.335032 | 0.329238 | 3,900 | 3,653 | 7,553 |

### Electricity, L=96

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.184393 | 0.157447 | -14.61% | 0.258763 | 0.247522 | 3,016 | 1,269 | 4,285 |
| 192 | 0.178030 | 0.164206 | -7.76% | 0.256725 | 0.252336 | 269,832 | 1,565 | 271,397 |
| 336 | 0.206783 | 0.187241 | -9.45% | 0.282938 | 0.276009 | 3,106 | 1,565 | 4,671 |
| 720 | 0.234283 | 0.222075 | -5.21% | 0.306655 | 0.302966 | 272,670 | 1,577 | 274,247 |

### Electricity, L=720

| H | Base MSE | +Asy2+Echo MSE | MSE change | Base MAE | +Asy2+Echo MAE | Base params | Added | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.128465 | 0.127460 | -0.78% | 0.220317 | 0.220729 | 3,666 | 3,765 | 7,431 |
| 192 | 0.146529 | 0.144539 | -1.36% | 0.236189 | 0.234771 | 273,160 | 4,061 | 277,221 |
| 336 | 0.166116 | 0.164075 | -1.23% | 0.258199 | 0.258211 | 3,756 | 4,061 | 7,817 |
| 720 | 0.198819 | 0.195948 | -1.44% | 0.284560 | 0.283729 | 275,998 | 4,073 | 280,071 |

## Overlay runtime evidence

| Backbone | Dataset | L | H | Train s | Forward ms/sample | Train peak MiB | Test peak MiB | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mixlinear | weather | 96 | 96 | 146.3 | 0.1421 | 74.4 | 39.7 | 0.3170 |
| mixlinear | weather | 96 | 192 | 155.5 | 0.1425 | 81.9 | 46.0 | 0.2654 |
| mixlinear | weather | 96 | 336 | 177.8 | 0.1436 | 95.8 | 57.1 | 0.3177 |
| mixlinear | weather | 96 | 720 | 185.3 | 0.1473 | 127.2 | 81.6 | 0.3411 |
| mixlinear | weather | 720 | 96 | 128.3 | 0.1441 | 105.8 | 59.3 | 0.0745 |
| mixlinear | weather | 720 | 192 | 151.4 | 0.1405 | 111.5 | 62.5 | 0.0990 |
| mixlinear | weather | 720 | 336 | 242.5 | 0.1450 | 122.2 | 68.7 | 0.1018 |
| mixlinear | weather | 720 | 720 | 155.4 | 0.1510 | 143.8 | 91.4 | 0.0808 |
| mixlinear | electricity | 96 | 96 | 346.8 | 0.2821 | 225.2 | 147.1 | -0.2008 |
| mixlinear | electricity | 96 | 192 | 381.7 | 0.2820 | 330.3 | 245.1 | 0.6188 |
| mixlinear | electricity | 96 | 336 | 402.2 | 0.3011 | 503.7 | 382.6 | 0.4450 |
| mixlinear | electricity | 96 | 720 | 574.1 | 0.3157 | 969.4 | 759.6 | 0.5611 |
| mixlinear | electricity | 720 | 96 | 1496.3 | 0.2894 | 717.2 | 435.8 | 0.2048 |
| mixlinear | electricity | 720 | 192 | 1824.4 | 0.2980 | 766.9 | 489.6 | 0.2612 |
| mixlinear | electricity | 720 | 336 | 1877.7 | 0.3124 | 877.8 | 556.5 | 0.3044 |
| mixlinear | electricity | 720 | 720 | 1648.0 | 0.1283 | 1248.6 | 903.8 | 0.2301 |
| phaseformer | weather | 96 | 96 | 656.7 | 0.2141 | 85.6 | 36.9 | 0.2016 |
| phaseformer | weather | 96 | 192 | 616.8 | 0.1831 | 79.9 | 37.4 | 0.2675 |
| phaseformer | weather | 96 | 336 | 852.4 | 0.1836 | 81.8 | 38.2 | 0.2175 |
| phaseformer | weather | 96 | 720 | 625.0 | 0.1531 | 86.9 | 40.6 | 0.2890 |
| phaseformer | weather | 720 | 96 | 352.6 | 0.2197 | 91.2 | 40.3 | 0.0541 |
| phaseformer | weather | 720 | 192 | 381.6 | 0.1883 | 85.1 | 40.9 | 0.1227 |
| phaseformer | weather | 720 | 336 | 538.1 | 0.1876 | 86.8 | 41.7 | 0.1234 |
| phaseformer | weather | 720 | 720 | 240.4 | 0.1857 | 91.4 | 43.9 | 0.0770 |
| phaseformer | electricity | 96 | 96 | 448.2 | 0.1701 | 291.8 | 103.0 | 0.2438 |
| phaseformer | electricity | 96 | 192 | 555.4 | 0.3743 | 1640.4 | 786.9 | 0.6587 |
| phaseformer | electricity | 96 | 336 | 600.7 | 0.2585 | 328.1 | 124.1 | 0.5736 |
| phaseformer | electricity | 96 | 720 | 574.0 | 0.3803 | 1670.5 | 827.9 | -0.1539 |
| phaseformer | electricity | 720 | 96 | 824.3 | 0.2259 | 367.2 | 152.8 | 0.0773 |
| phaseformer | electricity | 720 | 192 | 545.5 | 0.3645 | 1665.8 | 823.8 | 0.5720 |
| phaseformer | electricity | 720 | 336 | 540.9 | 0.2808 | 405.4 | 172.0 | 0.2518 |
| phaseformer | electricity | 720 | 720 | 564.1 | 0.3800 | 1696.0 | 864.7 | 0.4421 |

## Audit status

All 32 cells passed the artifact and protocol audit.

## Interpretation limits

- Parameter counts show registered trainable elements; the CSV also preserves MixLinear's complex real-scalar-equivalent count.
- Baseline elapsed time came from its official-repo harness, while detailed forward/memory timing is from the unified overlay harness; do not treat those timers as a strict speed ratio.
- This matrix uses one seed per backbone to match the official baselines. Replicate promising cells before making a paper-level robustness claim.

## Provenance

- Manifest: `outputs/asy2echo_stacks/manifests/a2estacks_phasee_fix0716v1_20260716_142824_merged32.csv`
- Run ID: `a2estacks_0716v1_20260716_024250`
- Code SHA-256 bundle: `96f8a84ac3ac894c5825555a597d79e2d03f01d91f09bacefd3e078e9f32829d`
- Official MixLinear commit: `71b5db62e38b1cb108faa1e1d4687287b4568f3b`.
- Official PhaseFormer commit: `ed1db61c6abfa9326d5ca2a56c6c4ba53ea592ab`.
