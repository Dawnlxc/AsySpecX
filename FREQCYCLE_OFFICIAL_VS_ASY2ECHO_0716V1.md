# FreqCycle official reproduction (L96 and L720)

Pinned upstream commit: `668ba8204bffbdf3e3967b7a617b2d8bc945ddf9`.

The L96 cells reproduce the public Weather/Electricity scripts. L720 keeps the same dataset-specific cycle, segment window/stride, MLP width, seed, optimizer schedule, batch size, and learning rate; only input length changes.

Registered parameters are the actual CUDA state-dict count. Active parameters exclude the duplicate official `model` MLP that is registered but not used by `forward`. The unused `Cycfilter` tensor is not registered on CUDA due to the upstream `Parameter(...).to(gpu)` assignment.

## Input length 96

| Dataset | H | Test MSE | Test MAE | Paper L96 MSE | Registered params | Active params | Train peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| electricity | 96 | 0.138942 | 0.233216 | 0.140 | 104,312 | 79,512 | 830.5 |
| electricity | 192 | 0.154767 | 0.247329 | 0.154 | 129,080 | 91,896 | 863.9 |
| electricity | 336 | 0.171835 | 0.264919 | 0.170 | 166,232 | 110,472 | 907.2 |
| electricity | 720 | 0.209347 | 0.296768 | 0.210 | 265,304 | 160,008 | 1029.4 |
| weather | 96 | 0.158477 | 0.202932 | 0.159 | 53,212 | 28,412 | 142.4 |
| weather | 192 | 0.205837 | 0.246245 | 0.208 | 77,980 | 40,796 | 147.1 |
| weather | 336 | 0.262490 | 0.288891 | 0.264 | 115,132 | 59,372 | 153.8 |
| weather | 720 | 0.344717 | 0.344579 | 0.343 | 214,204 | 108,908 | 171.6 |

## Input length 720

| Dataset | H | Test MSE | Test MAE | Paper L96 MSE | Registered params | Active params | Train peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| electricity | 96 | 0.129270 | 0.225331 | - | 306,592 | 201,920 | 40822.0 |
| electricity | 192 | 0.146843 | 0.241854 | - | 331,360 | 214,304 | 40854.4 |
| electricity | 336 | 0.162817 | 0.259287 | - | 368,512 | 232,880 | 40897.3 |
| electricity | 720 | 0.198580 | 0.292038 | - | 467,584 | 282,416 | 41019.1 |
| weather | 96 | 0.147196 | 0.202045 | - | 244,858 | 140,186 | 4071.6 |
| weather | 192 | 0.191169 | 0.243940 | - | 269,626 | 152,570 | 4076.2 |
| weather | 336 | 0.244089 | 0.283594 | - | 306,778 | 171,146 | 4084.6 |
| weather | 720 | 0.316879 | 0.334655 | - | 405,850 | 220,682 | 4101.2 |

## Comparison with chosen Asy2+Echo Full

Completed-cell MSE score: FreqCycle 5, Asy2+Echo 11, ties 0 (out of 16).

Positive Delta MSE means FreqCycle is worse. Active/ours is the conservative parameter ratio after excluding the official duplicate registered MLP; registered/ours is the literal trainable state-dict ratio.

| Dataset | L->H | FreqCycle MSE | Asy2+Echo MSE | Delta MSE | Winner | Freq active params | Freq registered params | Ours params | Active/ours | Registered/ours |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| electricity | 96->96 | 0.138942 | 0.138638 | 0.000304 | Asy2+Echo | 79,512 | 104,312 | 22,178 | 3.59x | 4.70x |
| electricity | 96->192 | 0.154767 | 0.153244 | 0.001522 | Asy2+Echo | 91,896 | 129,080 | 26,914 | 3.41x | 4.80x |
| electricity | 96->336 | 0.171835 | 0.170395 | 0.001440 | Asy2+Echo | 110,472 | 166,232 | 26,914 | 4.10x | 6.18x |
| electricity | 96->720 | 0.209347 | 0.209603 | -0.000256 | FreqCycle | 160,008 | 265,304 | 27,106 | 5.90x | 9.79x |
| weather | 96->96 | 0.158477 | 0.160774 | -0.002297 | FreqCycle | 28,412 | 53,212 | 3,470 | 8.19x | 15.33x |
| weather | 96->192 | 0.205837 | 0.212371 | -0.006534 | FreqCycle | 40,796 | 77,980 | 4,270 | 9.55x | 18.26x |
| weather | 96->336 | 0.262490 | 0.267135 | -0.004645 | FreqCycle | 59,372 | 115,132 | 4,286 | 13.85x | 26.86x |
| weather | 96->720 | 0.344717 | 0.348568 | -0.003852 | FreqCycle | 108,908 | 214,204 | 4,318 | 25.22x | 49.61x |
| electricity | 720->96 | 0.129270 | 0.126920 | 0.002351 | Asy2+Echo | 201,920 | 306,592 | 62,114 | 3.25x | 4.94x |
| electricity | 720->192 | 0.146843 | 0.144198 | 0.002645 | Asy2+Echo | 214,304 | 331,360 | 66,850 | 3.21x | 4.96x |
| electricity | 720->336 | 0.162817 | 0.159256 | 0.003561 | Asy2+Echo | 232,880 | 368,512 | 66,850 | 3.48x | 5.51x |
| electricity | 720->720 | 0.198580 | 0.193471 | 0.005109 | Asy2+Echo | 282,416 | 467,584 | 67,042 | 4.21x | 6.97x |
| weather | 720->96 | 0.147196 | 0.145168 | 0.002027 | Asy2+Echo | 140,186 | 244,858 | 13,454 | 10.42x | 18.20x |
| weather | 720->192 | 0.191169 | 0.188870 | 0.002299 | Asy2+Echo | 152,570 | 269,626 | 14,254 | 10.70x | 18.92x |
| weather | 720->336 | 0.244089 | 0.238102 | 0.005987 | Asy2+Echo | 171,146 | 306,778 | 14,270 | 11.99x | 21.50x |
| weather | 720->720 | 0.316879 | 0.307274 | 0.009605 | Asy2+Echo | 220,682 | 405,850 | 14,302 | 15.43x | 28.38x |

## Audit

Complete cells: 16/16.
