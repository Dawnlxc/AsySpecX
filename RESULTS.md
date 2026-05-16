# Results — promising models vs baselines

Three separate tables, one per `seq_len`. Within each table all numbers come from the same lookback setting → fair comparison.

Datasets: ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic, PEMS03, PEMS04, PEMS07, PEMS08.
Horizons: `{96, 192, 336, 720}` for non-PEMS, `{12, 24, 48, 96}` for PEMS.
Cell format: MSE (mean across seeds, `·Nsd` suffix if N>1 seeds). **Bold** = lowest in row. `—` = not run.

## Headline

| Line | Strongest claim | vs best baseline | Source |
| --- | --- | --- | --- |
| **AsySpecXResid** | PEMS_BAY mode B sensor outage (held-out 6/325): 0.472 | **−62 %** vs DLinear (1.247) | probe/FINAL_RESULTS_TABLE.md, 3 seeds |
| **AsySpecXResid** | ETTh2 pl=720 MTSF (sl=720): 0.419 | **−49 %** vs DLinear (0.819) | probe/FINAL_RESULTS_TABLE.md, 3 seeds |
| **JointMLP (JA v4)** | sl=336 standard MTSF: 8 wins / 32 settings | beats every baseline at sl=336 on those cells | this file |
| AsySpecX (lite) | sl=720 ETTh2 pl=192: 0.334 | best vs any sl=336 baseline (best 0.341 FITS) | this file, 2 seeds |

> AsySpecXResid is on a different protocol (3-seed sensor-outage + sl=720 long-horizon) — see [the dedicated table below](#asyspecxresid).

---

## Table @ sl=96

**Win count** (44 cells with ≥1 model):

| Model | Wins |
|---|---|
| TQNet | 13 |
| **JointMLP** | 12 |
| CycleNet | 9 |
| iTransformer | 5 |
| FreTS | 2 |
| SparseTSF | 1 |
| FilterNet | 1 |
| PatchTST | 1 |

### ETTh1

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.393·2sd | **0.371**·2sd | 0.372 | 0.379 | 0.388 | 0.385 | 0.398 | 0.387 | 0.388 | 0.387 | 0.404 | 0.402 |
| 192 | 0.443·2sd | 0.427·2sd | 0.430 | **0.427** | 0.438 | 0.435 | 0.447 | 0.443 | 0.440 | 0.436 | 0.447 | 0.433 |
| 336 | 0.483·2sd | 0.480·2sd | 0.476 | **0.465** | 0.477 | 0.481 | 0.500 | 0.520 | 0.490 | 0.480 | 0.516 | 0.475 |
| 720 | 0.475·2sd | 0.491·2sd | 0.491 | **0.462** | 0.476 | 0.464 | 0.575 | 0.635 | 0.514 | 0.489 | 0.543 | 0.476 |

### ETTh2

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.315·2sd | 0.294·2sd | 0.293 | **0.287** | 0.296 | 0.303 | 0.314 | 0.301 | 0.304 | 0.293 | 0.312 | 0.303 |
| 192 | 0.388·2sd | 0.371·2sd | **0.366** | 0.373 | 0.382 | 0.386 | 0.456 | 0.377 | 0.378 | 0.367 | 0.406 | 0.385 |
| 336 | 0.428·2sd | **0.417**·2sd | 0.419 | 0.425 | 0.420 | 0.421 | 0.509 | 0.423 | 0.424 | 0.420 | 0.490 | 0.426 |
| 720 | 0.428·2sd | 0.425·2sd | 0.440 | 0.452 | 0.423 | **0.420** | 0.728 | 0.449 | 0.435 | 0.423 | 0.694 | 0.428 |

### ETTm1

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.360·2sd | 0.314·2sd | **0.310** | 0.326 | 0.373 | 0.388 | 0.339 | 0.332 | 0.346 | 0.322 | 0.346 | 0.354 |
| 192 | 0.392·2sd | **0.357**·2sd | 0.363 | 0.367 | 0.404 | 0.415 | 0.387 | 0.368 | 0.386 | 0.362 | 0.383 | 0.399 |
| 336 | 0.420·2sd | 0.389·2sd | 0.391 | 0.396 | 0.434 | 0.441 | 0.424 | 0.412 | 0.423 | **0.388** | 0.419 | 0.437 |
| 720 | 0.485·2sd | **0.448**·2sd | 0.452 | 0.456 | 0.493 | 0.496 | 0.481 | 0.464 | 0.498 | 0.454 | 0.505 | 0.498 |

### ETTm2

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.182·2sd | 0.178·2sd | 0.173 | **0.168** | 0.189 | 0.198 | 0.182 | 0.178 | 0.183 | 0.177 | 0.193 | 0.184 |
| 192 | 0.245·2sd | 0.240·2sd | 0.241 | **0.233** | 0.251 | 0.260 | 0.257 | 0.241 | 0.252 | 0.248 | 0.253 | 0.248 |
| 336 | 0.305·2sd | 0.301·2sd | 0.299 | **0.294** | 0.310 | 0.317 | 0.360 | 0.297 | 0.312 | 0.305 | 0.314 | 0.309 |
| 720 | 0.406·2sd | 0.400·2sd | 0.399 | **0.396** | 0.410 | 0.414 | 0.506 | 0.396 | 0.410 | 0.404 | 0.416 | 0.410 |

### weather

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.192·2sd | 0.160·2sd | **0.158** | 0.171 | 0.209 | 0.207 | 0.173 | 0.165 | 0.174 | 0.173 | 0.202 | 0.196 |
| 192 | 0.242·2sd | 0.209·2sd | **0.206** | 0.222 | 0.252 | 0.253 | 0.213 | 0.211 | 0.226 | 0.219 | 0.239 | 0.243 |
| 336 | 0.295·2sd | 0.264·2sd | 0.264 | 0.276 | 0.302 | 0.302 | **0.263** | 0.272 | 0.281 | 0.275 | 0.303 | 0.293 |
| 720 | 0.367·2sd | 0.346·2sd | 0.343 | 0.350 | 0.372 | 0.372 | **0.339** | 0.356 | 0.360 | 0.352 | 0.358 | 0.367 |

### electricity

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.202·2sd | 0.152·2sd | **0.138** | 0.141 | 0.206 | 0.210 | 0.185 | 0.144 | 0.148 | 0.166 | 0.195 | 0.210 |
| 192 | 0.203·2sd | 0.169·2sd | 0.157 | **0.155** | 0.205 | 0.205 | 0.189 | 0.162 | 0.163 | 0.175 | 0.194 | 0.205 |
| 336 | 0.218·2sd | 0.187·2sd | **0.172** | 0.172 | 0.220 | 0.219 | 0.205 | 0.185 | 0.177 | 0.190 | 0.207 | 0.218 |
| 720 | 0.259·2sd | 0.217·2sd | 0.211 | 0.211 | 0.261 | 0.260 | 0.255 | 0.260 | **0.209** | 0.233 | 0.242 | 0.260 |

### traffic

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.625·2sd | 0.475·2sd | 0.422 | 0.481 | 0.658 | 0.663 | 0.546 | 0.417 | **0.387** | 0.474 | 0.662 | 0.663 |
| 192 | 0.595·2sd | 0.498·2sd | 0.445 | 0.483 | 0.608 | 0.611 | 0.548 | 0.439 | **0.409** | 0.478 | 0.611 | 0.612 |
| 336 | 0.607·2sd | 0.520·2sd | 0.462 | 0.477 | 0.616 | 0.617 | 0.573 | 0.458 | **0.413** | 0.491 | 0.615 | 0.617 |
| 720 | 0.641·2sd | 0.557·2sd | 0.498 | 0.503 | 0.661 | 0.655 | 0.625 | 0.495 | **0.438** | 0.525 | 0.657 | 0.654 |

### PEMS03

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | **0.058**·2sd | 0.060 | 0.081 | 0.120 | — | 0.081 | 0.068 | 0.067 | 0.087 | 0.103 | 0.185 |
| 24 | — | **0.074**·2sd | 0.076 | 0.125 | 0.239 | 0.322 | 0.122 | 0.095 | 0.093 | 0.139 | 0.180 | 0.324 |
| 48 | — | **0.106**·2sd | 0.108 | 0.208 | 0.547 | 0.444 | 0.198 | 0.150 | 0.149 | 0.253 | 0.318 | 0.650 |
| 96 | — | 0.153·2sd | **0.142** | 0.299 | 1.068 | 0.524 | 0.265 | 0.231 | 0.227 | 0.433 | 0.454 | 1.187 |

### PEMS04

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | 0.069 | **0.068** | 0.093 | 0.134 | — | 0.096 | 0.078 | 0.085 | 0.101 | 0.114 | 0.197 |
| 24 | — | 0.080·2sd | **0.079** | 0.130 | 0.255 | 0.324 | 0.144 | 0.097 | 0.116 | 0.161 | 0.188 | 0.336 |
| 48 | — | **0.097**·2sd | 0.099 | 0.196 | 0.576 | 0.445 | 0.224 | 0.136 | 0.177 | 0.296 | 0.318 | 0.675 |
| 96 | — | **0.117**·2sd | 0.123 | 0.246 | 1.173 | 0.507 | 0.289 | 0.201 | 0.273 | 0.513 | 0.424 | 1.267 |

### PEMS07

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | 0.053·2sd | **0.052** | 0.077 | 0.113 | — | 0.076 | 0.062 | 0.065 | 0.082 | 0.100 | 0.179 |
| 24 | — | 0.065·2sd | **0.063** | 0.122 | 0.236 | 0.350 | 0.126 | 0.086 | 0.090 | 0.142 | 0.188 | 0.327 |
| 48 | — | 0.088·2sd | **0.082** | 0.217 | 0.557 | 0.531 | 0.226 | 0.124 | 0.136 | 0.273 | 0.373 | 0.675 |
| 96 | — | 0.134·2sd | **0.110** | 0.307 | 1.117 | 0.666 | 0.323 | 0.180 | 0.199 | 0.450 | 0.577 | 1.253 |

### PEMS08

| pl | AsySpecX | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | **0.073**·2sd | 0.086 | 0.095 | 0.127 | — | 0.095 | 0.078 | 0.078 | 0.091 | 0.113 | 0.189 |
| 24 | — | **0.101**·2sd | 0.137 | 0.146 | 0.245 | 0.346 | 0.149 | 0.111 | 0.113 | 0.145 | 0.197 | 0.323 |
| 48 | — | **0.164**·2sd | 0.251 | 0.260 | 0.576 | 0.546 | 0.245 | 0.171 | 0.181 | 0.254 | 0.392 | 0.677 |
| 96 | — | 0.315·2sd | 0.392 | 0.359 | 1.244 | 0.756 | 0.355 | **0.292** | 0.303 | 0.438 | 0.651 | 1.345 |

---

## Table @ sl=336

**Win count** (44 cells with ≥1 model):

| Model | Wins |
|---|---|
| iTransformer | 13 |
| CycleNet | 9 |
| TQNet | 7 |
| **JointMLP** | 6 |
| FITS | 3 |
| FilterNet | 3 |
| MixLinear | 3 |
| FreTS | 2 |
| PatchTST | 1 |

### ETTh1

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.369·2sd | 0.371 | 0.372 | 0.376 | 0.368 | 0.405 | 0.395 | 0.409 | 0.371 | 0.386 | **0.363** |
| 192 | **0.408**·2sd | 0.414 | 0.408 | 0.408 | 0.425 | 0.446 | 0.443 | 0.445 | 0.410 | 0.430 | 0.428 |
| 336 | 0.440·2sd | 0.437 | 0.430 | 0.431 | 0.429 | 0.483 | 0.454 | 0.465 | 0.445 | 0.463 | **0.422** |
| 720 | 0.465·2sd | 0.480 | 0.451 | 0.426 | 0.421 | 0.636 | 0.489 | 0.549 | 0.456 | 0.515 | **0.419** |

### ETTh2

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.292·2sd | 0.296 | **0.277** | 0.281 | 0.297 | 0.301 | 0.317 | 0.302 | 0.282 | 0.334 | 0.290 |
| 192 | 0.354·2sd | 0.354 | 0.341 | **0.341** | 0.347 | 0.431 | 0.371 | 0.369 | 0.352 | 0.380 | 0.350 |
| 336 | 0.395·2sd | **0.362** | 0.372 | **0.362** | 0.364 | 0.483 | 0.389 | 0.422 | 0.378 | 0.422 | 0.365 |
| 720 | 0.410·2sd | 0.420 | 0.425 | **0.389** | 0.401 | 0.920 | 0.424 | 0.425 | 0.401 | 0.591 | 0.391 |

### ETTm1

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | **0.285**·2sd | 0.294 | 0.300 | 0.312 | — | 0.334 | 0.312 | 0.311 | 0.292 | 0.307 | 0.321 |
| 192 | **0.326**·2sd | 0.335 | 0.339 | 0.345 | — | 0.358 | 0.342 | 0.355 | 0.333 | 0.342 | 0.353 |
| 336 | **0.363**·2sd | 0.370 | 0.364 | 0.378 | — | 0.388 | 0.392 | 0.385 | 0.367 | 0.378 | 0.388 |
| 720 | 0.423·2sd | 0.427 | **0.418** | 0.432 | — | 0.447 | 0.455 | 0.437 | 0.425 | 0.474 | 0.445 |

### ETTm2

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.172·2sd | 0.170 | **0.159** | 0.170 | — | 0.175 | 0.174 | 0.178 | 0.167 | 0.189 | 0.169 |
| 192 | 0.228·2sd | 0.225 | **0.216** | 0.224 | — | 0.245 | 0.232 | 0.254 | 0.221 | 0.266 | 0.224 |
| 336 | 0.283·2sd | 0.294 | **0.269** | 0.277 | — | 0.333 | 0.284 | 0.290 | 0.276 | 0.309 | 0.278 |
| 720 | 0.377·2sd | 0.368 | **0.365** | 0.370 | — | 0.399 | 0.379 | 0.382 | 0.365 | 0.441 | 0.370 |

### weather

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | **0.149**·2sd | 0.161 | 0.167 | 0.180 | — | 0.154 | 0.160 | 0.160 | 0.151 | 0.177 | 0.177 |
| 192 | 0.198·2sd | **0.196** | 0.213 | 0.222 | — | **0.196** | 0.198 | 0.206 | **0.196** | 0.218 | 0.219 |
| 336 | **0.249**·2sd | 0.250 | 0.261 | 0.268 | — | 0.253 | 0.258 | 0.262 | 0.249 | 0.264 | 0.268 |
| 720 | 0.322·2sd | 0.321 | 0.328 | 0.335 | — | **0.317** | 0.336 | 0.326 | 0.322 | 0.326 | 0.337 |

### electricity

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.146·2sd | 0.131 | **0.129** | 0.149 | 0.147 | 0.136 | 0.132 | 0.132 | — | 0.140 | 0.147 |
| 192 | 0.187·2sd | 0.154 | **0.144** | 0.163 | 0.158 | 0.154 | 0.152 | 0.153 | — | 0.154 | 0.164 |
| 336 | 0.190·2sd | 0.168 | **0.161** | 0.179 | 0.174 | — | 0.172 | 0.172 | — | 0.169 | 0.178 |
| 720 | 0.224·2sd | 0.200 | 0.199 | 0.216 | 0.212 | — | 0.226 | **0.198** | — | 0.205 | 0.216 |

### traffic

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.422·2sd | 0.418 | 0.396 | 0.418 | 0.415 | — | 0.363 | **0.361** | — | 0.428 | 0.415 |
| 192 | 0.446·2sd | 0.440 | 0.411 | 0.431 | 0.427 | — | 0.391 | **0.377** | — | 0.441 | 0.430 |
| 336 | 0.465·2sd | 0.460 | 0.428 | 0.443 | 0.438 | — | 0.404 | **0.390** | — | 0.458 | — |
| 720 | 0.479·2sd | 0.459 | 0.451 | 0.469 | 0.465 | — | **0.445** | — | — | 0.486 | — |

### PEMS03

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | 0.062 | 0.073 | — | — | 0.064 | 0.064 | **0.060** | — | 0.076 | — |
| 24 | — | 0.078 | 0.101 | — | — | — | 0.080 | **0.074** | — | 0.107 | 0.156 |
| 48 | — | 0.107 | 0.143 | — | — | — | 0.107 | **0.090** | — | 0.163 | 0.186 |
| 96 | — | 0.243 | 0.178 | — | — | — | 0.129 | **0.113** | — | 0.189 | 0.211 |

### PEMS04

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | 0.069 | 0.090 | 0.091 | — | 0.076 | **0.069** | 0.070 | — | 0.089 | — |
| 24 | — | **0.073** | 0.116 | 0.123 | — | 0.096 | 0.080 | 0.082 | — | 0.123 | 0.172 |
| 48 | — | **0.090** | 0.155 | — | — | 0.128 | 0.095 | 0.102 | — | 0.163 | 0.202 |
| 96 | — | **0.114** | 0.190 | — | — | 0.160 | 0.117 | 0.120 | — | 0.209 | 0.228 |

### PEMS07

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | **0.050** | 0.069 | — | — | — | 0.050 | 0.050 | — | 0.072 | — |
| 24 | — | **0.056** | 0.098 | — | — | — | 0.059 | 0.058 | — | 0.103 | 0.151 |
| 48 | — | 0.067 | 0.144 | — | — | — | 0.069 | **0.067** | — | 0.157 | — |
| 96 | — | 0.095 | 0.186 | — | — | — | 0.078 | **0.076** | — | 0.194 | — |

### PEMS08

| pl | JointMLP | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | 0.107 | 0.093 | 0.097 | — | 0.077 | 0.075 | **0.066** | 0.072 | 0.092 | — |
| 24 | — | 0.158 | 0.134 | 0.150 | — | 0.106 | 0.092 | **0.084** | 0.087 | 0.141 | 0.229 |
| 48 | — | 0.342 | 0.217 | 0.257 | — | 0.160 | 0.128 | **0.117** | — | 0.234 | 0.330 |
| 96 | — | 0.332 | 0.312 | 0.385 | — | 0.232 | **0.153** | 0.178 | — | 0.332 | 0.431 |

---

## Table @ sl=720

**Win count** (28 cells with ≥1 model):

| Model | Wins |
|---|---|
| **AsySpecX** | 19 |
| **JointMLP** | 9 |

### ETTh1

| pl | AsySpecX | JointMLP |
|---|---|---|
| 96 | 0.384·2sd | **0.376**·2sd |
| 192 | 0.421·2sd | **0.412**·2sd |
| 336 | **0.451**·2sd | 0.458·2sd |
| 720 | **0.462**·2sd | 0.502·2sd |

### ETTh2

| pl | AsySpecX | JointMLP |
|---|---|---|
| 96 | **0.277**·2sd | 0.290·2sd |
| 192 | **0.334**·2sd | 0.350·2sd |
| 336 | **0.365**·2sd | 0.387·2sd |
| 720 | **0.389**·2sd | 0.418·2sd |

### ETTm1

| pl | AsySpecX | JointMLP |
|---|---|---|
| 96 | 0.321·2sd | **0.296**·2sd |
| 192 | 0.351·2sd | **0.340**·2sd |
| 336 | 0.380·2sd | **0.370**·2sd |
| 720 | **0.430**·2sd | 0.437·2sd |

### ETTm2

| pl | AsySpecX | JointMLP |
|---|---|---|
| 96 | 0.171·2sd | **0.169**·2sd |
| 192 | **0.234**·2sd | 0.238·2sd |
| 336 | 0.292·2sd | **0.284**·2sd |
| 720 | **0.373**·2sd | 0.374 |

### weather

| pl | AsySpecX | JointMLP |
|---|---|---|
| 96 | 0.156·2sd | **0.152**·2sd |
| 192 | **0.202**·2sd | 0.202·2sd |
| 336 | 0.252·2sd | **0.250**·2sd |
| 720 | **0.317**·2sd | 0.318·2sd |

### electricity

| pl | AsySpecX | JointMLP |
|---|---|---|
| 96 | **0.141**·2sd | 0.143·2sd |
| 192 | **0.155**·2sd | 0.159·2sd |
| 336 | **0.172**·2sd | 0.174·2sd |
| 720 | **0.201**·2sd | 0.203·2sd |

### traffic

| pl | AsySpecX | JointMLP |
|---|---|---|
| 96 | **0.390**·2sd | 0.403·2sd |
| 192 | **0.401**·2sd | 0.424·2sd |
| 336 | **0.412**·2sd | 0.442·2sd |
| 720 | **0.452**·2sd | 0.468·2sd |

---

## AsySpecXResid

Not in the sl=96/336/720 tables above — was benchmarked on a different protocol (3-seed sensor outage + sl=720 long-horizon MTSF). Headline numbers (from upstream `probe/FINAL_RESULTS_TABLE.md`):

| Setting | DLinear | iTransformer | FreqHerm | **AsySpecXResid** | Δ vs DLinear |
|---|---|---|---|---|---|
| ETTh2 pl=720 std MTSF | 0.819 | — | 0.423 | **0.419** ± 0.0002 | **−49 %** |
| PEMS_BAY mode B pl=12 (held-out 6/325 sensors) | 1.247 | 1.162 | 0.495 | **0.472** ± 0.007 | **−62 %** |
| METR_LA mode B pl=12 (held-out 6/207 sensors)   | 2.092 | 1.539 | 1.731 | **1.722** | −18 % |
| Beijing_AQ mode B (1 station × 11 modalities)   | 1.107 | 1.103 | 1.027 | 1.028 | −7 % |

Pattern: gain is largest where physical cross-channel structure (road graph) is strongest.

---

## How this was built

- **Baselines**: `note/sl96_results.csv` + `note/sl336_results.csv` from upstream `/scratch3/lin250/bldgFM/AsySpecX/` (seed 2026).
- **AsySpecX, JointMLP**: parsed canonical-tag runs from `logs/<Model>/*.log`. JointMLP requires `jmlp_v4` tag; AsySpecX requires no tag. Within each `(model, sl)` group, MSE is the mean across seeds (2026, 2027).
- **AsySpecXResid**: headline numbers from upstream `probe/FINAL_RESULTS_TABLE.md` (3-seed, sensor-outage benchmark — separate protocol).
- Each `seq_len` is reported as its own table so different lookback settings are never mixed in the same row.
