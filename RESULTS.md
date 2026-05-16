# Results — full comparison (all models, all datasets)

Per-(dataset, horizon) MSE, picking each model's best `seq_len` from available runs.
- **Bold** = lowest MSE in that row.
- Cell suffix `·slX` = best seq_len was X (default sl=96 omitted).
- Cell suffix `·Nsd` = mean over N seeds.
- `—` = no runs available in `/scratch3/lin250/bldgFM/AsySpecX/logs/`.

See `/scratch3/lin250/bldgFM/AsySpecX/probe/FINAL_RESULTS_TABLE.md` for AsySpecXResid sensor-outage and ETTh2-pl720 headline numbers (3-seed; separate sweep not in standard logs dir).

## Win count

| Model | Wins | / total |
|---|---|---|
| iTransformer | 12 | / 44 |
| JointMLP | 7 | / 44 |
| TQNet | 7 | / 44 |
| CycleNet | 7 | / 44 |
| FreqHerm | 4 | / 44 |
| MixLinear | 3 | / 44 |
| FreqHermCycle | 2 | / 44 |
| AsySpecX | 1 | / 44 |
| FITS | 1 | / 44 |
| FilterNet | 1 | / 44 |

---

## Per-active-model wins (where the active model beats every baseline)

### AsySpecX — 2 wins / 28 settings

| Dataset | pl | Best sl | MSE | vs best baseline | Δ |
|---|---|---|---|---|---|
| ETTh2 | 192 | 720 | **0.334** | FITS 0.341 | -2.1% |
| ETTh2 | 96 | 720 | **0.277** | CycleNet 0.277 | -0.2% |

### AsySpecXResid — _no logs in standard dir_

### JointMLP — 8 wins / 44 settings

| Dataset | pl | Best sl | MSE | vs best baseline | Δ |
|---|---|---|---|---|---|
| PEMS03 | 12 | 96 | **0.058** | iTransformer 0.060 | -3.0% |
| ETTm1 | 96 | 336 | **0.285** | PatchTST 0.292 | -2.2% |
| ETTm1 | 192 | 336 | **0.326** | PatchTST 0.333 | -1.9% |
| weather | 96 | 336 | **0.149** | PatchTST 0.151 | -1.4% |
| PEMS03 | 24 | 96 | **0.074** | iTransformer 0.074 | -0.5% |
| ETTm1 | 336 | 336 | **0.363** | CycleNet 0.364 | -0.1% |
| ETTh1 | 192 | 336 | **0.408** | FITS 0.408 | -0.1% |
| weather | 336 | 336 | **0.249** | PatchTST 0.249 | -0.1% |

### FreqHerm — 7 wins / 44 settings

| Dataset | pl | Best sl | MSE | vs best baseline | Δ |
|---|---|---|---|---|---|
| weather | 720 | 720 | **0.304** | FreTS 0.317 | -4.0% |
| weather | 336 | 720 | **0.240** | PatchTST 0.249 | -3.4% |
| weather | 192 | 720 | **0.192** | PatchTST 0.196 | -2.0% |
| ETTh2 | 96 | 720 | **0.272** | CycleNet 0.277 | -1.9% |
| ETTh2 | 192 | 720 | **0.335** | FITS 0.341 | -1.7% |
| ETTh2 | 720 | 720 | **0.385** | FITS 0.389 | -1.0% |
| weather | 96 | 720 | **0.150** | PatchTST 0.151 | -0.9% |

### FreqHermCycle — 7 wins / 24 settings

| Dataset | pl | Best sl | MSE | vs best baseline | Δ |
|---|---|---|---|---|---|
| weather | 720 | 720 | **0.305** | FreTS 0.317 | -3.6% |
| weather | 336 | 720 | **0.243** | PatchTST 0.249 | -2.5% |
| ETTh2 | 192 | 720 | **0.334** | FITS 0.341 | -1.9% |
| ETTh2 | 720 | 720 | **0.382** | FITS 0.389 | -1.8% |
| weather | 192 | 720 | **0.194** | PatchTST 0.196 | -1.0% |
| ETTh2 | 96 | 720 | **0.275** | CycleNet 0.277 | -0.6% |
| ETTm2 | 720 | 720 | **0.364** | CycleNet 0.365 | -0.2% |

### FreqHermCycleAttn — _no logs in standard dir_

---

## Full per-dataset table

### ETTh1

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.384·sl720·2sd | — | 0.369·sl336·2sd | 0.378·sl336·2sd | — | — | 0.371·sl336 | 0.372·sl336 | 0.376·sl336 | 0.368·sl336 | 0.398 | 0.387 | 0.388 | 0.371·sl336 | 0.386·sl336 | **0.363**·sl336 |
| 192 | 0.421·sl720·2sd | — | **0.408**·sl336·2sd | 0.415·sl336·2sd | — | — | 0.414·sl336 | 0.408·sl336 | 0.408·sl336 | 0.425·sl336 | 0.446·sl336 | 0.443 | 0.440 | 0.410·sl336 | 0.430·sl336 | 0.428·sl336 |
| 336 | 0.451·sl720·2sd | — | 0.440·sl336·2sd | 0.438·sl336·2sd | — | — | 0.437·sl336 | 0.430·sl336 | 0.431·sl336 | 0.429·sl336 | 0.483·sl336 | 0.454·sl336 | 0.465·sl336 | 0.445·sl336 | 0.463·sl336 | **0.422**·sl336 |
| 720 | 0.462·sl720·2sd | — | 0.465·sl336·2sd | 0.457·sl336·2sd | — | — | 0.480·sl336 | 0.451·sl336 | 0.426·sl336 | 0.421·sl336 | 0.575 | 0.489·sl336 | 0.514 | 0.456·sl336 | 0.515·sl336 | **0.419**·sl336 |

### ETTh2

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.277·sl720·2sd | — | 0.290·sl720·2sd | **0.272**·sl720·2sd | 0.275·sl720·2sd | — | 0.293 | 0.277·sl336 | 0.281·sl336 | 0.297·sl336 | 0.301·sl336 | 0.301 | 0.302·sl336 | 0.282·sl336 | 0.312 | 0.290·sl336 |
| 192 | **0.334**·sl720·2sd | — | 0.350·sl720·2sd | 0.335·sl720·2sd | 0.334·sl720·2sd | — | 0.354·sl336 | 0.341·sl336 | 0.341·sl336 | 0.347·sl336 | 0.431·sl336 | 0.371·sl336 | 0.369·sl336 | 0.352·sl336 | 0.380·sl336 | 0.350·sl336 |
| 336 | 0.365·sl720·2sd | — | 0.387·sl720·2sd | 0.364·sl720·2sd | 0.364·sl720·2sd | — | **0.362**·sl336 | 0.372·sl336 | **0.362**·sl336 | 0.364·sl336 | 0.483·sl336 | 0.389·sl336 | 0.422·sl336 | 0.378·sl336 | 0.422·sl336 | 0.365·sl336 |
| 720 | 0.389·sl720·2sd | — | 0.410·sl336·2sd | 0.385·sl720·2sd | **0.382**·sl720·2sd | — | 0.420·sl336 | 0.425·sl336 | 0.389·sl336 | 0.401·sl336 | 0.728 | 0.424·sl336 | 0.425·sl336 | 0.401·sl336 | 0.591·sl336 | 0.391·sl336 |

### ETTm1

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.321·sl720·2sd | — | **0.285**·sl336·2sd | 0.306·sl336·2sd | 0.307·sl336·2sd | — | 0.294·sl336 | 0.300·sl336 | 0.312·sl336 | 0.388 | 0.334·sl336 | 0.312·sl336 | 0.311·sl336 | 0.292·sl336 | 0.307·sl336 | 0.321·sl336 |
| 192 | 0.351·sl720·2sd | — | **0.326**·sl336·2sd | 0.348·sl336·2sd | 0.344·sl336·2sd | — | 0.335·sl336 | 0.339·sl336 | 0.345·sl336 | 0.415 | 0.358·sl336 | 0.342·sl336 | 0.355·sl336 | 0.333·sl336 | 0.342·sl336 | 0.353·sl336 |
| 336 | 0.380·sl720·2sd | — | **0.363**·sl336·2sd | 0.382·sl720·2sd | 0.381·sl720·2sd | — | 0.370·sl336 | 0.364·sl336 | 0.378·sl336 | 0.441 | 0.388·sl336 | 0.392·sl336 | 0.385·sl336 | 0.367·sl336 | 0.378·sl336 | 0.388·sl336 |
| 720 | 0.430·sl720·2sd | — | 0.423·sl336·2sd | 0.430·sl720·2sd | 0.428·sl720·2sd | — | 0.427·sl336 | **0.418**·sl336 | 0.432·sl336 | 0.496 | 0.447·sl336 | 0.455·sl336 | 0.437·sl336 | 0.425·sl336 | 0.474·sl336 | 0.445·sl336 |

### ETTm2

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.171·sl720·2sd | — | 0.169·sl720·2sd | 0.170·sl720·2sd | 0.168·sl720·2sd | — | 0.170·sl336 | **0.159**·sl336 | 0.170·sl336 | 0.198 | 0.175·sl336 | 0.174·sl336 | 0.178·sl336 | 0.167·sl336 | 0.189·sl336 | 0.169·sl336 |
| 192 | 0.234·sl720·2sd | — | 0.228·sl336·2sd | 0.232·sl336·2sd | 0.229·sl336·2sd | — | 0.225·sl336 | **0.216**·sl336 | 0.224·sl336 | 0.260 | 0.245·sl336 | 0.232·sl336 | 0.252 | 0.221·sl336 | 0.253 | 0.224·sl336 |
| 336 | 0.292·sl720·2sd | — | 0.283·sl336·2sd | 0.287·sl720·2sd | 0.287·sl336·2sd | — | 0.294·sl336 | **0.269**·sl336 | 0.277·sl336 | 0.317 | 0.333·sl336 | 0.284·sl336 | 0.290·sl336 | 0.276·sl336 | 0.309·sl336 | 0.278·sl336 |
| 720 | 0.373·sl720·2sd | — | 0.374·sl720 | 0.372·sl720·2sd | **0.364**·sl720·2sd | — | 0.368·sl336 | 0.365·sl336 | 0.370·sl336 | 0.414 | 0.399·sl336 | 0.379·sl336 | 0.382·sl336 | 0.365·sl336 | 0.416 | 0.370·sl336 |

### weather

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.156·sl720·2sd | — | **0.149**·sl336·2sd | 0.150·sl720·2sd | 0.153·sl720·2sd | — | 0.158 | 0.167·sl336 | 0.180·sl336 | 0.207 | 0.154·sl336 | 0.160·sl336 | 0.160·sl336 | 0.151·sl336 | 0.177·sl336 | 0.177·sl336 |
| 192 | 0.202·sl720·2sd | — | 0.198·sl336·2sd | **0.192**·sl720·2sd | 0.194·sl720·2sd | — | 0.196·sl336 | 0.213·sl336 | 0.222·sl336 | 0.253 | 0.196·sl336 | 0.198·sl336 | 0.206·sl336 | 0.196·sl336 | 0.218·sl336 | 0.219·sl336 |
| 336 | 0.252·sl720·2sd | — | 0.249·sl336·2sd | **0.240**·sl720·2sd | 0.243·sl720·2sd | — | 0.250·sl336 | 0.261·sl336 | 0.268·sl336 | 0.302 | 0.253·sl336 | 0.258·sl336 | 0.262·sl336 | 0.249·sl336 | 0.264·sl336 | 0.268·sl336 |
| 720 | 0.317·sl720·2sd | — | 0.318·sl720·2sd | **0.304**·sl720·2sd | 0.305·sl720·2sd | — | 0.321·sl336 | 0.328·sl336 | 0.335·sl336 | 0.372 | 0.317·sl336 | 0.336·sl336 | 0.326·sl336 | 0.322·sl336 | 0.326·sl336 | 0.337·sl336 |

### electricity

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.141·sl720·2sd | — | 0.143·sl720·2sd | 0.142·sl720·2sd | 0.137·sl720·2sd | — | 0.131·sl336 | **0.129**·sl336 | 0.149·sl336 | 0.147·sl336 | 0.136·sl336 | 0.132·sl336 | 0.132·sl336 | 0.166 | 0.140·sl336 | 0.147·sl336 |
| 192 | 0.155·sl720·2sd | — | 0.159·sl720·2sd | 0.156·sl720·2sd | 0.153·sl720·2sd | — | 0.154·sl336 | **0.144**·sl336 | 0.163·sl336 | 0.158·sl336 | 0.154·sl336 | 0.152·sl336 | 0.153·sl336 | 0.175 | 0.154·sl336 | 0.164·sl336 |
| 336 | 0.172·sl720·2sd | — | 0.174·sl720·2sd | 0.172·sl720·2sd | 0.168·sl720·2sd | — | 0.168·sl336 | **0.161**·sl336 | 0.179·sl336 | 0.174·sl336 | 0.205 | 0.172·sl336 | 0.172·sl336 | 0.190 | 0.169·sl336 | 0.178·sl336 |
| 720 | 0.201·sl720·2sd | — | 0.203·sl720·2sd | 0.209·sl720·2sd | 0.205·sl720·2sd | — | 0.200·sl336 | 0.199·sl336 | 0.216·sl336 | 0.212·sl336 | 0.255 | 0.226·sl336 | **0.198**·sl336 | 0.233 | 0.205·sl336 | 0.216·sl336 |

### traffic

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.390·sl720·2sd | — | 0.403·sl720·2sd | 0.391·sl720·2sd | — | — | 0.418·sl336 | 0.396·sl336 | 0.418·sl336 | 0.415·sl336 | 0.546 | 0.363·sl336 | **0.361**·sl336 | 0.474 | 0.428·sl336 | 0.415·sl336 |
| 192 | 0.401·sl720·2sd | — | 0.424·sl720·2sd | 0.404·sl720·2sd | — | — | 0.440·sl336 | 0.411·sl336 | 0.431·sl336 | 0.427·sl336 | 0.548 | 0.391·sl336 | **0.377**·sl336 | 0.478 | 0.441·sl336 | 0.430·sl336 |
| 336 | 0.412·sl720·2sd | — | 0.442·sl720·2sd | 0.417·sl720·2sd | — | — | 0.460·sl336 | 0.428·sl336 | 0.443·sl336 | 0.438·sl336 | 0.573 | 0.404·sl336 | **0.390**·sl336 | 0.491 | 0.458·sl336 | 0.617 |
| 720 | 0.452·sl720·2sd | — | 0.468·sl720·2sd | 0.452·sl720·2sd | — | — | 0.459·sl336 | 0.451·sl336 | 0.469·sl336 | 0.465·sl336 | 0.625 | 0.445·sl336 | **0.438** | 0.525 | 0.486·sl336 | 0.654 |

### PEMS03

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | — | **0.058**·2sd | 0.092·2sd | — | — | 0.060 | 0.073·sl336 | 0.120 | — | 0.064·sl336 | 0.064·sl336 | 0.060·sl336 | 0.087 | 0.076·sl336 | 0.185 |
| 24 | — | — | **0.074**·2sd | 0.174·2sd | — | — | 0.076 | 0.101·sl336 | 0.239 | 0.322 | 0.122 | 0.080·sl336 | 0.074·sl336 | 0.139 | 0.107·sl336 | 0.156·sl336 |
| 48 | — | — | 0.106·2sd | 0.416·2sd | — | — | 0.107·sl336 | 0.143·sl336 | 0.547 | 0.444 | 0.198 | 0.107·sl336 | **0.090**·sl336 | 0.253 | 0.163·sl336 | 0.186·sl336 |
| 96 | — | — | 0.153·2sd | 0.842·2sd | — | — | 0.142 | 0.178·sl336 | 1.068 | 0.524 | 0.265 | 0.129·sl336 | **0.113**·sl336 | 0.433 | 0.189·sl336 | 0.211·sl336 |

### PEMS04

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | — | 0.069 | 0.104·2sd | 0.103·2sd | — | **0.068** | 0.090·sl336 | 0.091·sl336 | — | 0.076·sl336 | 0.069·sl336 | 0.070·sl336 | 0.101 | 0.089·sl336 | 0.197 |
| 24 | — | — | 0.080·2sd | 0.177·2sd | 0.176·2sd | — | **0.073**·sl336 | 0.116·sl336 | 0.123·sl336 | 0.324 | 0.096·sl336 | 0.080·sl336 | 0.082·sl336 | 0.161 | 0.123·sl336 | 0.172·sl336 |
| 48 | — | — | 0.097·2sd | 0.385·2sd | 0.368·2sd | — | **0.090**·sl336 | 0.155·sl336 | 0.576 | 0.445 | 0.128·sl336 | 0.095·sl336 | 0.102·sl336 | 0.296 | 0.163·sl336 | 0.202·sl336 |
| 96 | — | — | 0.117·2sd | 0.813·2sd | 0.795·2sd | — | **0.114**·sl336 | 0.190·sl336 | 1.173 | 0.507 | 0.160·sl336 | 0.117·sl336 | 0.120·sl336 | 0.513 | 0.209·sl336 | 0.228·sl336 |

### PEMS07

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | — | 0.053·2sd | 0.084·2sd | — | — | **0.050**·sl336 | 0.069·sl336 | 0.113 | — | 0.076 | 0.050·sl336 | 0.050·sl336 | 0.082 | 0.072·sl336 | 0.179 |
| 24 | — | — | 0.065·2sd | 0.162·2sd | — | — | **0.056**·sl336 | 0.098·sl336 | 0.236 | 0.350 | 0.126 | 0.059·sl336 | 0.058·sl336 | 0.142 | 0.103·sl336 | 0.151·sl336 |
| 48 | — | — | 0.088·2sd | 0.399·2sd | — | — | 0.067·sl336 | 0.144·sl336 | 0.557 | 0.531 | 0.226 | 0.069·sl336 | **0.067**·sl336 | 0.273 | 0.157·sl336 | 0.675 |
| 96 | — | — | 0.134·2sd | 0.849·2sd | — | — | 0.095·sl336 | 0.186·sl336 | 1.117 | 0.666 | 0.323 | 0.078·sl336 | **0.076**·sl336 | 0.450 | 0.194·sl336 | 1.253 |

### PEMS08

| pl | AsySpecX | AsySpecXResid | JointMLP | FreqHerm | FreqHermCycle | FreqHermCycleAttn | TQNet | CycleNet | FITS | SparseTSF | FreTS | FilterNet | iTransformer | PatchTST | DLinear | MixLinear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | — | — | 0.073·2sd | 0.098·2sd | — | — | 0.086 | 0.093·sl336 | 0.097·sl336 | — | 0.077·sl336 | 0.075·sl336 | **0.066**·sl336 | 0.072·sl336 | 0.092·sl336 | 0.189 |
| 24 | — | — | 0.101·2sd | 0.174·2sd | — | — | 0.137 | 0.134·sl336 | 0.150·sl336 | 0.346 | 0.106·sl336 | 0.092·sl336 | **0.084**·sl336 | 0.087·sl336 | 0.141·sl336 | 0.229·sl336 |
| 48 | — | — | 0.164·2sd | 0.405·2sd | — | — | 0.251 | 0.217·sl336 | 0.257·sl336 | 0.546 | 0.160·sl336 | 0.128·sl336 | **0.117**·sl336 | 0.254 | 0.234·sl336 | 0.330·sl336 |
| 96 | — | — | 0.315·2sd | 0.954·2sd | — | — | 0.332·sl336 | 0.312·sl336 | 0.385·sl336 | 0.756 | 0.232·sl336 | **0.153**·sl336 | 0.178·sl336 | 0.438 | 0.332·sl336 | 0.431·sl336 |

---

## How this was built

Sources:
- **Baselines** (`TQNet … MixLinear`): `note/sl96_results.csv` + `note/sl336_results.csv` from upstream `/scratch3/lin250/bldgFM/AsySpecX/`.
- **Active line** (`AsySpecX, JointMLP, FreqHerm, FreqHermCycle`): parsed from `logs/<Model>/*.log`. Only canonical-tag runs were kept (`JointMLP` requires `jmlp_v4` tag; `FreqHerm*` and `AsySpecXResid` require no tag).
- **AsySpecXResid / FreqHermCycleAttn**: no canonical-tag logs in `logs/<Model>/`. Their results live in upstream `probe/FINAL_RESULTS_TABLE.md` (3-seed sensor-outage + ETTh2-pl720 headline).
- For each (dataset, horizon, model), the script picks the **best seq_len** across available runs (mean across seeds).

To regenerate this file: `bash scripts/slurm/submit_all.sh --all-baselines --full` + active-line sweeps, then re-run the aggregator (`/tmp/build_results_md.py` in this session).
