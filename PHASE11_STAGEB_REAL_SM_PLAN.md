# AsySpecX Phase 11 Stage B: Real Spectral-Mixture Kernel

## Frozen hypothesis

Stage A established that a channel-separable asymmetric past-to-future kernel
is useful. Stage B tests one additional time-series inductive bias only:
whether each low-rank past-analysis mode benefits from a smooth real spectral
envelope. Cross-channel mixing and complex phase are forbidden in this stage.

For rank mode `r`,

\[
V_r^{SM}=V_r+\operatorname{irfft}\left[(A_r(\omega)-1)
\operatorname{rfft}(V_r)\right],
\]

where `A_r` is a positive Gaussian-mixture envelope with geometric mean one.
Its signed log-gain is bounded by `tanh`; the gain parameter starts at exactly
zero. Therefore the Stage-B model initially implements the exact Stage-A
function, changes spectral amplitude only, and never changes Fourier phase.
The envelope is shared across channels.

## Experiment matrix

Primary cell remains Weather, `seq_len=96`, `pred_len=720`, `cut_freq=13`.
All other Stage-A winner settings are frozen.

| Arm | Mixtures | Envelope sharing | Base `V` | Purpose |
| --- | ---: | --- | --- | --- |
| `fk_r8_cs` | 0 | none | trainable | same-code Stage-A control |
| `fk_sm2_shared` | 2 | all modes | trainable | smallest real-SM hypothesis |
| `fk_sm2_mode` | 2 | per mode | trainable | mode-selective envelope |
| `fk_sm4_mode` | 4 | per mode | trainable | higher-resolution envelope |
| `fk_sm4_frozen` | 4 | per mode | frozen DCT init | compression/mechanism control |

The trainable-base arms test a smooth multiplicative spectral inductive bias;
the frozen arm tests whether the mixture can replace the free past basis rather
than merely reparameterize it.

## Execution and leakage gates

1. Unit tests: exact identity at zero gain, positive-real response, unchanged
   phase, channel isolation, gradients, state compatibility, and configuration.
2. Two-epoch H100 canary for the Stage-A control and all Stage-B families.
3. Seed-2026 screen with test evaluation disabled; rank by validation MSE only.
4. A Stage-B arm is promoted only if it strictly beats the same-code Stage-A
   validation MSE, stays below the dense-direct parameter count, has finite
   non-collapsed diagnostics, and fixed-work train/inference ratios are <=1.10.
5. Evaluate test only after the validation decision. Confirm the winner on
   seeds 2024/2025/2026 with the same protocol.

Stage B is a stable win only if at least two of three seed-matched test MSEs
beat `fk_r8_cs` and mean validation MSE also improves. Otherwise it is a tie,
resource-only result, or failure.

## Complex-phase lock

No complex parameter is implemented or run during the real-SM experiment.
A bounded within-variable phase ablation is unlocked only after a real-SM arm
passes the three-seed stable-win rule. Cross-variable lead--lag remains pruned
regardless of the outcome.
