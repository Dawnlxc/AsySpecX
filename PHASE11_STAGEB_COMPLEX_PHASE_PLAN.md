# Phase 11 Stage B.2: Bounded Within-Variable Complex Phase

## Unlock evidence

The real spectral-mixture stage passed its predeclared stable-win rule. Both
`fk_sm4_mode` and `fk_sm2_mode` beat the real Stage-A kernel on test MSE in
3/3 seeds and improve mean validation MSE. The complex ablation is anchored to
`fk_sm4_mode` because it has the lower replicated mean validation MSE
(`0.6852741` versus `0.6852830`); test metrics are not used to choose the
anchor.

## Frozen complex hypothesis

For each channel and temporal mode independently, the real SM analysis filter
produces an in-phase latent and its cyclic Hilbert quadrature:

\[
z^R_{bcr}=\langle X_{bc},V^{SM}_r\rangle,\qquad
z^Q_{bcr}=\langle X_{bc},\mathcal H(V^{SM}_r)\rangle.
\]

The horizon decoder applies a bounded phase rotation:

\[
\widehat Y_{bhc}=\sum_r U_{hr}s_{cr}
\left(\cos\phi_{hr}\,z^R_{bcr}+\sin\phi_{hr}\,z^Q_{bcr}\right).
\]

`phi[h,r]` is represented by a small fixed cosine basis over horizon and a
learned rank-by-basis coefficient table. Coefficients initialize at zero, so
the complex model initially equals the real-SM anchor exactly. The output is
real. No channel index is contracted, so cross-variable lead--lag remains
impossible.

## Experiment matrix

Primary cell remains Weather, `seq_len=96`, `pred_len=720`, `cut_freq=13`.

| Arm | Real envelope | Horizon phase basis | Maximum phase |
| --- | --- | ---: | ---: |
| `fk_sm4_mode` | 4 components/mode | 0 | 0 |
| `fk_sm4_ph2_q` | 4 components/mode | 2 | pi/4 |
| `fk_sm4_ph4_q` | 4 components/mode | 4 | pi/4 |
| `fk_sm4_ph4_h` | 4 components/mode | 4 | pi/2 |

The phase table adds only `rank * phase_basis` trainable parameters (16 or 32
for rank 8), not a dense horizon-by-rank table.

## Gates and protocol

1. Unit tests require exact zero-phase identity, analytic quadrature, bounded
   phase, channel isolation, gradients, cache invalidation, and legacy state
   compatibility.
2. Run two-epoch single-H100 canaries for the real anchor and all phase arms.
3. Run a seed-2026 validation-only screen with test deferred.
4. Promote at most two phase arms only if validation MSE strictly beats the
   same-code `fk_sm4_mode` anchor, parameters remain below dense direct,
   fixed-work train/inference are each <=1.10 of the real anchor, and learned
   phase is finite and non-collapsed.
5. Open test only after validation selection. A complex arm is retained only
   if a three-seed confirmation gets at least 2/3 paired test wins and lower
   mean validation MSE than the real-SM anchor.

If no complex arm passes, complex phase is discarded and real `fk_sm4_mode`
remains the Stage-B result. Cross-variable mixing is forbidden in either case.
