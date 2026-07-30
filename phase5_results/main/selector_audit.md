# Phase 6-Protocol Selector Audit

Oracle is analysis only and must not be reported as a valid selected model.

- best_fixed_single_arm: phase5_asx_individual_period (mse_mean=0.19578, mae_mean=0.265862)
- test_oracle_mse_mean (ANALYSIS ONLY): 0.193548
- test_oracle_mae_mean (ANALYSIS ONLY): 0.262155

## Selector Comparison

| selector | mse_mean | mae_mean | delta_vs_best_single_mse | delta_vs_oracle_mse | selected_arm_counts |
| --- | ---: | ---: | ---: | ---: | --- |
| unrestricted_mean | 0.195153 | 0.264011 | -0.000627302 | 0.00160421 | phase5_asx_individual_revin:2; phase5_asx_period_multi:6 |
| unrestricted_last_segment | 0.197953 | 0.267799 | 0.00217305 | 0.00440456 | phase5_asx_cross_clip05:3; phase5_asx_individual_period:1; phase5_asx_period_multi:4 |
| unrestricted_segment_robust | 0.196106 | 0.265371 | 0.000326479 | 0.00255799 | phase5_asx_cross:2; phase5_asx_individual_revin:1; phase5_asx_period_multi:5 |
| unrestricted_margin_prefer_simple | 0.195156 | 0.264032 | -0.00062402 | 0.00160749 | phase5_asx_cross:2; phase5_asx_individual_revin:2; phase5_asx_period_multi:4 |
| policy_family | 0.193601 | 0.261789 | -0.00217886 | 5.26458e-05 | phase5_asx_individual_revin:4; phase5_asx_period_multi:4 |

## Weather / Electricity Detail

| dataset | pred_len | selector | selected_arm | test_mse | test_mae | oracle_arm | oracle_mse | fixed_best_arm |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| electricity | 192 | unrestricted_mean | phase5_asx_period_multi | 0.151979 | 0.249248 | phase5_asx_period_multi | 0.151979 | phase5_asx_individual_period |
| electricity | 336 | unrestricted_mean | phase5_asx_period_multi | 0.166828 | 0.263942 | phase5_asx_period_multi | 0.166828 | phase5_asx_individual_period |
| electricity | 720 | unrestricted_mean | phase5_asx_period_multi | 0.203556 | 0.293765 | phase5_asx_period_multi | 0.203556 | phase5_asx_individual_period |
| electricity | 96 | unrestricted_mean | phase5_asx_period_multi | 0.138265 | 0.238301 | phase5_asx_period_multi | 0.138265 | phase5_asx_individual_period |
| weather | 192 | unrestricted_mean | phase5_asx_individual_revin | 0.190203 | 0.240335 | phase5_asx_individual_revin | 0.190203 | phase5_asx_individual_period |
| weather | 336 | unrestricted_mean | phase5_asx_period_multi | 0.249379 | 0.28983 | phase5_asx_individual_revin | 0.240016 | phase5_asx_individual_period |
| weather | 720 | unrestricted_mean | phase5_asx_period_multi | 0.313159 | 0.336044 | phase5_asx_individual | 0.309688 | phase5_asx_individual_period |
| weather | 96 | unrestricted_mean | phase5_asx_individual_revin | 0.147851 | 0.200624 | phase5_asx_individual_revin | 0.147851 | phase5_asx_individual_period |
| electricity | 192 | unrestricted_last_segment | phase5_asx_period_multi | 0.151979 | 0.249248 | phase5_asx_period_multi | 0.151979 | phase5_asx_individual_period |
| electricity | 336 | unrestricted_last_segment | phase5_asx_period_multi | 0.166828 | 0.263942 | phase5_asx_period_multi | 0.166828 | phase5_asx_individual_period |
| electricity | 720 | unrestricted_last_segment | phase5_asx_period_multi | 0.203556 | 0.293765 | phase5_asx_period_multi | 0.203556 | phase5_asx_individual_period |
| electricity | 96 | unrestricted_last_segment | phase5_asx_period_multi | 0.138265 | 0.238301 | phase5_asx_period_multi | 0.138265 | phase5_asx_individual_period |
| weather | 192 | unrestricted_last_segment | phase5_asx_cross_clip05 | 0.203098 | 0.256164 | phase5_asx_individual_revin | 0.190203 | phase5_asx_individual_period |
| weather | 336 | unrestricted_last_segment | phase5_asx_cross_clip05 | 0.252263 | 0.291876 | phase5_asx_individual_revin | 0.240016 | phase5_asx_individual_period |
| weather | 720 | unrestricted_last_segment | phase5_asx_individual_period | 0.309708 | 0.331879 | phase5_asx_individual | 0.309688 | phase5_asx_individual_period |
| weather | 96 | unrestricted_last_segment | phase5_asx_cross_clip05 | 0.157925 | 0.217216 | phase5_asx_individual_revin | 0.147851 | phase5_asx_individual_period |
| electricity | 192 | unrestricted_segment_robust | phase5_asx_period_multi | 0.151979 | 0.249248 | phase5_asx_period_multi | 0.151979 | phase5_asx_individual_period |
| electricity | 336 | unrestricted_segment_robust | phase5_asx_period_multi | 0.166828 | 0.263942 | phase5_asx_period_multi | 0.166828 | phase5_asx_individual_period |
| electricity | 720 | unrestricted_segment_robust | phase5_asx_period_multi | 0.203556 | 0.293765 | phase5_asx_period_multi | 0.203556 | phase5_asx_individual_period |
| electricity | 96 | unrestricted_segment_robust | phase5_asx_period_multi | 0.138265 | 0.238301 | phase5_asx_period_multi | 0.138265 | phase5_asx_individual_period |
| weather | 192 | unrestricted_segment_robust | phase5_asx_cross | 0.197634 | 0.250958 | phase5_asx_individual_revin | 0.190203 | phase5_asx_individual_period |
| weather | 336 | unrestricted_segment_robust | phase5_asx_cross | 0.249578 | 0.290083 | phase5_asx_individual_revin | 0.240016 | phase5_asx_individual_period |
| weather | 720 | unrestricted_segment_robust | phase5_asx_period_multi | 0.313159 | 0.336044 | phase5_asx_individual | 0.309688 | phase5_asx_individual_period |
| weather | 96 | unrestricted_segment_robust | phase5_asx_individual_revin | 0.147851 | 0.200624 | phase5_asx_individual_revin | 0.147851 | phase5_asx_individual_period |
| electricity | 192 | unrestricted_margin_prefer_simple | phase5_asx_period_multi | 0.151979 | 0.249248 | phase5_asx_period_multi | 0.151979 | phase5_asx_individual_period |
| electricity | 336 | unrestricted_margin_prefer_simple | phase5_asx_period_multi | 0.166828 | 0.263942 | phase5_asx_period_multi | 0.166828 | phase5_asx_individual_period |
| electricity | 720 | unrestricted_margin_prefer_simple | phase5_asx_period_multi | 0.203556 | 0.293765 | phase5_asx_period_multi | 0.203556 | phase5_asx_individual_period |
| electricity | 96 | unrestricted_margin_prefer_simple | phase5_asx_period_multi | 0.138265 | 0.238301 | phase5_asx_period_multi | 0.138265 | phase5_asx_individual_period |
| weather | 192 | unrestricted_margin_prefer_simple | phase5_asx_individual_revin | 0.190203 | 0.240335 | phase5_asx_individual_revin | 0.190203 | phase5_asx_individual_period |
| weather | 336 | unrestricted_margin_prefer_simple | phase5_asx_cross | 0.249578 | 0.290083 | phase5_asx_individual_revin | 0.240016 | phase5_asx_individual_period |
| weather | 720 | unrestricted_margin_prefer_simple | phase5_asx_cross | 0.312987 | 0.335959 | phase5_asx_individual | 0.309688 | phase5_asx_individual_period |
| weather | 96 | unrestricted_margin_prefer_simple | phase5_asx_individual_revin | 0.147851 | 0.200624 | phase5_asx_individual_revin | 0.147851 | phase5_asx_individual_period |
| electricity | 192 | policy_family | phase5_asx_period_multi | 0.151979 | 0.249248 | phase5_asx_period_multi | 0.151979 | phase5_asx_individual_period |
| electricity | 336 | policy_family | phase5_asx_period_multi | 0.166828 | 0.263942 | phase5_asx_period_multi | 0.166828 | phase5_asx_individual_period |
| electricity | 720 | policy_family | phase5_asx_period_multi | 0.203556 | 0.293765 | phase5_asx_period_multi | 0.203556 | phase5_asx_individual_period |
| electricity | 96 | policy_family | phase5_asx_period_multi | 0.138265 | 0.238301 | phase5_asx_period_multi | 0.138265 | phase5_asx_individual_period |
| weather | 192 | policy_family | phase5_asx_individual_revin | 0.190203 | 0.240335 | phase5_asx_individual_revin | 0.190203 | phase5_asx_individual_period |
| weather | 336 | policy_family | phase5_asx_individual_revin | 0.240016 | 0.279124 | phase5_asx_individual_revin | 0.240016 | phase5_asx_individual_period |
| weather | 720 | policy_family | phase5_asx_individual_revin | 0.31011 | 0.328973 | phase5_asx_individual | 0.309688 | phase5_asx_individual_period |
| weather | 96 | policy_family | phase5_asx_individual_revin | 0.147851 | 0.200624 | phase5_asx_individual_revin | 0.147851 | phase5_asx_individual_period |

## Fairness Note

Selectors use validation metrics aggregated over seeds. Test metrics (and the oracle) are shown only after selection, for analysis.
