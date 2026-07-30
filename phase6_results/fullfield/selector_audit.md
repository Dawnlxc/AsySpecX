# Phase 6-Protocol Selector Audit

Oracle is analysis only and must not be reported as a valid selected model.

- best_fixed_single_arm: phase6_asx_period_multi (mse_mean=0.338568, mae_mean=0.344618)
- test_oracle_mse_mean (ANALYSIS ONLY): 0.333085
- test_oracle_mae_mean (ANALYSIS ONLY): 0.341625

## Selector Comparison

| selector | mse_mean | mae_mean | delta_vs_best_single_mse | delta_vs_oracle_mse | selected_arm_counts |
| --- | ---: | ---: | ---: | ---: | --- |
| unrestricted_mean | 0.334771 | 0.342178 | -0.00379665 | 0.0016864 | phase6_asx_cross:2; phase6_asx_cross_clip05:6; phase6_asx_individual_period:8; phase6_asx_individual_revin:7; phase6_asx_period_multi:25 |
| unrestricted_segment_robust | 0.33493 | 0.342405 | -0.00363768 | 0.00184536 | phase6_asx_cross:4; phase6_asx_cross_clip05:6; phase6_asx_individual_period:8; phase6_asx_individual_revin:6; phase6_asx_period_multi:24 |
| unrestricted_margin_prefer_simple | 0.33493 | 0.342333 | -0.00363801 | 0.00184504 | phase6_asx_cross:8; phase6_asx_cross_clip05:6; phase6_asx_individual:2; phase6_asx_individual_period:4; phase6_asx_individual_revin:9; phase6_asx_period_multi:19 |
| policy_family | 0.334646 | 0.341964 | -0.00392204 | 0.001561 | phase6_asx_cross:6; phase6_asx_cross_clip05:6; phase6_asx_individual:3; phase6_asx_individual_period:6; phase6_asx_individual_revin:8; phase6_asx_period_multi:19 |

## Weather / Electricity Detail

| dataset | pred_len | selector | selected_arm | test_mse | test_mae | oracle_arm | oracle_mse | fixed_best_arm |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| electricity | 192 | unrestricted_mean | phase6_asx_period_multi | 0.151979 | 0.249248 | phase6_asx_period_multi | 0.151979 | phase6_asx_period_multi |
| electricity | 336 | unrestricted_mean | phase6_asx_period_multi | 0.166828 | 0.263942 | phase6_asx_period_multi | 0.166828 | phase6_asx_period_multi |
| electricity | 720 | unrestricted_mean | phase6_asx_period_multi | 0.203556 | 0.293765 | phase6_asx_period_multi | 0.203556 | phase6_asx_period_multi |
| electricity | 96 | unrestricted_mean | phase6_asx_period_multi | 0.138265 | 0.238301 | phase6_asx_period_multi | 0.138265 | phase6_asx_period_multi |
| electricity | 192 | unrestricted_mean | phase6_asx_individual_period | 0.194564 | 0.281049 | phase6_asx_individual_period | 0.194564 | phase6_asx_period_multi |
| electricity | 336 | unrestricted_mean | phase6_asx_individual_period | 0.208904 | 0.295086 | phase6_asx_individual_period | 0.208904 | phase6_asx_period_multi |
| electricity | 720 | unrestricted_mean | phase6_asx_individual_revin | 0.25078 | 0.327686 | phase6_asx_individual_period | 0.249786 | phase6_asx_period_multi |
| electricity | 96 | unrestricted_mean | phase6_asx_individual_period | 0.194684 | 0.279893 | phase6_asx_individual_period | 0.194684 | phase6_asx_period_multi |
| weather | 192 | unrestricted_mean | phase6_asx_individual_revin | 0.190203 | 0.240335 | phase6_asx_individual_revin | 0.190203 | phase6_asx_period_multi |
| weather | 336 | unrestricted_mean | phase6_asx_period_multi | 0.249379 | 0.28983 | phase6_asx_individual_revin | 0.240016 | phase6_asx_period_multi |
| weather | 720 | unrestricted_mean | phase6_asx_period_multi | 0.313159 | 0.336044 | phase6_asx_individual | 0.309688 | phase6_asx_period_multi |
| weather | 96 | unrestricted_mean | phase6_asx_individual_revin | 0.147851 | 0.200624 | phase6_asx_individual_revin | 0.147851 | phase6_asx_period_multi |
| weather | 192 | unrestricted_mean | phase6_asx_individual_revin | 0.231881 | 0.274434 | phase6_asx_individual_revin | 0.231881 | phase6_asx_period_multi |
| weather | 336 | unrestricted_mean | phase6_asx_individual_revin | 0.283318 | 0.308607 | phase6_asx_individual_revin | 0.283318 | phase6_asx_period_multi |
| weather | 720 | unrestricted_mean | phase6_asx_individual_revin | 0.3567 | 0.353665 | phase6_asx_individual_revin | 0.3567 | phase6_asx_period_multi |
| weather | 96 | unrestricted_mean | phase6_asx_individual_revin | 0.187501 | 0.239589 | phase6_asx_individual_revin | 0.187501 | phase6_asx_period_multi |
| electricity | 192 | unrestricted_segment_robust | phase6_asx_period_multi | 0.151979 | 0.249248 | phase6_asx_period_multi | 0.151979 | phase6_asx_period_multi |
| electricity | 336 | unrestricted_segment_robust | phase6_asx_period_multi | 0.166828 | 0.263942 | phase6_asx_period_multi | 0.166828 | phase6_asx_period_multi |
| electricity | 720 | unrestricted_segment_robust | phase6_asx_period_multi | 0.203556 | 0.293765 | phase6_asx_period_multi | 0.203556 | phase6_asx_period_multi |
| electricity | 96 | unrestricted_segment_robust | phase6_asx_period_multi | 0.138265 | 0.238301 | phase6_asx_period_multi | 0.138265 | phase6_asx_period_multi |
| electricity | 192 | unrestricted_segment_robust | phase6_asx_individual_period | 0.194564 | 0.281049 | phase6_asx_individual_period | 0.194564 | phase6_asx_period_multi |
| electricity | 336 | unrestricted_segment_robust | phase6_asx_individual_period | 0.208904 | 0.295086 | phase6_asx_individual_period | 0.208904 | phase6_asx_period_multi |
| electricity | 720 | unrestricted_segment_robust | phase6_asx_individual_revin | 0.25078 | 0.327686 | phase6_asx_individual_period | 0.249786 | phase6_asx_period_multi |
| electricity | 96 | unrestricted_segment_robust | phase6_asx_individual_period | 0.194684 | 0.279893 | phase6_asx_individual_period | 0.194684 | phase6_asx_period_multi |
| weather | 192 | unrestricted_segment_robust | phase6_asx_cross | 0.197634 | 0.250958 | phase6_asx_individual_revin | 0.190203 | phase6_asx_period_multi |
| weather | 336 | unrestricted_segment_robust | phase6_asx_cross | 0.249578 | 0.290083 | phase6_asx_individual_revin | 0.240016 | phase6_asx_period_multi |
| weather | 720 | unrestricted_segment_robust | phase6_asx_period_multi | 0.313159 | 0.336044 | phase6_asx_individual | 0.309688 | phase6_asx_period_multi |
| weather | 96 | unrestricted_segment_robust | phase6_asx_individual_revin | 0.147851 | 0.200624 | phase6_asx_individual_revin | 0.147851 | phase6_asx_period_multi |
| weather | 192 | unrestricted_segment_robust | phase6_asx_individual_revin | 0.231881 | 0.274434 | phase6_asx_individual_revin | 0.231881 | phase6_asx_period_multi |
| weather | 336 | unrestricted_segment_robust | phase6_asx_individual_revin | 0.283318 | 0.308607 | phase6_asx_individual_revin | 0.283318 | phase6_asx_period_multi |
| weather | 720 | unrestricted_segment_robust | phase6_asx_individual_revin | 0.3567 | 0.353665 | phase6_asx_individual_revin | 0.3567 | phase6_asx_period_multi |
| weather | 96 | unrestricted_segment_robust | phase6_asx_individual_revin | 0.187501 | 0.239589 | phase6_asx_individual_revin | 0.187501 | phase6_asx_period_multi |
| electricity | 192 | unrestricted_margin_prefer_simple | phase6_asx_period_multi | 0.151979 | 0.249248 | phase6_asx_period_multi | 0.151979 | phase6_asx_period_multi |
| electricity | 336 | unrestricted_margin_prefer_simple | phase6_asx_period_multi | 0.166828 | 0.263942 | phase6_asx_period_multi | 0.166828 | phase6_asx_period_multi |
| electricity | 720 | unrestricted_margin_prefer_simple | phase6_asx_period_multi | 0.203556 | 0.293765 | phase6_asx_period_multi | 0.203556 | phase6_asx_period_multi |
| electricity | 96 | unrestricted_margin_prefer_simple | phase6_asx_period_multi | 0.138265 | 0.238301 | phase6_asx_period_multi | 0.138265 | phase6_asx_period_multi |
| electricity | 192 | unrestricted_margin_prefer_simple | phase6_asx_individual_period | 0.194564 | 0.281049 | phase6_asx_individual_period | 0.194564 | phase6_asx_period_multi |
| electricity | 336 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.209877 | 0.29599 | phase6_asx_individual_period | 0.208904 | phase6_asx_period_multi |
| electricity | 720 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.25078 | 0.327686 | phase6_asx_individual_period | 0.249786 | phase6_asx_period_multi |
| electricity | 96 | unrestricted_margin_prefer_simple | phase6_asx_individual_period | 0.194684 | 0.279893 | phase6_asx_individual_period | 0.194684 | phase6_asx_period_multi |
| weather | 192 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.190203 | 0.240335 | phase6_asx_individual_revin | 0.190203 | phase6_asx_period_multi |
| weather | 336 | unrestricted_margin_prefer_simple | phase6_asx_cross | 0.249578 | 0.290083 | phase6_asx_individual_revin | 0.240016 | phase6_asx_period_multi |
| weather | 720 | unrestricted_margin_prefer_simple | phase6_asx_cross | 0.312987 | 0.335959 | phase6_asx_individual | 0.309688 | phase6_asx_period_multi |
| weather | 96 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.147851 | 0.200624 | phase6_asx_individual_revin | 0.147851 | phase6_asx_period_multi |
| weather | 192 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.231881 | 0.274434 | phase6_asx_individual_revin | 0.231881 | phase6_asx_period_multi |
| weather | 336 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.283318 | 0.308607 | phase6_asx_individual_revin | 0.283318 | phase6_asx_period_multi |
| weather | 720 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.3567 | 0.353665 | phase6_asx_individual_revin | 0.3567 | phase6_asx_period_multi |
| weather | 96 | unrestricted_margin_prefer_simple | phase6_asx_individual_revin | 0.187501 | 0.239589 | phase6_asx_individual_revin | 0.187501 | phase6_asx_period_multi |
| electricity | 192 | policy_family | phase6_asx_period_multi | 0.151979 | 0.249248 | phase6_asx_period_multi | 0.151979 | phase6_asx_period_multi |
| electricity | 336 | policy_family | phase6_asx_period_multi | 0.166828 | 0.263942 | phase6_asx_period_multi | 0.166828 | phase6_asx_period_multi |
| electricity | 720 | policy_family | phase6_asx_period_multi | 0.203556 | 0.293765 | phase6_asx_period_multi | 0.203556 | phase6_asx_period_multi |
| electricity | 96 | policy_family | phase6_asx_period_multi | 0.138265 | 0.238301 | phase6_asx_period_multi | 0.138265 | phase6_asx_period_multi |
| electricity | 192 | policy_family | phase6_asx_individual_period | 0.194564 | 0.281049 | phase6_asx_individual_period | 0.194564 | phase6_asx_period_multi |
| electricity | 336 | policy_family | phase6_asx_individual_period | 0.208904 | 0.295086 | phase6_asx_individual_period | 0.208904 | phase6_asx_period_multi |
| electricity | 720 | policy_family | phase6_asx_individual_period | 0.249786 | 0.327229 | phase6_asx_individual_period | 0.249786 | phase6_asx_period_multi |
| electricity | 96 | policy_family | phase6_asx_individual_period | 0.194684 | 0.279893 | phase6_asx_individual_period | 0.194684 | phase6_asx_period_multi |
| weather | 192 | policy_family | phase6_asx_individual_revin | 0.190203 | 0.240335 | phase6_asx_individual_revin | 0.190203 | phase6_asx_period_multi |
| weather | 336 | policy_family | phase6_asx_individual_revin | 0.240016 | 0.279124 | phase6_asx_individual_revin | 0.240016 | phase6_asx_period_multi |
| weather | 720 | policy_family | phase6_asx_individual_revin | 0.31011 | 0.328973 | phase6_asx_individual | 0.309688 | phase6_asx_period_multi |
| weather | 96 | policy_family | phase6_asx_individual_revin | 0.147851 | 0.200624 | phase6_asx_individual_revin | 0.147851 | phase6_asx_period_multi |
| weather | 192 | policy_family | phase6_asx_individual_revin | 0.231881 | 0.274434 | phase6_asx_individual_revin | 0.231881 | phase6_asx_period_multi |
| weather | 336 | policy_family | phase6_asx_individual_revin | 0.283318 | 0.308607 | phase6_asx_individual_revin | 0.283318 | phase6_asx_period_multi |
| weather | 720 | policy_family | phase6_asx_individual_revin | 0.3567 | 0.353665 | phase6_asx_individual_revin | 0.3567 | phase6_asx_period_multi |
| weather | 96 | policy_family | phase6_asx_individual_revin | 0.187501 | 0.239589 | phase6_asx_individual_revin | 0.187501 | phase6_asx_period_multi |

## Fairness Note

Selectors use validation metrics aggregated over seeds. Test metrics (and the oracle) are shown only after selection, for analysis.
