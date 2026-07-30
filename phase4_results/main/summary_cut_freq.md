# Phase 4-Finalize Cut_freq Diagnostics

Best cut_freq **by validation** is the selection-safe choice. Best-by-test is shown for analysis only and must NOT drive selection.

## Per cut_freq

| dataset | pred_len | arm | cut_freq | n | val_mse | mse | mae |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| electricity | 192 | phase4_asx_cross | 181 | 3 | 0.131551 | 0.156384 | 0.25649 |
| electricity | 192 | phase4_asx_cross_revin | 181 | 3 | 0.131513 | 0.156517 | 0.256449 |
| electricity | 192 | phase4_asx_individual | 181 | 3 | 0.132576 | 0.15903 | 0.258117 |
| electricity | 192 | phase4_asx_individual_period | 181 | 3 | 0.131225 | 0.157527 | 0.255902 |
| electricity | 192 | phase4_asx_individual_revin | 181 | 3 | 0.132509 | 0.15915 | 0.257916 |
| electricity | 192 | phase4_asx_period_multi | 181 | 3 | 0.127752 | 0.151979 | 0.249248 |
| electricity | 192 | phase4_asx_period_single | 181 | 3 | 0.128022 | 0.15228 | 0.249798 |
| electricity | 336 | phase4_asx_cross | 181 | 3 | 0.146601 | 0.171584 | 0.271645 |
| electricity | 336 | phase4_asx_cross_revin | 181 | 3 | 0.146357 | 0.171927 | 0.271631 |
| electricity | 336 | phase4_asx_individual | 181 | 3 | 0.147573 | 0.173817 | 0.27306 |
| electricity | 336 | phase4_asx_individual_period | 181 | 3 | 0.145905 | 0.171962 | 0.27044 |
| electricity | 336 | phase4_asx_individual_revin | 181 | 3 | 0.147353 | 0.174186 | 0.27284 |
| electricity | 336 | phase4_asx_period_multi | 181 | 3 | 0.142607 | 0.166828 | 0.263942 |
| electricity | 336 | phase4_asx_period_single | 181 | 3 | 0.142828 | 0.167059 | 0.264367 |
| electricity | 720 | phase4_asx_cross | 181 | 3 | 0.178851 | 0.209717 | 0.304132 |
| electricity | 720 | phase4_asx_cross_revin | 181 | 3 | 0.177342 | 0.210093 | 0.303634 |
| electricity | 720 | phase4_asx_individual | 181 | 3 | 0.179707 | 0.209144 | 0.302893 |
| electricity | 720 | phase4_asx_individual_period | 181 | 3 | 0.177137 | 0.206207 | 0.2989 |
| electricity | 720 | phase4_asx_individual_revin | 181 | 3 | 0.17811 | 0.209677 | 0.302399 |
| electricity | 720 | phase4_asx_period_multi | 181 | 3 | 0.174302 | 0.203556 | 0.293765 |
| electricity | 720 | phase4_asx_period_single | 181 | 3 | 0.174529 | 0.203699 | 0.294152 |
| electricity | 96 | phase4_asx_cross | 181 | 3 | 0.119219 | 0.141479 | 0.243351 |
| electricity | 96 | phase4_asx_cross_revin | 181 | 3 | 0.11917 | 0.141441 | 0.243249 |
| electricity | 96 | phase4_asx_individual | 181 | 3 | 0.11977 | 0.142635 | 0.244435 |
| electricity | 96 | phase4_asx_individual_period | 181 | 3 | 0.11868 | 0.141463 | 0.24263 |
| electricity | 96 | phase4_asx_individual_revin | 181 | 3 | 0.119742 | 0.142674 | 0.244322 |
| electricity | 96 | phase4_asx_period_multi | 181 | 3 | 0.116401 | 0.138265 | 0.238301 |
| electricity | 96 | phase4_asx_period_single | 181 | 3 | 0.116568 | 0.13846 | 0.238651 |
| weather | 192 | phase4_asx_cross | 31 | 3 | 0.437878 | 0.197634 | 0.250958 |
| weather | 192 | phase4_asx_cross_revin | 31 | 3 | 0.438163 | 0.196998 | 0.248284 |
| weather | 192 | phase4_asx_individual | 31 | 3 | 0.438676 | 0.190698 | 0.242888 |
| weather | 192 | phase4_asx_individual_period | 31 | 3 | 0.438569 | 0.190687 | 0.242814 |
| weather | 192 | phase4_asx_individual_revin | 31 | 3 | 0.437773 | 0.190203 | 0.240335 |
| weather | 192 | phase4_asx_period_multi | 31 | 3 | 0.438096 | 0.197595 | 0.250832 |
| weather | 192 | phase4_asx_period_single | 31 | 3 | 0.438066 | 0.197668 | 0.250993 |
| weather | 336 | phase4_asx_cross | 31 | 3 | 0.499075 | 0.249578 | 0.290083 |
| weather | 336 | phase4_asx_cross_revin | 31 | 3 | 0.500222 | 0.248298 | 0.286427 |
| weather | 336 | phase4_asx_individual | 31 | 3 | 0.503298 | 0.240569 | 0.282137 |
| weather | 336 | phase4_asx_individual_period | 31 | 3 | 0.503263 | 0.240582 | 0.282093 |
| weather | 336 | phase4_asx_individual_revin | 31 | 3 | 0.502111 | 0.240016 | 0.279124 |
| weather | 336 | phase4_asx_period_multi | 31 | 3 | 0.499061 | 0.249379 | 0.28983 |
| weather | 336 | phase4_asx_period_single | 31 | 3 | 0.499203 | 0.249561 | 0.290096 |
| weather | 720 | phase4_asx_cross | 31 | 3 | 0.591443 | 0.312987 | 0.335959 |
| weather | 720 | phase4_asx_cross_revin | 31 | 3 | 0.595247 | 0.313915 | 0.333501 |
| weather | 720 | phase4_asx_individual | 31 | 3 | 0.595775 | 0.309688 | 0.331898 |
| weather | 720 | phase4_asx_individual_period | 31 | 3 | 0.595722 | 0.309708 | 0.331879 |
| weather | 720 | phase4_asx_individual_revin | 31 | 3 | 0.596845 | 0.31011 | 0.328973 |
| weather | 720 | phase4_asx_period_multi | 31 | 3 | 0.591314 | 0.313159 | 0.336044 |
| weather | 720 | phase4_asx_period_single | 31 | 3 | 0.591341 | 0.313169 | 0.336087 |
| weather | 96 | phase4_asx_cross | 31 | 3 | 0.387855 | 0.152079 | 0.20958 |
| weather | 96 | phase4_asx_cross_revin | 31 | 3 | 0.38532 | 0.151491 | 0.207628 |
| weather | 96 | phase4_asx_individual | 31 | 3 | 0.382135 | 0.148135 | 0.202349 |
| weather | 96 | phase4_asx_individual_period | 31 | 3 | 0.382017 | 0.148103 | 0.20224 |
| weather | 96 | phase4_asx_individual_revin | 31 | 3 | 0.381475 | 0.147851 | 0.200624 |
| weather | 96 | phase4_asx_period_multi | 31 | 3 | 0.387529 | 0.151885 | 0.209185 |
| weather | 96 | phase4_asx_period_single | 31 | 3 | 0.387698 | 0.152048 | 0.20958 |

## Best cut_freq by validation (selection-safe)

| dataset | pred_len | arm | best_cut_freq_val | val_mse |
| --- | ---: | --- | ---: | ---: |
| electricity | 192 | phase4_asx_cross | 181 | 0.131551 |
| electricity | 192 | phase4_asx_cross_revin | 181 | 0.131513 |
| electricity | 192 | phase4_asx_individual | 181 | 0.132576 |
| electricity | 192 | phase4_asx_individual_period | 181 | 0.131225 |
| electricity | 192 | phase4_asx_individual_revin | 181 | 0.132509 |
| electricity | 192 | phase4_asx_period_multi | 181 | 0.127752 |
| electricity | 192 | phase4_asx_period_single | 181 | 0.128022 |
| electricity | 336 | phase4_asx_cross | 181 | 0.146601 |
| electricity | 336 | phase4_asx_cross_revin | 181 | 0.146357 |
| electricity | 336 | phase4_asx_individual | 181 | 0.147573 |
| electricity | 336 | phase4_asx_individual_period | 181 | 0.145905 |
| electricity | 336 | phase4_asx_individual_revin | 181 | 0.147353 |
| electricity | 336 | phase4_asx_period_multi | 181 | 0.142607 |
| electricity | 336 | phase4_asx_period_single | 181 | 0.142828 |
| electricity | 720 | phase4_asx_cross | 181 | 0.178851 |
| electricity | 720 | phase4_asx_cross_revin | 181 | 0.177342 |
| electricity | 720 | phase4_asx_individual | 181 | 0.179707 |
| electricity | 720 | phase4_asx_individual_period | 181 | 0.177137 |
| electricity | 720 | phase4_asx_individual_revin | 181 | 0.17811 |
| electricity | 720 | phase4_asx_period_multi | 181 | 0.174302 |
| electricity | 720 | phase4_asx_period_single | 181 | 0.174529 |
| electricity | 96 | phase4_asx_cross | 181 | 0.119219 |
| electricity | 96 | phase4_asx_cross_revin | 181 | 0.11917 |
| electricity | 96 | phase4_asx_individual | 181 | 0.11977 |
| electricity | 96 | phase4_asx_individual_period | 181 | 0.11868 |
| electricity | 96 | phase4_asx_individual_revin | 181 | 0.119742 |
| electricity | 96 | phase4_asx_period_multi | 181 | 0.116401 |
| electricity | 96 | phase4_asx_period_single | 181 | 0.116568 |
| weather | 192 | phase4_asx_cross | 31 | 0.437878 |
| weather | 192 | phase4_asx_cross_revin | 31 | 0.438163 |
| weather | 192 | phase4_asx_individual | 31 | 0.438676 |
| weather | 192 | phase4_asx_individual_period | 31 | 0.438569 |
| weather | 192 | phase4_asx_individual_revin | 31 | 0.437773 |
| weather | 192 | phase4_asx_period_multi | 31 | 0.438096 |
| weather | 192 | phase4_asx_period_single | 31 | 0.438066 |
| weather | 336 | phase4_asx_cross | 31 | 0.499075 |
| weather | 336 | phase4_asx_cross_revin | 31 | 0.500222 |
| weather | 336 | phase4_asx_individual | 31 | 0.503298 |
| weather | 336 | phase4_asx_individual_period | 31 | 0.503263 |
| weather | 336 | phase4_asx_individual_revin | 31 | 0.502111 |
| weather | 336 | phase4_asx_period_multi | 31 | 0.499061 |
| weather | 336 | phase4_asx_period_single | 31 | 0.499203 |
| weather | 720 | phase4_asx_cross | 31 | 0.591443 |
| weather | 720 | phase4_asx_cross_revin | 31 | 0.595247 |
| weather | 720 | phase4_asx_individual | 31 | 0.595775 |
| weather | 720 | phase4_asx_individual_period | 31 | 0.595722 |
| weather | 720 | phase4_asx_individual_revin | 31 | 0.596845 |
| weather | 720 | phase4_asx_period_multi | 31 | 0.591314 |
| weather | 720 | phase4_asx_period_single | 31 | 0.591341 |
| weather | 96 | phase4_asx_cross | 31 | 0.387855 |
| weather | 96 | phase4_asx_cross_revin | 31 | 0.38532 |
| weather | 96 | phase4_asx_individual | 31 | 0.382135 |
| weather | 96 | phase4_asx_individual_period | 31 | 0.382017 |
| weather | 96 | phase4_asx_individual_revin | 31 | 0.381475 |
| weather | 96 | phase4_asx_period_multi | 31 | 0.387529 |
| weather | 96 | phase4_asx_period_single | 31 | 0.387698 |

## Best cut_freq by test (ANALYSIS ONLY -- not for selection)

| dataset | pred_len | arm | best_cut_freq_test | test_mse |
| --- | ---: | --- | ---: | ---: |
| electricity | 192 | phase4_asx_cross | 181 | 0.156384 |
| electricity | 192 | phase4_asx_cross_revin | 181 | 0.156517 |
| electricity | 192 | phase4_asx_individual | 181 | 0.15903 |
| electricity | 192 | phase4_asx_individual_period | 181 | 0.157527 |
| electricity | 192 | phase4_asx_individual_revin | 181 | 0.15915 |
| electricity | 192 | phase4_asx_period_multi | 181 | 0.151979 |
| electricity | 192 | phase4_asx_period_single | 181 | 0.15228 |
| electricity | 336 | phase4_asx_cross | 181 | 0.171584 |
| electricity | 336 | phase4_asx_cross_revin | 181 | 0.171927 |
| electricity | 336 | phase4_asx_individual | 181 | 0.173817 |
| electricity | 336 | phase4_asx_individual_period | 181 | 0.171962 |
| electricity | 336 | phase4_asx_individual_revin | 181 | 0.174186 |
| electricity | 336 | phase4_asx_period_multi | 181 | 0.166828 |
| electricity | 336 | phase4_asx_period_single | 181 | 0.167059 |
| electricity | 720 | phase4_asx_cross | 181 | 0.209717 |
| electricity | 720 | phase4_asx_cross_revin | 181 | 0.210093 |
| electricity | 720 | phase4_asx_individual | 181 | 0.209144 |
| electricity | 720 | phase4_asx_individual_period | 181 | 0.206207 |
| electricity | 720 | phase4_asx_individual_revin | 181 | 0.209677 |
| electricity | 720 | phase4_asx_period_multi | 181 | 0.203556 |
| electricity | 720 | phase4_asx_period_single | 181 | 0.203699 |
| electricity | 96 | phase4_asx_cross | 181 | 0.141479 |
| electricity | 96 | phase4_asx_cross_revin | 181 | 0.141441 |
| electricity | 96 | phase4_asx_individual | 181 | 0.142635 |
| electricity | 96 | phase4_asx_individual_period | 181 | 0.141463 |
| electricity | 96 | phase4_asx_individual_revin | 181 | 0.142674 |
| electricity | 96 | phase4_asx_period_multi | 181 | 0.138265 |
| electricity | 96 | phase4_asx_period_single | 181 | 0.13846 |
| weather | 192 | phase4_asx_cross | 31 | 0.197634 |
| weather | 192 | phase4_asx_cross_revin | 31 | 0.196998 |
| weather | 192 | phase4_asx_individual | 31 | 0.190698 |
| weather | 192 | phase4_asx_individual_period | 31 | 0.190687 |
| weather | 192 | phase4_asx_individual_revin | 31 | 0.190203 |
| weather | 192 | phase4_asx_period_multi | 31 | 0.197595 |
| weather | 192 | phase4_asx_period_single | 31 | 0.197668 |
| weather | 336 | phase4_asx_cross | 31 | 0.249578 |
| weather | 336 | phase4_asx_cross_revin | 31 | 0.248298 |
| weather | 336 | phase4_asx_individual | 31 | 0.240569 |
| weather | 336 | phase4_asx_individual_period | 31 | 0.240582 |
| weather | 336 | phase4_asx_individual_revin | 31 | 0.240016 |
| weather | 336 | phase4_asx_period_multi | 31 | 0.249379 |
| weather | 336 | phase4_asx_period_single | 31 | 0.249561 |
| weather | 720 | phase4_asx_cross | 31 | 0.312987 |
| weather | 720 | phase4_asx_cross_revin | 31 | 0.313915 |
| weather | 720 | phase4_asx_individual | 31 | 0.309688 |
| weather | 720 | phase4_asx_individual_period | 31 | 0.309708 |
| weather | 720 | phase4_asx_individual_revin | 31 | 0.31011 |
| weather | 720 | phase4_asx_period_multi | 31 | 0.313159 |
| weather | 720 | phase4_asx_period_single | 31 | 0.313169 |
| weather | 96 | phase4_asx_cross | 31 | 0.152079 |
| weather | 96 | phase4_asx_cross_revin | 31 | 0.151491 |
| weather | 96 | phase4_asx_individual | 31 | 0.148135 |
| weather | 96 | phase4_asx_individual_period | 31 | 0.148103 |
| weather | 96 | phase4_asx_individual_revin | 31 | 0.147851 |
| weather | 96 | phase4_asx_period_multi | 31 | 0.151885 |
| weather | 96 | phase4_asx_period_single | 31 | 0.152048 |

