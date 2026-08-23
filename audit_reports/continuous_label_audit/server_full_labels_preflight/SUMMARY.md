# Continuous-label coverage audit

This report is outcome-blind: label value columns were not loaded.

## Source row counts

- `reports`: 1,070,181
- `report_fy_labels`: 2,625,006
- `report_confirmation_labels`: 15,750,036

## Task coverage and maturity

| task_id | source_rows | source_valid_rows | configured_valid_rows | positive_weight_valid_rows | feature_date_min | feature_date_max | label_date_min | label_date_max | maturity_days_p50 | maturity_days_p90 | maturity_days_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delta_log_dispersion__1m__active__fh0 | 1068134 | 341871 | 341871 | 341871 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 39 |
| delta_log_dispersion__1m__active__fh1 | 936003 | 317166 | 317166 | 317166 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 39 |
| delta_log_dispersion__1m__active__fh2 | 620869 | 153260 | 153260 | 153260 | 2014-01-02 | 2025-08-29 | 2014-03-26 | 2025-08-29 | 31 | 33 | 40 |
| delta_log_dispersion__1m__fixed__fh0 | 1068134 | 535740 | 535740 | 464904 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 39 |
| delta_log_dispersion__1m__fixed__fh1 | 936003 | 499844 | 499844 | 433325 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 39 |
| delta_log_dispersion__1m__fixed__fh2 | 620869 | 293466 | 293466 | 235714 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 40 |
| delta_log_dispersion__1m__market__fh0 | 1068134 | 529513 | 529513 | 462178 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 39 |
| delta_log_dispersion__1m__market__fh1 | 936003 | 494211 | 494211 | 430857 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 39 |
| delta_log_dispersion__1m__market__fh2 | 620869 | 291507 | 291507 | 234911 | 2014-01-02 | 2025-08-29 | 2014-03-07 | 2025-08-29 | 31 | 33 | 40 |
| delta_log_dispersion__3m__active__fh0 | 1068134 | 449860 | 449860 | 449860 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 99 |
| delta_log_dispersion__3m__active__fh1 | 936003 | 439178 | 439178 | 439178 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 99 |
| delta_log_dispersion__3m__active__fh2 | 620869 | 239237 | 239237 | 239237 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 100 |
| delta_log_dispersion__3m__fixed__fh0 | 1068134 | 530480 | 530480 | 505671 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 99 |
| delta_log_dispersion__3m__fixed__fh1 | 936003 | 497182 | 497182 | 483029 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 99 |
| delta_log_dispersion__3m__fixed__fh2 | 620869 | 291295 | 291295 | 276976 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 100 |
| delta_log_dispersion__3m__market__fh0 | 1068134 | 515898 | 515898 | 496489 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 99 |
| delta_log_dispersion__3m__market__fh1 | 936003 | 485706 | 485706 | 475079 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 99 |
| delta_log_dispersion__3m__market__fh2 | 620869 | 285261 | 285261 | 272791 | 2014-01-02 | 2025-08-29 | 2014-05-07 | 2025-08-29 | 92 | 94 | 100 |
| residual_signed_raw__fh0 | 1068134 | 845673 | 845673 | 845673 | 2014-01-02 | 2025-08-29 | 2015-01-24 | 2026-06-18 | 286 | 420 | 1341 |
| residual_signed_raw__fh1 | 936003 | 745818 | 745818 | 745818 | 2014-01-02 | 2025-08-29 | 2016-01-18 | 2026-06-18 | 634 | 779 | 1700 |
| residual_signed_raw__fh2 | 620869 | 375376 | 375376 | 375376 | 2014-01-02 | 2025-08-29 | 2017-01-24 | 2026-06-18 | 966 | 1089 | 1833 |

## Configured split counts

| task_id | split | configured_valid_rows_in_feature_window | eligible_rows | positive_weight_eligible_rows | excluded_label_too_late | excluded_missing_or_invalid_label_date |
| --- | --- | --- | --- | --- | --- | --- |
| delta_log_dispersion__1m__active__fh0 | test | 8808 | 8565 | 8565 | 243 | 0 |
| delta_log_dispersion__1m__active__fh0 | train | 16439 | 15892 | 15892 | 547 | 0 |
| delta_log_dispersion__1m__active__fh0 | validation | 11608 | 6563 | 6563 | 5045 | 0 |
| delta_log_dispersion__1m__active__fh1 | test | 9056 | 8476 | 8476 | 580 | 0 |
| delta_log_dispersion__1m__active__fh1 | train | 15583 | 15067 | 15067 | 516 | 0 |
| delta_log_dispersion__1m__active__fh1 | validation | 11547 | 6438 | 6438 | 5109 | 0 |
| delta_log_dispersion__1m__active__fh2 | test | 7666 | 7290 | 7290 | 376 | 0 |
| delta_log_dispersion__1m__active__fh2 | train | 3981 | 3692 | 3692 | 289 | 0 |
| delta_log_dispersion__1m__active__fh2 | validation | 9755 | 5437 | 5437 | 4318 | 0 |
| delta_log_dispersion__1m__fixed__fh0 | test | 13634 | 12080 | 10716 | 1554 | 0 |
| delta_log_dispersion__1m__fixed__fh0 | train | 26057 | 24241 | 21070 | 1816 | 0 |
| delta_log_dispersion__1m__fixed__fh0 | validation | 16854 | 8777 | 7905 | 8077 | 0 |
| delta_log_dispersion__1m__fixed__fh1 | test | 13544 | 12059 | 10708 | 1485 | 0 |
| delta_log_dispersion__1m__fixed__fh1 | train | 25003 | 23203 | 19952 | 1800 | 0 |
| delta_log_dispersion__1m__fixed__fh1 | validation | 16738 | 8557 | 7738 | 8181 | 0 |
| delta_log_dispersion__1m__fixed__fh2 | test | 12150 | 10887 | 9466 | 1263 | 0 |
| delta_log_dispersion__1m__fixed__fh2 | train | 10362 | 9034 | 6218 | 1328 | 0 |
| delta_log_dispersion__1m__fixed__fh2 | validation | 15170 | 7716 | 6852 | 7454 | 0 |
| delta_log_dispersion__1m__market__fh0 | test | 13493 | 11956 | 10666 | 1537 | 0 |
| delta_log_dispersion__1m__market__fh0 | train | 25696 | 23894 | 20913 | 1802 | 0 |
| delta_log_dispersion__1m__market__fh0 | validation | 16754 | 8744 | 7891 | 8010 | 0 |
| delta_log_dispersion__1m__market__fh1 | test | 13412 | 11936 | 10658 | 1476 | 0 |
| delta_log_dispersion__1m__market__fh1 | train | 24641 | 22857 | 19781 | 1784 | 0 |
| delta_log_dispersion__1m__market__fh1 | validation | 16637 | 8530 | 7729 | 8107 | 0 |
| delta_log_dispersion__1m__market__fh2 | test | 12028 | 10771 | 9411 | 1257 | 0 |
| delta_log_dispersion__1m__market__fh2 | train | 10362 | 9034 | 6218 | 1328 | 0 |
| delta_log_dispersion__1m__market__fh2 | validation | 15076 | 7701 | 6841 | 7375 | 0 |
| delta_log_dispersion__3m__active__fh0 | test | 9361 | 0 | 0 | 9361 | 0 |
| delta_log_dispersion__3m__active__fh0 | train | 22464 | 7863 | 7863 | 14601 | 0 |
| delta_log_dispersion__3m__active__fh0 | validation | 15955 | 0 | 0 | 15955 | 0 |
| delta_log_dispersion__3m__active__fh1 | test | 10916 | 0 | 0 | 10916 | 0 |
| delta_log_dispersion__3m__active__fh1 | train | 21413 | 7224 | 7224 | 14189 | 0 |
| delta_log_dispersion__3m__active__fh1 | validation | 15819 | 0 | 0 | 15819 | 0 |
| delta_log_dispersion__3m__active__fh2 | test | 9147 | 0 | 0 | 9147 | 0 |
| delta_log_dispersion__3m__active__fh2 | train | 6875 | 525 | 525 | 6350 | 0 |
| delta_log_dispersion__3m__active__fh2 | validation | 14263 | 0 | 0 | 14263 | 0 |
| delta_log_dispersion__3m__fixed__fh0 | test | 13501 | 0 | 0 | 13501 | 0 |
| delta_log_dispersion__3m__fixed__fh0 | train | 26057 | 8672 | 8445 | 17385 | 0 |
| delta_log_dispersion__3m__fixed__fh0 | validation | 16854 | 0 | 0 | 16854 | 0 |
| delta_log_dispersion__3m__fixed__fh1 | test | 13544 | 0 | 0 | 13544 | 0 |
| delta_log_dispersion__3m__fixed__fh1 | train | 25003 | 8071 | 7841 | 16932 | 0 |
| delta_log_dispersion__3m__fixed__fh1 | validation | 16738 | 0 | 0 | 16738 | 0 |
| delta_log_dispersion__3m__fixed__fh2 | test | 12150 | 0 | 0 | 12150 | 0 |
| delta_log_dispersion__3m__fixed__fh2 | train | 10362 | 601 | 594 | 9761 | 0 |
| delta_log_dispersion__3m__fixed__fh2 | validation | 15170 | 0 | 0 | 15170 | 0 |
| delta_log_dispersion__3m__market__fh0 | test | 13154 | 0 | 0 | 13154 | 0 |
| delta_log_dispersion__3m__market__fh0 | train | 25492 | 8444 | 8299 | 17048 | 0 |
| delta_log_dispersion__3m__market__fh0 | validation | 16434 | 0 | 0 | 16434 | 0 |
| delta_log_dispersion__3m__market__fh1 | test | 13330 | 0 | 0 | 13330 | 0 |
| delta_log_dispersion__3m__market__fh1 | train | 24486 | 7854 | 7705 | 16632 | 0 |
| delta_log_dispersion__3m__market__fh1 | validation | 16326 | 0 | 0 | 16326 | 0 |
| delta_log_dispersion__3m__market__fh2 | test | 11947 | 0 | 0 | 11947 | 0 |
| delta_log_dispersion__3m__market__fh2 | train | 10360 | 601 | 594 | 9759 | 0 |
| delta_log_dispersion__3m__market__fh2 | validation | 14773 | 0 | 0 | 14773 | 0 |
| residual_signed_raw__fh0 | test | 20658 | 0 | 0 | 20658 | 0 |
| residual_signed_raw__fh0 | train | 39771 | 0 | 0 | 39771 | 0 |
| residual_signed_raw__fh0 | validation | 24979 | 0 | 0 | 24979 | 0 |
| residual_signed_raw__fh1 | test | 20350 | 0 | 0 | 20350 | 0 |
| residual_signed_raw__fh1 | train | 37936 | 0 | 0 | 37936 | 0 |
| residual_signed_raw__fh1 | validation | 24713 | 0 | 0 | 24713 | 0 |
| residual_signed_raw__fh2 | test | 0 | 0 | 0 | 0 | 0 |
| residual_signed_raw__fh2 | train | 0 | 0 | 0 | 0 | 0 |
| residual_signed_raw__fh2 | validation | 0 | 0 | 0 | 0 | 0 |

## Tasks with zero configured validation rows

| task_id | configured_valid_rows_in_feature_window | excluded_label_too_late | excluded_missing_or_invalid_label_date |
| --- | --- | --- | --- |
| delta_log_dispersion__3m__active__fh0 | 15955 | 15955 | 0 |
| delta_log_dispersion__3m__active__fh1 | 15819 | 15819 | 0 |
| delta_log_dispersion__3m__active__fh2 | 14263 | 14263 | 0 |
| delta_log_dispersion__3m__fixed__fh0 | 16854 | 16854 | 0 |
| delta_log_dispersion__3m__fixed__fh1 | 16738 | 16738 | 0 |
| delta_log_dispersion__3m__fixed__fh2 | 15170 | 15170 | 0 |
| delta_log_dispersion__3m__market__fh0 | 16434 | 16434 | 0 |
| delta_log_dispersion__3m__market__fh1 | 16326 | 16326 | 0 |
| delta_log_dispersion__3m__market__fh2 | 14773 | 14773 | 0 |
| residual_signed_raw__fh0 | 24979 | 24979 | 0 |
| residual_signed_raw__fh1 | 24713 | 24713 | 0 |
| residual_signed_raw__fh2 | 0 | 0 | 0 |

## Invalid reasons

| task_id | scope | reason | count |
| --- | --- | --- | --- |
| delta_log_dispersion__1m__active__fh0 | main_validity | insufficient_active_peers | 346370 |
| delta_log_dispersion__1m__active__fh1 | main_validity | insufficient_active_peers | 329327 |
| delta_log_dispersion__1m__fixed__fh0 | main_validity | report_inline_with_consensus | 301429 |
| delta_log_dispersion__1m__market__fh0 | main_validity | report_inline_with_consensus | 299275 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | report_inline_with_consensus | 297695 |
| delta_log_dispersion__3m__market__fh0 | main_validity | report_inline_with_consensus | 292265 |
| delta_log_dispersion__1m__fixed__fh1 | main_validity | report_inline_with_consensus | 279550 |
| delta_log_dispersion__3m__fixed__fh1 | main_validity | report_inline_with_consensus | 277852 |
| delta_log_dispersion__1m__market__fh1 | main_validity | report_inline_with_consensus | 277552 |
| delta_log_dispersion__3m__market__fh1 | main_validity | report_inline_with_consensus | 273663 |
| delta_log_dispersion__1m__active__fh2 | main_validity | insufficient_active_peers | 234614 |
| delta_log_dispersion__3m__active__fh0 | main_validity | report_inline_with_consensus | 225777 |
| delta_log_dispersion__3m__active__fh1 | main_validity | report_inline_with_consensus | 217781 |
| delta_log_dispersion__1m__market__fh0 | main_validity | insufficient_pre_peers | 215174 |
| residual_signed_raw__fh0 | main_validity | insufficient_pre_peers | 215174 |
| delta_log_dispersion__3m__market__fh0 | main_validity | insufficient_pre_peers | 215174 |
| delta_log_dispersion__3m__active__fh0 | main_validity | insufficient_pre_peers | 215174 |
| delta_log_dispersion__1m__fixed__fh0 | main_validity | insufficient_pre_peers | 215174 |
| delta_log_dispersion__1m__active__fh0 | main_validity | insufficient_pre_peers | 215174 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | insufficient_pre_peers | 215174 |
| delta_log_dispersion__3m__active__fh2 | main_validity | insufficient_pre_peers | 164343 |
| delta_log_dispersion__1m__fixed__fh2 | main_validity | insufficient_pre_peers | 164343 |
| delta_log_dispersion__1m__market__fh2 | main_validity | insufficient_pre_peers | 164343 |
| delta_log_dispersion__3m__fixed__fh2 | main_validity | insufficient_pre_peers | 164343 |
| residual_signed_raw__fh2 | main_validity | insufficient_pre_peers | 164343 |
| delta_log_dispersion__1m__active__fh2 | main_validity | insufficient_pre_peers | 164343 |
| delta_log_dispersion__3m__market__fh2 | main_validity | insufficient_pre_peers | 164343 |
| delta_log_dispersion__1m__fixed__fh2 | main_validity | report_inline_with_consensus | 154823 |
| delta_log_dispersion__1m__market__fh2 | main_validity | report_inline_with_consensus | 154112 |
| delta_log_dispersion__3m__fixed__fh2 | main_validity | report_inline_with_consensus | 153585 |
| delta_log_dispersion__3m__active__fh0 | main_validity | insufficient_active_peers | 152538 |
| delta_log_dispersion__3m__market__fh2 | main_validity | report_inline_with_consensus | 151297 |
| delta_log_dispersion__1m__fixed__fh0 | probe_validity | insufficient_peer_updates | 149142 |
| delta_log_dispersion__1m__active__fh0 | main_validity | report_inline_with_consensus | 148928 |
| delta_log_dispersion__3m__active__fh1 | main_validity | insufficient_pre_peers | 144385 |
| delta_log_dispersion__1m__market__fh1 | main_validity | insufficient_pre_peers | 144385 |
| residual_signed_raw__fh1 | main_validity | insufficient_pre_peers | 144385 |
| delta_log_dispersion__1m__active__fh1 | main_validity | insufficient_pre_peers | 144385 |
| delta_log_dispersion__1m__fixed__fh1 | main_validity | insufficient_pre_peers | 144385 |
| delta_log_dispersion__3m__fixed__fh1 | main_validity | insufficient_pre_peers | 144385 |
| delta_log_dispersion__3m__market__fh1 | main_validity | insufficient_pre_peers | 144385 |
| delta_log_dispersion__1m__market__fh0 | probe_validity | insufficient_peer_updates | 143677 |
| delta_log_dispersion__1m__fixed__fh1 | probe_validity | insufficient_peer_updates | 142782 |
| delta_log_dispersion__1m__market__fh1 | probe_validity | insufficient_peer_updates | 137844 |
| delta_log_dispersion__1m__active__fh1 | main_validity | report_inline_with_consensus | 132901 |
| delta_log_dispersion__3m__active__fh1 | main_validity | insufficient_active_peers | 118075 |
| delta_log_dispersion__1m__fixed__fh2 | probe_validity | insufficient_peer_updates | 111450 |
| delta_log_dispersion__3m__active__fh2 | main_validity | report_inline_with_consensus | 110629 |
| delta_log_dispersion__1m__market__fh2 | probe_validity | insufficient_peer_updates | 109704 |
| delta_log_dispersion__3m__active__fh2 | main_validity | insufficient_active_peers | 95014 |
| residual_signed_raw__fh2 | main_validity | actual_missing | 81150 |
| delta_log_dispersion__1m__active__fh2 | main_validity | report_inline_with_consensus | 60415 |
| delta_log_dispersion__3m__fixed__fh0 | probe_validity | insufficient_peer_updates | 59035 |
| delta_log_dispersion__3m__market__fh0 | probe_validity | insufficient_peer_updates | 48478 |
| residual_signed_raw__fh1 | main_validity | actual_missing | 42702 |
| delta_log_dispersion__3m__fixed__fh1 | probe_validity | insufficient_peer_updates | 41952 |
| delta_log_dispersion__3m__fixed__fh2 | probe_validity | insufficient_peer_updates | 37054 |
| delta_log_dispersion__3m__market__fh1 | probe_validity | insufficient_peer_updates | 34044 |
| delta_log_dispersion__3m__market__fh2 | probe_validity | insufficient_peer_updates | 32985 |
| delta_log_dispersion__3m__market__fh0 | main_validity | insufficient_future_peers | 20012 |
| delta_log_dispersion__3m__market__fh1 | main_validity | insufficient_future_peers | 15665 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | right_censored | 13706 |
| delta_log_dispersion__3m__active__fh0 | main_validity | right_censored | 13706 |
| delta_log_dispersion__3m__market__fh0 | main_validity | right_censored | 13706 |
| delta_log_dispersion__3m__active__fh1 | main_validity | right_censored | 13486 |
| delta_log_dispersion__3m__fixed__fh1 | main_validity | right_censored | 13486 |
| delta_log_dispersion__3m__market__fh1 | main_validity | right_censored | 13486 |
| delta_log_dispersion__3m__active__fh2 | main_validity | right_censored | 11646 |
| delta_log_dispersion__3m__fixed__fh2 | main_validity | right_censored | 11646 |
| delta_log_dispersion__3m__market__fh2 | main_validity | right_censored | 11646 |
| delta_log_dispersion__1m__fixed__fh0 | main_validity | right_censored | 9181 |
| delta_log_dispersion__1m__active__fh0 | main_validity | right_censored | 9181 |
| delta_log_dispersion__1m__market__fh0 | main_validity | right_censored | 9181 |
| delta_log_dispersion__1m__fixed__fh1 | main_validity | right_censored | 9126 |
| delta_log_dispersion__1m__active__fh1 | main_validity | right_censored | 9126 |
| delta_log_dispersion__1m__market__fh1 | main_validity | right_censored | 9126 |
| delta_log_dispersion__1m__market__fh0 | main_validity | insufficient_future_peers | 8381 |
| delta_log_dispersion__3m__market__fh2 | main_validity | insufficient_future_peers | 8322 |
| delta_log_dispersion__1m__market__fh2 | main_validity | right_censored | 8237 |
| delta_log_dispersion__1m__active__fh2 | main_validity | right_censored | 8237 |
| delta_log_dispersion__1m__fixed__fh2 | main_validity | right_censored | 8237 |
| delta_log_dispersion__1m__market__fh1 | main_validity | insufficient_future_peers | 7631 |
| delta_log_dispersion__3m__market__fh0 | main_validity | crosses_actual_disclosure | 4481 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | crosses_actual_disclosure | 4481 |
| delta_log_dispersion__3m__active__fh0 | main_validity | crosses_actual_disclosure | 4481 |
| delta_log_dispersion__3m__market__fh0 | main_validity | insufficient_scale_history | 4025 |
| delta_log_dispersion__1m__active__fh0 | main_validity | insufficient_scale_history | 4025 |
| residual_signed_raw__fh0 | main_validity | insufficient_scale_history | 4025 |
| delta_log_dispersion__1m__fixed__fh0 | main_validity | insufficient_scale_history | 4025 |
| delta_log_dispersion__3m__active__fh0 | main_validity | insufficient_scale_history | 4025 |
| delta_log_dispersion__1m__market__fh0 | main_validity | insufficient_scale_history | 4025 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | insufficient_scale_history | 4025 |
| delta_log_dispersion__1m__fixed__fh1 | main_validity | insufficient_scale_history | 2830 |
| delta_log_dispersion__3m__fixed__fh1 | main_validity | insufficient_scale_history | 2830 |
| residual_signed_raw__fh1 | main_validity | insufficient_scale_history | 2830 |
| delta_log_dispersion__1m__market__fh1 | main_validity | insufficient_scale_history | 2830 |
| delta_log_dispersion__3m__active__fh1 | main_validity | insufficient_scale_history | 2830 |
| delta_log_dispersion__3m__market__fh1 | main_validity | insufficient_scale_history | 2830 |
| delta_log_dispersion__1m__active__fh1 | main_validity | insufficient_scale_history | 2830 |
| delta_log_dispersion__1m__market__fh2 | main_validity | insufficient_future_peers | 2670 |

## Candidate-window interpretation

`candidate_window_counts.json` enumerates calendar-year and half-year feature cohorts with fixed 1–48 month maturity cutoffs. It is descriptive only and does not select a split automatically.
