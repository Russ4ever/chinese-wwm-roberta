# Continuous-label coverage audit

This report is outcome-blind: label value columns were not loaded.

## Source row counts

- `reports`: 93,731
- `report_fy_labels`: 258,161
- `report_confirmation_labels`: 1,548,966

## Task coverage and maturity

| task_id | source_rows | source_valid_rows | configured_valid_rows | positive_weight_valid_rows | feature_date_min | feature_date_max | label_date_min | label_date_max | maturity_days_p50 | maturity_days_p90 | maturity_days_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delta_log_dispersion__1m__active__fh0 | 93722 | 36855 | 36855 | 36855 | 2024-01-02 | 2024-12-31 | 2024-02-02 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__active__fh1 | 91556 | 36186 | 36186 | 36186 | 2024-01-02 | 2024-12-31 | 2024-02-02 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__active__fh2 | 72883 | 21402 | 21402 | 21402 | 2024-01-05 | 2024-12-31 | 2024-04-15 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__fixed__fh0 | 93722 | 56545 | 56545 | 48382 | 2024-01-02 | 2024-12-31 | 2024-02-02 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__fixed__fh1 | 91556 | 55285 | 55285 | 47467 | 2024-01-02 | 2024-12-31 | 2024-02-02 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__fixed__fh2 | 72883 | 37682 | 37682 | 30086 | 2024-01-05 | 2024-12-31 | 2024-02-29 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__market__fh0 | 93722 | 55943 | 55943 | 48134 | 2024-01-02 | 2024-12-31 | 2024-02-02 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__market__fh1 | 91556 | 54690 | 54690 | 47209 | 2024-01-02 | 2024-12-31 | 2024-02-02 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__1m__market__fh2 | 72883 | 37466 | 37466 | 29995 | 2024-01-05 | 2024-12-31 | 2024-02-29 | 2025-02-05 | 31.0 | 34.0 | 36.0 |
| delta_log_dispersion__3m__active__fh0 | 93722 | 47780 | 47780 | 47780 | 2024-01-02 | 2024-12-31 | 2024-04-02 | 2025-03-31 | 92.0 | 96.0 | 100.0 |
| delta_log_dispersion__3m__active__fh1 | 91556 | 48148 | 48148 | 48148 | 2024-01-02 | 2024-12-31 | 2024-04-02 | 2025-03-31 | 92.0 | 96.0 | 100.0 |
| delta_log_dispersion__3m__active__fh2 | 72883 | 30285 | 30285 | 30285 | 2024-01-05 | 2024-12-31 | 2024-04-30 | 2025-03-31 | 92.0 | 98.0 | 100.0 |
| delta_log_dispersion__3m__fixed__fh0 | 93722 | 56412 | 56412 | 53480 | 2024-01-02 | 2024-12-31 | 2024-04-02 | 2025-03-31 | 92.0 | 96.0 | 100.0 |
| delta_log_dispersion__3m__fixed__fh1 | 91556 | 55285 | 55285 | 53325 | 2024-01-02 | 2024-12-31 | 2024-04-02 | 2025-03-31 | 92.0 | 96.0 | 100.0 |
| delta_log_dispersion__3m__fixed__fh2 | 72883 | 37682 | 37682 | 35202 | 2024-01-05 | 2024-12-31 | 2024-04-30 | 2025-03-31 | 92.0 | 97.0 | 100.0 |
| delta_log_dispersion__3m__market__fh0 | 93722 | 55080 | 55080 | 52672 | 2024-01-02 | 2024-12-31 | 2024-04-02 | 2025-03-31 | 92.0 | 96.0 | 100.0 |
| delta_log_dispersion__3m__market__fh1 | 91556 | 54142 | 54142 | 52563 | 2024-01-02 | 2024-12-31 | 2024-04-02 | 2025-03-31 | 92.0 | 96.0 | 100.0 |
| delta_log_dispersion__3m__market__fh2 | 72883 | 37080 | 37080 | 34784 | 2024-01-05 | 2024-12-31 | 2024-04-30 | 2025-03-31 | 92.0 | 97.0 | 100.0 |
| residual_signed_raw__fh0 | 93722 | 85408 | 85408 | 85408 | 2024-01-02 | 2024-12-31 | 2025-01-25 | 2026-04-30 | 267.0 | 403.0 | 476.0 |
| residual_signed_raw__fh1 | 91556 | 82999 | 82999 | 82999 | 2024-01-02 | 2024-12-31 | 2026-01-31 | 2026-04-30 | 623.0 | 757.0 | 829.0 |
| residual_signed_raw__fh2 | 72883 | 0 | 0 | 0 | 2024-01-05 | 2024-12-31 |  |  |  |  |  |

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
| residual_signed_raw__fh2 | main_validity | actual_missing | 56596 |
| delta_log_dispersion__1m__active__fh0 | main_validity | insufficient_active_peers | 34664 |
| delta_log_dispersion__1m__active__fh1 | main_validity | insufficient_active_peers | 33833 |
| delta_log_dispersion__1m__fixed__fh0 | main_validity | report_inline_with_consensus | 28897 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | report_inline_with_consensus | 28794 |
| delta_log_dispersion__1m__market__fh0 | main_validity | report_inline_with_consensus | 28742 |
| delta_log_dispersion__3m__market__fh0 | main_validity | report_inline_with_consensus | 28438 |
| delta_log_dispersion__1m__fixed__fh1 | main_validity | report_inline_with_consensus | 27756 |
| delta_log_dispersion__3m__fixed__fh1 | main_validity | report_inline_with_consensus | 27756 |
| delta_log_dispersion__1m__active__fh2 | main_validity | insufficient_active_peers | 27650 |
| delta_log_dispersion__1m__market__fh1 | main_validity | report_inline_with_consensus | 27613 |
| delta_log_dispersion__3m__market__fh1 | main_validity | report_inline_with_consensus | 27426 |
| delta_log_dispersion__3m__active__fh0 | main_validity | report_inline_with_consensus | 21602 |
| delta_log_dispersion__3m__active__fh1 | main_validity | report_inline_with_consensus | 21365 |
| delta_log_dispersion__1m__fixed__fh2 | main_validity | report_inline_with_consensus | 18914 |
| delta_log_dispersion__3m__fixed__fh2 | main_validity | report_inline_with_consensus | 18914 |
| delta_log_dispersion__1m__market__fh2 | main_validity | report_inline_with_consensus | 18865 |
| delta_log_dispersion__3m__market__fh2 | main_validity | report_inline_with_consensus | 18725 |
| residual_signed_raw__fh2 | main_validity | insufficient_pre_peers | 16287 |
| delta_log_dispersion__3m__market__fh2 | main_validity | insufficient_pre_peers | 16287 |
| delta_log_dispersion__3m__fixed__fh2 | main_validity | insufficient_pre_peers | 16287 |
| delta_log_dispersion__1m__fixed__fh2 | main_validity | insufficient_pre_peers | 16287 |
| delta_log_dispersion__3m__active__fh2 | main_validity | insufficient_pre_peers | 16287 |
| delta_log_dispersion__1m__active__fh2 | main_validity | insufficient_pre_peers | 16287 |
| delta_log_dispersion__1m__market__fh2 | main_validity | insufficient_pre_peers | 16287 |
| delta_log_dispersion__1m__fixed__fh0 | probe_validity | insufficient_peer_updates | 15990 |
| delta_log_dispersion__3m__active__fh0 | main_validity | insufficient_active_peers | 15824 |
| delta_log_dispersion__1m__fixed__fh1 | probe_validity | insufficient_peer_updates | 15534 |
| delta_log_dispersion__1m__market__fh0 | probe_validity | insufficient_peer_updates | 15461 |
| delta_log_dispersion__1m__market__fh1 | probe_validity | insufficient_peer_updates | 15018 |
| delta_log_dispersion__1m__active__fh0 | main_validity | report_inline_with_consensus | 13923 |
| delta_log_dispersion__3m__active__fh1 | main_validity | insufficient_active_peers | 13528 |
| delta_log_dispersion__1m__fixed__fh2 | probe_validity | insufficient_peer_updates | 13470 |
| delta_log_dispersion__3m__active__fh2 | main_validity | report_inline_with_consensus | 13332 |
| delta_log_dispersion__1m__market__fh2 | probe_validity | insufficient_peer_updates | 13276 |
| delta_log_dispersion__1m__active__fh1 | main_validity | report_inline_with_consensus | 13022 |
| delta_log_dispersion__3m__active__fh2 | main_validity | insufficient_active_peers | 12979 |
| residual_signed_raw__fh1 | main_validity | insufficient_pre_peers | 8499 |
| delta_log_dispersion__1m__fixed__fh1 | main_validity | insufficient_pre_peers | 8499 |
| delta_log_dispersion__3m__market__fh1 | main_validity | insufficient_pre_peers | 8499 |
| delta_log_dispersion__1m__active__fh1 | main_validity | insufficient_pre_peers | 8499 |
| delta_log_dispersion__3m__active__fh1 | main_validity | insufficient_pre_peers | 8499 |
| delta_log_dispersion__3m__fixed__fh1 | main_validity | insufficient_pre_peers | 8499 |
| delta_log_dispersion__1m__market__fh1 | main_validity | insufficient_pre_peers | 8499 |
| delta_log_dispersion__1m__market__fh0 | main_validity | insufficient_pre_peers | 8276 |
| delta_log_dispersion__1m__fixed__fh0 | main_validity | insufficient_pre_peers | 8276 |
| delta_log_dispersion__3m__market__fh0 | main_validity | insufficient_pre_peers | 8276 |
| delta_log_dispersion__3m__active__fh0 | main_validity | insufficient_pre_peers | 8276 |
| delta_log_dispersion__1m__active__fh0 | main_validity | insufficient_pre_peers | 8276 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | insufficient_pre_peers | 8276 |
| residual_signed_raw__fh0 | main_validity | insufficient_pre_peers | 8276 |
| delta_log_dispersion__1m__active__fh2 | main_validity | report_inline_with_consensus | 7544 |
| delta_log_dispersion__3m__fixed__fh0 | probe_validity | insufficient_peer_updates | 6578 |
| delta_log_dispersion__3m__fixed__fh2 | probe_validity | insufficient_peer_updates | 5670 |
| delta_log_dispersion__3m__market__fh0 | probe_validity | insufficient_peer_updates | 5621 |
| delta_log_dispersion__3m__fixed__fh1 | probe_validity | insufficient_peer_updates | 5272 |
| delta_log_dispersion__3m__market__fh2 | probe_validity | insufficient_peer_updates | 5264 |
| delta_log_dispersion__3m__market__fh1 | probe_validity | insufficient_peer_updates | 4478 |
| delta_log_dispersion__3m__market__fh0 | main_validity | insufficient_future_peers | 1688 |
| delta_log_dispersion__3m__market__fh1 | main_validity | insufficient_future_peers | 1473 |
| delta_log_dispersion__3m__market__fh2 | main_validity | insufficient_future_peers | 791 |
| delta_log_dispersion__1m__market__fh0 | main_validity | insufficient_future_peers | 757 |
| delta_log_dispersion__1m__market__fh1 | main_validity | insufficient_future_peers | 738 |
| delta_log_dispersion__1m__market__fh2 | main_validity | insufficient_future_peers | 265 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | crosses_actual_disclosure | 236 |
| delta_log_dispersion__3m__active__fh0 | main_validity | crosses_actual_disclosure | 236 |
| delta_log_dispersion__3m__market__fh0 | main_validity | crosses_actual_disclosure | 236 |
| residual_signed_raw__fh1 | main_validity | actual_missing | 42 |
| residual_signed_raw__fh0 | main_validity | actual_missing | 34 |
| delta_log_dispersion__1m__fixed__fh1 | main_validity | conflicting_report_forecast | 16 |
| residual_signed_raw__fh1 | main_validity | conflicting_report_forecast | 16 |
| delta_log_dispersion__1m__active__fh1 | main_validity | conflicting_report_forecast | 16 |
| delta_log_dispersion__3m__market__fh1 | main_validity | conflicting_report_forecast | 16 |
| delta_log_dispersion__1m__market__fh1 | main_validity | conflicting_report_forecast | 16 |
| delta_log_dispersion__3m__active__fh1 | main_validity | conflicting_report_forecast | 16 |
| delta_log_dispersion__3m__fixed__fh1 | main_validity | conflicting_report_forecast | 16 |
| delta_log_dispersion__1m__active__fh0 | main_validity | conflicting_report_forecast | 4 |
| delta_log_dispersion__3m__market__fh0 | main_validity | conflicting_report_forecast | 4 |
| residual_signed_raw__fh0 | main_validity | conflicting_report_forecast | 4 |
| delta_log_dispersion__1m__market__fh0 | main_validity | conflicting_report_forecast | 4 |
| delta_log_dispersion__3m__fixed__fh0 | main_validity | conflicting_report_forecast | 4 |
| delta_log_dispersion__1m__fixed__fh0 | main_validity | conflicting_report_forecast | 4 |
| delta_log_dispersion__3m__active__fh0 | main_validity | conflicting_report_forecast | 4 |

## Candidate-window interpretation

`candidate_window_counts.json` enumerates calendar-year and half-year feature cohorts with fixed 1–48 month maturity cutoffs. It is descriptive only and does not select a split automatically.
