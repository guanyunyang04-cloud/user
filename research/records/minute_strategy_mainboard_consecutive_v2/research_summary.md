# Main-board consecutive limit-up filter study

The existing minute samples are already point-in-time Shanghai/Shenzhen main-board data. This study adds only a causal prior-day limit-up streak label; it does not regenerate minute events.

## Label contract

`prior_limit_up_streak` counts completed prior trading days whose unadjusted close equals the reconstructed upper limit. Ordinary main-board shares use 10%, ST shares 5%, the theoretical limit is rounded to two decimals, and half a tick is tolerated. The signal-day bar is not used. A touched-but-not-closed flag is retained for sensitivity only.

`control_comparison.csv` is a same-filter cross-sectional baseline comparison. It is weaker than the matched-control test in the main existing-sample report and must not be read as causal proof.

Samples: primary_development_2022_2023_partial, supplemental_2023_09_legacy
Label rows: 1,041,401; unobserved daily-label rows: 0; non-main audit rows: 0; unresolved board rows: 0.

## Reading the result

A better conditional mean on `geN` can be a narrow high-volatility selection effect. Require enough events, positive day-level consistency, a positive control difference, and a fresh forward sample before treating it as a candidate.

## Highest conditional rows (descriptive only)

| Sample | Strategy | MA | Filter | Horizon | N | Mean | Median | Win rate |
|---|---|---:|---|---|---:|---:|---:|---:|
| primary_development_2022_2023_partial | ng_r1_auction_reclaim | 20 | ge3 | 5d | 1 | 11.758% | n/a | 100.00% |
| primary_development_2022_2023_partial | s1_near_reversal | 20 | ge3 | 5d | 1 | 11.758% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_reclaim_bull_stack | 240 | ge1 | 5d | 2 | 9.104% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_reclaim_bull_stack | 120 | ge1 | 5d | 2 | 8.830% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_first_touch_reclaim | 20 | ge3 | 5d | 1 | 7.905% | n/a | 100.00% |
| primary_development_2022_2023_partial | ht_r3_core_pullback | 120 | ge3 | 1d | 1 | 6.309% | n/a | 100.00% |
| primary_development_2022_2023_partial | cs_r3_leader_confirmation | 120 | ge3 | 1d | 2 | 6.047% | n/a | 100.00% |
| primary_development_2022_2023_partial | fu_r3_core_pullback | 120 | ge3 | 1d | 2 | 6.047% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_first_touch_reclaim | 20 | ge3 | 60m | 1 | 6.006% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_first_touch_reclaim | 20 | ge3 | 1d | 1 | 6.006% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_reclaim_daily_trend | 120 | ge2 | 5d | 17 | 5.084% | n/a | 64.71% |
| primary_development_2022_2023_partial | s1_break_reclaim_stable3 | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s1_touch_immediate | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s1_touch_reclaim | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s2_reclaim_breakout_recent | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s2_reclaim_bull_stack | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s2_reclaim_daily_trend | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s2_reclaim_positive_slope | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s4_auction_confirmed_reclaim | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s4_reclaim_amount_acceleration | 20 | ge3 | 5d | 3 | 3.649% | n/a | 66.67% |
| primary_development_2022_2023_partial | s4_reclaim_vwap_support | 40 | ge2 | 1d | 25 | 3.527% | n/a | 80.00% |
| primary_development_2022_2023_partial | s2_reclaim_breakout_recent | 120 | ge2 | 5d | 10 | 3.367% | n/a | 50.00% |
| primary_development_2022_2023_partial | s2_reclaim_acceleration | 40 | ge3 | 5d | 3 | 3.175% | n/a | 100.00% |
| primary_development_2022_2023_partial | s4_reclaim_volume_proxy | 40 | ge3 | 5d | 3 | 2.889% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_reclaim_bull_stack | 240 | ge1 | 1d | 2 | 2.834% | n/a | 100.00% |
| primary_development_2022_2023_partial | s4_reclaim_vwap_support | 40 | ge3 | 5d | 3 | 2.719% | n/a | 100.00% |
| primary_development_2022_2023_partial | ng_r1_auction_reclaim | 60 | ge2 | 5d | 9 | 2.712% | n/a | 44.44% |
| primary_development_2022_2023_partial | s2_reclaim_acceleration | 40 | ge2 | 1d | 16 | 2.692% | n/a | 75.00% |
| primary_development_2022_2023_partial | s2_reclaim_bull_stack | 120 | ge1 | 1d | 2 | 2.576% | n/a | 100.00% |
| primary_development_2022_2023_partial | s2_reclaim_acceleration | 40 | ge3 | 1d | 3 | 2.525% | n/a | 100.00% |
