# Short Open-Known Factor Rebuild

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_short_open_known_factor_rebuild_open_print_filter.md`.


## Summary

- Profile: `open_print_filter`
- Years: `[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]`
- Event rows: 114457
- OOS trades: 85990
- Executable event rate: 79.0%
- Min signal amount: 20000000.0
- Stable cache: `True`
- Train start year: `2016`
- Event panel write mode: `selected_years`
- Decision: `open_known_diagnostic`
- T+1 limit-up is a label only; buy features passed leakage audit.

## Candidate

| profile           |     n | mean_net_ret_pct   | win_rate   |   payoff | positive_year_rate   | min_year_mean_net_ret_pct   | evidence_grade        | failed_gates                                                                    |
|:------------------|------:|:-------------------|:-----------|---------:|:---------------------|:----------------------------|:----------------------|:--------------------------------------------------------------------------------|
| open_print_filter | 85990 | 1.46%              | 35.6%      |     2.73 | 100.0%               | 0.16%                       | open_known_diagnostic | incomplete_oos_years,contains_unfilled_near_limit_entries,open_known_diagnostic |

## OOS Yearly

|   eval_year |     n | mean_net_ret_pct   | median_net_ret_pct   | win_rate   |   payoff |
|------------:|------:|:-------------------|:---------------------|:-----------|---------:|
|        2017 |  6478 | 8.09%              | 0.72%                | 52.2%      |     4.56 |
|        2018 |  7120 | 0.88%              | -4.24%               | 32.7%      |     2.66 |
|        2019 |  9517 | 1.10%              | -4.55%               | 34.2%      |     2.63 |
|        2020 | 11819 | 1.66%              | -3.53%               | 36.5%      |     2.76 |
|        2021 | 14080 | 0.99%              | -3.57%               | 35.7%      |     2.43 |
|        2022 | 14423 | 0.69%              | -4.12%               | 33.9%      |     2.41 |
|        2023 |  7609 | 0.16%              | -3.62%               | 32.1%      |     2.23 |
|        2024 | 14937 | 0.78%              | -5.30%               | 33.0%      |     2.54 |
|        2025 |     7 | 5.65%              | -2.30%               | 28.6%      |     7.48 |

## Portfolio Top-K

|   top_k |   periods | mean_period_net_ret_pct   | median_period_net_ret_pct   | win_rate   |   mean_trade_count |   mean_utilization | cumulative_ret_pct   | max_drawdown_pct   |   longest_losing_streak |
|--------:|----------:|:--------------------------|:----------------------------|:-----------|-------------------:|-------------------:|:---------------------|:-------------------|------------------------:|
|       1 |      1918 | 2.28%                     | -5.30%                      | 34.4%      |            1       |           1        | 3856096859487086.00% | -85.85%            |                      23 |
|       3 |      1918 | 1.91%                     | -0.26%                      | 48.7%      |            2.9927  |           0.997567 | 5234065152892841.00% | -65.97%            |                      15 |
|       5 |      1918 | 1.72%                     | 1.11%                       | 55.3%      |            4.98488 |           0.996976 | 839582153451482.88%  | -56.57%            |                      14 |

## Factor Diagnostics Snapshot

| feature                      | bucket   |     n | entry_limit_up_rate   |   sell1_close_mean |   sell3_close_mean |   sell20_close_mean |   sell20_max_high_mean |
|:-----------------------------|:---------|------:|:----------------------|-------------------:|-------------------:|--------------------:|-----------------------:|
| one_word_limit_like          | True     | 15040 | 74.4%                 |            3.65284 |            9.59799 |            21.9205  |                50.8745 |
| near_one_word_limit_like     | True     | 15652 | 72.7%                 |            3.48923 |            9.17994 |            20.9366  |                49.5571 |
| entry_open_near_limit        | True     | 19208 | 83.0%                 |            3.33612 |            8.77013 |            20.0961  |                49.544  |
| next_open_gap_pct            | high20   | 22897 | 77.4%                 |            2.40715 |            6.82172 |            15.9783  |                44.2626 |
| signal_range_pct             | low20    | 22892 | 58.5%                 |            2.08158 |            5.71772 |            13.0969  |                39.0454 |
| signal_day_ret_from_open_pct | low20    | 22892 | 59.4%                 |            2.08619 |            5.60893 |            13.0187  |                39.392  |
| signal_open_gap_pct          | high20   | 22684 | 58.7%                 |            1.98535 |            5.31125 |            11.9367  |                38.1282 |
| kama_bias_0_3                | True     |  5559 | 27.8%                 |            1.79948 |            4.95078 |            15.3007  |                35.3015 |
| ma60_trend_positive          | False    | 35400 | 33.4%                 |            1.46906 |            4.09323 |            10.649   |                30.3533 |
| bias_kama_pct                | high20   | 22892 | 49.3%                 |            1.25313 |            3.18533 |             6.37529 |                33.8626 |
| high_cross_atr_upper         | False    | 46262 | 24.5%                 |            1.00541 |            2.53748 |             6.56668 |                25.4275 |
| close_cross_atr_upper        | False    | 46264 | 24.5%                 |            1.00551 |            2.53727 |             6.56633 |                25.4265 |

## Factor Interaction Snapshot

| feature_a                | feature_b              |     n | entry_limit_up_rate   |   sell1_close_mean |   sell3_close_mean |   sell20_close_mean |   sell20_max_high_mean |
|:-------------------------|:-----------------------|------:|:----------------------|-------------------:|-------------------:|--------------------:|-----------------------:|
| ma_compression_tight     | second_plus_board      |  6361 | 61.8%                 |           3.11015  |            8.93968 |            24.3335  |                51.8445 |
| kama_slope_turn_positive | second_plus_board      |  2611 | 57.0%                 |           2.89785  |            8.26421 |            25.1072  |                51.9103 |
| second_plus_board        | industry_heat_positive | 27014 | 51.9%                 |           1.32871  |            3.39014 |             7.34789 |                34.3058 |
| kama_slope_positive      | second_plus_board      | 30325 | 51.2%                 |           1.21262  |            3.08859 |             6.61594 |                33.4802 |
| second_plus_board        | open_above_kama        | 30981 | 50.8%                 |           1.20274  |            3.06972 |             6.59343 |                33.2556 |
| second_plus_board        | market_heat_positive   | 31234 | 50.6%                 |           1.19614  |            3.04246 |             6.56157 |                33.1617 |
| moderate_volume          | second_plus_board      | 10217 | 43.6%                 |           0.665868 |            2.09131 |             4.6507  |                30.0686 |
| ma_compression_tight     | industry_heat_positive | 22491 | 28.2%                 |           0.606004 |            2.03478 |             5.44156 |                24.0398 |
| kama_bias_0_8            | industry_heat_positive | 16123 | 18.9%                 |           0.600767 |            1.60775 |             4.70612 |                21.7767 |
| kama_slope_positive      | ma_compression_tight   | 28423 | 26.9%                 |           0.392552 |            1.50453 |             3.974   |                22.1614 |
| ma_compression_tight     | open_above_kama        | 31589 | 25.7%                 |           0.385794 |            1.39366 |             3.65467 |                21.5172 |
| ma_compression_tight     | market_heat_positive   | 32340 | 25.3%                 |           0.387102 |            1.37298 |             3.60645 |                21.3916 |

## Sell Rule Diagnostics

| exit_rule    |   sell_window |   grid_rows | best_mean_net_ret_pct   | median_mean_net_ret_pct   |   best_payoff |   mean_avg_holding_days | mean_target_hit_rate   | mean_stop_hit_rate   |
|:-------------|--------------:|------------:|:------------------------|:--------------------------|--------------:|------------------------:|:-----------------------|:---------------------|
| kama_break   |            20 |         384 | 2.14%                   | -0.56%                    |       4.60379 |                 8.41139 | 21.0%                  | 44.0%                |
| upper_shadow |            20 |         384 | 1.72%                   | -0.47%                    |       3.33398 |                 6.57795 | 15.6%                  | 39.9%                |
| ma10_break   |            20 |         384 | 1.59%                   | -0.52%                    |       3.77146 |                 6.74095 | 19.4%                  | 40.6%                |
| weak_close   |            20 |         384 | 1.56%                   | -0.26%                    |       3.06178 |                 2.91762 | 14.2%                  | 26.3%                |
| kama_break   |            10 |         384 | 1.56%                   | -0.56%                    |       4.25928 |                 6.44175 | 18.3%                  | 43.0%                |
| upper_shadow |            10 |         384 | 1.54%                   | -0.48%                    |       3.30862 |                 5.40335 | 15.0%                  | 38.9%                |
| weak_close   |             3 |         384 | 1.53%                   | -0.18%                    |       2.83804 |                 2.10739 | 11.7%                  | 25.6%                |
| weak_close   |             5 |         384 | 1.48%                   | -0.29%                    |       3.20568 |                 2.58408 | 13.2%                  | 26.4%                |
| ma5_break    |            20 |         384 | 1.48%                   | -0.34%                    |       3.22043 |                 3.65678 | 16.1%                  | 30.8%                |
| weak_close   |            10 |         384 | 1.46%                   | -0.30%                    |       3.08301 |                 2.87574 | 14.0%                  | 26.4%                |
| window_close |            20 |         384 | 1.45%                   | -1.35%                    |       4.57333 |                20       | 19.3%                  | 67.3%                |
| kama_break   |             5 |         384 | 1.45%                   | -0.45%                    |       3.55883 |                 4.08643 | 14.4%                  | 38.4%                |

## Notes

- One-word and near-limit paths remain diagnostic unless the candidate summary proves executable gates.
- `open_print_filter` may be useful operationally, but is reported as open-known diagnostic rather than full pre-open evidence.