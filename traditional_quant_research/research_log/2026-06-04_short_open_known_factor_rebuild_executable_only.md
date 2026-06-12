# Short Open-Known Factor Rebuild

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_short_open_known_factor_rebuild_executable_only.md`.


## Summary

- Profile: `executable_only`
- Years: `[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]`
- Event rows: 114457
- OOS trades: 15250
- Executable event rate: 79.0%
- Min signal amount: 20000000.0
- Stable cache: `True`
- Train start year: `2016`
- Event panel write mode: `selected_years`
- Decision: `open_known_diagnostic`
- T+1 limit-up is a label only; buy features passed leakage audit.

## Candidate

| profile         |     n | mean_net_ret_pct   | win_rate   |   payoff | positive_year_rate   | min_year_mean_net_ret_pct   | evidence_grade        | failed_gates                                                                                     |
|:----------------|------:|:-------------------|:-----------|---------:|:---------------------|:----------------------------|:----------------------|:-------------------------------------------------------------------------------------------------|
| executable_only | 15250 | -0.14%             | 44.4%      |     1.17 | 50.0%                | -1.29%                      | open_known_diagnostic | nonpositive_mean,positive_year_rate_lt_0.8,negative_worst_year,payoff_lt_2,open_known_diagnostic |

## OOS Yearly

|   eval_year |    n | mean_net_ret_pct   | median_net_ret_pct   | win_rate   |   payoff |
|------------:|-----:|:-------------------|:---------------------|:-----------|---------:|
|        2017 | 1141 | -1.29%             | -2.50%               | 35.1%      |     1.16 |
|        2018 | 1708 | -0.58%             | -1.23%               | 37.8%      |     1.27 |
|        2019 | 1010 | -0.27%             | -0.88%               | 42.9%      |     1.18 |
|        2020 | 1430 | 0.16%              | -0.18%               | 47.5%      |     1.2  |
|        2021 | 2078 | -0.07%             | -0.59%               | 44.9%      |     1.19 |
|        2022 | 2208 | 0.17%              | -0.30%               | 47.7%      |     1.19 |
|        2023 | 1219 | 0.03%              | -0.37%               | 46.3%      |     1.18 |
|        2024 | 1826 | 0.18%              | -0.34%               | 46.8%      |     1.23 |
|        2025 | 1678 | 0.06%              | -0.42%               | 46.8%      |     1.17 |
|        2026 |  952 | -0.27%             | -0.67%               | 45.0%      |     1.08 |

## Portfolio Top-K

|   top_k |   periods | mean_period_net_ret_pct   | median_period_net_ret_pct   | win_rate   |   mean_trade_count |   mean_utilization | cumulative_ret_pct   | max_drawdown_pct   |   longest_losing_streak |
|--------:|----------:|:--------------------------|:----------------------------|:-----------|-------------------:|-------------------:|:---------------------|:-------------------|------------------------:|
|       1 |      2154 | -0.41%                    | -1.14%                      | 40.9%      |            1       |           1        | -100.00%             | -100.00%           |                      14 |
|       3 |      2154 | -0.31%                    | -0.56%                      | 44.4%      |            2.71681 |           0.905602 | -99.98%              | -99.99%            |                      11 |
|       5 |      2154 | -0.31%                    | -0.51%                      | 44.1%      |            3.9935  |           0.7987   | -99.97%              | -99.98%            |                      13 |

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
| window_close |             3 |          96 | 0.00%                   | -0.30%                    |       2.01056 |                 3       | 10.0%                  | 32.9%                |
| kama_break   |             1 |          96 | -0.01%                  | -0.20%                    |       1.57274 |                 1       | 5.1%                   | 19.3%                |
| ma10_break   |             1 |          96 | -0.01%                  | -0.20%                    |       1.57274 |                 1       | 5.1%                   | 19.3%                |
| ma5_break    |             1 |          96 | -0.01%                  | -0.20%                    |       1.57274 |                 1       | 5.1%                   | 19.3%                |
| upper_shadow |             1 |          96 | -0.01%                  | -0.20%                    |       1.57274 |                 1       | 5.1%                   | 19.3%                |
| weak_close   |             1 |          96 | -0.01%                  | -0.20%                    |       1.57274 |                 1       | 5.1%                   | 19.3%                |
| window_close |             1 |          96 | -0.01%                  | -0.20%                    |       1.57274 |                 1       | 5.1%                   | 19.3%                |
| ma10_break   |             3 |          96 | -0.02%                  | -0.31%                    |       2.01133 |                 2.7345  | 10.0%                  | 30.5%                |
| kama_break   |             3 |          96 | -0.03%                  | -0.30%                    |       2.01371 |                 2.70309 | 10.0%                  | 30.1%                |
| weak_close   |             3 |          96 | -0.03%                  | -0.24%                    |       1.97256 |                 2.10283 | 9.7%                   | 23.3%                |
| upper_shadow |             3 |          96 | -0.03%                  | -0.29%                    |       1.96799 |                 2.56207 | 9.8%                   | 28.8%                |
| ma5_break    |             3 |          96 | -0.04%                  | -0.28%                    |       2.07189 |                 2.51674 | 10.1%                  | 27.6%                |

## Notes

- One-word and near-limit paths remain diagnostic unless the candidate summary proves executable gates.
- `open_print_filter` may be useful operationally, but is reported as open-known diagnostic rather than full pre-open evidence.