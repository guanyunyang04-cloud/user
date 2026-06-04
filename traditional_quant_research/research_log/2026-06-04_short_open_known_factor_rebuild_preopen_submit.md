# Short Open-Known Factor Rebuild

## Summary

- Profile: `preopen_submit`
- Years: `[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]`
- Event rows: 115311
- OOS trades: 64501
- Decision: `shortline_diagnostic_only`
- T+1 limit-up is a label only; buy features passed leakage audit.

## Candidate

| profile        |     n | mean_net_ret_pct   | win_rate   |   payoff | positive_year_rate   | min_year_mean_net_ret_pct   | evidence_grade            | failed_gates                                                                  |
|:---------------|------:|:-------------------|:-----------|---------:|:---------------------|:----------------------------|:--------------------------|:------------------------------------------------------------------------------|
| preopen_submit | 64501 | 1.59%              | 34.9%      |      2.8 | 88.9%                | -3.44%                      | shortline_diagnostic_only | incomplete_oos_years,negative_worst_year,contains_unfilled_near_limit_entries |

## OOS Yearly

|   eval_year |     n | mean_net_ret_pct   | median_net_ret_pct   | win_rate   |   payoff |
|------------:|------:|:-------------------|:---------------------|:-----------|---------:|
|        2017 |  6606 | 7.83%              | 0.33%                | 50.9%      |     4.26 |
|        2018 |  7200 | 0.57%              | -5.30%               | 30.9%      |     2.62 |
|        2019 |  9659 | 0.95%              | -5.30%               | 32.6%      |     2.66 |
|        2020 | 12079 | 1.50%              | -5.30%               | 35.0%      |     2.73 |
|        2021 | 14333 | 0.91%              | -5.30%               | 34.3%      |     2.47 |
|        2022 | 14585 | 0.43%              | -5.30%               | 31.7%      |     2.43 |
|        2023 |     3 | -3.44%             | -5.30%               | 33.3%      |     0.06 |
|        2024 |    26 | 6.14%              | -0.83%               | 50.0%      |     3.38 |
|        2025 |    10 | 5.20%              | -5.30%               | 30.0%      |     5.6  |

## Portfolio Top-K

|   top_k |   periods | mean_period_net_ret_pct   | median_period_net_ret_pct   | win_rate   |   mean_trade_count |
|--------:|----------:|:--------------------------|:----------------------------|:-----------|-------------------:|
|       1 |      1481 | 2.06%                     | -5.30%                      | 32.8%      |            1       |
|       3 |      1481 | 1.66%                     | -0.35%                      | 48.6%      |            2.97569 |
|       5 |      1481 | 1.48%                     | 0.98%                       | 54.2%      |            4.94801 |
|      10 |      1481 | 1.31%                     | 0.54%                       | 54.8%      |            9.86293 |

## Factor Diagnostics Snapshot

| feature                      | bucket   |     n | entry_limit_up_rate   |   sell1_close_mean |   sell3_close_mean |   sell20_close_mean |   sell20_max_high_mean |
|:-----------------------------|:---------|------:|:----------------------|-------------------:|-------------------:|--------------------:|-----------------------:|
| one_word_limit_like          | True     | 15143 | 74.4%                 |           3.65963  |            9.59174 |            21.0924  |                50.3019 |
| near_one_word_limit_like     | True     | 15757 | 72.7%                 |           3.49578  |            9.1743  |            20.1514  |                49.0321 |
| entry_open_near_limit        | True     | 19357 | 83.1%                 |           3.34164  |            8.77192 |            19.2665  |                49.1045 |
| signal_range_pct             | low20    | 23063 | 58.5%                 |           2.08644  |            5.71412 |            12.45    |                38.6901 |
| signal_day_ret_from_open_pct | low20    | 23063 | 59.4%                 |           2.09506  |            5.61597 |            12.3364  |                39.05   |
| signal_open_gap_pct          | high20   | 22861 | 58.8%                 |           1.99116  |            5.31744 |            11.2725  |                37.8207 |
| kama_bias_0_3                | True     |  5566 | 27.7%                 |           1.79864  |            4.96938 |            15.0022  |                35.2067 |
| ma60_trend_positive          | False    | 35529 | 33.5%                 |           1.47356  |            4.11464 |            10.5035  |                30.3504 |
| bias_kama_pct                | high20   | 23063 | 49.4%                 |           1.26026  |            3.1821  |             5.68668 |                33.5698 |
| high_cross_atr_upper         | False    | 46593 | 24.6%                 |           1.01066  |            2.5364  |             6.11494 |                25.2571 |
| close_cross_atr_upper        | False    | 46595 | 24.5%                 |           1.01075  |            2.53619 |             6.11464 |                25.2562 |
| kama_slope_pct               | low20    | 22861 | 21.7%                 |           0.993833 |            2.38499 |             6.46081 |                23.9356 |

## Notes

- One-word and near-limit paths remain diagnostic unless the candidate summary proves executable gates.
- `open_print_filter` may be useful operationally, but is reported as open-known diagnostic rather than full pre-open evidence.