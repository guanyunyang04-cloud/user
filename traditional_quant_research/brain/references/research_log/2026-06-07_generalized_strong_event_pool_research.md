# Generalized Strong-Event Pool Research

- run_id: `generalized_strong_event_pool_research_20260607_222241`
- event_count: `312573`
- prediction_count: `267659`
- profile: `open_print_filter`
- feature_column_count: `83`
- best_strategy_id: `all_strong_events__generalized_lgbm_rank__pos1`
- best_practical_strategy_id: `executable_only__generalized_lgbm_rank__pos1`
- decision: `generalized_strong_event_pool_diagnostic`
- strategy_candidate_count: `0`

## Event Pools

| pool_name           |   event_count |   mean_net_ret_pct |   median_net_ret_pct |   win_rate |   big_loss_rate |   near_limit_open_rate |   executable_rate |   limit_up_share |
|:--------------------|--------------:|-------------------:|---------------------:|-----------:|----------------:|-----------------------:|------------------:|-----------------:|
| all_strong_events   |        312573 |          -0.193783 |            -0.6125   |   0.443987 |       0.166509  |              0.0612721 |          0.924411 |         0.340436 |
| executable_only     |        288946 |          -0.25479  |            -0.667959 |   0.434605 |       0.152739  |              0         |          1        |         0.293944 |
| limit_up_core       |        106411 |          -0.182673 |            -0.881959 |   0.454693 |       0.266365  |              0.159777  |          0.798169 |         1        |
| non_limit_strong    |        206162 |          -0.19951  |            -0.553807 |   0.438468 |       0.115027  |              0.0104287 |          0.989571 |         0        |
| near_limit          |         40809 |          -0.427809 |            -0.577689 |   0.451989 |       0.178878  |              0.036487  |          0.963513 |         0        |
| volume_atr_breakout |         67586 |          -0.566534 |            -0.849451 |   0.408777 |       0.151464  |              0.0263664 |          0.973634 |         0        |
| new_high_breakout   |        119102 |          -0.358092 |            -0.695257 |   0.423119 |       0.129897  |              0.0161962 |          0.983804 |         0        |
| kama_breakout       |         74236 |          -0.127936 |            -0.400351 |   0.449059 |       0.0801709 |              0.0015087 |          0.998491 |         0        |

## Event Types

| primary_event_type   |   event_count |   mean_net_ret_pct |   median_net_ret_pct |   win_rate |   big_loss_rate |   near_limit_open_rate |   executable_rate |   mean_event_strength_score |
|:---------------------|--------------:|-------------------:|---------------------:|-----------:|----------------:|-----------------------:|------------------:|----------------------------:|
| new_high_breakout    |         40978 |          -0.108803 |            -0.580899 |   0.432199 |       0.0952602 |             0.00202548 |          0.997975 |                     7.96661 |
| kama_breakout        |         41392 |          -0.123026 |            -0.38489  |   0.447908 |       0.0704464 |             0.00135292 |          0.998647 |                     7.50191 |
| big_up               |         47908 |          -0.128662 |            -0.542248 |   0.443219 |       0.118451  |             0.00576104 |          0.994239 |                    10.255   |
| limit_up_core        |        106411 |          -0.182673 |            -0.881959 |   0.454693 |       0.266365  |             0.159777   |          0.798169 |                    15.7815  |
| volume_atr_breakout  |         33207 |          -0.215539 |            -0.713936 |   0.41313  |       0.109597  |             0.00722739 |          0.992773 |                     9.54979 |
| near_limit           |         40809 |          -0.427809 |            -0.577689 |   0.451989 |       0.178878  |             0.036487   |          0.963513 |                    13.2661  |
| trend_accel          |          1868 |          -0.42918  |            -1.01634  |   0.399786 |       0.150589  |             0.00321199 |          0.996788 |                     7.7214  |

## Portfolio Ranking

| strategy_id                                      |   trade_count |   period_count |   mean_period_net_ret_pct |   median_period_net_ret_pct |   period_win_rate |   positive_year_rate |   min_year_period_ret_pct |   p10_trade_net_ret_pct |   p90_trade_net_ret_pct |   near_limit_open_rate |   executable_rate |
|:-------------------------------------------------|--------------:|---------------:|--------------------------:|----------------------------:|------------------:|---------------------:|--------------------------:|------------------------:|------------------------:|-----------------------:|------------------:|
| all_strong_events__generalized_lgbm_rank__pos1   |           891 |            891 |                 4.68953   |                    9.6872   |          0.768799 |                1     |                  1.37446  |                -9.64328 |                 9.72812 |              0.892256  |         0.106622  |
| limit_up_core__generalized_lgbm_rank__pos1       |           891 |            891 |                 4.63986   |                    9.68926  |          0.771044 |                1     |                  1.26187  |               -10.2806  |                 9.72849 |              0.929293  |         0.0695847 |
| limit_up_core__generalized_lgbm_rank__pos2       |          1782 |            891 |                 3.70733   |                    5.3473   |          0.709315 |                1     |                  0.787259 |               -10.2883  |                 9.72548 |              0.872615  |         0.122334  |
| all_strong_events__generalized_lgbm_rank__pos2   |          1782 |            891 |                 3.67902   |                    5.07599  |          0.718294 |                1     |                  0.986886 |               -10.217   |                 9.72428 |              0.815937  |         0.180696  |
| new_high_breakout__generalized_lgbm_rank__pos1   |           891 |            891 |                -0.0270667 |                   -0.3      |          0.452301 |                0.375 |                 -0.345112 |                -4.0218  |                 4.14507 |              0.0673401 |         0.93266   |
| kama_breakout__generalized_lgbm_rank__pos2       |          1776 |           1017 |                -0.0341477 |                   -0.250884 |          0.46116  |                0.5   |                 -0.632661 |                -3.9036  |                 4.17626 |              0.018018  |         0.981982  |
| non_limit_strong__generalized_lgbm_rank__pos1    |           891 |            891 |                -0.0460787 |                   -0.463043 |          0.434343 |                0.5   |                 -0.506875 |                -4.16068 |                 4.74808 |              0.0875421 |         0.912458  |
| kama_breakout__generalized_lgbm_rank__pos1       |           889 |            889 |                -0.0807127 |                   -0.348792 |          0.43982  |                0.5   |                 -0.7547   |                -3.89688 |                 3.96811 |              0.0247469 |         0.975253  |
| non_limit_strong__generalized_lgbm_rank__pos2    |          1782 |            891 |                -0.180914  |                   -0.285555 |          0.43771  |                0.25  |                 -0.502974 |                -4.16065 |                 4.28475 |              0.0533109 |         0.946689  |
| volume_atr_breakout__generalized_lgbm_rank__pos1 |           890 |            890 |                -0.201353  |                   -0.637468 |          0.410112 |                0.25  |                 -0.430568 |                -4.1894  |                 4.34192 |              0.0494382 |         0.950562  |
| executable_only__generalized_lgbm_rank__pos1     |           891 |            891 |                -0.206899  |                   -0.582031 |          0.414141 |                0.25  |                 -0.650766 |                -4.16039 |                 3.9508  |              0         |         1         |
| near_limit__generalized_lgbm_rank__pos1          |           891 |            891 |                -0.259841  |                   -0.626531 |          0.4422   |                0.25  |                 -0.647179 |                -5.14765 |                 5.37544 |              0.0179574 |         0.982043  |
| executable_only__generalized_lgbm_rank__pos2     |          1782 |            891 |                -0.260888  |                   -0.354771 |          0.421998 |                0.125 |                 -0.496083 |                -4.10736 |                 3.91503 |              0         |         1         |
| new_high_breakout__generalized_lgbm_rank__pos2   |          1781 |           1111 |                -0.271473  |                   -0.329558 |          0.438344 |                0     |                 -0.43282  |                -4.27727 |                 3.92442 |              0.0443571 |         0.955643  |
| volume_atr_breakout__generalized_lgbm_rank__pos2 |          1779 |           1121 |                -0.276545  |                   -0.468074 |          0.432649 |                0.125 |                 -0.513637 |                -4.26484 |                 4.15479 |              0.0337268 |         0.966273  |
| near_limit__generalized_lgbm_rank__pos2          |          1776 |           1229 |                -0.470174  |                   -0.537812 |          0.441009 |                0.125 |                 -0.917359 |                -5.57516 |                 5.13246 |              0.0146396 |         0.98536   |

## Practical Yearly

| strategy_id                                  |   year |   trade_count |   period_count |   mean_trade_net_ret_pct |   mean_period_net_ret_pct |   median_period_net_ret_pct |   period_win_rate |   min_trade_pct |   max_trade_pct |
|:---------------------------------------------|-------:|--------------:|---------------:|-------------------------:|--------------------------:|----------------------------:|------------------:|----------------:|----------------:|
| executable_only__generalized_lgbm_rank__pos1 |   2019 |           121 |            121 |                -0.650766 |                 -0.650766 |                   -0.677359 |          0.38843  |       -17.3829  |        20.825   |
| executable_only__generalized_lgbm_rank__pos1 |   2020 |           121 |            121 |                -0.340592 |                 -0.340592 |                   -0.925    |          0.404959 |       -14.2485  |        13.8228  |
| executable_only__generalized_lgbm_rank__pos1 |   2021 |           121 |            121 |                 0.45981  |                  0.45981  |                   -0.476991 |          0.438017 |       -10.6064  |        23.7343  |
| executable_only__generalized_lgbm_rank__pos1 |   2022 |           120 |            120 |                -0.501266 |                 -0.501266 |                   -0.612272 |          0.4      |       -11.4588  |        14.7188  |
| executable_only__generalized_lgbm_rank__pos1 |   2023 |           120 |            120 |                -0.587724 |                 -0.587724 |                   -0.669105 |          0.391667 |       -15.8455  |        23.4391  |
| executable_only__generalized_lgbm_rank__pos1 |   2024 |           120 |            120 |                 0.355056 |                  0.355056 |                   -0.3      |          0.441667 |       -20.3492  |        21.8939  |
| executable_only__generalized_lgbm_rank__pos1 |   2025 |           121 |            121 |                -0.188462 |                 -0.188462 |                   -0.56738  |          0.438017 |       -12.2565  |         9.39163 |
| executable_only__generalized_lgbm_rank__pos1 |   2026 |            47 |             47 |                -0.194759 |                 -0.194759 |                   -0.563158 |          0.404255 |        -6.73275 |         9.56985 |

## Best Strategy Yearly

| strategy_id                                    |   year |   trade_count |   period_count |   mean_trade_net_ret_pct |   mean_period_net_ret_pct |   median_period_net_ret_pct |   period_win_rate |   min_trade_pct |   max_trade_pct |
|:-----------------------------------------------|-------:|--------------:|---------------:|-------------------------:|--------------------------:|----------------------------:|------------------:|----------------:|----------------:|
| all_strong_events__generalized_lgbm_rank__pos1 |   2019 |           121 |            121 |                  7.63538 |                   7.63538 |                     9.69678 |          0.900826 |        -14.0536 |        12.3576  |
| all_strong_events__generalized_lgbm_rank__pos1 |   2020 |           121 |            121 |                  7.08685 |                   7.08685 |                     9.69756 |          0.876033 |        -24.0009 |        10.4143  |
| all_strong_events__generalized_lgbm_rank__pos1 |   2021 |           121 |            121 |                  6.78191 |                   6.78191 |                     9.7     |          0.867769 |        -22.5691 |         9.83699 |
| all_strong_events__generalized_lgbm_rank__pos1 |   2022 |           120 |            120 |                  5.29271 |                   5.29271 |                     9.68968 |          0.791667 |        -26.6407 |        14.7188  |
| all_strong_events__generalized_lgbm_rank__pos1 |   2023 |           120 |            120 |                  3.55439 |                   3.55439 |                     7.73146 |          0.708333 |        -18.9916 |         9.76622 |
| all_strong_events__generalized_lgbm_rank__pos1 |   2024 |           120 |            120 |                  1.72242 |                   1.72242 |                     8.70649 |          0.625    |        -26.6593 |        19.4802  |
| all_strong_events__generalized_lgbm_rank__pos1 |   2025 |           121 |            121 |                  1.37446 |                   1.37446 |                     6.36667 |          0.652893 |        -26.7138 |         9.82048 |
| all_strong_events__generalized_lgbm_rank__pos1 |   2026 |            47 |             47 |                  3.0153  |                   3.0153  |                     2.28985 |          0.659574 |        -19.3141 |         9.72928 |

## Top Feature Importance

| feature                      |   mean_importance |   eval_year_count |
|:-----------------------------|------------------:|------------------:|
| market_breadth_5d            |           199.75  |                 8 |
| market_limitup_rate_ma20     |           193.5   |                 8 |
| market_limitup_count_ma5     |           188.5   |                 8 |
| market_breadth_20d           |           174.625 |                 8 |
| market_limitup_count         |           126.75  |                 8 |
| next_open_gap_pct            |           118.5   |                 8 |
| market_limitup_rate          |            99.625 |                 8 |
| signal_amount_log10          |            79.375 |                 8 |
| industry_ret5_mean           |            36.875 |                 8 |
| signal_open_gap_pct          |            36.75  |                 8 |
| turn                         |            30.25  |                 8 |
| industry_limitup_rate        |            27.125 |                 8 |
| price                        |            26.625 |                 8 |
| industry_limitup_count_ma5   |            23.875 |                 8 |
| signal_range_pct             |            21.625 |                 8 |
| limit_up_run_ending_today    |            21.375 |                 8 |
| signal_ret5_before_pct       |            20.75  |                 8 |
| signal_close_position        |            19.625 |                 8 |
| volatility_20_pct            |            19.125 |                 8 |
| ma60_bias_pct                |            18.75  |                 8 |
| industry_limitup_count       |            17.375 |                 8 |
| turn_x60                     |            16.25  |                 8 |
| ma5_slope_pct                |            16.25  |                 8 |
| ma10_bias_pct                |            13.75  |                 8 |
| next_gap_bucket_gap_ge_6     |            13.25  |                 8 |
| ma5_bias_pct                 |            13.25  |                 8 |
| volatility_compression_20_60 |            12     |                 8 |
| signal_vrat5                 |            11.875 |                 8 |
| ma30_bias_pct                |            11.625 |                 8 |
| kama_slope_pct               |            11.625 |                 8 |

## Best Trades Sample

| date                | entry_date          | code      | name_on_date   | primary_event_type   |   event_strength_score |   ml_score |   predicted_ret_pct |   predicted_big_loss_prob |   target_net_ret_pct |   entry_open_near_limit |   executable_entry |
|:--------------------|:--------------------|:----------|:---------------|:---------------------|-----------------------:|-----------:|--------------------:|--------------------------:|---------------------:|------------------------:|-------------------:|
| 2019-01-02 00:00:00 | 2019-01-03 00:00:00 | 600776.SH | 东方通信       | limit_up_core        |               15.2998  |  -0.377802 |            1.44605  |                0.364771   |              1.62308 |                       0 |                  1 |
| 2019-01-04 00:00:00 | 2019-01-07 00:00:00 | 600614.SH | 退市鹏起       | limit_up_core        |               12.376   |   2.92914  |            4.03092  |                0.220356   |              9.75693 |                       1 |                  0 |
| 2019-01-08 00:00:00 | 2019-01-09 00:00:00 | 601860.SH | 紫金银行       | limit_up_core        |               13.8048  |   0.403463 |            1.1257   |                0.144447   |              9.66979 |                       1 |                  0 |
| 2019-01-10 00:00:00 | 2019-01-11 00:00:00 | 600547.SH | 山东黄金       | kama_breakout        |                6.18497 |  -0.496499 |           -0.261149 |                0.04707    |              0.46412 |                       0 |                  1 |
| 2019-01-14 00:00:00 | 2019-01-15 00:00:00 | 603121.SH | 华培动力       | limit_up_core        |               12.0118  |   7.80742  |            8.09468  |                0.0574524  |              9.72433 |                       1 |                  0 |
| 2019-01-16 00:00:00 | 2019-01-17 00:00:00 | 603739.SH | 蔚蓝生物       | limit_up_core        |               14       |   8.85107  |            8.94114  |                0.0180152  |              9.67522 |                       1 |                  0 |
| 2019-01-18 00:00:00 | 2019-01-21 00:00:00 | 603739.SH | 蔚蓝生物       | limit_up_core        |               12.4077  |   8.50944  |            8.69027  |                0.0361661  |              9.68464 |                       1 |                  0 |
| 2019-01-22 00:00:00 | 2019-01-23 00:00:00 | 603700.SH | 宁水集团       | limit_up_core        |               16       |   8.18651  |            8.49423  |                0.0615436  |              9.71898 |                       1 |                  0 |
| 2019-01-24 00:00:00 | 2019-01-25 00:00:00 | 601615.SH | 明阳智能       | limit_up_core        |               11.9415  |   8.36702  |            8.62982  |                0.0525595  |              9.73628 |                       1 |                  0 |
| 2019-01-28 00:00:00 | 2019-01-29 00:00:00 | 002946.SZ | 新乳业         | limit_up_core        |               12.0637  |   9.04681  |            9.13024  |                0.0166866  |              9.7     |                       1 |                  0 |
| 2019-01-30 00:00:00 | 2019-01-31 00:00:00 | 002946.SZ | 新乳业         | limit_up_core        |               12.7988  |   6.65034  |            7.23843  |                0.117618   |              9.7     |                       1 |                  0 |
| 2019-02-01 00:00:00 | 2019-02-11 00:00:00 | 002947.SZ | 恒铭达         | limit_up_core        |               16       |   8.08624  |            8.41478  |                0.065709   |              9.71349 |                       1 |                  0 |
| 2019-02-12 00:00:00 | 2019-02-13 00:00:00 | 002947.SZ | 恒铭达         | limit_up_core        |               12.2624  |   8.96774  |            9.03325  |                0.0131019  |              9.70279 |                       1 |                  0 |
| 2019-02-14 00:00:00 | 2019-02-15 00:00:00 | 002947.SZ | 恒铭达         | limit_up_core        |               15.9479  |   3.83526  |            4.58493  |                0.149935   |              9.69309 |                       1 |                  0 |
| 2019-02-18 00:00:00 | 2019-02-19 00:00:00 | 601865.SH | 福莱特         | limit_up_core        |               12.0694  |   9.59953  |            9.63768  |                0.00762838 |              9.72865 |                       1 |                  0 |
| 2019-02-20 00:00:00 | 2019-02-21 00:00:00 | 601865.SH | 福莱特         | limit_up_core        |               12.3473  |   9.62099  |            9.66963  |                0.00972844 |              9.65261 |                       1 |                  0 |
| 2019-02-22 00:00:00 | 2019-02-25 00:00:00 | 603956.SH | 威派格         | limit_up_core        |               16       |   9.46743  |            9.50872  |                0.00825832 |              9.66678 |                       1 |                  0 |
| 2019-02-26 00:00:00 | 2019-02-27 00:00:00 | 002949.SZ | 华阳国际       | limit_up_core        |               16       |   9.20904  |            9.24982  |                0.00815498 |              9.67596 |                       1 |                  0 |
| 2019-02-28 00:00:00 | 2019-03-01 00:00:00 | 002949.SZ | 华阳国际       | limit_up_core        |               12.189   |   9.05575  |            9.10278  |                0.00940722 |              9.6851  |                       1 |                  0 |
| 2019-03-04 00:00:00 | 2019-03-05 00:00:00 | 603956.SH | 威派格         | limit_up_core        |               13.5867  |   6.78001  |            7.06097  |                0.0561918  |              9.71252 |                       1 |                  0 |
| 2019-03-06 00:00:00 | 2019-03-07 00:00:00 | 002949.SZ | 华阳国际       | limit_up_core        |               13.9192  |   5.62879  |            6.02105  |                0.0784505  |              9.71018 |                       1 |                  0 |
| 2019-03-08 00:00:00 | 2019-03-11 00:00:00 | 000715.SZ | 中兴商业       | limit_up_core        |               12.1184  |   4.45173  |            5.4536   |                0.200374   |              4.58372 |                       1 |                  0 |
| 2019-03-12 00:00:00 | 2019-03-13 00:00:00 | 002950.SZ | 奥美医疗       | limit_up_core        |               12.0126  |   8.2944   |            8.34329  |                0.0097779  |              9.68959 |                       1 |                  0 |
| 2019-03-14 00:00:00 | 2019-03-15 00:00:00 | 002950.SZ | 奥美医疗       | limit_up_core        |               13.0336  |   8.01227  |            8.14649  |                0.0268449  |              9.72151 |                       1 |                  0 |
| 2019-03-18 00:00:00 | 2019-03-19 00:00:00 | 002951.SZ | 金时科技       | limit_up_core        |               11.993   |   8.64704  |            8.69357  |                0.00930753 |              9.69422 |                       1 |                  0 |
| 2019-03-20 00:00:00 | 2019-03-21 00:00:00 | 002951.SZ | 金时科技       | limit_up_core        |               12.4036  |   8.58415  |            8.64765  |                0.0127005  |              9.6809  |                       1 |                  0 |
| 2019-03-22 00:00:00 | 2019-03-25 00:00:00 | 002951.SZ | 金时科技       | limit_up_core        |               15.1422  |   6.24266  |            6.58539  |                0.0685471  |              9.68816 |                       1 |                  0 |
| 2019-03-26 00:00:00 | 2019-03-27 00:00:00 | 603681.SH | 永冠新材       | limit_up_core        |               16       |   9.08589  |            9.14415  |                0.0116504  |              9.67475 |                       1 |                  0 |
| 2019-03-28 00:00:00 | 2019-03-29 00:00:00 | 603681.SH | 永冠新材       | limit_up_core        |               12.5917  |   8.91664  |            8.98995  |                0.0146623  |              9.72088 |                       1 |                  0 |
| 2019-04-01 00:00:00 | 2019-04-02 00:00:00 | 002952.SZ | 亚世光电       | limit_up_core        |               12.3917  |   7.6733   |            7.75133  |                0.0156043  |              9.70335 |                       1 |                  0 |

## Interpretation Boundary

- This is a diagnostic expansion from pure limit-up events to a broader strong-event pool.
- Current event types are daily-bar approximations; touch-board, failed-board, and refill events require five-minute data.
- The open-known profile includes next-open fields, so it is valid only after the Day+1 open print is known.
- Strategy candidate count remains zero until execution and stability gates are strengthened.

## Research Interpretation

- Event space expanded materially: `limit_up_core` has `106411` events, while `all_strong_events` has `312573` events and `non_limit_strong` has `206162` events.
- Raw next-day net returns remain negative for every major pool. `kama_breakout` is the least weak raw pool (`-0.127936%`) and has much lower big-loss rate (`0.080171`) than limit-up core (`0.266365`), but it is not positive on its own.
- The best headline ML result, `all_strong_events__generalized_lgbm_rank__pos1`, is not a tradable result: mean period return is `+4.68953%`, but `near_limit_open_rate=0.892256` and `executable_rate=0.106622`. Its selected trades are still dominated by `limit_up_core` (`798/891` trades).
- The practical `executable_only__generalized_lgbm_rank__pos1` result is weak: `-0.206899%` mean period return, `0.414141` win rate, and only `0.25` positive-year rate. This is worse than the prior all-limit-up practical ML result (`+0.6778%`, positive-year rate `0.875`).
- Decision: the generalized strong-event pool is useful as a research taxonomy and as an early-warning expansion, but it should not replace the current limit-up-centered pool for strategy development. The next practical path is to keep limit-up as the core tradable research line, then study non-limit branches (`big_up`, `kama_breakout`, `new_high_breakout`) as separate side pools with stricter execution filters and five-minute event features.
