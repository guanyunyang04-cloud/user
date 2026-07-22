# 2026-06-02 Multifactor Baseline

- Hypothesis: 方向校正后的传统价量因子 rank 合成，应该比第一阶段等权 z-score `baseline_score` 更适合作为第二阶段多因子诊断基线。
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2024-01-01` to `2026-06-01`.
- Panel: `1779897` rows, `581` dates, `3233` securities.
- Label: `fwd_ret_5d`; label mode: `raw`; raw label: `fwd_ret_5d`.
- Label Mode Note: `raw` keeps the same-date market move in the label, so IC and quantile returns are closer to absolute return diagnostics but still need execution-aware portfolio validation.
- Backtest Return Note: Top-N backtests use the generated 5-day forward return as each selected rebalance-date return. Daily or otherwise overlapping rebalance schedules can overstate portfolio PnL; treat these rows as holding-period ranking diagnostics until a horizon-aligned portfolio simulator is used.
- Horizon Backtest Note: Horizon-aligned Top-N backtests use explicit next-open entry and 5-tradeable-day close exit with non-overlapping baskets. They are stricter than label-column diagnostic Top-N rows. Buffer multipliers above 1.0 keep prior holdings that remain inside the expanded rank buffer to reduce turnover. These rows still need limit-up/down, suspension holding, slippage, and impact-cost constraints before strategy-candidate promotion.
- Rolling IC: window `252`, min periods `60`, fallback date rate `0.110155`.
- Low-Correlation Factors: max abs corr `0.75`, selected `['momentum_20d_z', 'log_amount_mean_20d_z', 'neg_volatility_20d_z', 'reversal_5d_z']`.
- Neutralization: proxy columns `['log_amount_mean_20d_z']`, neutralized factors `['reversal_5d_z_neutral', 'momentum_20d_z_neutral', 'ma20_gap_z_neutral', 'neg_volatility_20d_z_neutral', 'neg_amplitude_20d_z_neutral']`.
- Signals: `['baseline_score', 'multifactor_equal_rank_score', 'multifactor_ic_weighted_score', 'multifactor_rolling_ic_weighted_score', 'multifactor_low_corr_rank_score', 'multifactor_neutral_rank_score']`.
- Factor Directions: `{'reversal_5d_z': 1, 'momentum_20d_z': -1, 'ma20_gap_z': -1, 'neg_volatility_20d_z': 1, 'log_amount_mean_20d_z': -1, 'neg_amplitude_20d_z': 1}`.
- Result: best IC signal `multifactor_low_corr_rank_score`; best quantile-spread signal `multifactor_low_corr_rank_score`; best backtest signal `multifactor_rolling_ic_weighted_score` under the tested Top-N/cost/frequency grid.
- Consistency Check: IC、分位 spread 和 Top-N 最优信号 `不一致`；若不一致，本实验只作为诊断证据，不能升级为策略结论。
- Assessment: this is a phase-2 diagnostic baseline; it is not yet a traditional ML model or production strategy candidate.
- Next Step: add industry/size neutralization and longer execution-constrained backtest grids.

## Factor Coverage

| factor                |    rows |   available_rows |   missing_rows |   coverage_rate |
|:----------------------|--------:|-----------------:|---------------:|----------------:|
| reversal_5d_z         | 1779897 |          1763736 |          16161 |        0.99092  |
| momentum_20d_z        | 1779897 |          1715373 |          64524 |        0.963748 |
| ma20_gap_z            | 1779897 |          1766968 |          12929 |        0.992736 |
| neg_volatility_20d_z  | 1779897 |          1763736 |          16161 |        0.99092  |
| log_amount_mean_20d_z | 1779897 |          1766968 |          12929 |        0.992736 |
| neg_amplitude_20d_z   | 1779897 |          1766968 |          12929 |        0.992736 |

## Single Factor IC

| signal                | label      |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:----------------------|:-----------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| reversal_5d_z         | fwd_ret_5d |     571 |        1747582 |  0.034002 |       0.038218 |      0.155595 |     3.8992  |
| momentum_20d_z        | fwd_ret_5d |     556 |        1699308 | -0.034229 |      -0.062525 |      0.171637 |    -5.78283 |
| ma20_gap_z            | fwd_ret_5d |     572 |        1750812 | -0.033239 |      -0.048509 |      0.166173 |    -4.63409 |
| neg_volatility_20d_z  | fwd_ret_5d |     571 |        1747582 |  0.005839 |       0.046885 |      0.209474 |     3.55307 |
| log_amount_mean_20d_z | fwd_ret_5d |     572 |        1750812 | -0.015752 |      -0.050422 |      0.170305 |    -4.69996 |
| neg_amplitude_20d_z   | fwd_ret_5d |     572 |        1750812 | -0.002593 |       0.043964 |      0.220912 |     3.15918 |

## Multifactor IC

| signal                                | label      |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:--------------------------------------|:-----------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| baseline_score                        | fwd_ret_5d |     572 |        1750812 | -0.013948 |      -0.002236 |      0.221414 |   -0.160316 |
| multifactor_equal_rank_score          | fwd_ret_5d |     572 |        1750812 |  0.017332 |       0.071564 |      0.151898 |    7.47897  |
| multifactor_ic_weighted_score         | fwd_ret_5d |     572 |        1750812 |  0.018611 |       0.072937 |      0.153238 |    7.55584  |
| multifactor_rolling_ic_weighted_score | fwd_ret_5d |     572 |        1750812 |  0.018168 |       0.072146 |      0.18766  |    6.10295  |
| multifactor_low_corr_rank_score       | fwd_ret_5d |     571 |        1747582 |  0.022431 |       0.074406 |      0.155219 |    7.6096   |
| multifactor_neutral_rank_score        | fwd_ret_5d |     571 |        1747582 |  0.013544 |       0.055229 |      0.140237 |    6.25184  |

## Quantile Returns

| signal                                |   quantile |   mean_return |   periods |   observations |
|:--------------------------------------|-----------:|--------------:|----------:|---------------:|
| baseline_score                        |          1 |      0.005567 |       572 |         350396 |
| baseline_score                        |          2 |      0.00507  |       572 |         350046 |
| baseline_score                        |          3 |      0.004254 |       572 |         350055 |
| baseline_score                        |          4 |      0.003389 |       572 |         350046 |
| baseline_score                        |          5 |      0.003256 |       572 |         350269 |
| multifactor_equal_rank_score          |          1 |      0.002217 |       572 |         350396 |
| multifactor_equal_rank_score          |          2 |      0.005137 |       572 |         350046 |
| multifactor_equal_rank_score          |          3 |      0.004971 |       572 |         350055 |
| multifactor_equal_rank_score          |          4 |      0.004838 |       572 |         350046 |
| multifactor_equal_rank_score          |          5 |      0.004375 |       572 |         350269 |
| multifactor_ic_weighted_score         |          1 |      0.00208  |       572 |         350396 |
| multifactor_ic_weighted_score         |          2 |      0.005032 |       572 |         350046 |
| multifactor_ic_weighted_score         |          3 |      0.005006 |       572 |         350055 |
| multifactor_ic_weighted_score         |          4 |      0.004869 |       572 |         350046 |
| multifactor_ic_weighted_score         |          5 |      0.004552 |       572 |         350269 |
| multifactor_rolling_ic_weighted_score |          1 |      0.002485 |       572 |         350396 |
| multifactor_rolling_ic_weighted_score |          2 |      0.004747 |       572 |         350046 |
| multifactor_rolling_ic_weighted_score |          3 |      0.004365 |       572 |         350055 |
| multifactor_rolling_ic_weighted_score |          4 |      0.004501 |       572 |         350046 |
| multifactor_rolling_ic_weighted_score |          5 |      0.005438 |       572 |         350269 |
| multifactor_low_corr_rank_score       |          1 |      0.001942 |       571 |         349742 |
| multifactor_low_corr_rank_score       |          2 |      0.004823 |       571 |         349403 |
| multifactor_low_corr_rank_score       |          3 |      0.004788 |       571 |         349411 |
| multifactor_low_corr_rank_score       |          4 |      0.004953 |       571 |         349403 |
| multifactor_low_corr_rank_score       |          5 |      0.005015 |       571 |         349623 |
| multifactor_neutral_rank_score        |          1 |      0.002211 |       571 |         349742 |
| multifactor_neutral_rank_score        |          2 |      0.005182 |       571 |         349403 |
| multifactor_neutral_rank_score        |          3 |      0.004967 |       571 |         349411 |
| multifactor_neutral_rank_score        |          4 |      0.005007 |       571 |         349403 |
| multifactor_neutral_rank_score        |          5 |      0.004154 |       571 |         349623 |

## Quantile Spread

| signal                                |   low_quantile |   high_quantile |   low_mean_return |   high_mean_return |   q_high_minus_low |
|:--------------------------------------|---------------:|----------------:|------------------:|-------------------:|-------------------:|
| baseline_score                        |              1 |               5 |          0.005567 |           0.003256 |          -0.00231  |
| multifactor_equal_rank_score          |              1 |               5 |          0.002217 |           0.004375 |           0.002158 |
| multifactor_ic_weighted_score         |              1 |               5 |          0.00208  |           0.004552 |           0.002471 |
| multifactor_low_corr_rank_score       |              1 |               5 |          0.001942 |           0.005015 |           0.003073 |
| multifactor_neutral_rank_score        |              1 |               5 |          0.002211 |           0.004154 |           0.001943 |
| multifactor_rolling_ic_weighted_score |              1 |               5 |          0.002485 |           0.005438 |           0.002954 |

## Backtest Summary

| signal                                | rebalance_frequency   |   top_n |   fee_bps |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:--------------------------------------|:----------------------|--------:|----------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score                        | weekly                |     100 |         0 |            0.076988 |     0.196003 |  0.476984 |      -0.260036 |       123 |        0.664715 |            0.001798 |          0.001798 |
| baseline_score                        | weekly                |     100 |        10 |            0.040414 |     0.19606  |  0.300545 |      -0.277333 |       123 |        0.664715 |            0.001798 |          0.001133 |
| baseline_score                        | weekly                |     100 |        30 |           -0.029122 |     0.1962   | -0.052016 |      -0.318401 |       123 |        0.664715 |            0.001798 |         -0.000196 |
| multifactor_equal_rank_score          | weekly                |     100 |         0 |            0.172942 |     0.273914 |  0.718412 |      -0.218191 |       123 |        1.37024  |            0.003784 |          0.003784 |
| multifactor_equal_rank_score          | weekly                |     100 |        10 |            0.092434 |     0.273809 |  0.458458 |      -0.222709 |       123 |        1.37024  |            0.003784 |          0.002414 |
| multifactor_equal_rank_score          | weekly                |     100 |        30 |           -0.052667 |     0.273623 | -0.062038 |      -0.271523 |       123 |        1.37024  |            0.003784 |         -0.000326 |
| multifactor_ic_weighted_score         | weekly                |     100 |         0 |            0.185128 |     0.280364 |  0.744676 |      -0.223101 |       123 |        1.32195  |            0.004015 |          0.004015 |
| multifactor_ic_weighted_score         | weekly                |     100 |        10 |            0.106571 |     0.28025  |  0.499693 |      -0.227496 |       123 |        1.32195  |            0.004015 |          0.002693 |
| multifactor_ic_weighted_score         | weekly                |     100 |        30 |           -0.035535 |     0.28004  |  0.009127 |      -0.248075 |       123 |        1.32195  |            0.004015 |          4.9e-05  |
| multifactor_rolling_ic_weighted_score | weekly                |     100 |         0 |            0.328218 |     0.206832 |  1.47949  |      -0.167668 |       123 |        1.2761   |            0.005885 |          0.005885 |
| multifactor_rolling_ic_weighted_score | weekly                |     100 |        10 |            0.24333  |     0.206768 |  1.15903  |      -0.184881 |       123 |        1.2761   |            0.005885 |          0.004609 |
| multifactor_rolling_ic_weighted_score | weekly                |     100 |        30 |            0.089196 |     0.206695 |  0.517358 |      -0.218308 |       123 |        1.2761   |            0.005885 |          0.002056 |
| multifactor_low_corr_rank_score       | weekly                |     100 |         0 |            0.204796 |     0.288236 |  0.790184 |      -0.241051 |       123 |        1.45171  |            0.00438  |          0.00438  |
| multifactor_low_corr_rank_score       | weekly                |     100 |        10 |            0.117402 |     0.288156 |  0.528431 |      -0.24546  |       123 |        1.45171  |            0.00438  |          0.002928 |
| multifactor_low_corr_rank_score       | weekly                |     100 |        30 |           -0.03915  |     0.288016 |  0.004488 |      -0.281939 |       123 |        1.45171  |            0.00438  |          2.5e-05  |
| multifactor_neutral_rank_score        | weekly                |     100 |         0 |            0.185307 |     0.257221 |  0.786325 |      -0.155549 |       123 |        1.27122  |            0.00389  |          0.00389  |
| multifactor_neutral_rank_score        | weekly                |     100 |        10 |            0.109687 |     0.257041 |  0.529704 |      -0.184271 |       123 |        1.27122  |            0.00389  |          0.002618 |
| multifactor_neutral_rank_score        | weekly                |     100 |        30 |           -0.027642 |     0.256719 |  0.015382 |      -0.261052 |       123 |        1.27122  |            0.00389  |          7.6e-05  |

## Horizon-Aligned Backtest Summary

| signal                                |   horizon | rebalance_frequency   |   top_n |   fee_bps | non_overlapping   |   buffer_multiplier |   periods_per_year |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:--------------------------------------|----------:|:----------------------|--------:|----------:|:------------------|--------------------:|-------------------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score                        |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |           -0.118272 |     0.182196 | -0.596821 |      -0.33844  |        92 |        0.75     |           -0.002158 |         -0.002158 |
| baseline_score                        |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |           -0.0885   |     0.179871 | -0.422697 |      -0.312719 |        89 |        0.531236 |           -0.001509 |         -0.001509 |
| baseline_score                        |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |           -0.089975 |     0.177308 | -0.440671 |      -0.300066 |        92 |        0.41913  |           -0.00155  |         -0.00155  |
| baseline_score                        |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.151112 |     0.182379 | -0.803485 |      -0.361282 |        92 |        0.75     |           -0.002158 |         -0.002908 |
| baseline_score                        |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |           -0.112663 |     0.180021 | -0.571075 |      -0.329001 |        89 |        0.531236 |           -0.001509 |         -0.00204  |
| baseline_score                        |         5 | weekly                |     100 |        10 | True              |                 2   |               50.4 |           -0.109074 |     0.177497 | -0.559213 |      -0.313249 |        92 |        0.41913  |           -0.00155  |         -0.001969 |
| baseline_score                        |         5 | weekly                |     100 |        30 | True              |                 1   |               50.4 |           -0.21324  |     0.182789 | -1.21527  |      -0.431959 |        92 |        0.75     |           -0.002158 |         -0.004408 |
| baseline_score                        |         5 | weekly                |     100 |        30 | True              |                 1.5 |               50.4 |           -0.15913  |     0.180369 | -0.866853 |      -0.360447 |        89 |        0.531236 |           -0.001509 |         -0.003102 |
| baseline_score                        |         5 | weekly                |     100 |        30 | True              |                 2   |               50.4 |           -0.146105 |     0.177915 | -0.795361 |      -0.340826 |        92 |        0.41913  |           -0.00155  |         -0.002808 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |           -0.052022 |     0.225665 | -0.116866 |      -0.225384 |        51 |        1.43569  |           -0.000523 |         -0.000523 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |           -0.05342  |     0.224165 | -0.12551  |      -0.212897 |        51 |        1.26588  |           -0.000558 |         -0.000558 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |           -0.054463 |     0.223391 | -0.131656 |      -0.213345 |        51 |        1.13961  |           -0.000584 |         -0.000584 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.11835  |     0.225657 | -0.437528 |      -0.235479 |        51 |        1.43569  |           -0.000523 |         -0.001959 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |           -0.112072 |     0.224198 | -0.410064 |      -0.222356 |        51 |        1.26588  |           -0.000558 |         -0.001824 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        10 | True              |                 2   |               50.4 |           -0.10739  |     0.223503 | -0.388572 |      -0.222213 |        51 |        1.13961  |           -0.000584 |         -0.001723 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        30 | True              |                 1   |               50.4 |           -0.237651 |     0.225666 | -1.0788   |      -0.255321 |        51 |        1.43569  |           -0.000523 |         -0.00483  |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        30 | True              |                 1.5 |               50.4 |           -0.218896 |     0.224301 | -0.978757 |      -0.240974 |        51 |        1.26588  |           -0.000558 |         -0.004356 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        30 | True              |                 2   |               50.4 |           -0.204687 |     0.223772 | -0.901453 |      -0.239687 |        51 |        1.13961  |           -0.000584 |         -0.004002 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |            0.026953 |     0.224647 |  0.237023 |      -0.233775 |        58 |        1.39724  |            0.001056 |          0.001056 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |            0.028034 |     0.223806 |  0.241893 |      -0.225553 |        58 |        1.22276  |            0.001074 |          0.001074 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |            0.015092 |     0.219028 |  0.184266 |      -0.218529 |        58 |        1.09345  |            0.000801 |          0.000801 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.042927 |     0.224608 | -0.076464 |      -0.243704 |        58 |        1.39724  |            0.001056 |         -0.000341 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |           -0.033448 |     0.223774 | -0.033471 |      -0.234665 |        58 |        1.22276  |            0.001074 |         -0.000149 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        10 | True              |                 2   |               50.4 |           -0.039388 |     0.219026 | -0.067345 |      -0.227211 |        58 |        1.09345  |            0.000801 |         -0.000293 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        30 | True              |                 1   |               50.4 |           -0.168997 |     0.224557 | -0.703679 |      -0.263222 |        58 |        1.39724  |            0.001056 |         -0.003135 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        30 | True              |                 1.5 |               50.4 |           -0.145802 |     0.223751 | -0.584328 |      -0.252604 |        58 |        1.22276  |            0.001074 |         -0.002594 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        30 | True              |                 2   |               50.4 |           -0.139898 |     0.21907  | -0.570457 |      -0.244321 |        58 |        1.09345  |            0.000801 |         -0.00248  |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |            0.380077 |     0.153011 |  2.18833  |      -0.076641 |        59 |        1.33492  |            0.006644 |          0.006644 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |            0.346897 |     0.15301  |  2.02822  |      -0.07782  |        59 |        1.1339   |            0.006157 |          0.006157 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |            0.327781 |     0.151024 |  1.95773  |      -0.080023 |        59 |        1.00271  |            0.005866 |          0.005866 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |            0.290725 |     0.153187 |  1.74662  |      -0.080584 |        59 |        1.33492  |            0.006644 |          0.005309 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |            0.272414 |     0.153235 |  1.6523   |      -0.081117 |        59 |        1.1339   |            0.006157 |          0.005024 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        10 | True              |                 2   |               50.4 |            0.262619 |     0.151252 |  1.62066  |      -0.082919 |        59 |        1.00271  |            0.005866 |          0.004864 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        30 | True              |                 1   |               50.4 |            0.128685 |     0.153626 |  0.86574  |      -0.104835 |        59 |        1.33492  |            0.006644 |          0.002639 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        30 | True              |                 1.5 |               50.4 |            0.135336 |     0.153804 |  0.903045 |      -0.105498 |        59 |        1.1339   |            0.006157 |          0.002756 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        30 | True              |                 2   |               50.4 |            0.141538 |     0.151832 |  0.94878  |      -0.101225 |        59 |        1.00271  |            0.005866 |          0.002858 |
| multifactor_low_corr_rank_score       |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |            0.073098 |     0.246329 |  0.415208 |      -0.241051 |        57 |        1.53404  |            0.002029 |          0.002029 |
| multifactor_low_corr_rank_score       |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |            0.073069 |     0.242726 |  0.417668 |      -0.234757 |        58 |        1.37483  |            0.002011 |          0.002011 |
| multifactor_low_corr_rank_score       |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |            0.081285 |     0.238231 |  0.452786 |      -0.223237 |        58 |        1.23069  |            0.00214  |          0.00214  |
| multifactor_low_corr_rank_score       |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.006694 |     0.246115 |  0.101426 |      -0.253296 |        57 |        1.53404  |            0.002029 |          0.000495 |

## Horizon-Aligned Yearly Summary

| signal                       |   year |   horizon | rebalance_frequency   |   top_n |   fee_bps | non_overlapping   |   buffer_multiplier |   periods_per_year |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:-----------------------------|-------:|----------:|:----------------------|--------:|----------:|:------------------|--------------------:|-------------------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score               |   2024 |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |           -0.143804 |     0.192605 | -0.7061   |      -0.210628 |        39 |        0.694359 |           -0.002698 |         -0.002698 |
| baseline_score               |   2025 |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |           -0.152623 |     0.180789 | -0.821788 |      -0.161916 |        38 |        0.776842 |           -0.002948 |         -0.002948 |
| baseline_score               |   2026 |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |            0.052526 |     0.153763 |  0.410001 |      -0.06647  |        15 |        0.826667 |            0.001251 |          0.001251 |
| baseline_score               |   2024 |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |           -0.13663  |     0.195211 | -0.651463 |      -0.19635  |        39 |        0.488205 |           -0.002523 |         -0.002523 |
| baseline_score               |   2025 |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |           -0.130697 |     0.177147 | -0.698742 |      -0.1448   |        36 |        0.547778 |           -0.002456 |         -0.002456 |
| baseline_score               |   2026 |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |            0.197627 |     0.13103  |  1.44409  |      -0.0339   |        14 |        0.608571 |            0.003754 |          0.003754 |
| baseline_score               |   2024 |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |           -0.113526 |     0.19469  | -0.518676 |      -0.185617 |        39 |        0.38359  |           -0.002004 |         -0.002004 |
| baseline_score               |   2025 |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |           -0.111849 |     0.170432 | -0.607715 |      -0.140534 |        38 |        0.431053 |           -0.002055 |         -0.002055 |
| baseline_score               |   2026 |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |            0.036162 |     0.142546 |  0.320702 |      -0.06185  |        15 |        0.481333 |            0.000907 |          0.000907 |
| baseline_score               |   2024 |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.173473 |     0.193302 | -0.884596 |      -0.226982 |        39 |        0.694359 |           -0.002698 |         -0.003393 |
| baseline_score               |   2025 |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.18523  |     0.180413 | -1.04052  |      -0.172761 |        38 |        0.776842 |           -0.002948 |         -0.003725 |
| baseline_score               |   2026 |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |            0.009549 |     0.154026 |  0.138802 |      -0.073214 |        15 |        0.826667 |            0.001251 |          0.000424 |
| baseline_score               |   2024 |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |           -0.157792 |     0.195819 | -0.775095 |      -0.2084   |        39 |        0.488205 |           -0.002523 |         -0.003011 |
| baseline_score               |   2025 |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |           -0.154392 |     0.17679  | -0.856317 |      -0.152352 |        36 |        0.547778 |           -0.002456 |         -0.003004 |
| baseline_score               |   2026 |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |            0.161531 |     0.131235 |  1.20812  |      -0.035112 |        14 |        0.608571 |            0.003754 |          0.003146 |
| baseline_score               |   2024 |         5 | weekly                |     100 |        10 | True              |                 2   |               50.4 |           -0.130652 |     0.195272 | -0.616135 |      -0.195276 |        39 |        0.38359  |           -0.002004 |         -0.002387 |
| baseline_score               |   2025 |         5 | weekly                |     100 |        10 | True              |                 2   |               50.4 |           -0.130963 |     0.170249 | -0.735978 |      -0.146601 |        38 |        0.431053 |           -0.002055 |         -0.002486 |
| baseline_score               |   2026 |         5 | weekly                |     100 |        10 | True              |                 2   |               50.4 |            0.011337 |     0.142535 |  0.150527 |      -0.063967 |        15 |        0.481333 |            0.000907 |          0.000426 |
| baseline_score               |   2024 |         5 | weekly                |     100 |        30 | True              |                 1   |               50.4 |           -0.229828 |     0.19473  | -1.23753  |      -0.258719 |        39 |        0.694359 |           -0.002698 |         -0.004781 |
| baseline_score               |   2025 |         5 | weekly                |     100 |        30 | True              |                 1   |               50.4 |           -0.246802 |     0.179698 | -1.48042  |      -0.198928 |        38 |        0.776842 |           -0.002948 |         -0.005278 |
| baseline_score               |   2026 |         5 | weekly                |     100 |        30 | True              |                 1   |               50.4 |           -0.071318 |     0.154615 | -0.400665 |      -0.086578 |        15 |        0.826667 |            0.001251 |         -0.001229 |
| baseline_score               |   2024 |         5 | weekly                |     100 |        30 | True              |                 1.5 |               50.4 |           -0.198609 |     0.197081 | -1.01983  |      -0.231981 |        39 |        0.488205 |           -0.002523 |         -0.003988 |
| baseline_score               |   2025 |         5 | weekly                |     100 |        30 | True              |                 1.5 |               50.4 |           -0.199906 |     0.176115 | -1.17312  |      -0.167269 |        36 |        0.547778 |           -0.002456 |         -0.004099 |
| baseline_score               |   2026 |         5 | weekly                |     100 |        30 | True              |                 1.5 |               50.4 |            0.092498 |     0.131708 |  0.738021 |      -0.037534 |        14 |        0.608571 |            0.003754 |          0.001929 |
| baseline_score               |   2024 |         5 | weekly                |     100 |        30 | True              |                 2   |               50.4 |           -0.163943 |     0.196471 | -0.809176 |      -0.214266 |        39 |        0.38359  |           -0.002004 |         -0.003154 |
| baseline_score               |   2025 |         5 | weekly                |     100 |        30 | True              |                 2   |               50.4 |           -0.167995 |     0.169921 | -0.993102 |      -0.158616 |        38 |        0.431053 |           -0.002055 |         -0.003348 |
| baseline_score               |   2026 |         5 | weekly                |     100 |        30 | True              |                 2   |               50.4 |           -0.036584 |     0.142558 | -0.189839 |      -0.068473 |        15 |        0.481333 |            0.000907 |         -0.000537 |
| multifactor_equal_rank_score |   2024 |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |           -0.669321 |     0.379321 | -2.67902  |      -0.225384 |        10 |        1.412    |           -0.020163 |         -0.020163 |
| multifactor_equal_rank_score |   2025 |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |            0.43937  |     0.137928 |  2.71902  |      -0.04025  |        24 |        1.44333  |            0.007441 |          0.007441 |
| multifactor_equal_rank_score |   2026 |         5 | weekly                |     100 |         0 | True              |                 1   |               50.4 |           -0.023226 |     0.15914  | -0.067865 |      -0.072855 |        17 |        1.43882  |           -0.000214 |         -0.000214 |
| multifactor_equal_rank_score |   2024 |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |           -0.645601 |     0.379036 | -2.50147  |      -0.212897 |        10 |        1.308    |           -0.018812 |         -0.018812 |
| multifactor_equal_rank_score |   2025 |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |            0.415198 |     0.137491 |  2.60321  |      -0.041133 |        24 |        1.25667  |            0.007102 |          0.007102 |
| multifactor_equal_rank_score |   2026 |         5 | weekly                |     100 |         0 | True              |                 1.5 |               50.4 |           -0.043779 |     0.159678 | -0.200202 |      -0.077338 |        17 |        1.25412  |           -0.000634 |         -0.000634 |
| multifactor_equal_rank_score |   2024 |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |           -0.645218 |     0.380113 | -2.49047  |      -0.213345 |        10 |        1.222    |           -0.018783 |         -0.018783 |
| multifactor_equal_rank_score |   2025 |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |            0.392418 |     0.135745 |  2.51459  |      -0.041037 |        24 |        1.11417  |            0.006773 |          0.006773 |
| multifactor_equal_rank_score |   2026 |         5 | weekly                |     100 |         0 | True              |                 2   |               50.4 |           -0.02547  |     0.158209 | -0.083863 |      -0.074005 |        17 |        1.12706  |           -0.000263 |         -0.000263 |
| multifactor_equal_rank_score |   2024 |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.692637 |     0.379726 | -2.86358  |      -0.235479 |        10 |        1.412    |           -0.020163 |         -0.021575 |
| multifactor_equal_rank_score |   2025 |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |            0.339029 |     0.137737 |  2.19466  |      -0.04179  |        24 |        1.44333  |            0.007441 |          0.005998 |
| multifactor_equal_rank_score |   2026 |         5 | weekly                |     100 |        10 | True              |                 1   |               50.4 |           -0.091609 |     0.158883 | -0.524389 |      -0.080127 |        17 |        1.43882  |           -0.000214 |         -0.001653 |
| multifactor_equal_rank_score |   2024 |         5 | weekly                |     100 |        10 | True              |                 1.5 |               50.4 |           -0.668788 |     0.37947  | -2.67233  |      -0.222356 |        10 |        1.308    |           -0.018812 |         -0.02012  |

Artifacts: `traditional_quant_research\output\experiments\multifactor_baseline\multifactor_baseline_20260602_123305`
