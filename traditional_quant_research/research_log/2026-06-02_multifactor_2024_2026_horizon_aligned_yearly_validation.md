# 2026-06-02 Multifactor Baseline

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_multifactor_2024_2026_horizon_aligned_yearly_validation.md`.


- Hypothesis: 方向校正后的传统价量因子 rank 合成，应该比第一阶段等权 z-score `baseline_score` 更适合作为第二阶段多因子诊断基线。
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2024-01-01` to `2026-06-01`.
- Panel: `1779897` rows, `581` dates, `3233` securities.
- Label: `fwd_ret_5d`; label mode: `raw`; raw label: `fwd_ret_5d`.
- Label Mode Note: `raw` keeps the same-date market move in the label, so IC and quantile returns are closer to absolute return diagnostics but still need execution-aware portfolio validation.
- Backtest Return Note: Top-N backtests use the generated 5-day forward return as each selected rebalance-date return. Daily or otherwise overlapping rebalance schedules can overstate portfolio PnL; treat these rows as holding-period ranking diagnostics until a horizon-aligned portfolio simulator is used.
- Horizon Backtest Note: Horizon-aligned Top-N backtests use explicit next-open entry and 5-tradeable-day close exit with non-overlapping baskets. They are stricter than label-column diagnostic Top-N rows, but still need limit-up/down, suspension holding, slippage, and impact-cost constraints before strategy-candidate promotion.
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
| baseline_score                        | daily                 |     100 |         0 |            1.15     |     0.458715 |  1.89248  |      -0.56321  |       572 |        0.334825 |            0.003445 |          0.003445 |
| baseline_score                        | daily                 |     100 |        10 |            0.976437 |     0.458724 |  1.70851  |      -0.584505 |       572 |        0.334825 |            0.003445 |          0.00311  |
| baseline_score                        | daily                 |     100 |        30 |            0.670058 |     0.458764 |  1.34052  |      -0.624051 |       572 |        0.334825 |            0.003445 |          0.00244  |
| baseline_score                        | weekly                |     100 |         0 |            0.076988 |     0.196003 |  0.476984 |      -0.260036 |       123 |        0.664715 |            0.001798 |          0.001798 |
| baseline_score                        | weekly                |     100 |        10 |            0.040414 |     0.19606  |  0.300545 |      -0.277333 |       123 |        0.664715 |            0.001798 |          0.001133 |
| baseline_score                        | weekly                |     100 |        30 |           -0.029122 |     0.1962   | -0.052016 |      -0.318401 |       123 |        0.664715 |            0.001798 |         -0.000196 |
| multifactor_equal_rank_score          | daily                 |     100 |         0 |            1.444    |     0.623233 |  1.74294  |      -0.765792 |       572 |        0.666049 |            0.004311 |          0.004311 |
| multifactor_equal_rank_score          | daily                 |     100 |        10 |            1.06713  |     0.623317 |  1.47343  |      -0.769173 |       572 |        0.666049 |            0.004311 |          0.003644 |
| multifactor_equal_rank_score          | daily                 |     100 |        30 |            0.478252 |     0.623512 |  0.934584 |      -0.775963 |       572 |        0.666049 |            0.004311 |          0.002312 |
| multifactor_equal_rank_score          | weekly                |     100 |         0 |            0.172942 |     0.273914 |  0.718412 |      -0.218191 |       123 |        1.37024  |            0.003784 |          0.003784 |
| multifactor_equal_rank_score          | weekly                |     100 |        10 |            0.092434 |     0.273809 |  0.458458 |      -0.222709 |       123 |        1.37024  |            0.003784 |          0.002414 |
| multifactor_equal_rank_score          | weekly                |     100 |        30 |           -0.052667 |     0.273623 | -0.062038 |      -0.271523 |       123 |        1.37024  |            0.003784 |         -0.000326 |
| multifactor_ic_weighted_score         | daily                 |     100 |         0 |            1.61113  |     0.638977 |  1.81858  |      -0.774659 |       572 |        0.638252 |            0.004611 |          0.004611 |
| multifactor_ic_weighted_score         | daily                 |     100 |        10 |            1.22415  |     0.638987 |  1.56684  |      -0.777747 |       572 |        0.638252 |            0.004611 |          0.003973 |
| multifactor_ic_weighted_score         | daily                 |     100 |        30 |            0.613235 |     0.639033 |  1.06334  |      -0.784285 |       572 |        0.638252 |            0.004611 |          0.002696 |
| multifactor_ic_weighted_score         | weekly                |     100 |         0 |            0.185128 |     0.280364 |  0.744676 |      -0.223101 |       123 |        1.32195  |            0.004015 |          0.004015 |
| multifactor_ic_weighted_score         | weekly                |     100 |        10 |            0.106571 |     0.28025  |  0.499693 |      -0.227496 |       123 |        1.32195  |            0.004015 |          0.002693 |
| multifactor_ic_weighted_score         | weekly                |     100 |        30 |           -0.035535 |     0.28004  |  0.009127 |      -0.248075 |       123 |        1.32195  |            0.004015 |          4.9e-05  |
| multifactor_rolling_ic_weighted_score | daily                 |     100 |         0 |            3.46438  |     0.458277 |  3.49816  |      -0.603981 |       572 |        0.618951 |            0.006362 |          0.006362 |
| multifactor_rolling_ic_weighted_score | daily                 |     100 |        10 |            2.82274  |     0.458289 |  3.15773  |      -0.621946 |       572 |        0.618951 |            0.006362 |          0.005743 |
| multifactor_rolling_ic_weighted_score | daily                 |     100 |        30 |            1.802    |     0.458363 |  2.47664  |      -0.655497 |       572 |        0.618951 |            0.006362 |          0.004505 |
| multifactor_rolling_ic_weighted_score | weekly                |     100 |         0 |            0.328218 |     0.206832 |  1.47949  |      -0.167668 |       123 |        1.2761   |            0.005885 |          0.005885 |
| multifactor_rolling_ic_weighted_score | weekly                |     100 |        10 |            0.24333  |     0.206768 |  1.15903  |      -0.184881 |       123 |        1.2761   |            0.005885 |          0.004609 |
| multifactor_rolling_ic_weighted_score | weekly                |     100 |        30 |            0.089196 |     0.206695 |  0.517358 |      -0.218308 |       123 |        1.2761   |            0.005885 |          0.002056 |
| multifactor_low_corr_rank_score       | daily                 |     100 |         0 |            2.07668  |     0.652281 |  2.04955  |      -0.790284 |       571 |        0.727075 |            0.005305 |          0.005305 |
| multifactor_low_corr_rank_score       | daily                 |     100 |        10 |            1.56322  |     0.652247 |  1.76875  |      -0.793173 |       571 |        0.727075 |            0.005305 |          0.004578 |
| multifactor_low_corr_rank_score       | daily                 |     100 |        30 |            0.77833  |     0.652207 |  1.207    |      -0.798876 |       571 |        0.727075 |            0.005305 |          0.003124 |
| multifactor_low_corr_rank_score       | weekly                |     100 |         0 |            0.204796 |     0.288236 |  0.790184 |      -0.241051 |       123 |        1.45171  |            0.00438  |          0.00438  |
| multifactor_low_corr_rank_score       | weekly                |     100 |        10 |            0.117402 |     0.288156 |  0.528431 |      -0.24546  |       123 |        1.45171  |            0.00438  |          0.002928 |
| multifactor_low_corr_rank_score       | weekly                |     100 |        30 |           -0.03915  |     0.288016 |  0.004488 |      -0.281939 |       123 |        1.45171  |            0.00438  |          2.5e-05  |
| multifactor_neutral_rank_score        | daily                 |     100 |         0 |            1.054    |     0.555073 |  1.56322  |      -0.607046 |       571 |        0.594606 |            0.003443 |          0.003443 |
| multifactor_neutral_rank_score        | daily                 |     100 |        10 |            0.768795 |     0.554958 |  1.29354  |      -0.627666 |       571 |        0.594606 |            0.003443 |          0.002849 |
| multifactor_neutral_rank_score        | daily                 |     100 |        30 |            0.311321 |     0.554764 |  0.753796 |      -0.665755 |       571 |        0.594606 |            0.003443 |          0.001659 |
| multifactor_neutral_rank_score        | weekly                |     100 |         0 |            0.185307 |     0.257221 |  0.786325 |      -0.155549 |       123 |        1.27122  |            0.00389  |          0.00389  |
| multifactor_neutral_rank_score        | weekly                |     100 |        10 |            0.109687 |     0.257041 |  0.529704 |      -0.184271 |       123 |        1.27122  |            0.00389  |          0.002618 |
| multifactor_neutral_rank_score        | weekly                |     100 |        30 |           -0.027642 |     0.256719 |  0.015382 |      -0.261052 |       123 |        1.27122  |            0.00389  |          7.6e-05  |

## Horizon-Aligned Backtest Summary

| signal                                |   horizon | rebalance_frequency   |   top_n |   fee_bps | non_overlapping   |   periods_per_year |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:--------------------------------------|----------:|:----------------------|--------:|----------:|:------------------|-------------------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score                        |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.254391 |     0.235555 |  1.07348  |      -0.145885 |       101 |        0.720594 |            0.005017 |          0.005017 |
| baseline_score                        |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.209869 |     0.235311 |  0.92025  |      -0.151443 |       101 |        0.720594 |            0.005017 |          0.004297 |
| baseline_score                        |         5 | daily                 |     100 |        30 | True              |               50.4 |            0.125416 |     0.234849 |  0.612773 |      -0.162464 |       101 |        0.720594 |            0.005017 |          0.002855 |
| baseline_score                        |         5 | weekly                |     100 |         0 | True              |               50.4 |           -0.118272 |     0.182196 | -0.596821 |      -0.33844  |        92 |        0.75     |           -0.002158 |         -0.002158 |
| baseline_score                        |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.151112 |     0.182379 | -0.803485 |      -0.361282 |        92 |        0.75     |           -0.002158 |         -0.002908 |
| baseline_score                        |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.21324  |     0.182789 | -1.21527  |      -0.431959 |        92 |        0.75     |           -0.002158 |         -0.004408 |
| multifactor_equal_rank_score          |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.208749 |     0.322803 |  0.753959 |      -0.24083  |        58 |        1.39552  |            0.004829 |          0.004829 |
| multifactor_equal_rank_score          |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.12672  |     0.322978 |  0.535784 |      -0.244884 |        58 |        1.39552  |            0.004829 |          0.003433 |
| multifactor_equal_rank_score          |         5 | daily                 |     100 |        30 | True              |               50.4 |           -0.021313 |     0.323345 |  0.100135 |      -0.252951 |        58 |        1.39552  |            0.004829 |          0.000642 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |         0 | True              |               50.4 |           -0.052022 |     0.225665 | -0.116866 |      -0.225384 |        51 |        1.43569  |           -0.000523 |         -0.000523 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.11835  |     0.225657 | -0.437528 |      -0.235479 |        51 |        1.43569  |           -0.000523 |         -0.001959 |
| multifactor_equal_rank_score          |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.237651 |     0.225666 | -1.0788   |      -0.255321 |        51 |        1.43569  |           -0.000523 |         -0.00483  |
| multifactor_ic_weighted_score         |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.2407   |     0.336371 |  0.813971 |      -0.246252 |        58 |        1.35724  |            0.005432 |          0.005432 |
| multifactor_ic_weighted_score         |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.158752 |     0.336572 |  0.610244 |      -0.250204 |        58 |        1.35724  |            0.005432 |          0.004075 |
| multifactor_ic_weighted_score         |         5 | daily                 |     100 |        30 | True              |               50.4 |            0.010447 |     0.336993 |  0.203509 |      -0.258066 |        58 |        1.35724  |            0.005432 |          0.001361 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |         0 | True              |               50.4 |            0.026953 |     0.224647 |  0.237023 |      -0.233775 |        58 |        1.39724  |            0.001056 |          0.001056 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.042927 |     0.224608 | -0.076464 |      -0.243704 |        58 |        1.39724  |            0.001056 |         -0.000341 |
| multifactor_ic_weighted_score         |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.168997 |     0.224557 | -0.703679 |      -0.263222 |        58 |        1.39724  |            0.001056 |         -0.003135 |
| multifactor_rolling_ic_weighted_score |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.319117 |     0.167629 |  1.7404   |      -0.084008 |        64 |        1.29156  |            0.005789 |          0.005789 |
| multifactor_rolling_ic_weighted_score |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.236348 |     0.167673 |  1.35171  |      -0.095207 |        64 |        1.29156  |            0.005789 |          0.004497 |
| multifactor_rolling_ic_weighted_score |         5 | daily                 |     100 |        30 | True              |               50.4 |            0.085776 |     0.167848 |  0.574664 |      -0.125855 |        64 |        1.29156  |            0.005789 |          0.001914 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |         0 | True              |               50.4 |            0.380077 |     0.153011 |  2.18833  |      -0.076641 |        59 |        1.33492  |            0.006644 |          0.006644 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        10 | True              |               50.4 |            0.290725 |     0.153187 |  1.74662  |      -0.080584 |        59 |        1.33492  |            0.006644 |          0.005309 |
| multifactor_rolling_ic_weighted_score |         5 | weekly                |     100 |        30 | True              |               50.4 |            0.128685 |     0.153626 |  0.86574  |      -0.104835 |        59 |        1.33492  |            0.006644 |          0.002639 |
| multifactor_low_corr_rank_score       |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.129589 |     0.291142 |  0.569747 |      -0.258517 |        61 |        1.51279  |            0.003291 |          0.003291 |
| multifactor_low_corr_rank_score       |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.046737 |     0.291092 |  0.30792  |      -0.262412 |        61 |        1.51279  |            0.003291 |          0.001778 |
| multifactor_low_corr_rank_score       |         5 | daily                 |     100 |        30 | True              |               50.4 |           -0.101497 |     0.291004 | -0.215997 |      -0.270294 |        61 |        1.51279  |            0.003291 |         -0.001247 |
| multifactor_low_corr_rank_score       |         5 | weekly                |     100 |         0 | True              |               50.4 |            0.073098 |     0.246329 |  0.415208 |      -0.241051 |        57 |        1.53404  |            0.002029 |          0.002029 |
| multifactor_low_corr_rank_score       |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.006694 |     0.246115 |  0.101426 |      -0.253296 |        57 |        1.53404  |            0.002029 |          0.000495 |
| multifactor_low_corr_rank_score       |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.149227 |     0.245704 | -0.527742 |      -0.277588 |        57 |        1.53404  |            0.002029 |         -0.002573 |
| multifactor_neutral_rank_score        |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.200621 |     0.20778  |  0.98195  |      -0.110337 |        54 |        1.26111  |            0.004048 |          0.004048 |
| multifactor_neutral_rank_score        |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.126963 |     0.207338 |  0.677491 |      -0.115097 |        54 |        1.26111  |            0.004048 |          0.002787 |
| multifactor_neutral_rank_score        |         5 | daily                 |     100 |        30 | True              |               50.4 |           -0.007321 |     0.206503 |  0.064647 |      -0.151436 |        54 |        1.26111  |            0.004048 |          0.000265 |
| multifactor_neutral_rank_score        |         5 | weekly                |     100 |         0 | True              |               50.4 |            0.117837 |     0.193969 |  0.674074 |      -0.116633 |        49 |        1.30082  |            0.002594 |          0.002594 |
| multifactor_neutral_rank_score        |         5 | weekly                |     100 |        10 | True              |               50.4 |            0.047036 |     0.193678 |  0.33658  |      -0.120467 |        49 |        1.30082  |            0.002594 |          0.001293 |
| multifactor_neutral_rank_score        |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.081643 |     0.193156 | -0.341352 |      -0.136771 |        49 |        1.30082  |            0.002594 |         -0.001308 |

## Horizon-Aligned Yearly Summary

| signal                        |   year |   horizon | rebalance_frequency   |   top_n |   fee_bps | non_overlapping   |   periods_per_year |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:------------------------------|-------:|----------:|:----------------------|--------:|----------:|:------------------|-------------------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score                |   2024 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.225789 |     0.316786 |  0.787933 |      -0.101768 |        44 |        0.69     |            0.004953 |          0.004953 |
| baseline_score                |   2025 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.193814 |     0.123353 |  1.49991  |      -0.051157 |        40 |        0.734    |            0.003671 |          0.003671 |
| baseline_score                |   2026 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.496028 |     0.182584 |  2.30541  |      -0.038096 |        17 |        0.768235 |            0.008352 |          0.008352 |
| baseline_score                |   2024 |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.184067 |     0.31656  |  0.678639 |      -0.106771 |        44 |        0.69     |            0.004953 |          0.004263 |
| baseline_score                |   2025 |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.150614 |     0.123142 |  1.20206  |      -0.057881 |        40 |        0.734    |            0.003671 |          0.002937 |
| baseline_score                |   2026 |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.43974  |     0.182096 |  2.09895  |      -0.040444 |        17 |        0.768235 |            0.008352 |          0.007584 |
| baseline_score                |   2024 |         5 | daily                 |     100 |        30 | True              |               50.4 |            0.104749 |     0.316132 |  0.459549 |      -0.126554 |        44 |        0.69     |            0.004953 |          0.002883 |
| baseline_score                |   2025 |         5 | daily                 |     100 |        30 | True              |               50.4 |            0.068758 |     0.12275  |  0.603153 |      -0.071199 |        40 |        0.734    |            0.003671 |          0.001469 |
| baseline_score                |   2026 |         5 | daily                 |     100 |        30 | True              |               50.4 |            0.33331  |     0.18116  |  1.68233  |      -0.045128 |        17 |        0.768235 |            0.008352 |          0.006047 |
| baseline_score                |   2024 |         5 | weekly                |     100 |         0 | True              |               50.4 |           -0.143804 |     0.192605 | -0.7061   |      -0.210628 |        39 |        0.694359 |           -0.002698 |         -0.002698 |
| baseline_score                |   2025 |         5 | weekly                |     100 |         0 | True              |               50.4 |           -0.152623 |     0.180789 | -0.821788 |      -0.161916 |        38 |        0.776842 |           -0.002948 |         -0.002948 |
| baseline_score                |   2026 |         5 | weekly                |     100 |         0 | True              |               50.4 |            0.052526 |     0.153763 |  0.410001 |      -0.06647  |        15 |        0.826667 |            0.001251 |          0.001251 |
| baseline_score                |   2024 |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.173473 |     0.193302 | -0.884596 |      -0.226982 |        39 |        0.694359 |           -0.002698 |         -0.003393 |
| baseline_score                |   2025 |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.18523  |     0.180413 | -1.04052  |      -0.172761 |        38 |        0.776842 |           -0.002948 |         -0.003725 |
| baseline_score                |   2026 |         5 | weekly                |     100 |        10 | True              |               50.4 |            0.009549 |     0.154026 |  0.138802 |      -0.073214 |        15 |        0.826667 |            0.001251 |          0.000424 |
| baseline_score                |   2024 |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.229828 |     0.19473  | -1.23753  |      -0.258719 |        39 |        0.694359 |           -0.002698 |         -0.004781 |
| baseline_score                |   2025 |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.246802 |     0.179698 | -1.48042  |      -0.198928 |        38 |        0.776842 |           -0.002948 |         -0.005278 |
| baseline_score                |   2026 |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.071318 |     0.154615 | -0.400665 |      -0.086578 |        15 |        0.826667 |            0.001251 |         -0.001229 |
| multifactor_equal_rank_score  |   2024 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.055722 |     0.596177 |  0.398744 |      -0.24083  |        14 |        1.45714  |            0.004717 |          0.004717 |
| multifactor_equal_rank_score  |   2025 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.488802 |     0.147594 |  2.77873  |      -0.050175 |        25 |        1.3936   |            0.008137 |          0.008137 |
| multifactor_equal_rank_score  |   2026 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.01525  |     0.160894 |  0.174954 |      -0.082519 |        19 |        1.35263  |            0.000559 |          0.000559 |
| multifactor_equal_rank_score  |   2024 |         5 | daily                 |     100 |        10 | True              |               50.4 |           -0.019706 |     0.596775 |  0.275283 |      -0.244884 |        14 |        1.45714  |            0.004717 |          0.00326  |
| multifactor_equal_rank_score  |   2025 |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.388527 |     0.147467 |  2.30483  |      -0.056547 |        25 |        1.3936   |            0.008137 |          0.006744 |
| multifactor_equal_rank_score  |   2026 |         5 | daily                 |     100 |        10 | True              |               50.4 |           -0.051641 |     0.160535 | -0.249313 |      -0.097362 |        19 |        1.35263  |            0.000559 |         -0.000794 |
| multifactor_equal_rank_score  |   2024 |         5 | daily                 |     100 |        30 | True              |               50.4 |           -0.155071 |     0.597983 |  0.029101 |      -0.252951 |        14 |        1.45714  |            0.004717 |          0.000345 |
| multifactor_equal_rank_score  |   2025 |         5 | daily                 |     100 |        30 | True              |               50.4 |            0.207429 |     0.147241 |  1.35433  |      -0.069188 |        25 |        1.3936   |            0.008137 |          0.003957 |
| multifactor_equal_rank_score  |   2026 |         5 | daily                 |     100 |        30 | True              |               50.4 |           -0.172727 |     0.159855 | -1.10331  |      -0.126391 |        19 |        1.35263  |            0.000559 |         -0.003499 |
| multifactor_equal_rank_score  |   2024 |         5 | weekly                |     100 |         0 | True              |               50.4 |           -0.669321 |     0.379321 | -2.67902  |      -0.225384 |        10 |        1.412    |           -0.020163 |         -0.020163 |
| multifactor_equal_rank_score  |   2025 |         5 | weekly                |     100 |         0 | True              |               50.4 |            0.43937  |     0.137928 |  2.71902  |      -0.04025  |        24 |        1.44333  |            0.007441 |          0.007441 |
| multifactor_equal_rank_score  |   2026 |         5 | weekly                |     100 |         0 | True              |               50.4 |           -0.023226 |     0.15914  | -0.067865 |      -0.072855 |        17 |        1.43882  |           -0.000214 |         -0.000214 |
| multifactor_equal_rank_score  |   2024 |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.692637 |     0.379726 | -2.86358  |      -0.235479 |        10 |        1.412    |           -0.020163 |         -0.021575 |
| multifactor_equal_rank_score  |   2025 |         5 | weekly                |     100 |        10 | True              |               50.4 |            0.339029 |     0.137737 |  2.19466  |      -0.04179  |        24 |        1.44333  |            0.007441 |          0.005998 |
| multifactor_equal_rank_score  |   2026 |         5 | weekly                |     100 |        10 | True              |               50.4 |           -0.091609 |     0.158883 | -0.524389 |      -0.080127 |        17 |        1.43882  |           -0.000214 |         -0.001653 |
| multifactor_equal_rank_score  |   2024 |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.734543 |     0.380567 | -3.23124  |      -0.255321 |        10 |        1.412    |           -0.020163 |         -0.024399 |
| multifactor_equal_rank_score  |   2025 |         5 | weekly                |     100 |        30 | True              |               50.4 |            0.158479 |     0.137386 |  1.14129  |      -0.050555 |        24 |        1.44333  |            0.007441 |          0.003111 |
| multifactor_equal_rank_score  |   2026 |         5 | weekly                |     100 |        30 | True              |               50.4 |           -0.214597 |     0.158397 | -1.44163  |      -0.111906 |        17 |        1.43882  |           -0.000214 |         -0.004531 |
| multifactor_ic_weighted_score |   2024 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.078045 |     0.623265 |  0.440057 |      -0.246252 |        14 |        1.41714  |            0.005442 |          0.005442 |
| multifactor_ic_weighted_score |   2025 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.537113 |     0.152565 |  2.90374  |      -0.046847 |        25 |        1.3496   |            0.00879  |          0.00879  |
| multifactor_ic_weighted_score |   2026 |         5 | daily                 |     100 |         0 | True              |               50.4 |            0.038045 |     0.163696 |  0.310307 |      -0.079678 |        19 |        1.32316  |            0.001008 |          0.001008 |
| multifactor_ic_weighted_score |   2024 |         5 | daily                 |     100 |        10 | True              |               50.4 |            0.003028 |     0.623877 |  0.325142 |      -0.250204 |        14 |        1.41714  |            0.005442 |          0.004025 |

Artifacts: `traditional_quant_research\output\experiments\multifactor_baseline\multifactor_baseline_20260602_115939`
