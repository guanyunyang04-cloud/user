# 2026-06-02 Multifactor Baseline

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_multifactor_2024_2026_validation.md`.


- Hypothesis: 方向校正后的传统价量因子 rank 合成，应该比第一阶段等权 z-score `baseline_score` 更适合作为第二阶段多因子诊断基线。
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2024-01-01` to `2026-06-01`.
- Panel: `1779897` rows, `581` dates, `3233` securities.
- Label: `xsec_excess_ret_5d`; label mode: `xsec-excess`; raw label: `fwd_ret_5d`.
- Label Mode Note: `xsec-excess` subtracts the same-date tradeable-universe mean forward return, so Top-N metrics are alpha-style ranking diagnostics rather than standalone investable portfolio returns.
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

| signal                | label              |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:----------------------|:-------------------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| reversal_5d_z         | xsec_excess_ret_5d |     571 |        1747582 |  0.034002 |       0.038218 |      0.155595 |     3.8992  |
| momentum_20d_z        | xsec_excess_ret_5d |     556 |        1699308 | -0.034229 |      -0.062525 |      0.171637 |    -5.78283 |
| ma20_gap_z            | xsec_excess_ret_5d |     572 |        1750812 | -0.033239 |      -0.048509 |      0.166173 |    -4.63409 |
| neg_volatility_20d_z  | xsec_excess_ret_5d |     571 |        1747582 |  0.005839 |       0.046885 |      0.209474 |     3.55307 |
| log_amount_mean_20d_z | xsec_excess_ret_5d |     572 |        1750812 | -0.015752 |      -0.050422 |      0.170305 |    -4.69996 |
| neg_amplitude_20d_z   | xsec_excess_ret_5d |     572 |        1750812 | -0.002593 |       0.043964 |      0.220912 |     3.15918 |

## Multifactor IC

| signal                                | label              |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:--------------------------------------|:-------------------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| baseline_score                        | xsec_excess_ret_5d |     572 |        1750812 | -0.013948 |      -0.002236 |      0.221414 |   -0.160316 |
| multifactor_equal_rank_score          | xsec_excess_ret_5d |     572 |        1750812 |  0.017332 |       0.071564 |      0.151898 |    7.47897  |
| multifactor_ic_weighted_score         | xsec_excess_ret_5d |     572 |        1750812 |  0.018611 |       0.072937 |      0.153238 |    7.55584  |
| multifactor_rolling_ic_weighted_score | xsec_excess_ret_5d |     572 |        1750812 |  0.018168 |       0.072146 |      0.18766  |    6.10295  |
| multifactor_low_corr_rank_score       | xsec_excess_ret_5d |     571 |        1747582 |  0.022431 |       0.074406 |      0.155219 |    7.6096   |
| multifactor_neutral_rank_score        | xsec_excess_ret_5d |     571 |        1747582 |  0.013544 |       0.055229 |      0.140237 |    6.25184  |

## Quantile Returns

| signal                                |   quantile |   mean_return |   periods |   observations |
|:--------------------------------------|-----------:|--------------:|----------:|---------------:|
| baseline_score                        |          1 |      0.001278 |       572 |         350396 |
| baseline_score                        |          2 |      0.000781 |       572 |         350046 |
| baseline_score                        |          3 |     -3.5e-05  |       572 |         350055 |
| baseline_score                        |          4 |     -0.0009   |       572 |         350046 |
| baseline_score                        |          5 |     -0.001033 |       572 |         350269 |
| multifactor_equal_rank_score          |          1 |     -0.002071 |       572 |         350396 |
| multifactor_equal_rank_score          |          2 |      0.000848 |       572 |         350046 |
| multifactor_equal_rank_score          |          3 |      0.000682 |       572 |         350055 |
| multifactor_equal_rank_score          |          4 |      0.000549 |       572 |         350046 |
| multifactor_equal_rank_score          |          5 |      8.6e-05  |       572 |         350269 |
| multifactor_ic_weighted_score         |          1 |     -0.002209 |       572 |         350396 |
| multifactor_ic_weighted_score         |          2 |      0.000743 |       572 |         350046 |
| multifactor_ic_weighted_score         |          3 |      0.000717 |       572 |         350055 |
| multifactor_ic_weighted_score         |          4 |      0.000581 |       572 |         350046 |
| multifactor_ic_weighted_score         |          5 |      0.000263 |       572 |         350269 |
| multifactor_rolling_ic_weighted_score |          1 |     -0.001804 |       572 |         350396 |
| multifactor_rolling_ic_weighted_score |          2 |      0.000458 |       572 |         350046 |
| multifactor_rolling_ic_weighted_score |          3 |      7.6e-05  |       572 |         350055 |
| multifactor_rolling_ic_weighted_score |          4 |      0.000213 |       572 |         350046 |
| multifactor_rolling_ic_weighted_score |          5 |      0.00115  |       572 |         350269 |
| multifactor_low_corr_rank_score       |          1 |     -0.002343 |       571 |         349742 |
| multifactor_low_corr_rank_score       |          2 |      0.000539 |       571 |         349403 |
| multifactor_low_corr_rank_score       |          3 |      0.000504 |       571 |         349411 |
| multifactor_low_corr_rank_score       |          4 |      0.000669 |       571 |         349403 |
| multifactor_low_corr_rank_score       |          5 |      0.00073  |       571 |         349623 |
| multifactor_neutral_rank_score        |          1 |     -0.002073 |       571 |         349742 |
| multifactor_neutral_rank_score        |          2 |      0.000898 |       571 |         349403 |
| multifactor_neutral_rank_score        |          3 |      0.000683 |       571 |         349411 |
| multifactor_neutral_rank_score        |          4 |      0.000723 |       571 |         349403 |
| multifactor_neutral_rank_score        |          5 |     -0.00013  |       571 |         349623 |

## Quantile Spread

| signal                                |   low_quantile |   high_quantile |   low_mean_return |   high_mean_return |   q_high_minus_low |
|:--------------------------------------|---------------:|----------------:|------------------:|-------------------:|-------------------:|
| baseline_score                        |              1 |               5 |          0.001278 |          -0.001033 |          -0.00231  |
| multifactor_equal_rank_score          |              1 |               5 |         -0.002071 |           8.6e-05  |           0.002158 |
| multifactor_ic_weighted_score         |              1 |               5 |         -0.002209 |           0.000263 |           0.002471 |
| multifactor_low_corr_rank_score       |              1 |               5 |         -0.002343 |           0.00073  |           0.003073 |
| multifactor_neutral_rank_score        |              1 |               5 |         -0.002073 |          -0.00013  |           0.001943 |
| multifactor_rolling_ic_weighted_score |              1 |               5 |         -0.001804 |           0.00115  |           0.002954 |

## Backtest Summary

| signal                                | rebalance_frequency   |   top_n |   fee_bps |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:--------------------------------------|:----------------------|--------:|----------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score                        | daily                 |      50 |         0 |           -0.344391 |     0.495862 | -0.604292 |      -0.943393 |       572 |        0.378671 |           -0.001189 |         -0.001189 |
| baseline_score                        | daily                 |      50 |        10 |           -0.404169 |     0.495805 | -0.796826 |      -0.953029 |       572 |        0.378671 |           -0.001189 |         -0.001568 |
| baseline_score                        | daily                 |     100 |         0 |           -0.262869 |     0.43176  | -0.49259  |      -0.918687 |       572 |        0.334825 |           -0.000844 |         -0.000844 |
| baseline_score                        | daily                 |     100 |        10 |           -0.322608 |     0.431775 | -0.687989 |      -0.93125  |       572 |        0.334825 |           -0.000844 |         -0.001179 |
| baseline_score                        | weekly                |      50 |         0 |           -0.1085   |     0.200216 | -0.47364  |      -0.457871 |       123 |        0.757073 |           -0.001824 |         -0.001824 |
| baseline_score                        | weekly                |      50 |        10 |           -0.142989 |     0.200079 | -0.670726 |      -0.495226 |       123 |        0.757073 |           -0.001824 |         -0.002581 |
| baseline_score                        | weekly                |     100 |         0 |           -0.090621 |     0.176522 | -0.450028 |      -0.432278 |       123 |        0.664715 |           -0.001528 |         -0.001528 |
| baseline_score                        | weekly                |     100 |        10 |           -0.121592 |     0.17654  | -0.645775 |      -0.464205 |       123 |        0.664715 |           -0.001528 |         -0.002192 |
| baseline_score                        | monthly               |      50 |         0 |           -0.020688 |     0.124221 | -0.106562 |      -0.245415 |        29 |        1.30345  |           -0.001103 |         -0.001103 |
| baseline_score                        | monthly               |      50 |        10 |           -0.035953 |     0.124406 | -0.232132 |      -0.262537 |        29 |        1.30345  |           -0.001103 |         -0.002407 |
| baseline_score                        | monthly               |     100 |         0 |            0.001444 |     0.109853 |  0.066265 |      -0.198796 |        29 |        1.15517  |            0.000607 |          0.000607 |
| baseline_score                        | monthly               |     100 |        10 |           -0.012369 |     0.109987 | -0.05985  |      -0.21421  |        29 |        1.15517  |            0.000607 |         -0.000549 |
| multifactor_equal_rank_score          | daily                 |      50 |         0 |           -0.019629 |     0.270057 |  0.057651 |      -0.447119 |       572 |        0.748741 |            6.2e-05  |          6.2e-05  |
| multifactor_equal_rank_score          | daily                 |      50 |        10 |           -0.18831  |     0.270118 | -0.640882 |      -0.548056 |       572 |        0.748741 |            6.2e-05  |         -0.000687 |
| multifactor_equal_rank_score          | daily                 |     100 |         0 |           -0.024312 |     0.247564 |  0.022075 |      -0.348548 |       572 |        0.666049 |            2.2e-05  |          2.2e-05  |
| multifactor_equal_rank_score          | daily                 |     100 |        10 |           -0.175161 |     0.247637 | -0.655715 |      -0.461696 |       572 |        0.666049 |            2.2e-05  |         -0.000644 |
| multifactor_equal_rank_score          | weekly                |      50 |         0 |            0.012519 |     0.140539 |  0.155129 |      -0.111062 |       123 |        1.48748  |            0.000419 |          0.000419 |
| multifactor_equal_rank_score          | weekly                |      50 |        10 |           -0.062899 |     0.14056  | -0.395187 |      -0.220147 |       123 |        1.48748  |            0.000419 |         -0.001068 |
| multifactor_equal_rank_score          | weekly                |     100 |         0 |            0.016929 |     0.120772 |  0.197504 |      -0.086325 |       123 |        1.37024  |            0.000459 |          0.000459 |
| multifactor_equal_rank_score          | weekly                |     100 |        10 |           -0.053047 |     0.120805 | -0.392367 |      -0.180789 |       123 |        1.37024  |            0.000459 |         -0.000912 |
| multifactor_equal_rank_score          | monthly               |      50 |         0 |           -0.025852 |     0.054108 | -0.456022 |      -0.087886 |        29 |        1.81793  |           -0.002056 |         -0.002056 |
| multifactor_equal_rank_score          | monthly               |      50 |        10 |           -0.046916 |     0.053662 | -0.86635  |      -0.125114 |        29 |        1.81793  |           -0.002056 |         -0.003874 |
| multifactor_equal_rank_score          | monthly               |     100 |         0 |           -0.026735 |     0.055389 | -0.460312 |      -0.08568  |        29 |        1.75172  |           -0.002125 |         -0.002125 |
| multifactor_equal_rank_score          | monthly               |     100 |        10 |           -0.047023 |     0.054978 | -0.846103 |      -0.128686 |        29 |        1.75172  |           -0.002125 |         -0.003876 |
| multifactor_ic_weighted_score         | daily                 |      50 |         0 |            0.028884 |     0.275644 |  0.236447 |      -0.393796 |       572 |        0.715524 |            0.000259 |          0.000259 |
| multifactor_ic_weighted_score         | daily                 |      50 |        10 |           -0.140934 |     0.275644 | -0.417702 |      -0.497881 |       572 |        0.715524 |            0.000259 |         -0.000457 |
| multifactor_ic_weighted_score         | daily                 |     100 |         0 |            0.050045 |     0.25737  |  0.31564  |      -0.327664 |       572 |        0.638252 |            0.000322 |          0.000322 |
| multifactor_ic_weighted_score         | daily                 |     100 |        10 |           -0.106015 |     0.257444 | -0.309206 |      -0.421574 |       572 |        0.638252 |            0.000322 |         -0.000316 |
| multifactor_ic_weighted_score         | weekly                |      50 |         0 |            0.023602 |     0.142728 |  0.230824 |      -0.093079 |       123 |        1.43642  |            0.000634 |          0.000634 |
| multifactor_ic_weighted_score         | weekly                |      50 |        10 |           -0.050108 |     0.14277  | -0.292422 |      -0.202226 |       123 |        1.43642  |            0.000634 |         -0.000803 |
| multifactor_ic_weighted_score         | weekly                |     100 |         0 |            0.028679 |     0.125362 |  0.285973 |      -0.087785 |       123 |        1.32195  |            0.000689 |          0.000689 |
| multifactor_ic_weighted_score         | weekly                |     100 |        10 |           -0.039676 |     0.125354 | -0.262387 |      -0.167004 |       123 |        1.32195  |            0.000689 |         -0.000633 |
| multifactor_ic_weighted_score         | monthly               |      50 |         0 |           -0.027718 |     0.056096 | -0.471981 |      -0.095878 |        29 |        1.83586  |           -0.002206 |         -0.002206 |
| multifactor_ic_weighted_score         | monthly               |      50 |        10 |           -0.04895  |     0.055662 | -0.871448 |      -0.129564 |        29 |        1.83586  |           -0.002206 |         -0.004042 |
| multifactor_ic_weighted_score         | monthly               |     100 |         0 |           -0.023898 |     0.055735 | -0.404928 |      -0.080939 |        29 |        1.76207  |           -0.001881 |         -0.001881 |
| multifactor_ic_weighted_score         | monthly               |     100 |        10 |           -0.044358 |     0.055307 | -0.790379 |      -0.122632 |        29 |        1.76207  |           -0.001881 |         -0.003643 |
| multifactor_rolling_ic_weighted_score | daily                 |      50 |         0 |            0.605639 |     0.353212 |  1.51373  |      -0.512174 |       572 |        0.689301 |            0.002122 |          0.002122 |
| multifactor_rolling_ic_weighted_score | daily                 |      50 |        10 |            0.34982  |     0.35348  |  1.02117  |      -0.519482 |       572 |        0.689301 |            0.002122 |          0.001432 |
| multifactor_rolling_ic_weighted_score | daily                 |     100 |         0 |            0.593572 |     0.338546 |  1.54288  |      -0.525802 |       572 |        0.618951 |            0.002073 |          0.002073 |
| multifactor_rolling_ic_weighted_score | daily                 |     100 |        10 |            0.363591 |     0.338877 |  1.0811   |      -0.531368 |       572 |        0.618951 |            0.002073 |          0.001454 |

Artifacts: `traditional_quant_research\output\experiments\multifactor_baseline\multifactor_baseline_20260602_112236`
