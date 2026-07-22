# 2026-06-02 Multifactor Baseline

- Hypothesis: 方向校正后的传统价量因子 rank 合成，应该比第一阶段等权 z-score `baseline_score` 更适合作为第二阶段多因子诊断基线。
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2024-01-01` to `2026-06-01`.
- Panel: `1779897` rows, `581` dates, `3233` securities.
- Label: `fwd_ret_5d`; label mode: `raw`; raw label: `fwd_ret_5d`.
- Label Mode Note: `raw` keeps the same-date market move in the label, so IC and quantile returns are closer to absolute return diagnostics but still need execution-aware portfolio validation.
- Backtest Return Note: Top-N backtests use the generated 5-day forward return as each selected rebalance-date return. Daily or otherwise overlapping rebalance schedules can overstate portfolio PnL; treat these rows as holding-period ranking diagnostics until a horizon-aligned portfolio simulator is used.
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

| signal                       | rebalance_frequency   |   top_n |   fee_bps |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:-----------------------------|:----------------------|--------:|----------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score               | daily                 |      50 |         0 |            0.913291 |     0.519667 |  1.50317  |      -0.705247 |       572 |        0.378671 |            0.0031   |          0.0031   |
| baseline_score               | daily                 |      50 |         5 |            0.824339 |     0.519637 |  1.41144  |      -0.713259 |       572 |        0.378671 |            0.0031   |          0.00291  |
| baseline_score               | daily                 |      50 |        10 |            0.739504 |     0.519609 |  1.31969  |      -0.721055 |       572 |        0.378671 |            0.0031   |          0.002721 |
| baseline_score               | daily                 |      50 |        20 |            0.581438 |     0.51956  |  1.13615  |      -0.736022 |       572 |        0.378671 |            0.0031   |          0.002342 |
| baseline_score               | daily                 |      50 |        30 |            0.437676 |     0.519521 |  0.952552 |      -0.750194 |       572 |        0.378671 |            0.0031   |          0.001964 |
| baseline_score               | daily                 |     100 |         0 |            1.15     |     0.458715 |  1.89248  |      -0.56321  |       572 |        0.334825 |            0.003445 |          0.003445 |
| baseline_score               | daily                 |     100 |         5 |            1.0614   |     0.458718 |  1.8005   |      -0.57399  |       572 |        0.334825 |            0.003445 |          0.003277 |
| baseline_score               | daily                 |     100 |        10 |            0.976437 |     0.458724 |  1.70851  |      -0.584505 |       572 |        0.334825 |            0.003445 |          0.00311  |
| baseline_score               | daily                 |     100 |        20 |            0.81683  |     0.45874  |  1.52452  |      -0.604769 |       572 |        0.334825 |            0.003445 |          0.002775 |
| baseline_score               | daily                 |     100 |        30 |            0.670058 |     0.458764 |  1.34052  |      -0.624051 |       572 |        0.334825 |            0.003445 |          0.00244  |
| baseline_score               | weekly                |      50 |         0 |            0.056097 |     0.216047 |  0.361495 |      -0.270194 |       123 |        0.757073 |            0.001502 |          0.001502 |
| baseline_score               | weekly                |      50 |         5 |            0.035523 |     0.216034 |  0.270401 |      -0.277498 |       123 |        0.757073 |            0.001502 |          0.001123 |
| baseline_score               | weekly                |      50 |        10 |            0.015342 |     0.216025 |  0.179294 |      -0.28695  |       123 |        0.757073 |            0.001502 |          0.000745 |
| baseline_score               | weekly                |      50 |        20 |           -0.023871 |     0.216014 | -0.002943 |      -0.317123 |       123 |        0.757073 |            0.001502 |         -1.2e-05  |
| baseline_score               | weekly                |      50 |        30 |           -0.0616   |     0.216015 | -0.185189 |      -0.348619 |       123 |        0.757073 |            0.001502 |         -0.000769 |
| baseline_score               | weekly                |     100 |         0 |            0.076988 |     0.196003 |  0.476984 |      -0.260036 |       123 |        0.664715 |            0.001798 |          0.001798 |
| baseline_score               | weekly                |     100 |         5 |            0.058546 |     0.19603  |  0.388753 |      -0.268052 |       123 |        0.664715 |            0.001798 |          0.001466 |
| baseline_score               | weekly                |     100 |        10 |            0.040414 |     0.19606  |  0.300545 |      -0.277333 |       123 |        0.664715 |            0.001798 |          0.001133 |
| baseline_score               | weekly                |     100 |        20 |            0.005057 |     0.196126 |  0.124204 |      -0.295551 |       123 |        0.664715 |            0.001798 |          0.000468 |
| baseline_score               | weekly                |     100 |        30 |           -0.029122 |     0.1962   | -0.052016 |      -0.318401 |       123 |        0.664715 |            0.001798 |         -0.000196 |
| baseline_score               | monthly               |      50 |         0 |           -0.056567 |     0.118794 | -0.428249 |      -0.227    |        29 |        1.30345  |           -0.004239 |         -0.004239 |
| baseline_score               | monthly               |      50 |         5 |           -0.063965 |     0.118842 | -0.493884 |      -0.235686 |        29 |        1.30345  |           -0.004239 |         -0.004891 |
| baseline_score               | monthly               |      50 |        10 |           -0.07131  |     0.118891 | -0.559461 |      -0.245344 |        29 |        1.30345  |           -0.004239 |         -0.005543 |
| baseline_score               | monthly               |      50 |        20 |           -0.085842 |     0.118992 | -0.690436 |      -0.264314 |        29 |        1.30345  |           -0.004239 |         -0.006846 |
| baseline_score               | monthly               |      50 |        30 |           -0.100166 |     0.119097 | -0.821159 |      -0.282832 |        29 |        1.30345  |           -0.004239 |         -0.00815  |
| baseline_score               | monthly               |     100 |         0 |           -0.03562  |     0.106652 | -0.284634 |      -0.204685 |        29 |        1.15517  |           -0.00253  |         -0.00253  |
| baseline_score               | monthly               |     100 |         5 |           -0.042312 |     0.106706 | -0.349445 |      -0.212599 |        29 |        1.15517  |           -0.00253  |         -0.003107 |
| baseline_score               | monthly               |     100 |        10 |           -0.048962 |     0.10676  | -0.414188 |      -0.220589 |        29 |        1.15517  |           -0.00253  |         -0.003685 |
| baseline_score               | monthly               |     100 |        20 |           -0.062135 |     0.106872 | -0.543464 |      -0.23634  |        29 |        1.15517  |           -0.00253  |         -0.00484  |
| baseline_score               | monthly               |     100 |        30 |           -0.075142 |     0.106986 | -0.672452 |      -0.251792 |        29 |        1.15517  |           -0.00253  |         -0.005995 |
| multifactor_equal_rank_score | daily                 |      50 |         0 |            1.45241  |     0.636291 |  1.72305  |      -0.7605   |       572 |        0.748741 |            0.004351 |          0.004351 |
| multifactor_equal_rank_score | daily                 |      50 |         5 |            1.23217  |     0.636296 |  1.57477  |      -0.762363 |       572 |        0.748741 |            0.004351 |          0.003976 |
| multifactor_equal_rank_score | daily                 |      50 |        10 |            1.03164  |     0.636304 |  1.42649  |      -0.764323 |       572 |        0.748741 |            0.004351 |          0.003602 |
| multifactor_equal_rank_score | daily                 |      50 |        20 |            0.68281  |     0.63633  |  1.12991  |      -0.768196 |       572 |        0.748741 |            0.004351 |          0.002853 |
| multifactor_equal_rank_score | daily                 |      50 |        30 |            0.393665 |     0.636369 |  0.833342 |      -0.77201  |       572 |        0.748741 |            0.004351 |          0.002104 |
| multifactor_equal_rank_score | daily                 |     100 |         0 |            1.444    |     0.623233 |  1.74294  |      -0.765792 |       572 |        0.666049 |            0.004311 |          0.004311 |
| multifactor_equal_rank_score | daily                 |     100 |         5 |            1.24771  |     0.623274 |  1.60818  |      -0.767488 |       572 |        0.666049 |            0.004311 |          0.003978 |
| multifactor_equal_rank_score | daily                 |     100 |        10 |            1.06713  |     0.623317 |  1.47343  |      -0.769173 |       572 |        0.666049 |            0.004311 |          0.003644 |
| multifactor_equal_rank_score | daily                 |     100 |        20 |            0.74817  |     0.62341  |  1.20397  |      -0.772508 |       572 |        0.666049 |            0.004311 |          0.002978 |
| multifactor_equal_rank_score | daily                 |     100 |        30 |            0.478252 |     0.623512 |  0.934584 |      -0.775963 |       572 |        0.666049 |            0.004311 |          0.002312 |

Artifacts: `traditional_quant_research\output\experiments\multifactor_baseline\multifactor_baseline_20260602_113107`
