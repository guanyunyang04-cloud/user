# 2026-06-02 Multifactor Baseline

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_multifactor_baseline.md`.


- Hypothesis: 方向校正后的传统价量因子 rank 合成，应该比第一阶段等权 z-score `baseline_score` 更适合作为第二阶段多因子诊断基线。
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2026-01-01` to `2026-06-01`.
- Panel: `292483` rows, `96` dates, `3098` securities.
- Label: `xsec_excess_ret_5d`; label mode: `xsec-excess`; raw label: `fwd_ret_5d`.
- Label Mode Note: `xsec-excess` subtracts the same-date tradeable-universe mean forward return, so Top-N metrics are alpha-style ranking diagnostics rather than standalone investable portfolio returns.
- Rolling IC: window `60`, min periods `20`, fallback date rate `0.250000`.
- Low-Correlation Factors: max abs corr `0.75`, selected `['log_amount_mean_20d_z', 'neg_volatility_20d_z', 'ma20_gap_z', 'reversal_5d_z']`.
- Neutralization: proxy columns `['log_amount_mean_20d_z']`, neutralized factors `['reversal_5d_z_neutral', 'momentum_20d_z_neutral', 'ma20_gap_z_neutral', 'neg_volatility_20d_z_neutral', 'neg_amplitude_20d_z_neutral']`.
- Signals: `['baseline_score', 'multifactor_equal_rank_score', 'multifactor_ic_weighted_score', 'multifactor_rolling_ic_weighted_score', 'multifactor_low_corr_rank_score', 'multifactor_neutral_rank_score']`.
- Factor Directions: `{'reversal_5d_z': 1, 'momentum_20d_z': 1, 'ma20_gap_z': -1, 'neg_volatility_20d_z': 1, 'log_amount_mean_20d_z': -1, 'neg_amplitude_20d_z': 1}`.
- Result: best IC signal `multifactor_equal_rank_score`; best quantile-spread signal `baseline_score`; best backtest signal `multifactor_low_corr_rank_score` under the tested Top-N/cost/frequency grid.
- Consistency Check: IC、分位 spread 和 Top-N 最优信号 `不一致`；若不一致，本实验只作为诊断证据，不能升级为策略结论。
- Assessment: this is a phase-2 diagnostic baseline; it is not yet a traditional ML model or production strategy candidate.
- Next Step: add industry/size neutralization and longer execution-constrained backtest grids.

## Factor Coverage

| factor                |   rows |   available_rows |   missing_rows |   coverage_rate |
|:----------------------|-------:|-----------------:|---------------:|----------------:|
| reversal_5d_z         | 292483 |           277013 |          15470 |        0.947108 |
| momentum_20d_z        | 292483 |           230784 |          61699 |        0.789051 |
| ma20_gap_z            | 292483 |           280106 |          12377 |        0.957683 |
| neg_volatility_20d_z  | 292483 |           277013 |          15470 |        0.947108 |
| log_amount_mean_20d_z | 292483 |           280106 |          12377 |        0.957683 |
| neg_amplitude_20d_z   | 292483 |           280106 |          12377 |        0.957683 |

## Single Factor IC

| signal                | label              |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:----------------------|:-------------------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| reversal_5d_z         | xsec_excess_ret_5d |      86 |         261566 |  0.00318  |       0.013865 |      0.131975 |     1.66771 |
| momentum_20d_z        | xsec_excess_ret_5d |      71 |         215414 |  0.041155 |       0.008644 |      0.129444 |     1.06001 |
| ma20_gap_z            | xsec_excess_ret_5d |      87 |         264650 |  0.006624 |      -0.017037 |      0.126356 |    -2.14046 |
| neg_volatility_20d_z  | xsec_excess_ret_5d |      86 |         261566 | -0.020256 |       0.031269 |      0.159177 |     3.11843 |
| log_amount_mean_20d_z | xsec_excess_ret_5d |      87 |         264650 |  0.009631 |      -0.03175  |      0.150244 |    -3.35468 |
| neg_amplitude_20d_z   | xsec_excess_ret_5d |      87 |         264650 | -0.030046 |       0.027014 |      0.169583 |     2.52875 |

## Multifactor IC

| signal                                | label              |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:--------------------------------------|:-------------------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| baseline_score                        | xsec_excess_ret_5d |      87 |         264650 |  0.004228 |       0.018153 |      0.129737 |    2.22119  |
| multifactor_equal_rank_score          | xsec_excess_ret_5d |      87 |         264650 | -0.017604 |       0.03874  |      0.154785 |    3.97306  |
| multifactor_ic_weighted_score         | xsec_excess_ret_5d |      87 |         264650 | -0.023173 |       0.036503 |      0.163825 |    3.53716  |
| multifactor_rolling_ic_weighted_score | xsec_excess_ret_5d |      87 |         264650 | -0.034308 |       0.006788 |      0.149212 |    0.722151 |
| multifactor_low_corr_rank_score       | xsec_excess_ret_5d |      86 |         261566 | -0.019553 |       0.035049 |      0.148274 |    3.75242  |
| multifactor_neutral_rank_score        | xsec_excess_ret_5d |      86 |         261566 | -0.021482 |       0.015924 |      0.105499 |    2.39616  |

## Quantile Returns

| signal                                |   quantile |   mean_return |   periods |   observations |
|:--------------------------------------|-----------:|--------------:|----------:|---------------:|
| baseline_score                        |          1 |     -1.5e-05  |        87 |          52968 |
| baseline_score                        |          2 |      0.000159 |        87 |          52907 |
| baseline_score                        |          3 |     -0.000533 |        87 |          52917 |
| baseline_score                        |          4 |     -0.00069  |        87 |          52907 |
| baseline_score                        |          5 |      0.001133 |        87 |          52951 |
| multifactor_equal_rank_score          |          1 |      0.00277  |        87 |          52968 |
| multifactor_equal_rank_score          |          2 |      0.000443 |        87 |          52907 |
| multifactor_equal_rank_score          |          3 |     -0.000715 |        87 |          52917 |
| multifactor_equal_rank_score          |          4 |     -0.00122  |        87 |          52907 |
| multifactor_equal_rank_score          |          5 |     -0.001222 |        87 |          52951 |
| multifactor_ic_weighted_score         |          1 |      0.003726 |        87 |          52968 |
| multifactor_ic_weighted_score         |          2 |     -5e-06    |        87 |          52907 |
| multifactor_ic_weighted_score         |          3 |     -0.00104  |        87 |          52917 |
| multifactor_ic_weighted_score         |          4 |     -0.001196 |        87 |          52907 |
| multifactor_ic_weighted_score         |          5 |     -0.00143  |        87 |          52951 |
| multifactor_rolling_ic_weighted_score |          1 |      0.004221 |        87 |          52968 |
| multifactor_rolling_ic_weighted_score |          2 |      0.000218 |        87 |          52907 |
| multifactor_rolling_ic_weighted_score |          3 |     -0.001118 |        87 |          52917 |
| multifactor_rolling_ic_weighted_score |          4 |     -0.001131 |        87 |          52907 |
| multifactor_rolling_ic_weighted_score |          5 |     -0.002135 |        87 |          52951 |
| multifactor_low_corr_rank_score       |          1 |      0.003171 |        86 |          52350 |
| multifactor_low_corr_rank_score       |          2 |      0.000226 |        86 |          52290 |
| multifactor_low_corr_rank_score       |          3 |     -0.001032 |        86 |          52302 |
| multifactor_low_corr_rank_score       |          4 |     -0.00176  |        86 |          52290 |
| multifactor_low_corr_rank_score       |          5 |     -0.000525 |        86 |          52334 |
| multifactor_neutral_rank_score        |          1 |      0.001929 |        86 |          52350 |
| multifactor_neutral_rank_score        |          2 |      0.001598 |        86 |          52290 |
| multifactor_neutral_rank_score        |          3 |     -0.000421 |        86 |          52302 |
| multifactor_neutral_rank_score        |          4 |     -0.000915 |        86 |          52290 |
| multifactor_neutral_rank_score        |          5 |     -0.002109 |        86 |          52334 |

## Quantile Spread

| signal                                |   low_quantile |   high_quantile |   low_mean_return |   high_mean_return |   q_high_minus_low |
|:--------------------------------------|---------------:|----------------:|------------------:|-------------------:|-------------------:|
| baseline_score                        |              1 |               5 |         -1.5e-05  |           0.001133 |           0.001148 |
| multifactor_equal_rank_score          |              1 |               5 |          0.00277  |          -0.001222 |          -0.003992 |
| multifactor_ic_weighted_score         |              1 |               5 |          0.003726 |          -0.00143  |          -0.005157 |
| multifactor_low_corr_rank_score       |              1 |               5 |          0.003171 |          -0.000525 |          -0.003696 |
| multifactor_neutral_rank_score        |              1 |               5 |          0.001929 |          -0.002109 |          -0.004038 |
| multifactor_rolling_ic_weighted_score |              1 |               5 |          0.004221 |          -0.002135 |          -0.006356 |

## Backtest Summary

| signal                                | rebalance_frequency   |   top_n |   fee_bps |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   mean_turnover |   mean_gross_return |   mean_net_return |
|:--------------------------------------|:----------------------|--------:|----------:|--------------------:|-------------:|----------:|---------------:|----------:|----------------:|--------------------:|------------------:|
| baseline_score                        | daily                 |      50 |         0 |            1.19288  |     0.36867  |  2.31721  |      -0.266903 |        87 |        0.426207 |            0.00339  |          0.00339  |
| baseline_score                        | daily                 |      50 |        10 |            0.970194 |     0.368559 |  2.02649  |      -0.276323 |        87 |        0.426207 |            0.00339  |          0.002964 |
| baseline_score                        | daily                 |     100 |         0 |            1.48387  |     0.330871 |  2.91963  |      -0.23277  |        87 |        0.350805 |            0.003833 |          0.003833 |
| baseline_score                        | daily                 |     100 |        10 |            1.27418  |     0.331111 |  2.65053  |      -0.241603 |        87 |        0.350805 |            0.003833 |          0.003483 |
| baseline_score                        | weekly                |      50 |         0 |           -0.040498 |     0.160371 | -0.176099 |      -0.118543 |        20 |        0.848    |           -0.000543 |         -0.000543 |
| baseline_score                        | weekly                |      50 |        10 |           -0.082148 |     0.161665 | -0.447451 |      -0.129254 |        20 |        0.848    |           -0.000543 |         -0.001391 |
| baseline_score                        | weekly                |     100 |         0 |            0.110758 |     0.150947 |  0.772698 |      -0.075572 |        20 |        0.694    |            0.002243 |          0.002243 |
| baseline_score                        | weekly                |     100 |        10 |            0.071268 |     0.151995 |  0.529942 |      -0.078229 |        20 |        0.694    |            0.002243 |          0.001549 |
| baseline_score                        | monthly               |      50 |         0 |            0.091068 |     0.050607 |  1.75382  |      -0.018339 |         5 |        1.168    |            0.007396 |          0.007396 |
| baseline_score                        | monthly               |      50 |        10 |            0.075985 |     0.050546 |  1.47867  |      -0.019539 |         5 |        1.168    |            0.007396 |          0.006228 |
| baseline_score                        | monthly               |     100 |         0 |            0.072684 |     0.0442   |  1.61414  |      -0.014758 |         5 |        1.06     |            0.005945 |          0.005945 |
| baseline_score                        | monthly               |     100 |        10 |            0.059195 |     0.044216 |  1.32586  |      -0.015858 |         5 |        1.06     |            0.005945 |          0.004885 |
| multifactor_equal_rank_score          | daily                 |      50 |         0 |            0.165026 |     0.201753 |  0.857922 |      -0.184028 |        87 |        0.793103 |            0.000687 |          0.000687 |
| multifactor_equal_rank_score          | daily                 |      50 |        10 |           -0.045923 |     0.201351 | -0.132972 |      -0.204841 |        87 |        0.793103 |            0.000687 |         -0.000106 |
| multifactor_equal_rank_score          | daily                 |     100 |         0 |            0.024913 |     0.192165 |  0.223939 |      -0.214731 |        87 |        0.688736 |            0.000171 |          0.000171 |
| multifactor_equal_rank_score          | daily                 |     100 |        10 |           -0.138398 |     0.191921 | -0.680114 |      -0.231998 |        87 |        0.688736 |            0.000171 |         -0.000518 |
| multifactor_equal_rank_score          | weekly                |      50 |         0 |            0.088013 |     0.091374 |  0.969382 |      -0.045317 |        20 |        1.528    |            0.001703 |          0.001703 |
| multifactor_equal_rank_score          | weekly                |      50 |        10 |            0.004904 |     0.092105 |  0.099024 |      -0.054606 |        20 |        1.528    |            0.001703 |          0.000175 |
| multifactor_equal_rank_score          | weekly                |     100 |         0 |            0.033942 |     0.094266 |  0.401184 |      -0.060436 |        20 |        1.345    |            0.000727 |          0.000727 |
| multifactor_equal_rank_score          | weekly                |     100 |        10 |           -0.03595  |     0.094709 | -0.339168 |      -0.068888 |        20 |        1.345    |            0.000727 |         -0.000618 |
| multifactor_equal_rank_score          | monthly               |      50 |         0 |            0.021964 |     0.046255 |  0.493185 |      -0.020176 |         5 |        1.544    |            0.001901 |          0.001901 |
| multifactor_equal_rank_score          | monthly               |      50 |        10 |            0.003209 |     0.046504 |  0.092124 |      -0.02518  |         5 |        1.544    |            0.001901 |          0.000357 |
| multifactor_equal_rank_score          | monthly               |     100 |         0 |           -0.004409 |     0.043694 | -0.07923  |      -0.0249   |         5 |        1.428    |           -0.000288 |         -0.000288 |
| multifactor_equal_rank_score          | monthly               |     100 |        10 |           -0.021352 |     0.043881 | -0.469408 |      -0.029456 |         5 |        1.428    |           -0.000288 |         -0.001716 |
| multifactor_ic_weighted_score         | daily                 |      50 |         0 |           -0.091625 |     0.206025 | -0.363576 |      -0.200577 |        87 |        0.56046  |           -0.000297 |         -0.000297 |
| multifactor_ic_weighted_score         | daily                 |      50 |        10 |           -0.211299 |     0.205682 | -1.05085  |      -0.21478  |        87 |        0.56046  |           -0.000297 |         -0.000858 |
| multifactor_ic_weighted_score         | daily                 |     100 |         0 |           -0.131951 |     0.201966 | -0.599706 |      -0.208169 |        87 |        0.436322 |           -0.000481 |         -0.000481 |
| multifactor_ic_weighted_score         | daily                 |     100 |        10 |           -0.222352 |     0.201599 | -1.1462   |      -0.219084 |        87 |        0.436322 |           -0.000481 |         -0.000917 |
| multifactor_ic_weighted_score         | weekly                |      50 |         0 |            0.050244 |     0.10533  |  0.518072 |      -0.052624 |        20 |        1.09     |            0.001049 |          0.001049 |
| multifactor_ic_weighted_score         | weekly                |      50 |        10 |           -0.007645 |     0.105653 | -0.019985 |      -0.060234 |        20 |        1.09     |            0.001049 |         -4.1e-05  |
| multifactor_ic_weighted_score         | weekly                |     100 |         0 |            0.033931 |     0.108243 |  0.362219 |      -0.056577 |        20 |        0.882    |            0.000754 |          0.000754 |
| multifactor_ic_weighted_score         | weekly                |     100 |        10 |           -0.012468 |     0.108749 | -0.061208 |      -0.062464 |        20 |        0.882    |            0.000754 |         -0.000128 |
| multifactor_ic_weighted_score         | monthly               |      50 |         0 |            0.009617 |     0.050108 |  0.21596  |      -0.024247 |         5 |        1.376    |            0.000902 |          0.000902 |
| multifactor_ic_weighted_score         | monthly               |      50 |        10 |           -0.006928 |     0.050323 | -0.113084 |      -0.028649 |         5 |        1.376    |            0.000902 |         -0.000474 |
| multifactor_ic_weighted_score         | monthly               |     100 |         0 |            0.001193 |     0.052556 |  0.048798 |      -0.026856 |         5 |        1.188    |            0.000214 |          0.000214 |
| multifactor_ic_weighted_score         | monthly               |     100 |        10 |           -0.012998 |     0.052757 | -0.221606 |      -0.030546 |         5 |        1.188    |            0.000214 |         -0.000974 |
| multifactor_rolling_ic_weighted_score | daily                 |      50 |         0 |           -0.522435 |     0.245559 | -2.88217  |      -0.27189  |        87 |        0.718161 |           -0.002809 |         -0.002809 |
| multifactor_rolling_ic_weighted_score | daily                 |      50 |        10 |           -0.601783 |     0.246031 | -3.61222  |      -0.314986 |        87 |        0.718161 |           -0.002809 |         -0.003527 |
| multifactor_rolling_ic_weighted_score | daily                 |     100 |         0 |           -0.414516 |     0.223648 | -2.27916  |      -0.221704 |        87 |        0.638161 |           -0.002023 |         -0.002023 |
| multifactor_rolling_ic_weighted_score | daily                 |     100 |        10 |           -0.501798 |     0.224601 | -2.9855   |      -0.260628 |        87 |        0.638161 |           -0.002023 |         -0.002661 |

Artifacts: `traditional_quant_research\output\experiments\multifactor_baseline\multifactor_baseline_20260602_111956`
