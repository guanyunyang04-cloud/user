# 2026-06-02 Full Cycle Factor Diagnostics

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_full_cycle_factor_diagnostics.md`.


- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2016-01-04` to `2026-06-01`.
- Panel: `7031085` rows, `2526` dates, `3392` securities.
- Horizons: `[1, 5, 20]`; Top-N values: `[100]`; include Top-N: `False`; fee bps: `10.0`.
- Method: baseline price/volume factors are cross-sectional z-scored per date; `baseline_score` is an equal-weight factor score; diagnostics cover IC, yearly IC stability, yearly quantiles, and Top-N baselines for `['baseline_score']`.
- Assessment: this is still diagnostic evidence; extreme long-horizon labels from the v2 audit require follow-up before upgrading conclusions.
- Next Step: add return-adjusted labels/excess returns, cost sensitivity, and rebalancing-frequency comparison before entering multi-factor or traditional ML.

## IC Summary

|   horizon | signal                | label       |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|----------:|:----------------------|:------------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
|         1 | reversal_5d_z         | fwd_ret_1d  |    2520 |        7010733 | -0.00034  |       0.020403 |      0.146991 |    2.20345  |
|         1 | momentum_20d_z        | fwd_ret_1d  |    2505 |        6959917 |  0.011372 |      -0.016932 |      0.152638 |   -1.76096  |
|         1 | ma20_gap_z            | fwd_ret_1d  |    2521 |        7014125 |  0.006609 |      -0.019752 |      0.156813 |   -1.99958  |
|         1 | neg_volatility_20d_z  | fwd_ret_1d  |    2520 |        7010733 | -0.026063 |       0.015828 |      0.174119 |    1.44308  |
|         1 | log_amount_mean_20d_z | fwd_ret_1d  |    2521 |        7014125 |  0.00934  |      -0.014274 |      0.146451 |   -1.54728  |
|         1 | neg_amplitude_20d_z   | fwd_ret_1d  |    2521 |        7014125 | -0.029631 |       0.018119 |      0.184935 |    1.55527  |
|         1 | baseline_score        | fwd_ret_1d  |    2521 |        7014125 | -0.011694 |       0.00275  |      0.167586 |    0.260537 |
|         5 | reversal_5d_z         | fwd_ret_5d  |    2516 |        6997171 |  0.001851 |       0.032027 |      0.131205 |    3.87497  |
|         5 | momentum_20d_z        | fwd_ret_5d  |    2501 |        6946377 | -0.027006 |      -0.047529 |      0.144596 |   -5.21804  |
|         5 | ma20_gap_z            | fwd_ret_5d  |    2517 |        7000561 | -0.01169  |      -0.042924 |      0.14306  |   -4.76296  |
|         5 | neg_volatility_20d_z  | fwd_ret_5d  |    2516 |        6997171 |  0.020314 |       0.059201 |      0.168019 |    5.59332  |
|         5 | log_amount_mean_20d_z | fwd_ret_5d  |    2517 |        7000561 | -0.044526 |      -0.060572 |      0.145199 |   -6.6223   |
|         5 | neg_amplitude_20d_z   | fwd_ret_5d  |    2517 |        7000561 |  0.017191 |       0.058549 |      0.17935  |    5.18222  |
|         5 | baseline_score        | fwd_ret_5d  |    2517 |        7000561 | -0.019054 |       0.002506 |      0.173514 |    0.229249 |
|        20 | reversal_5d_z         | fwd_ret_20d |    2501 |        6946377 |  0.018156 |       0.036148 |      0.123973 |    4.62866  |
|        20 | momentum_20d_z        | fwd_ret_20d |    2486 |        6895649 | -0.048636 |      -0.068207 |      0.138687 |   -7.80719  |
|        20 | ma20_gap_z            | fwd_ret_20d |    2502 |        6949762 | -0.035464 |      -0.060049 |      0.134131 |   -7.10689  |
|        20 | neg_volatility_20d_z  | fwd_ret_20d |    2501 |        6946377 |  0.041987 |       0.087005 |      0.167343 |    8.25346  |
|        20 | log_amount_mean_20d_z | fwd_ret_20d |    2502 |        6949762 | -0.070813 |      -0.099759 |      0.157502 |  -10.0546   |
|        20 | neg_amplitude_20d_z   | fwd_ret_20d |    2502 |        6949762 |  0.043589 |       0.08981  |      0.177575 |    8.02865  |
|        20 | baseline_score        | fwd_ret_20d |    2502 |        6949762 | -0.020611 |       0.00104  |      0.171739 |    0.096088 |

## Stability

|   horizon | signal                |   years |   positive_year_rate |   mean_yearly_rank_ic |   min_yearly_rank_ic |   max_yearly_rank_ic |
|----------:|:----------------------|--------:|---------------------:|----------------------:|---------------------:|---------------------:|
|         1 | baseline_score        |      11 |             0.636364 |              0.003105 |            -0.020332 |             0.012588 |
|         1 | log_amount_mean_20d_z |      11 |             0.090909 |             -0.013788 |            -0.039606 |             0.008495 |
|         1 | ma20_gap_z            |      11 |             0        |             -0.019285 |            -0.042821 |            -0.007484 |
|         1 | momentum_20d_z        |      11 |             0        |             -0.016479 |            -0.03899  |            -0.001866 |
|         1 | neg_amplitude_20d_z   |      11 |             0.909091 |              0.018063 |            -0.001629 |             0.03153  |
|         1 | neg_volatility_20d_z  |      11 |             0.909091 |              0.015819 |            -9.8e-05  |             0.025725 |
|         1 | reversal_5d_z         |      11 |             0.909091 |              0.019688 |            -0.004931 |             0.042115 |
|         5 | baseline_score        |      11 |             0.454545 |              0.002836 |            -0.040126 |             0.034597 |
|         5 | log_amount_mean_20d_z |      11 |             0        |             -0.058941 |            -0.113569 |            -0.025622 |
|         5 | ma20_gap_z            |      11 |             0        |             -0.041218 |            -0.07315  |            -0.012538 |
|         5 | momentum_20d_z        |      11 |             0        |             -0.045056 |            -0.080608 |            -0.001908 |
|         5 | neg_amplitude_20d_z   |      11 |             1        |              0.056393 |             0.020554 |             0.079718 |
|         5 | neg_volatility_20d_z  |      11 |             1        |              0.057371 |             0.027011 |             0.074172 |
|         5 | reversal_5d_z         |      11 |             1        |              0.030837 |             0.010566 |             0.055172 |
|        20 | baseline_score        |      11 |             0.545455 |              0.001083 |            -0.044565 |             0.073168 |
|        20 | log_amount_mean_20d_z |      11 |             0        |             -0.098367 |            -0.187588 |            -0.020391 |
|        20 | ma20_gap_z            |      11 |             0        |             -0.056908 |            -0.090321 |            -0.009319 |
|        20 | momentum_20d_z        |      11 |             0        |             -0.064842 |            -0.100853 |            -0.013105 |
|        20 | neg_amplitude_20d_z   |      11 |             1        |              0.086202 |             0.031328 |             0.133897 |
|        20 | neg_volatility_20d_z  |      11 |             1        |              0.084457 |             0.039204 |             0.122566 |
|        20 | reversal_5d_z         |      11 |             1        |              0.034539 |             0.010058 |             0.057666 |

## Top-N Summary

_No rows._

Artifacts: `traditional_quant_research\output\experiments\full_cycle_factor_diagnostics\full_cycle_factor_diagnostics_20260602_101008`
