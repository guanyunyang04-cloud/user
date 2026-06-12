# 2026-06-02 First Loop Baseline

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_first_loop_baseline.md`.


- Hypothesis: 基础传统价量因子在沪深主板 PIT 可交易股票池上具有可诊断的横截面排序信息。
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2026-01-01` to `2026-06-01`.
- Method: z-score baseline factors, average into `baseline_score`, evaluate `fwd_ret_1d` with IC, quantile returns, Top-100 equal-weight baseline, and IS/OOS split diagnostics.
- Cost/Risk Assumptions: daily rebalance, next-open-to-horizon-close label, `10.0` bps per one-way turnover unit, no industry/size neutralization yet.
- Result: RankIC mean `0.015100`, RankICIR `1.510067`, Top-N annualized return `0.212635`, Sharpe `1.446618`, max drawdown `-0.056063`.
- OOS Result: split date `2026-04-01`, OOS RankIC mean `0.010413`, OOS RankICIR `1.140195`, OOS Top-N annualized return `0.198493`, OOS Sharpe `1.385825`.
- Failure Modes: 因子未做行业/市值中性化，Top-N 回测是研究基线而非完整生产执行模型，涨跌停成交约束尚未建模。
- Next Step: 增加行业/市值数据或替代暴露 proxy，扩展单因子报告，并加入样本内/样本外切分。

## Panel

- Rows: `292483`
- Dates: `96`
- Securities: `3098`

## IC Summary

| signal                | label      |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:----------------------|:-----------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| reversal_5d_z         | fwd_ret_1d |      90 |         273920 | -0.018768 |       0.008345 |      0.135101 |    0.980509 |
| momentum_20d_z        | fwd_ret_1d |      75 |         227710 |  0.035559 |      -0.004462 |      0.12506  |   -0.566359 |
| ma20_gap_z            | fwd_ret_1d |      91 |         277013 |  0.026721 |      -0.012232 |      0.129114 |   -1.50394  |
| neg_volatility_20d_z  | fwd_ret_1d |      90 |         273920 | -0.025565 |       0.021887 |      0.19414  |    1.78962  |
| log_amount_mean_20d_z | fwd_ret_1d |      91 |         277013 |  0.021416 |      -0.010021 |      0.181209 |   -0.877833 |
| neg_amplitude_20d_z   | fwd_ret_1d |      91 |         277013 | -0.029485 |       0.024008 |      0.201579 |    1.89068  |
| baseline_score        | fwd_ret_1d |      91 |         277013 |  0.002206 |       0.0151   |      0.158742 |    1.51007  |

## IC By Split

| sample_split   | signal                | label      |   dates |   observations |   mean_ic |   mean_rank_ic |   std_rank_ic |   rank_icir |
|:---------------|:----------------------|:-----------|--------:|---------------:|----------:|---------------:|--------------:|------------:|
| in_sample      | reversal_5d_z         | fwd_ret_1d |      51 |         155974 | -0.000881 |       0.026388 |      0.1306   |    3.20749  |
| in_sample      | momentum_20d_z        | fwd_ret_1d |      36 |         109976 |  0.034443 |      -0.006197 |      0.131691 |   -0.747024 |
| in_sample      | ma20_gap_z            | fwd_ret_1d |      52 |         159042 |  0.015305 |      -0.026999 |      0.130439 |   -3.28582  |
| in_sample      | neg_volatility_20d_z  | fwd_ret_1d |      51 |         155974 | -0.012775 |       0.035099 |      0.204489 |    2.72471  |
| in_sample      | log_amount_mean_20d_z | fwd_ret_1d |      52 |         159042 |  0.006056 |      -0.026941 |      0.183697 |   -2.32813  |
| in_sample      | neg_amplitude_20d_z   | fwd_ret_1d |      52 |         159042 | -0.013863 |       0.039382 |      0.210625 |    2.96816  |
| in_sample      | baseline_score        | fwd_ret_1d |      52 |         159042 |  0.008665 |       0.018616 |      0.168246 |    1.75649  |
| out_of_sample  | reversal_5d_z         | fwd_ret_1d |      39 |         117946 | -0.042159 |      -0.015251 |      0.137235 |   -1.7641   |
| out_of_sample  | momentum_20d_z        | fwd_ret_1d |      39 |         117734 |  0.03659  |      -0.00286  |      0.118587 |   -0.382839 |
| out_of_sample  | ma20_gap_z            | fwd_ret_1d |      39 |         117971 |  0.041942 |       0.007457 |      0.124634 |    0.94982  |
| out_of_sample  | neg_volatility_20d_z  | fwd_ret_1d |      39 |         117946 | -0.042291 |       0.004609 |      0.178239 |    0.410517 |
| out_of_sample  | log_amount_mean_20d_z | fwd_ret_1d |      39 |         117971 |  0.041897 |       0.01254  |      0.175315 |    1.13545  |
| out_of_sample  | neg_amplitude_20d_z   | fwd_ret_1d |      39 |         117971 | -0.050313 |       0.00351  |      0.186888 |    0.298161 |
| out_of_sample  | baseline_score        | fwd_ret_1d |      39 |         117971 | -0.006407 |       0.010413 |      0.144973 |    1.1402   |

## Quantile Returns

|   quantile |   mean_return |   periods |   observations |
|-----------:|--------------:|----------:|---------------:|
|          1 |      0.000165 |        91 |          55441 |
|          2 |      5.4e-05  |        91 |          55377 |
|          3 |      1e-06    |        91 |          55391 |
|          4 |      1.8e-05  |        91 |          55377 |
|          5 |      0.00052  |        91 |          55427 |

## Top-N By Split

| sample_split   | signal                |   annualized_return |   volatility |    sharpe |   max_drawdown |   periods |   top_n |   fee_bps |   mean_turnover |   mean_gross_return |   mean_net_return |
|:---------------|:----------------------|--------------------:|-------------:|----------:|---------------:|----------:|--------:|----------:|----------------:|--------------------:|------------------:|
| in_sample      | reversal_5d_z         |           -0.156372 |     0.2721   | -0.488448 |      -0.112547 |        51 |     100 |        10 |        0.95098  |            0.000424 |         -0.000527 |
| in_sample      | momentum_20d_z        |            1.1153   |     0.269354 |  2.9202   |      -0.071166 |        36 |     100 |        10 |        0.397222 |            0.003519 |          0.003121 |
| in_sample      | ma20_gap_z            |            0.868529 |     0.263208 |  2.51009  |      -0.073161 |        52 |     100 |        10 |        0.637692 |            0.003259 |          0.002622 |
| in_sample      | neg_volatility_20d_z  |           -0.054263 |     0.1181   | -0.412824 |      -0.054201 |        51 |     100 |        10 |        0.260784 |            6.7e-05  |         -0.000193 |
| in_sample      | log_amount_mean_20d_z |            0.095634 |     0.235098 |  0.506703 |      -0.088432 |        52 |     100 |        10 |        0.073077 |            0.000546 |          0.000473 |
| in_sample      | neg_amplitude_20d_z   |           -0.050003 |     0.120184 | -0.366218 |      -0.053172 |        52 |     100 |        10 |        0.157692 |           -1.7e-05  |         -0.000175 |
| in_sample      | baseline_score        |            0.218857 |     0.141994 |  1.46514  |      -0.056063 |        52 |     100 |        10 |        0.374615 |            0.0012   |          0.000826 |
| out_of_sample  | reversal_5d_z         |           -0.368924 |     0.23413  | -1.84628  |      -0.110776 |        39 |     100 |        10 |        0.885641 |           -0.00083  |         -0.001715 |
| out_of_sample  | momentum_20d_z        |            1.35556  |     0.273032 |  3.28005  |      -0.049532 |        39 |     100 |        10 |        0.382051 |            0.003936 |          0.003554 |
| out_of_sample  | ma20_gap_z            |            1.56907  |     0.261017 |  3.75234  |      -0.041125 |        39 |     100 |        10 |        0.561538 |            0.004448 |          0.003887 |
| out_of_sample  | neg_volatility_20d_z  |           -0.248232 |     0.099725 | -2.80974  |      -0.071258 |        39 |     100 |        10 |        0.215385 |           -0.000897 |         -0.001112 |
| out_of_sample  | log_amount_mean_20d_z |            0.954533 |     0.283784 |  2.50704  |      -0.082121 |        39 |     100 |        10 |        0.074359 |            0.002898 |          0.002823 |
| out_of_sample  | neg_amplitude_20d_z   |           -0.178792 |     0.095001 | -2.02526  |      -0.059809 |        39 |     100 |        10 |        0.134872 |           -0.000629 |         -0.000763 |
| out_of_sample  | baseline_score        |            0.198493 |     0.137567 |  1.38583  |      -0.035509 |        39 |     100 |        10 |        0.335897 |            0.001092 |          0.000757 |

Artifacts: `traditional_quant_research\output\experiments\baseline_first_loop\baseline_first_loop_20260602_084836`
