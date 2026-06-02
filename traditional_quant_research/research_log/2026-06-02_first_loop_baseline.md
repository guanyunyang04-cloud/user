# 2026-06-02 First Loop Baseline

- Hypothesis: 基础传统价量因子在沪深主板 PIT 可交易股票池上具有可诊断的横截面排序信息。
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2026-01-01` to `2026-06-01`.
- Method: z-score baseline factors, average into `baseline_score`, evaluate `fwd_ret_1d` with IC, quantile returns, and Top-100 equal-weight baseline.
- Cost/Risk Assumptions: daily rebalance, next-open-to-horizon-close label, `10.0` bps per one-way turnover unit, no industry/size neutralization yet.
- Result: RankIC mean `0.015100`, RankICIR `1.510067`, Top-N annualized return `0.212635`, Sharpe `1.446618`, max drawdown `-0.056063`.
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

## Quantile Returns

|   quantile |   mean_return |   periods |   observations |
|-----------:|--------------:|----------:|---------------:|
|          1 |      0.000165 |        91 |          55441 |
|          2 |      5.4e-05  |        91 |          55377 |
|          3 |      1e-06    |        91 |          55391 |
|          4 |      1.8e-05  |        91 |          55377 |
|          5 |      0.00052  |        91 |          55427 |

Artifacts: `traditional_quant_research\output\experiments\baseline_first_loop\baseline_first_loop_20260602_083912`
