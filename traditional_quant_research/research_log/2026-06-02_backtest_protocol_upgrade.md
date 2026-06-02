# 2026-06-02 Backtest Protocol Upgrade

- Hypothesis: `baseline_score` can support a simple Top-N protocol whose behavior changes meaningfully with rebalance frequency and transaction costs.
- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2026-01-01` to `2026-06-01` for the smoke run.
- Method: rank `baseline_score`, run equal-weight Top-N long-only backtests on daily/weekly/monthly rebalance schedules, compare 0/10 bps cost sensitivity, and record the best configurations.
- Cost/Risk Assumptions: next-open-to-horizon-close label convention, turnover costs applied on weight changes, no limit-up/down execution model yet, no overlapping multi-day holdings model yet.
- Result: daily Top-50 and Top-100 were the strongest configurations in the smoke run; daily rebalancing outperformed weekly/monthly on this short window, while costs reduced but did not eliminate the edge.
- Failure Modes: the protocol still ignores actual tradability at execution time, limit-up/down effects, and overlapping holding periods for longer horizons.
- Next Step: use the protocol results together with the full-cycle diagnostic report to decide whether to move into multi-factor ranking or traditional ML, and add excess-return / execution-constrained variants if needed.

## Summary Table

| horizon | signal | rebalance_frequency | top_n | fee_bps | annualized_return | volatility | sharpe | max_drawdown | periods | mean_turnover | mean_gross_return | mean_net_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline_score | daily | 50 | 0 | 0.510077 | 0.163879 | 2.598712 | -0.059011 | 91 | 0.418022 | 0.001690 | 0.001690 |
| 1 | baseline_score | weekly | 50 | 0 | 0.148353 | 0.059044 | 2.375191 | -0.014092 | 20 | 0.868000 | 0.002697 | 0.002697 |
| 1 | baseline_score | weekly | 100 | 0 | 0.114270 | 0.049581 | 2.209333 | -0.018035 | 20 | 0.710000 | 0.002107 | 0.002107 |
| 1 | baseline_score | daily | 100 | 0 | 0.324164 | 0.140495 | 2.069867 | -0.050901 | 91 | 0.349670 | 0.001154 | 0.001154 |
| 1 | baseline_score | daily | 50 | 10 | 0.359481 | 0.162994 | 1.966526 | -0.064179 | 91 | 0.418022 | 0.001690 | 0.001272 |
| 1 | baseline_score | weekly | 50 | 10 | 0.097808 | 0.058635 | 1.622003 | -0.017483 | 20 | 0.868000 | 0.002697 | 0.001829 |
| 1 | baseline_score | weekly | 100 | 10 | 0.073979 | 0.049005 | 1.481918 | -0.019203 | 20 | 0.710000 | 0.002107 | 0.001397 |
| 1 | baseline_score | daily | 100 | 10 | 0.212635 | 0.140113 | 1.446618 | -0.056063 | 91 | 0.349670 | 0.001154 | 0.000804 |
| 1 | baseline_score | monthly | 50 | 0 | 0.012943 | 0.034537 | 0.389701 | -0.008071 | 5 | 1.184000 | 0.001122 | 0.001122 |
| 1 | baseline_score | monthly | 100 | 0 | 0.004840 | 0.026179 | 0.197545 | -0.010614 | 5 | 1.080000 | 0.000431 | 0.000431 |
