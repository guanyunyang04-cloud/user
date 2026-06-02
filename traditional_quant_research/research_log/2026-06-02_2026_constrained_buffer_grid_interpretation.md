# 2026-06-02 2026 Constrained Buffer Grid Interpretation

## Question

Can Top-N buffer reduce the execution-constrained 2026 damage enough to restore a viable candidate input?

## Evidence

- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`.
- History warm-up: `2025-01-01`.
- Evaluation window: `2026-01-01` to `2026-06-01`.
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_low_corr_rank_score`.
- Portfolio: weekly Top-100, 5-day horizon, non-overlapping.
- Buffer grid: `1.0, 1.5, 2.0`.
- Fee grid: `0, 30` bps.
- Execution approximation: `limit_threshold=0.095`.
- Experiment output: `traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_134401`.
- Auto research log: `traditional_quant_research/research_log/2026-06-02_multifactor_2026_constrained_buffer_grid.md`.

The experiment code now reuses fee-independent horizon trade paths, then reapplies fee rates to the same trades. This keeps results equivalent while reducing repeated constrained-path simulation.

## Summary

| Signal | Buffer | 0 bps AnnRet | 0 bps Sharpe | 30 bps AnnRet | 30 bps Sharpe | Mean Turnover |
|---|---:|---:|---:|---:|---:|---:|
| `multifactor_rolling_ic_weighted_score` | 1.0 | -0.137811 | -0.853836 | -0.300674 | -2.184832 | 1.378667 |
| `multifactor_rolling_ic_weighted_score` | 1.5 | -0.165401 | -1.062109 | -0.303146 | -2.220899 | 1.188000 |
| `multifactor_rolling_ic_weighted_score` | 2.0 | -0.169672 | -1.107552 | -0.293945 | -2.161364 | 1.068000 |
| `multifactor_low_corr_rank_score` | 1.0 | -0.181404 | -1.013576 | -0.353211 | -2.317064 | 1.548571 |
| `multifactor_low_corr_rank_score` | 1.5 | -0.193017 | -1.114242 | -0.351453 | -2.350810 | 1.437143 |
| `multifactor_low_corr_rank_score` | 2.0 | -0.128338 | -0.703060 | -0.288036 | -1.879388 | 1.333333 |

## Interpretation

Buffer helps turnover, but it does not repair the alpha problem.

- Rolling IC: buffer `1.5/2.0` lowers turnover, but gross return gets worse. At 30 bps, buffer `2.0` is slightly better than `1.0`, but still deeply negative.
- Low-corr: buffer `2.0` is the best constrained setting in this grid. It lowers turnover, reduces exit-delay count from `3` to `2`, and improves both 0 bps and 30 bps results. Still, the best result remains negative before and after cost.
- Monthly slices still show March, May, and delayed June exits as key damage periods.

This means buffer is useful as a cost-control and execution-stability knob, but it cannot promote these candidate inputs to strategy status.

## Decision

Current status remains `candidate_input_watchlist`.

Strategy candidate count remains `0`.

No candidate is eligible for strategy promotion.

Next research step should move from pure buffer tuning to exposure/regime controls:

- Test simple exposure caps for liquidity, momentum, volatility, and amplitude.
- Test a regime filter that avoids the March/May-like defensive-small-cap failure state.
- Keep execution-constrained horizon results as the promotion gate.
