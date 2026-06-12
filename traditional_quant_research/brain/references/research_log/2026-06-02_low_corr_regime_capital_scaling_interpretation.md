# 2026-06-02 Low-Corr Regime Capital Scaling Interpretation

## Context

- Objective: test whether dynamic regime-based capital scaling can improve the current `multifactor_low_corr_rank_score` watchlist without fully skipping weak-regime rebalance dates.
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`.
- Fit window: `2025-01-01` to `2025-12-31`.
- Evaluation window: `2026-01-01` to `2026-06-01`.
- Stock filter: `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8`.
- Backtest protocol: horizon-aligned weekly Top-100, 5 tradeable-day holding period, `buffer_multiplier=2.0`, execution constraints enabled, fees `0/30` bps.
- Artifacts: `traditional_quant_research/output/experiments/low_corr_regime_capital_scaling/low_corr_regime_capital_scaling_20260602_153300`.

## Result

Best 2026 plan was `breadth_soft`:

```text
breadth_20d_positive_rate>=0.50 @ 1.0
breadth_20d_positive_rate>=0.45 @ 0.6
default @ 0.3
```

Key rows:

| plan | fee_bps | annualized_return | baseline | delta | max_drawdown | mean_turnover | mean_capital_scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_full_capital | 0 | -0.100812 | -0.100812 | 0.000000 | -0.070929 | 1.232000 | 1.000000 |
| breadth_soft | 0 | -0.004522 | -0.100812 | +0.096290 | -0.050817 | 0.886667 | 0.666667 |
| baseline_full_capital | 30 | -0.254097 | -0.254097 | 0.000000 | -0.104863 | 1.232000 | 1.000000 |
| breadth_soft | 30 | -0.129649 | -0.254097 | +0.124449 | -0.062797 | 0.886667 | 0.666667 |

## Interpretation

- Dynamic capital scaling is directionally useful for weak-regime damage control: it reduced 2026 drawdown, turnover, and negative annualized return.
- It did not pass the promotion gate. The 30 bps row is still materially negative, so this remains `backtest_only`.
- The improvement mostly comes from exposure reduction, not from repairing gross alpha. At 0 bps the best plan is only near breakeven, which means cost pressure still dominates.
- The next required check is multi-year validation, because a soft de-risking rule may still hurt strong years less than static all-or-nothing regime skips.

## Decision

- Candidate count remains `0`.
- Keep `breadth_soft` as the dynamic-capital watchlist plan.
- Do not enter traditional ML yet.
- Continue with 2024/2025/2026 prior-year-fit validation before any further tuning.
