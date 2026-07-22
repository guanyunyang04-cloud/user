# 2026-06-02 Low-Corr Regime Filter Interpretation

## Context

- Objective: test whether market-state filters can reduce the March/May weakness of the low-corr watchlist setup.
- Run: `traditional_quant_research/output/experiments/low_corr_regime_filter/low_corr_regime_filter_20260602_144834`
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- Fit window: `2025-01-01` to `2025-12-31`
- Evaluation window: `2026-01-01` to `2026-06-01`
- Signal: `multifactor_low_corr_rank_score`
- Stock filter: `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8`
- Protocol: 5-day horizon, weekly Top-100, buffer `2.0`, execution constraints enabled, fees `0/30` bps.

The regime experiment fixes weekly rebalance dates from the full evaluation calendar first. If a scheduled date fails the regime rule, the strategy skips that scheduled rebalance rather than shifting to an earlier date in the same week.

## Best Regime Rule

```text
market_ret_20d_mean>=-0.02,
breadth_20d_positive_rate>=0.45
```

This rule allowed `11 / 21` scheduled weekly rebalances and blocked `10 / 21`.

Blocked dates:

```text
2026-03-06, 2026-03-13, 2026-03-20, 2026-03-27,
2026-04-03, 2026-04-10, 2026-04-17,
2026-05-22, 2026-05-29, 2026-06-01
```

## Result Summary

Baseline with stock filter but no regime filter:

- 0 bps annualized return: `-0.100812`; Sharpe: `-0.548200`; max drawdown: `-0.070929`
- 30 bps annualized return: `-0.254097`; Sharpe: `-1.667652`; max drawdown: `-0.104863`
- Mean turnover: `1.232000`
- Exit delayed count: `1`

Best regime filter:

- 0 bps annualized return: `0.007264`; Sharpe: `0.121349`; max drawdown: `-0.053958`
- 30 bps annualized return: `-0.164644`; Sharpe: `-1.242916`; max drawdown: `-0.061593`
- Mean turnover: `1.235556`
- Exit delayed count: `0`

Relative improvement:

- 0 bps annualized delta: `+0.108076`
- 30 bps annualized delta: `+0.089454`
- 30 bps max-drawdown delta: `+0.043270`

## Period Diagnostics

At 30 bps, the best regime filter materially reduced the worst March drawdown:

- Baseline March annualized return: `-0.547144`
- Regime-filtered March annualized return: `0.020041`

It did not fully fix May:

- Baseline May annualized return: `-0.573192`
- Regime-filtered May annualized return: `-0.546460`

The rule is therefore mostly a March/early-April damage-control filter. It also blocks late May and June 1, but remaining May trades are still weak.

## Interpretation

This is the strongest evidence so far that the current low-corr line needs market-regime control. The best rule is simple and interpretable: avoid scheduled weekly rebalances when 20-day market return is below `-2%` or 20-day breadth is below `45%`.

However, the result still fails the strategy-candidate gate. It turns slightly positive only before costs, and remains negative at 30 bps. The experiment improves drawdown and removes one delayed-exit event, but does not prove cost robustness or sample-out stability.

Candidate count remains `0`.

## Next Step

- Promote this setup only to `watchlist`, not candidate.
- Extend regime validation to multiple years, especially 2024 and 2025, using the same fit/evaluation separation.
- Add a compact regime threshold grid around `market_ret_20d_mean >= -0.03/-0.02/-0.01` and `breadth_20d_positive_rate >= 0.40/0.45/0.50`.
- Keep the current stock filter `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8` as the base guard for this line.
