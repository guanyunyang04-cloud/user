# 2026-06-02 Low-Corr Compact Exposure Grid Interpretation

## Context

- Objective: compare looser liquidity and momentum exposure filters under fit/evaluation separation.
- Run: `traditional_quant_research/output/experiments/low_corr_exposure_grid/low_corr_exposure_grid_20260602_142235`
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- Fit window: `2025-01-01` to `2025-12-31`
- Evaluation window: `2026-01-01` to `2026-06-01`
- Protocol: 5-day horizon, weekly Top-100, buffer `2.0`, execution constraints enabled, fees `0/30` bps.
- Grid size: `8` configs, including the no-filter baseline.

## Best Rule

```text
log_amount_mean_20d_z>=-0.8,
momentum_20d_z>=-0.8
```

This rule kept `194198 / 292483` rows, or `66.40%`, and retained all `96` evaluation dates.

## Result Summary

No-filter baseline:

- 0 bps annualized return: `-0.175190`; Sharpe: `-1.063846`; max drawdown: `-0.085836`
- 30 bps annualized return: `-0.314117`; Sharpe: `-2.183667`; max drawdown: `-0.115621`
- Mean turnover: `1.214667`

Best compact filter:

- 0 bps annualized return: `-0.100812`; Sharpe: `-0.548200`; max drawdown: `-0.070929`
- 30 bps annualized return: `-0.254097`; Sharpe: `-1.667652`; max drawdown: `-0.104863`
- Mean turnover: `1.232000`

Relative improvement:

- 0 bps annualized delta: `+0.074378`
- 30 bps annualized delta: `+0.060020`
- Turnover delta: `+0.017333`

## Period Diagnostics

At 30 bps, the best compact filter improved January materially and reduced the overall drawdown, but it did not solve the weak months:

- March remained negative: annualized `-0.547144`
- May remained negative: annualized `-0.573192`
- April stayed positive but weaker than the no-filter baseline

The filter reduced the selected basket's extreme small-liquidity exposure:

- Baseline Q1 `log_amount_mean_20d_z` active exposure: `-1.189728`
- Filtered Q1 `log_amount_mean_20d_z` active exposure: `-0.513231`
- Baseline Q2 `log_amount_mean_20d_z` active exposure: `-1.143143`
- Filtered Q2 `log_amount_mean_20d_z` active exposure: `-0.458995`

It also reduced weak-momentum exposure, but did not remove the low-volatility and reversal tilts.

## Interpretation

The compact grid shows that a simple liquidity-plus-momentum eligibility filter is directionally useful for `multifactor_low_corr_rank_score` under 2025 fit / 2026 evaluation. It improves annualized return, Sharpe, and drawdown versus the no-filter baseline at both 0 bps and 30 bps.

However, the result remains negative after costs and still fails in March and May. This cannot be promoted to a strategy candidate. It is a watchlist configuration and a clue: the current low-corr signal is too exposed to small/liquidity-constrained and weak-momentum names, but exposure caps alone do not repair regime failure.

Wide defensive caps did not help. The `-1.2/-1.2` liquidity/momentum rule with `neg_volatility_20d_z<=1.4` and `neg_amplitude_20d_z<=1.6` was close to baseline, while the `-1.0/-1.0` version was worse. Defensive caps should not be the next main tuning axis.

Candidate count remains `0`.

## Next Step

- Treat `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8` as a watchlist filter, not a candidate.
- Add a regime filter experiment before traditional ML. The next hypothesis should test whether bad March/May-like market states can be avoided or down-weighted.
- Keep using fit/evaluation separation for any exposure or regime rule.
- Continue reporting execution-constrained horizon results as the promotion gate.
