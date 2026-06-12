# 2026-06-02 Low-Corr Horizon Cost Yearly Validation Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_horizon_cost_yearly_validation_interpretation.md`.


## Context

- Objective: validate the promising 20-day low-corr horizon/cost protocols across multiple years.
- Evaluation years: `2024`, `2025`, `2026`, with `2026` ending on `2026-06-01`.
- Fit protocol: each evaluation year uses the prior calendar year for low-corr factor selection and factor direction.
- Tested horizon: `20`.
- Tested protocols: weekly/monthly, Top-100/Top-200, buffer `2.0/3.0`, fees `0/30` bps, execution constraints enabled.
- Filter configs: full low-corr universe baseline and `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8`.
- Artifacts: `traditional_quant_research/output/experiments/low_corr_horizon_cost_yearly_validation/low_corr_horizon_cost_yearly_validation_20260602_162415`.

## Main Finding

The strongest row is:

```text
config_id = baseline_no_filter
horizon = 20
rebalance_frequency = monthly
top_n = 200
buffer_multiplier = 3.0
fee_bps = 30
execution_constraints = true
```

Aggregate:

| metric | value |
|---|---:|
| mean_annualized_return | 0.262745 |
| min_annualized_return | 0.076594 |
| positive_year_rate | 1.000000 |
| mean_sharpe | 2.144099 |
| worst_max_drawdown | -0.114807 |
| mean_turnover | 1.125774 |
| mean_cost_drag_vs_0bps | -0.051135 |

Year rows:

| year | ann_return | sharpe | max_drawdown | periods | mean_turnover |
|---:|---:|---:|---:|---:|---:|
| 2024 | 0.282025 | 0.976543 | -0.114807 | 7 | 1.138571 |
| 2025 | 0.429615 | 3.335735 | -0.010150 | 8 | 1.083750 |
| 2026 | 0.076594 | 2.120018 | -0.003944 | 2 | 1.155000 |

## Important Caveats

- This is a strong `backtest_only` result, but not yet a strategy candidate.
- There is selection bias: the 20-day protocol was promoted after observing the 2026 cost/horizon grid.
- 2026 has only two completed monthly 20-day baskets, so the half-year evidence is thin even though it is positive.
- The best result uses `baseline_no_filter`, while the earlier 5-day watchlist filter underperforms. This reopens exposure and capacity risk: the strategy may depend on names the watchlist filter removed.
- Current execution constraints are still approximate: no queue priority, partial fills, slippage, market impact, or capital capacity model.

## Decision

- Candidate count remains `0`.
- Mark `low_corr_20d_monthly_top200_buffer3_no_filter` as the first candidate-frontier protocol.
- Required before promotion:
  - selected basket exposure and capacity audit,
  - turnover and trade-count review by year/month,
  - stricter slippage/impact stress beyond 30 bps,
  - comparison against rolling IC and equal-rank alternatives under the same 20-day monthly protocol,
  - preferably expand to earlier years if data quality allows.
