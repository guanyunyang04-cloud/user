# 2026-06-02 Low-Corr Horizon Cost Grid Interpretation

## Context

- Objective: test whether lower-turnover horizon and rebalance choices can recover the low-corr signal under 30 bps costs.
- Fit window: `2025-01-01` to `2025-12-31`.
- Evaluation window: `2026-01-01` to `2026-06-01`.
- Tested horizons: `5/10/20`.
- Tested protocols: weekly/monthly, Top-100/Top-200, buffer `2.0/3.0`, fees `0/30` bps, execution constraints enabled.
- Filter configs: full low-corr universe baseline and `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8`.
- Artifacts: `traditional_quant_research/output/experiments/low_corr_horizon_cost_grid/low_corr_horizon_cost_grid_20260602_155437`.

## Key Result

The strongest 2026 rows came from `horizon=20`, not from the 5-day protocol used in earlier regime/capital tests.

Best 30 bps row:

| config | horizon | frequency | top_n | buffer | ann_return | sharpe | max_drawdown | periods |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline_no_filter | 20 | weekly | 200 | 3.0 | 0.089440 | 0.783871 | -0.038034 | 4 |

Best monthly 30 bps rows were also positive:

| config | horizon | frequency | top_n | buffer | ann_return | sharpe | max_drawdown | periods |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline_no_filter | 20 | monthly | 100 | 3.0 | 0.082924 | 1.321117 | -0.011062 | 2 |
| baseline_no_filter | 20 | monthly | 200 | 2.0 | 0.082610 | 2.179686 | -0.003944 | 2 |
| baseline_no_filter | 20 | monthly | 200 | 3.0 | 0.076594 | 2.120018 | -0.003944 | 2 |

The prior watchlist filter was weaker. Its best 30 bps 2026 row was only near breakeven:

| config | horizon | frequency | top_n | buffer | ann_return |
|---|---:|---|---:|---:|---:|
| `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8` | 20 | monthly | 200 | 2.0 | 0.0004 |

## Interpretation

- The earlier 5-day weekly protocol appears too cost-sensitive for the current low-corr signal.
- Extending the horizon to 20 tradeable days changes the evidence materially: gross alpha survives costs in 2026 under several low-turnover protocols.
- The stock-level watchlist filter that helped in 5-day weak-regime tests appears to remove useful 20-day candidates. This is a regime/horizon interaction, not a universal filter.
- The 2026 result alone is not enough: monthly 20-day rows have only two completed trades in the 2026 half-year window.

## Decision

- Candidate count remains `0`.
- Promote `horizon=20` low-corr protocols to candidate-frontier audit, not to strategy candidate.
- Next required check: prior-year-fit validation across 2024/2025/2026 and exposure/capacity audit for the best `baseline_no_filter` rows.
