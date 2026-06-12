# Low-Corr Frontier Combined Constraint Audit Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_low_corr_frontier_combined_constraint_audit_interpretation.md`.


## Summary

- Run: `low_corr_frontier_combined_constraint_audit_20260603_094139`.
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`.
- Protocol: `2024,2025,2026`, `horizon=20`, `monthly`, `top_n=200`, `buffer=3.0`, `fee_bps=30`, execution constraints enabled.
- Combined constraints:
  - Rolling IC: `exposure_penalty_strength=0.25`.
  - IC-weighted: `exposure_penalty_strength=0.0`.
  - Low-corr: `exposure_penalty_strength=1.0`.
  - Penalty columns: `log_amount_mean_20d_z,neg_volatility_20d_z,momentum_20d_z,turn_xsec_z`.
  - Industry cap: `group_col=industry,max_group_weight=0.10`.
  - Capacity stress: `capital=100m`, impact `0` and `10` bps per 1 pct participation.
- Artifacts: `traditional_quant_research/output/experiments/low_corr_frontier_combined_constraint_audit/low_corr_frontier_combined_constraint_audit_20260603_094139`.

This run is the first combined portfolio-construction gate for the current frontier. It combines the best signal-specific exposure penalty settings with an industry cap and participation-based impact stress.

## Return Evidence

At `30 bps / 100m / no extra impact`, all three frontier signals remain positive in each tested year:

| signal | penalty strength | mean annualized return | min annualized return | worst max drawdown | mean turnover | periods |
|:--|--:|--:|--:|--:|--:|--:|
| rolling IC | `0.25` | `0.352647` | `0.070722` | `-0.093437` | `1.015357` | `16` |
| IC-weighted | `0.0` | `0.284596` | `0.116312` | `-0.085351` | `1.002708` | `18` |
| low-corr | `1.0` | `0.277423` | `0.106522` | `-0.112062` | `1.136310` | `17` |

At `30 bps / 100m / 10 bps per 1 pct participation`, all three also remain positive in each tested year:

| signal | mean annualized return | min annualized return | worst max drawdown | mean impact drag |
|:--|--:|--:|--:|--:|
| rolling IC | `0.309775` | `0.040104` | `-0.102264` | `-0.042871` |
| IC-weighted | `0.247792` | `0.084162` | `-0.091905` | `-0.036804` |
| low-corr | `0.229035` | `0.073808` | `-0.129854` | `-0.048388` |

Rolling IC remains the return leader. IC-weighted remains the smoother cost/impact robustness control because its worst year and drawdown are better than rolling IC in several stress views. Low-corr remains useful as a diversified diagnostic but is not the leading candidate under this combined gate.

## Exposure Evidence

The industry cap is doing its job as a concentration control. Mean absolute active industry weight is low:

- Rolling IC monthly mean abs active industry weight: `0.008826`; max abs active industry weight: about `0.080904`.
- IC-weighted monthly mean abs active industry weight: `0.009079`; max: about `0.080904`.
- Low-corr monthly mean abs active industry weight: `0.008549`; max: about `0.080026`.

However, style exposure is still not neutralized:

- Rolling IC monthly active `log_amount_mean_20d_z`: `-0.998691`.
- IC-weighted monthly active `log_amount_mean_20d_z`: `-0.984496`.
- Low-corr monthly active `log_amount_mean_20d_z`: `-0.985770`.
- Rolling IC monthly active `neg_volatility_20d_z`: `0.797025`.
- IC-weighted monthly active `neg_volatility_20d_z`: `0.845810`.
- Low-corr monthly active `momentum_20d_z`: `-0.575090`.

So the combined gate solves industry concentration reasonably well, but it does not solve the low-liquidity / low-amount proxy, low-volatility, and weak-momentum style exposure issue.

## Capacity Evidence

The 100m participation stress remains survivable in this proxy model:

- 2026 p95 participation at 100m is about `0.021` for all three signals.
- Worst max participation at 100m in 2026 is roughly `0.036` to `0.044`.
- 2024 is the stressed year, with worst max participation about `0.119` to `0.149`.

This is acceptable as a proxy stress result, but it is not a final capacity proof. The model still uses Baostock `amount` and p95 participation approximations; it does not model partial fills, queues, market impact curvature, real float, or order scheduling.

## Decision

- Candidate count remains `0`.
- Frontier status remains `candidate-frontier/backtest_only`.
- The combined gate strengthens the frontier evidence because return survives industry cap plus impact stress.
- It does not complete strategy promotion because:
  - Completed monthly periods are still only `16-18` across 2024-2026, and 2026 has only `2`.
  - The basket still has large style active exposure to low amount/liquidity proxy, low volatility, weak momentum, and turnover structure.
  - True PIT market cap / float cap is still missing.
  - Industry data is still `month-start` forward-filled, not exact daily industry PIT.
  - Capacity stress remains proxy-based.

## Next Step

Do not enter traditional ML yet. The next highest-value work is a true-size/data-source gate:

1. Evaluate external PIT market cap / float cap sources.
2. If a source is usable, add `market_cap` and `float_market_cap` to v2.2.
3. Re-run this combined gate with true size/float-size exposure diagnostics.
4. If external size is unavailable, define a stricter proxy-only promotion policy and keep strategy status below `production_candidate`.
