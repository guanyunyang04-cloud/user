# Low-Corr Frontier Exposure Penalty Grid Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_low_corr_frontier_exposure_penalty_grid_interpretation.md`.


## Summary

- Run: `low_corr_frontier_exposure_penalty_grid_20260603_085614`.
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`.
- Protocol: `2024,2025,2026`, `horizon=20`, `monthly`, `top_n=200`, `buffer=3.0`, `fee_bps=30`, execution constraints enabled.
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_ic_weighted_score`, `multifactor_low_corr_rank_score`.
- Penalty columns: `log_amount_mean_20d_z,neg_volatility_20d_z,momentum_20d_z,turn_xsec_z`.
- Strengths: `0.0,0.25,0.5,1.0`.
- Artifacts: `traditional_quant_research/output/experiments/low_corr_frontier_exposure_penalty_grid/low_corr_frontier_exposure_penalty_grid_20260603_085614`.

This run upgrades the previous single-strength portfolio exposure penalty smoke into a real grid. The result supports keeping portfolio-level exposure penalty as a useful control mechanism, but it does not promote any signal to strategy candidate status.

## Aggregate Evidence

At 30 bps, rolling IC remains the strongest frontier signal. A weak penalty improves it slightly:

| signal | best strength | mean annualized return | min annualized return | worst max drawdown | mean turnover | delta vs strength 0 |
|:--|--:|--:|--:|--:|--:|--:|
| rolling IC | `0.25` | `0.353015` | `0.071950` | `-0.092856` | `1.014405` | `+0.002838` |
| IC-weighted | `0.0` | `0.280980` | `0.113754` | `-0.085134` | `1.001458` | `0.000000` |
| low-corr | `1.0` | `0.275757` | `0.097905` | `-0.111611` | `1.133333` | `+0.013012` |

Full strength ordering:

- Rolling IC: `0.25 > 0.5 > 1.0 > 0.0`; all strengths keep positive years, and `0.25/0.5` beat no-penalty in `2/3` years.
- IC-weighted: `0.0` remains best; all tested penalties reduce mean annualized return.
- Low-corr: `1.0` is best on mean return and min year; `0.5` also improves mean return, while `0.25` is worse than no penalty.

## Exposure Evidence

The penalty moves targeted basket exposures in the expected direction, but the effect is moderate rather than decisive.

Examples from monthly mean active exposure:

- Rolling IC `log_amount_mean_20d_z`: `-1.001891` at strength `0.0` to `-0.977177` at strength `1.0`.
- Rolling IC `neg_volatility_20d_z`: `0.800373` to `0.780835`.
- Rolling IC `turn_xsec_z`: `-0.469745` to `-0.455069`.
- Low-corr `log_amount_mean_20d_z`: `-1.019779` to `-0.987043`.
- Low-corr `turn_xsec_z`: `-0.438705` to `-0.418304`.

This confirms that the selector is reacting to basket-level active exposure. It also confirms the limitation: even at strength `1.0`, the selected portfolios remain meaningfully tilted toward low amount/liquidity proxy, low volatility, and low turnover style. The penalty is a steering force, not an exposure-neutral optimizer.

## Interpretation

The result is better than hard multi-metrics residualization because it preserves most of the original frontier performance while nudging basket exposure. However, it is not strong enough to become a candidate gate by itself.

The main reading by signal:

- Rolling IC: `strength=0.25` is the current best watchlist setting. It slightly improves return and drawdown versus no-penalty, with only a small turnover increase.
- IC-weighted: no-penalty remains preferred. This signal already has the best cost robustness among the original frontier set, and extra penalty mostly subtracts alpha.
- Low-corr: `strength=1.0` improves mean and worst-year return, but turnover and drawdown remain less attractive than rolling IC / IC-weighted. Treat it as an exploratory setting, not a promoted configuration.

## Decision

- Candidate count remains `0`.
- Frontier status remains `candidate-frontier/backtest_only`.
- Portfolio exposure penalty is accepted as useful infrastructure.
- The grid does not complete the exposure gate because:
  - 2026 still has only `2` completed monthly periods.
  - Total non-overlapping trade count is still small: `16-18` periods depending on signal.
  - Basket active exposure is reduced but not neutralized.
  - True PIT market cap / float cap is still missing.
  - This grid has not yet been combined with industry cap, industry-neutral signal variants, impact stress, or a broader out-of-sample selection protocol.

## Next Step

The next useful gate is not traditional ML. It should be a combined portfolio-construction audit:

1. Rolling IC with `exposure_penalty_strength=0.25`.
2. IC-weighted with no exposure penalty as cost-robust control.
3. Low-corr with `exposure_penalty_strength=1.0` as exploratory control.
4. Combine with `group_col=industry,max_group_weight=0.10`.
5. Re-run impact stress and basket exposure summaries.
6. Keep candidate count at `0` until combined constraints pass stability, cost, exposure, and capacity checks.
