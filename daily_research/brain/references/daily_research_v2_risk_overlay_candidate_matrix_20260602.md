# Daily Research V2 Risk Overlay Candidate Matrix - 2026-06-02

## Summary

- Status: `v2_risk_overlay_candidate_matrix / evidence_grade_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_candidate_review_matrix`.
- Matrix run tag: `v2_candidate_review_matrix_risk_overlay_20260602_01`.
- Source bridge: `v2_score_backtest_bridge_20260602_01`.
- Source attribution: `v2_bad_month_attribution_20260602_01`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_candidate_review_matrix_risk_overlay_20260602_01`.
- `v2_score_backtest_bridge_20260602_01`.
- `v2_bad_month_attribution_20260602_01`.

## Matrix Scope

- Fixed holding count: `20`.
- Max weights tested: `0.04`, `0.06`, `0.08`.
- Rebalance frequencies tested: `3d`, `5d`, `10d`.
- Offset mode: `all`.
- Market regime filter: `off`, `on`.
- Cost assumptions: transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`.
- Variant count: `18`.
- Completed shared backtests: `18`.
- Promotion-review eligible variants: `0`.

## Best By Excess Sharpe

| variant | excess_sharpe | excess_annual_return | max_drawdown | positive_month_ratio | negative_months | worst_month | cost_drag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h20_mw080_rb5d_all_c3_7_10_regime_off` | `0.952747` | `0.267906` | `-0.260202` | `0.636364` | `4` | `-0.098235` | `0.149831` |
| `h20_mw080_rb10d_all_c3_7_10_regime_off` | `0.931986` | `0.238898` | `-0.264113` | `0.636364` | `4` | `-0.074040` | `0.104964` |
| `h20_mw080_rb3d_all_c3_7_10_regime_off` | `0.926132` | `0.268056` | `-0.254487` | `0.636364` | `4` | `-0.103990` | `0.188658` |
| `h20_mw060_rb10d_all_c3_7_10_regime_off` | `0.925946` | `0.232845` | `-0.264738` | `0.636364` | `4` | `-0.069287` | `0.102449` |

## Best By Drawdown

| variant | excess_sharpe | excess_annual_return | max_drawdown | positive_month_ratio | negative_months | worst_month | cost_drag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h20_mw040_rb5d_all_c3_7_10_regime_on` | `0.174742` | `0.032723` | `-0.119922` | `0.636364` | `4` | `-0.088989` | `0.053492` |
| `h20_mw040_rb10d_all_c3_7_10_regime_on` | `0.002925` | `0.000547` | `-0.125680` | `0.454545` | `6` | `-0.088989` | `0.041742` |
| `h20_mw040_rb3d_all_c3_7_10_regime_on` | `0.090238` | `0.016851` | `-0.126622` | `0.636364` | `4` | `-0.088989` | `0.060812` |

## Interpretation

- No tested risk-overlay candidate is promotion-review eligible.
- The original `mw0.08 / 5d / regime_off` candidate remains best by excess Sharpe, but it still fails monthly stability, negative month count, drawdown, and deep bad-month checks.
- Market-regime filtering materially reduces drawdown and cost drag, but it is too blunt: it also collapses excess return and excess Sharpe and introduces or preserves multi-month drawdown-streak flags.
- Lower max weight helps drawdown only when combined with regime filtering, but does not repair monthly positive ratio or negative month count.
- `10d` rebalance without regime filter slightly reduces worst month and cost drag versus `5d`, but still fails the same monthly stability gate.
- The next repair should not be a coarse global risk-off switch. It should target high-score/high-weight exposure more selectively with confidence throttles, reversal/volatility guards, or state-specific sizing.

## Boundaries

- Do not promote to live/default from this matrix.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Do not treat the score panel as a production target-weight panel.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Next Allowed Actions

- Implement a candidate-level high-confidence throttle or soft-state sizing variant that reduces high-score/high-weight exposure only under adverse state.
- Add finer attribution dimensions before the next matrix:
  - sector;
  - liquidity bucket;
  - volatility bucket;
  - reversal / recent drawdown bucket;
  - predicted horizon bucket.
- Test selective filters instead of coarse regime filter:
  - high-score max-weight cap under recent negative breadth;
  - skip or downweight recent reversal losers;
  - volatility-scaled target weights;
  - monthly drawdown stop / exposure taper.

## Verification

- `python -m daily_research.path_policy.v2_candidate_review_matrix --run-tag v2_candidate_review_matrix_risk_overlay_20260602_01 --run-backtests --holding-counts 20 --max-weights 0.04,0.06,0.08 --rebalance-freqs 3d,5d,10d --rebalance-offset-modes all --transaction-cost-bps-values 3 --slippage-bps-values 7 --sell-tax-bps-values 10 --market-regime-filter-modes off,on --json`: completed, `18` backtests completed, `0` promotion-review eligible variants.
- `git diff -- daily_research/output/active_execution_strategy.json`: must remain empty.
