# Frontier ML Signal Diagnostic Closure

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_frontier_ml_signal_diagnostic_closure.md`.


- ML rebuild run: `frontier_ml_signal_rebuild_20260604_165434`
- smoke personal grid: `frontier_personal_protocol_grid_20260604_150232`
- formal personal grid: `frontier_personal_protocol_grid_20260604_165943`
- ml_signal: `ml_lgbm_xsec_excess_score_h20_prior_fit`
- model: `lightgbm.LGBMRegressor`
- factor_set: `expanded`
- target: `xsec_excess_ret_20d`
- evidence_scope: `formal_personal_backtest_candidate_gate`
- personal_backtest_candidate_count: `0`
- personal_paper_candidate_count: `0`
- strategy_candidate_count: `0`

## Summary

The Baostock-only early ML diagnostic line is now implemented and tested against the existing personal gate chain. The formal run used `2017-2026 / 20d / monthly / buffer=3.0 / top_n=50,100,200 / 30bps / 100m / 10bps impact`, with `baseline` and `capital_scaled` variants.

The ML signal produced valid prior-fit predictions for all 10 eval years:

- prediction_rows: `6489959`
- feature_count: `14`
- ready_eval_year_count: `10`
- fit_uses_eval_year_count: `0`

The best formal row was:

| protocol_id | mean_annualized_return | min_annualized_return | positive_year_rate | worst_max_drawdown | failed_gates |
|:--|--:|--:|--:|--:|:--|
| `top200_ml_lgbm_xsec_excess_score_h20_prior_fit_capital_scaled_penalty_0` | `0.186804` | `-0.361206` | `0.6` | `-0.206932` | `weak_year_damage_gate` |

## Interpretation

This is a meaningful model-strength diagnostic, but not a new candidate. LightGBM materially improved average long-sample return relative to many prior simple rebuild attempts, especially at `top_n=200`, but it did not repair weakest-year damage enough for formal personal promotion.

`capital_scaled` helped the ML signal at `top_n=200` (`0.186804` vs baseline `0.171036`) and at smaller baskets, but the best row still failed `weak_year_damage_gate`. `top_n=50/100` remained too fragile and failed return, weak-year, or drawdown gates.

The generic market-regime fallback was used for `capital_scaled` where available, preserving `weak_year_rule_source_signal=__generic_market_regime__` and `weak_year_rule_scope=generic_market_regime`. It reduced some weak-year damage, but did not make the ML signal eligible for paper tracking.

Top mean feature importances were:

| feature | mean_importance |
|:--|--:|
| `neg_turn_mean_20d_z` | `191.8` |
| `range_position_20d_z` | `129.1` |
| `neg_volatility_20d_z` | `127.6` |
| `log_amount_mean_20d_z` | `123.7` |
| `log_amount_mean_5d_z` | `116.5` |
| `momentum_20d_z` | `116.4` |

This suggests the ML model is leaning heavily on turnover, range position, volatility, liquidity/amount, and momentum structure. Those are legitimate Baostock-only signals, but the remaining failure still points to regime/weak-year handling rather than data-source insufficiency.

## Decision

- Do not create a new `personal_backtest_candidate`.
- Do not bootstrap ML paper tracking from this run.
- Keep existing 3 `personal_backtest_candidate` entries unchanged and continue paper tracking.
- Treat ML v1 as `personal_research/backtest_only`.
- Next ML work should focus on weak-year repair: regime-conditioned training, separate weak-regime model heads, down-market loss weighting, or regime-specific post-model capital scaling. It should not be promoted from smoke or relaxed evidence.
