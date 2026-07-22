# Frontier Personal Candidate Gate

- run_id: `frontier_personal_candidate_gate_20260604_083341`
- north_star: `Baostock-only personal quant strategy research`
- decision: `keep_personal_research_backtest_only`
- personal_backtest_candidate_count: `0`
- strategy_candidate_count: `0`
- required_year_window: `2017-2026`
- required_fee_bps: `30.0`
- required_impact_bps_per_1pct: `10.0`
- personal_capital_amount: `1000000.0`
- evidence_scope: `formal_personal_backtest_candidate_gate`
- formal_gate_profile: `True`
- gate_profile_detail: `formal_defaults_or_stricter`
- best_signal_by_personal_gate: `multifactor_rolling_ic_weighted_score`
- best_top_n_by_personal_gate: `200`
- best_signal_mean_annualized_return: `0.078193`

## Failed Gates

- `weak_year_damage_gate`: `3`
- `return_gate`: `2`

## Gate Summary

| constraint_variant      |   top_n | signal                                |   fee_bps |   capital_amount |   impact_bps_per_1pct |   personal_capital_amount |   exposure_penalty_strength |   constraint_fallback_count |   constraint_fallback_rate |   mean_annualized_return |   min_annualized_return |   positive_year_rate |   worst_max_drawdown |   total_periods |   eval_year_count |   max_proxy_mean_abs_active_exposure | formal_profile_gate   | evidence_scope                          | gate_profile_detail         | baostock_source_gate   | walk_forward_gate   | execution_gate   | capital_stress_gate   | return_gate   | weak_year_damage_gate   | sample_gate   | drawdown_gate   | proxy_exposure_sanity_gate   | optimizer_fallback_gate   | explainability_gate   | promoted   | failed_gates                      | exposure_failures   | walk_forward_detail     | promotion_level                 | paper_tracking_recommendation   |
|:------------------------|--------:|:--------------------------------------|----------:|-----------------:|----------------------:|--------------------------:|----------------------------:|----------------------------:|---------------------------:|-------------------------:|------------------------:|---------------------:|---------------------:|----------------:|------------------:|-------------------------------------:|:----------------------|:----------------------------------------|:----------------------------|:-----------------------|:--------------------|:-----------------|:----------------------|:--------------|:------------------------|:--------------|:----------------|:-----------------------------|:--------------------------|:----------------------|:-----------|:----------------------------------|:--------------------|:------------------------|:--------------------------------|:--------------------------------|
| factor_family_prior_fit |     200 | multifactor_rolling_ic_weighted_score |        30 |            1e+08 |                    10 |                     1e+06 |                        0.25 |                           0 |                          0 |                 0.078193 |               -0.486795 |                  0.6 |            -0.138549 |              58 |                10 |                              1.11785 | True                  | formal_personal_backtest_candidate_gate | formal_defaults_or_stricter | True                   | True                | True             | True                  | True          | False                   | True          | True            | True                         | True                      | True                  | False      | weak_year_damage_gate             |                     | covered_years=2017-2026 | personal_research/backtest_only | continue_research               |
| factor_family_prior_fit |     200 | multifactor_ic_weighted_score         |        30 |            1e+08 |                    10 |                     1e+06 |                        0    |                           0 |                          0 |                 0.018409 |               -0.336486 |                  0.5 |            -0.171939 |              61 |                10 |                              1.13228 | True                  | formal_personal_backtest_candidate_gate | formal_defaults_or_stricter | True                   | True                | True             | True                  | False         | False                   | True          | True            | True                         | True                      | True                  | False      | return_gate,weak_year_damage_gate |                     | covered_years=2017-2026 | personal_research/backtest_only | continue_research               |
| factor_family_prior_fit |     200 | multifactor_low_corr_rank_score       |        30 |            1e+08 |                    10 |                     1e+06 |                        1    |                           0 |                          0 |                -0.022299 |               -0.368741 |                  0.5 |            -0.215864 |              61 |                10 |                              0.89339 | True                  | formal_personal_backtest_candidate_gate | formal_defaults_or_stricter | True                   | True                | True             | True                  | False         | False                   | True          | True            | True                         | True                      | True                  | False      | return_gate,weak_year_damage_gate |                     | covered_years=2017-2026 | personal_research/backtest_only | continue_research               |

## Interpretation

This is a pragmatic personal-trading gate. It keeps the checks that prevent fake profitability (walk-forward evidence, execution constraints, cost stress, drawdown, weak-year damage, and proxy exposure disclosure) while explicitly not requiring true market-cap or float-cap data before paper tracking.

## Closure Decision

The `factor_family_prior_fit` combined evidence did not create a new `personal_backtest_candidate`. The formal gate stayed at `keep_personal_research_backtest_only` with `personal_backtest_candidate_count=0` for this run and `strategy_candidate_count=0`.

The best row, `multifactor_rolling_ic_weighted_score`, passed return, sample, drawdown, proxy exposure, optimizer fallback, execution, and walk-forward checks, but failed `weak_year_damage_gate` because the weakest selected-family year fell to `-0.486795`. `multifactor_ic_weighted_score` and `multifactor_low_corr_rank_score` both failed `return_gate` and `weak_year_damage_gate`.

Research conclusion: prior-fit factor-family selection is now formally falsified as a standalone upgrade path for the current frontier. It remains useful as a diagnostic scaffold, but the next model-strength work should focus on weak-year alpha/regime reconstruction rather than more family switching or blind factor expansion.
