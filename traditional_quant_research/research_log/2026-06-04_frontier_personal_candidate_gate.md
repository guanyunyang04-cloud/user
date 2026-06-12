# Frontier Personal Candidate Gate

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_frontier_personal_candidate_gate.md`.


- run_id: `frontier_personal_candidate_gate_20260604_004311`
- north_star: `Baostock-only personal quant strategy research`
- decision: `personal_paper_tracking_ready`
- personal_backtest_candidate_count: `1`
- strategy_candidate_count: `0`
- required_year_window: `2017-2026`
- required_fee_bps: `30.0`
- required_impact_bps_per_1pct: `10.0`
- personal_capital_amount: `1000000.0`
- best_signal_by_personal_gate: `multifactor_rolling_ic_weighted_score`
- best_signal_mean_annualized_return: `0.096737`

## Failed Gates

- `weak_year_damage_gate`: `1`
- `return_gate`: `1`

## Gate Summary

| constraint_variant   | signal                                |   fee_bps |   capital_amount |   impact_bps_per_1pct |   personal_capital_amount |   exposure_penalty_strength |   constraint_fallback_count |   constraint_fallback_rate |   mean_annualized_return |   min_annualized_return |   positive_year_rate |   worst_max_drawdown |   total_periods |   eval_year_count |   max_proxy_mean_abs_active_exposure | baostock_source_gate   | walk_forward_gate   | execution_gate   | capital_stress_gate   | return_gate   | weak_year_damage_gate   | sample_gate   | drawdown_gate   | proxy_exposure_sanity_gate   | optimizer_fallback_gate   | explainability_gate   | promoted   | failed_gates          | exposure_failures   | walk_forward_detail     | promotion_level                 | paper_tracking_recommendation   |
|:---------------------|:--------------------------------------|----------:|-----------------:|----------------------:|--------------------------:|----------------------------:|----------------------------:|---------------------------:|-------------------------:|------------------------:|---------------------:|---------------------:|----------------:|------------------:|-------------------------------------:|:-----------------------|:--------------------|:-----------------|:----------------------|:--------------|:------------------------|:--------------|:----------------|:-----------------------------|:--------------------------|:----------------------|:-----------|:----------------------|:--------------------|:------------------------|:--------------------------------|:--------------------------------|
| baseline             | multifactor_rolling_ic_weighted_score |        30 |            1e+08 |                    10 |                     1e+06 |                        0.25 |                           0 |                          0 |                 0.096737 |               -0.301352 |                  0.6 |            -0.138549 |              60 |                10 |                              1.11138 | True                   | True                | True             | True                  | True          | True                    | True          | True            | True                         | True                      | True                  | True       |                       |                     | covered_years=2017-2026 | personal_backtest_candidate     | start_paper_tracking            |
| baseline             | multifactor_low_corr_rank_score       |        30 |            1e+08 |                    10 |                     1e+06 |                        1    |                           0 |                          0 |                 0.056183 |               -0.385395 |                  0.6 |            -0.183577 |              64 |                10 |                              1.05854 | True                   | True                | True             | True                  | True          | False                   | True          | True            | True                         | True                      | True                  | False      | weak_year_damage_gate |                     | covered_years=2017-2026 | personal_research/backtest_only | continue_research               |
| baseline             | multifactor_ic_weighted_score         |        30 |            1e+08 |                    10 |                     1e+06 |                        0    |                           0 |                          0 |                 0.023878 |               -0.336486 |                  0.6 |            -0.169636 |              64 |                10 |                              1.05204 | True                   | True                | True             | True                  | False         | True                    | True          | True            | True                         | True                      | True                  | False      | return_gate           |                     | covered_years=2017-2026 | personal_research/backtest_only | continue_research               |

## Interpretation

This is a pragmatic personal-trading gate. It keeps the checks that prevent fake profitability (walk-forward evidence, execution constraints, cost stress, drawdown, weak-year damage, and proxy exposure disclosure) while explicitly not requiring true market-cap or float-cap data before paper tracking.
