# Frontier Promotion Gate

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_frontier_promotion_gate.md`.


- run_id: `frontier_promotion_gate_20260603_202428`
- decision: `keep_candidate_frontier_backtest_only`
- candidate_count: `0`
- daily_size_status: `daily_size_absent`
- daily_size_ready_for_research: `False`
- required_impact_bps: `10.0`
- min_total_periods: `24`
- max_monthly_mean_abs_active_exposure: `0.5`
- best_signal_by_return: `multifactor_rolling_ic_weighted_score`
- best_signal_mean_annualized_return: `0.096737`

## Failed Gates

- `size_gate`: `3`
- `return_gate`: `3`
- `year_gate`: `3`
- `style_exposure_gate`: `3`

## Gate Summary

| signal                                |   impact_bps_per_1pct |   fee_bps |   exposure_penalty_strength |   mean_annualized_return |   min_annualized_return |   positive_year_rate |   worst_max_drawdown |   total_periods |   max_monthly_mean_abs_active_exposure | size_gate   | return_gate   | year_gate   | sample_gate   | drawdown_gate   | style_exposure_gate   | promoted   | failed_gates                                        | exposure_failures                                         | promotion_level                  |
|:--------------------------------------|----------------------:|----------:|----------------------------:|-------------------------:|------------------------:|---------------------:|---------------------:|----------------:|---------------------------------------:|:------------|:--------------|:------------|:--------------|:----------------|:----------------------|:-----------|:----------------------------------------------------|:----------------------------------------------------------|:---------------------------------|
| multifactor_rolling_ic_weighted_score |                    10 |        30 |                        0.25 |                 0.096737 |               -0.301352 |                  0.6 |            -0.138549 |              60 |                                1.11138 | False       | False         | False       | True          | True            | False                 | False      | size_gate,return_gate,year_gate,style_exposure_gate | log_amount_mean_20d_z,neg_volatility_20d_z                | candidate-frontier/backtest_only |
| multifactor_low_corr_rank_score       |                    10 |        30 |                        1    |                 0.056183 |               -0.385395 |                  0.6 |            -0.183577 |              64 |                                1.05854 | False       | False         | False       | True          | True            | False                 | False      | size_gate,return_gate,year_gate,style_exposure_gate | log_amount_mean_20d_z,momentum_20d_z,neg_volatility_20d_z | candidate-frontier/backtest_only |
| multifactor_ic_weighted_score         |                    10 |        30 |                        0    |                 0.023878 |               -0.336486 |                  0.6 |            -0.169636 |              64 |                                1.05204 | False       | False         | False       | True          | True            | False                 | False      | size_gate,return_gate,year_gate,style_exposure_gate | log_amount_mean_20d_z,neg_volatility_20d_z                | candidate-frontier/backtest_only |

## Interpretation

This gate is a promotion decision layer. Positive constrained returns alone are insufficient: the frontier must also have enough independent periods, execution/cost evidence, true size readiness, and controlled active exposures. Rows that fail remain `candidate-frontier/backtest_only`.
