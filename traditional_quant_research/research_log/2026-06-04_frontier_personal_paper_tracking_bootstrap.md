# Frontier Personal Paper Tracking Bootstrap

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_frontier_personal_paper_tracking_bootstrap.md`.


- run_id: `frontier_personal_paper_tracking_bootstrap_20260604_004320`
- decision: `paper_tracking_bootstrap_ready`
- north_star: `Baostock-only personal quant strategy research`
- personal_backtest_candidate_count: `1`
- personal_paper_candidate_count: `0`
- strategy_candidate_count: `0`
- tracking_status: `paper_tracking_bootstrapped`
- personal_gate_run_id: `frontier_personal_candidate_gate_20260604_004311`
- combined_run_id: `low_corr_frontier_combined_constraint_audit_20260603_122047`

## Candidates

| candidate_id                                                | tracking_status             | current_level               | target_next_level        | paper_tracking_recommendation   | north_star                                     | snapshot_id                                                             | required_year_window   | signal                                | constraint_variant   |   exposure_penalty_strength |   fee_bps |   impact_bps_per_1pct |   backtest_capital_stress_amount |   personal_capital_amount |   horizon | rebalance_frequency   |   top_n |   buffer_multiplier |   mean_annualized_return |   min_annualized_return |   positive_year_rate |   worst_max_drawdown |   total_periods |   eval_year_count |   max_proxy_mean_abs_active_exposure | paper_tracking_started   | personal_paper_candidate   | strategy_candidate   |
|:------------------------------------------------------------|:----------------------------|:----------------------------|:-------------------------|:--------------------------------|:-----------------------------------------------|:------------------------------------------------------------------------|:-----------------------|:--------------------------------------|:---------------------|----------------------------:|----------:|----------------------:|---------------------------------:|--------------------------:|----------:|:----------------------|--------:|--------------------:|-------------------------:|------------------------:|---------------------:|---------------------:|----------------:|------------------:|-------------------------------------:|:-------------------------|:---------------------------|:---------------------|
| multifactor_rolling_ic_weighted_score_baseline_penalty_0_25 | paper_tracking_bootstrapped | personal_backtest_candidate | personal_paper_candidate | start_paper_tracking            | Baostock-only personal quant strategy research | baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603 | 2017-2026              | multifactor_rolling_ic_weighted_score | baseline             |                        0.25 |        30 |                    10 |                            1e+08 |                     1e+06 |        20 | monthly               |     200 |                   3 |                 0.096737 |               -0.301352 |                  0.6 |            -0.138549 |              60 |                10 |                              1.11138 | False                    | False                      | False                |

## Protocol

| candidate_id                                                | tracking_status             | current_level               | target_next_level        | data_source   | signal_generation_source      | execution_mode      | rebalance_frequency   |   horizon |   top_n |   buffer_multiplier |   fee_bps |   impact_bps_per_1pct |   personal_capital_amount |   min_tracking_periods |   min_tracking_days |   max_paper_drawdown |   max_single_period_loss |   min_paper_excess_return | promotion_rule                                                    | downgrade_rule                                                                     | true_size_required   | institutional_promotion_allowed   |
|:------------------------------------------------------------|:----------------------------|:----------------------------|:-------------------------|:--------------|:------------------------------|:--------------------|:----------------------|----------:|--------:|--------------------:|----------:|----------------------:|--------------------------:|-----------------------:|--------------------:|---------------------:|-------------------------:|--------------------------:|:------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------|:----------------------------------|
| multifactor_rolling_ic_weighted_score_baseline_penalty_0_25 | paper_tracking_bootstrapped | personal_backtest_candidate | personal_paper_candidate | baostock_only | combined_constraint_artifacts | paper_tracking_only | monthly               |        20 |     200 |                   3 |        30 |                    10 |                     1e+06 |                      6 |                 120 |                 -0.2 |                    -0.12 |                     -0.02 | promote only after complete paper evidence meets all review rules | downgrade to personal_research/backtest_only on execution drift or drawdown breach | False                | False                             |

## Review Rules

| dimension               | pass_condition                                                                                           | remain_or_downgrade_condition                                                                    |
|:------------------------|:---------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| minimum_tracking_sample | >=6 completed rebalance records and >=120 calendar days                                                  | remain personal_backtest_candidate until the paper sample is complete                            |
| execution_integrity     | record selected_count, blocked entries, limit-up/down delays, turnover, fees, and notes for every period | downgrade if execution records are missing or materially inconsistent with the backtest protocol |
| paper_return_sanity     | paper excess return >= -0.0200 after costs over the completed tracking window                            | remain research-only if paper return is too weak even when execution is clean                    |
| drawdown_control        | max paper drawdown stays above -0.2000                                                                   | downgrade if drawdown breaches the paper limit                                                   |
| single_period_damage    | no single completed tracking period is below -0.1200                                                     | trigger review if a single period breaches the damage threshold                                  |
| promotion_boundary      | only personal_paper_candidate is allowed after paper evidence passes                                     | never call the row strategy_candidate without true-size and institutional gates                  |

## Interpretation

This package is the first operational step after a personal backtest candidate passes. It defines what to track, how to record execution, and what evidence is required before any future `personal_paper_candidate` decision. It deliberately keeps `strategy_candidate_count=0`.
