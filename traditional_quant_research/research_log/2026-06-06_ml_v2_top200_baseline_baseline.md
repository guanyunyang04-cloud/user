# Frontier Personal Protocol Grid

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-06_ml_v2_top200_baseline_baseline.md`.


- run_id: `frontier_personal_protocol_grid_20260606_210925`
- north_star: `Baostock-only personal quant strategy research`
- decision: `keep_personal_research_backtest_only`
- top_n_values: `[200]`
- protocol: `horizon=20`, `monthly`, `buffer=3.0`
- required stress: `30.0` fee bps, `10.0` impact bps per 1 pct, personal capital `1000000.0`
- personal_backtest_candidate_count: `0`
- personal_paper_candidate_count: `0`
- strategy_candidate_count: `0`
- post_selection_boundary: `agent_selects_models_and_strategies_only; user_handles_risk_recording_and_live_decisions`
- evidence_scopes: `['formal_personal_backtest_candidate_gate']`
- best_protocol_id: `top200_ml_lgbm_xsec_excess_score_h20_prior_fit_baseline_penalty_top_n_penalty_0_25`
- Artifacts: `traditional_quant_research\output\experiments\frontier_personal_protocol_grid\frontier_personal_protocol_grid_20260606_210925`

## Top-N Summary

|   top_n |   evaluated_rows |   personal_backtest_candidate_count | best_protocol_id                                                                   | best_signal                             | best_promotion_level            |   best_mean_annualized_return |   best_min_annualized_return |   best_positive_year_rate |   best_worst_max_drawdown | decision          |
|--------:|-----------------:|------------------------------------:|:-----------------------------------------------------------------------------------|:----------------------------------------|:--------------------------------|------------------------------:|-----------------------------:|--------------------------:|--------------------------:|:------------------|
|     200 |                1 |                                   0 | top200_ml_lgbm_xsec_excess_score_h20_prior_fit_baseline_penalty_top_n_penalty_0_25 | ml_lgbm_xsec_excess_score_h20_prior_fit | personal_research/backtest_only |                      0.201848 |                    -0.352624 |                       0.6 |                 -0.195403 | continue_research |

## Protocol Ledger

| protocol_id                                                                        | research_track                       |   top_n |   horizon | rebalance_frequency   |   buffer_multiplier | constraint_variant   | portfolio_constraint_mode   | signal                                  |   exposure_penalty_strength |   fee_bps |   capital_amount |   impact_bps_per_1pct |   personal_capital_amount |   mean_annualized_return |   min_annualized_return |   positive_year_rate |   worst_max_drawdown |   total_periods |   eval_year_count |   max_proxy_mean_abs_active_exposure | formal_profile_gate   | evidence_scope                          | gate_profile_detail         | promoted   | promotion_level                 | paper_tracking_recommendation   | failed_gates          | evidence_grade                  | strategy_candidate   | combined_run_dir                                                                                                                                                                                                         | personal_gate_run_dir                                                                                                                                                                                  |   rank_overall |   rank_within_top_n |
|:-----------------------------------------------------------------------------------|:-------------------------------------|--------:|----------:|:----------------------|--------------------:|:---------------------|:----------------------------|:----------------------------------------|----------------------------:|----------:|-----------------:|----------------------:|--------------------------:|-------------------------:|------------------------:|---------------------:|---------------------:|----------------:|------------------:|-------------------------------------:|:----------------------|:----------------------------------------|:----------------------------|:-----------|:--------------------------------|:--------------------------------|:----------------------|:--------------------------------|:---------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------:|--------------------:|
| top200_ml_lgbm_xsec_excess_score_h20_prior_fit_baseline_penalty_top_n_penalty_0_25 | baostock_only_personal_protocol_grid |     200 |        20 | monthly               |                   3 | baseline             | penalty_top_n               | ml_lgbm_xsec_excess_score_h20_prior_fit |                        0.25 |        30 |            1e+08 |                    10 |                     1e+06 |                 0.201848 |               -0.352624 |                  0.6 |            -0.195403 |              60 |                10 |                             0.470949 | True                  | formal_personal_backtest_candidate_gate | formal_defaults_or_stricter | False      | personal_research/backtest_only | none                            | weak_year_damage_gate | personal_research/backtest_only | False                | traditional_quant_research\output\experiments\frontier_personal_protocol_grid\frontier_personal_protocol_grid_20260606_210925\combined_constraint_merged\frontier_personal_protocol_grid_20260606_210925_combined_merged | traditional_quant_research\output\experiments\frontier_personal_protocol_grid\frontier_personal_protocol_grid_20260606_210925\personal_candidate_gate\frontier_personal_candidate_gate_20260606_214953 |              1 |                   1 |

## Interpretation

This experiment compares the current frontier under personal small-capital assumptions. Passing rows are selected candidates for the user to judge after delivery; they are not institutional strategy candidates.
