# Frontier Trial Ledger

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_frontier_trial_ledger.md`.


- run_id: `frontier_trial_ledger_20260603_202437`
- decision: `keep_candidate_frontier_backtest_only`
- trial_count: `6`
- candidate_count: `0`

## Selection Bias Report

| experiment_family            |   trial_count |   signal_count | best_signal                           |   best_mean_annualized_return |   median_mean_annualized_return |   best_minus_median_return | selected_trial_id                                |   weak_year_count |   positive_year_rate_min | selection_bias_risk   |
|:-----------------------------|--------------:|---------------:|:--------------------------------------|------------------------------:|--------------------------------:|---------------------------:|:-------------------------------------------------|------------------:|-------------------------:|:----------------------|
| frontier_combined_constraint |             6 |              3 | multifactor_rolling_ic_weighted_score |                      0.173626 |                        0.096108 |                   0.077518 | trial_0001_multifactor_rolling_ic_weighted_score |                 4 |                      0.6 | moderate              |

## Research Maturity Report

| dimension                    | status   | evidence                                                                                                                                |
|:-----------------------------|:---------|:----------------------------------------------------------------------------------------------------------------------------------------|
| retail_research_baseline     | ahead    | PIT universe, execution constraints, impact stress, promotion gate and failure attribution are already explicit.                        |
| academic_factor_standard     | partial  | Trial ledger and selection-bias report are now explicit, but cross-market replication and formal significance correction remain absent. |
| professional_quant_standard  | blocked  | True size/float-size, optimizer-grade risk controls and paper/live shadow evidence are not yet complete.                                |
| strategy_candidate_readiness | blocked  | Promotion gate must pass all size, return, year, sample, drawdown and style-exposure gates before candidate upgrade.                    |
| governance_visibility        | passed   | Trial ledger and selection-bias report artifacts are required before promotion review.                                                  |

## Interpretation

This ledger is a governance artifact. It does not rerun backtests and does not promote candidates by itself.
