# Traditional Quant Research 状态程序

## Object Instances
### object `traditional_quant_project`
`type`: traditional_quant_lab_instance
`state`: mounted to workspace main brain; current North Star is `Baostock-only personal quant strategy research`.
`scope`: research and candidate selection stop at `personal_backtest_candidate`; no agent-driven `personal_paper_candidate`, `personal_trading_candidate` or `strategy_candidate`.

### object `data_substrate_v2_1`
`type`: research_data_substrate
`state`: full-cycle PIT-like Baostock-based daily panel with industry + metrics snapshot strengthened; true market-cap/float-cap source audit found Baostock single-source insufficient.
`future_upgrade`: Tushare `daily_basic`, JoinQuant, RQData or equivalent authenticated PIT size source for true-size gate.

### object `frontier_personal_candidate_pool`
`type`: current_candidate_pool
`state`: 4 Baostock-only `personal_backtest_candidate` entries; `strategy_candidate_count = 0`.
`candidates`: `ml_lgbm_xsec_excess_score_h20_prior_fit_capital_scaled_penalty_top_n_top200_penalty_0_25`; `multifactor_rolling_ic_weighted_score_capital_scaled_penalty_top_n_top200_penalty_0_25`; `multifactor_rolling_ic_weighted_score_baseline_penalty_top_n_top200_penalty_0_25`; `multifactor_low_corr_rank_score_capital_scaled_penalty_top_n_top200_penalty_1`.
`best_current`: ML v2 regime-weighted / capital-scaled / penalty_top_n top200, mean annualized about `0.181658`, weakest year about `-0.329566`, positive-year ratio `0.6`, worst drawdown about `-0.154592`.
`user_boundary`: historical paper tracking and lifecycle registry are cancelled; candidate follow-up is user discretion.

### object `weak_year_problem`
`type`: persistent_blocker
`years`: `2017/2018/2022/2023`
`state`: frontier failure attribution shows weak years and style exposure remain the key blocker; high mean return does not replace weak-year gate.

### object `research_artifact_routes`
`type`: evidence_entrypoints
`framework`: `brain/references/research_framework.md`
`logs`: `brain/references/research_log/`
`data_catalog`: `brain/references/data_catalog.md`
`tests`: `traditional_quant_research/tests`

## Pure Functions
- `classify_candidate(row)`: returns diagnostic, personal_research/backtest_only, personal_backtest_candidate or strategy_candidate.
- `derive_next_research_action(state)`: improve weak-year robustness, regime-conditioned training, alpha/factor rebuild or post-model scaling; keep user-discretion boundary.
- `requires_size_source(task)`: true only for true-size/promotion/market-cap/float-cap claims.
- `resolve_evidence_entry(claim)`: prefer explicit research log, run dir, gate report and candidate selection report.

## Procedures
### procedure `formal_personal_candidate_review`
`input`: protocol grid, personal gate, trial ledger, candidate selection report.
`steps`: inspect formal profile coverage；check costs and weak-year damage；classify candidate；write research log.
`side_effects`: research evidence only.

### procedure `data_source_upgrade_review`
`input`: size/industry/status/metric source proposal.
`steps`: audit PIT/source grade/coverage/units；build cache only if auth and field semantics pass；write catalog and research log.
`side_effects`: data artifacts and references.
