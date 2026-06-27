# Traditional Quant Research 治理对象

## Governed Objects
### object `research_evidence`
`scope`: backtests, diagnostics, candidate gates, protocol grids and run summaries.
`invariant`: evidence grade stays explicit; sample-in, no-fee, loose latest, failed or interrupted evidence remains diagnostic.

### object `data_semantics`
`scope`: data range, PIT/source grade, universe, fees/slippage, holding constraints, labels and metrics.
`invariant`: data semantics changes create research log or reference entries.

### object `candidate_boundary`
`scope`: `personal_backtest_candidate`, `strategy_candidate`, paper/live/trading plans.
`invariant`: agent may select personal backtest candidates; strategy/paper/live decisions are not inferred from research artifacts.

## Pure Functions
- `select_governed_object(task) -> research_evidence|data_semantics|candidate_boundary`
- `classify_evidence(result) -> diagnostic|formal_personal|candidate|strategy`
- `requires_research_log(change) -> bool`

## Procedures
### procedure `research_conclusion`
`input`: result and evidence path
`steps`: separate fact/inference/assumption；classify evidence；write summary and reference if durable.

### procedure `candidate_boundary_review`
`input`: gate output or candidate selection report
`steps`: inspect formal profile；classify candidate status；preserve user-discretion boundary.
