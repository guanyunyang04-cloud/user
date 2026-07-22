# Factor-Pruned Formal Personal Grid

- run_id: `frontier_personal_protocol_grid_20260604_124941`
- pruning_plan_run: `frontier_factor_pruning_rebuild_20260604_121809`
- signal: `factor_pruned_rank_score_h20_prior_fit`
- protocol: `2017-2026 / factor_set=core / 20d / monthly / top_n=100,200 / buffer=3.0 / 30bps / 100m / 10bps impact / execution constraints`
- decision: `keep_personal_research_backtest_only`
- personal_backtest_candidate_count: `0`
- personal_paper_candidate_count: `0`
- strategy_candidate_count: `0`

## Result

| top_n | mean_annualized_return | min_annualized_return | positive_year_rate | worst_max_drawdown | total_periods | failed_gates |
|---:|---:|---:|---:|---:|---:|---|
| 200 | 0.010121 | -0.290374 | 0.400000 | -0.169396 | 63 | return_gate,weak_year_damage_gate |
| 100 | -0.087086 | -0.400523 | 0.200000 | -0.207391 | 64 | return_gate,weak_year_damage_gate |

## Interpretation

- The pruned signal wiring works end-to-end through the formal personal gate.
- Simple prior-fit factor pruning is not a sufficient alpha upgrade. It improves compactness and removes unstable factors, but it does not solve weak-year damage.
- `top_n=200` is materially better than `top_n=100`, matching the broader lesson that smaller personal baskets are not automatically stronger.
- No new `personal_backtest_candidate` should be created from this run. Existing paper-tracked candidates remain unchanged.

## Next Step

Use factor pruning as a component, not as a standalone model upgrade. The next model-strength work should test regime-conditioned pruning, capital-scaled pruned variants, or new Baostock-only alpha features against the same 2017-2026 formal personal gate.
