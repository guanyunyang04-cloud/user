# Path Policy Alpha v2 July Archived Runs Reconciliation

- Date: `2026-07-10`.
- Status: `reconciled / completed historical research / shadow_only / no_active_change`.
- Research program: `alpha_path20_neural_policy_v1` historical line.
- Run tags:
  - `qdp_v2_alpha_v2_no_symbol_h256_t4_b512_full_e12_w0_20260702_01`
  - `qdp_v2_alpha_v2_symbol_h256_t4_b512_full_e12_w0_20260702_01`
  - `qdp_v2_alpha_score_symbol_h256_t4_b512_w1_full_e12_20260703_01`
- Shared data pointer: `qdp_v2_active_20260626`.
- Shared boundary: all three runs are `shadow_only=true`, `promotion_allowed=false`, and declare `active_execution_strategy_expected_diff=none`.

## Evidence Summary

The no-symbol alpha-v2 run completed with validation/test 20d rank IC `0.0786 / 0.1278`; its decision-score rank IC was `0.0785 / 0.0953` and the decision utility profile passed.

The symbol alpha-v2 run completed with validation/test 20d rank IC `0.1138 / 0.1436`; its decision-score rank IC was `0.0880 / 0.0919` and the decision utility profile passed. Symbol context improved the validation 20d path ranking in this single-seed comparison, but did not establish multi-seed stability or execution value.

The later alpha-score symbol run completed with validation/test 20d rank IC `0.0785 / 0.1649` and upside-20d rank IC `0.1203 / 0.2040`. Its decision-utility profile was unavailable and the study verdict was only `forecast_promising`, so it is not a stronger promotion-grade conclusion than the two alpha-v2 comparison runs.

## Interpretation

These runs belong to the superseded Path20/alpha-v2 research lineage. They are preserved as historical feature/model comparisons, not as the current seq100 today-close path-value pointer. Single-seed forecast results do not authorize allocator, replay, execution-candidate, live/default, paper/broker, or promotion changes.

## Retention

- Keep the study summaries, compact metrics, selected checkpoint metadata, and this reconciliation reference.
- Large per-row predictions may be trimmed under research artifact GC.
- The runs do not protect unrelated shared packs or superseded QDP datasets from manifest-aware cleanup.

## Active Artifact Impact

`daily_research/output/active_execution_strategy.json` remains absent in the frozen execution skeleton and was not created or modified by these runs or this reconciliation.
