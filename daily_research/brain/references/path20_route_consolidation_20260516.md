# path20 Route Consolidation 2026-05-16

## Decision
- `alpha_path20_sequence_policy_v1` is the path20 main research line.
- `alpha_path20_neural_policy_v1` is legacy diagnostic-only and has `no-new-mainline-budget`.
- The near-term objective is a reproducible full-year sequence/RL loop, not live/default promotion.

## Facts
- The sequence/RL implementation writes explicit trajectory manifests and replay summaries under `daily_research/output/path_policy`.
- Fixed full-year request windows are:
  - `2019`: `20190101 -> 20191231`
  - `2020`: `20200101 -> 20201231`
  - `2022`: `20220101 -> 20221231`
  - `2024`: `20240101 -> 20241231`
- Sequence/RL dataset inputs forbid `future_*`, `oracle_path20_*`, `path_q*`, and `path_mu_*` columns.
- The deterministic projection and `PortfolioState.step` remain the replay safety layer.
- `portfolio_daily_target_weight` is the execution truth after projection.
- Source/receiver fields are derived diagnostics only.
- `daily_research/output/active_execution_strategy.json` is not part of this research loop and must remain unchanged.

## Inferences
- The neural v1 route is valuable for oracle upper-bound, path-label schema, and forecaster baseline diagnostics, but it is not aligned with the current pure sequence/RL objective.
- The sequence/RL route is more valuable for the project because it directly optimizes target weights from historical market and portfolio state, avoids oracle imitation as the default teacher, and produces replay evidence in the same units the simulator executes.
- Annual completeness is part of evidence quality. A year with fewer than 180 trading days is an incomplete artifact, not a failed strategy and not aggregate evidence.

## Assumptions
- `policy_input_bundle__0f116a9b78c92ff045a6853d` remains the fixed lake source for near-term full-year smokes.
- First-pass annual runs use fixed smoke-scale parameters for auditability before broad hyperparameter search.
- Deterministic safety projection is compatible with the phrase pure neural because the neural policy owns the decision intent and the projection only enforces trading constraints.

## Boundaries
- No live/default promotion.
- No writes to `daily_research/output/active_execution_strategy.json`.
- No loose `latest_*` or `default` dataset ids as evidence.
- No deletion of old neural v1 evidence.
- No oracle/future/path-label inputs in sequence/RL training or replay.
- Legacy stages `dataset-smoke`, `oracle-smoke`, and `tiny-smoke` require `--allow-legacy-neural-policy`.

## Evidence Contract
- Mainline CLI stages:
  - `rl-dataset-smoke`
  - `rl-train-smoke`
  - `rl-replay-smoke`
  - `rl-multiyear-smoke`
- Required annual manifest fields include policy version, dataset id, source lake id, year, requested date range, actual signal range, trading-day count, feature columns, portfolio feature columns, reward profile, leakage guard, and `loose_latest_allowed=false`.
- Required replay summary fields include yearly metrics, projection diagnostics path, returns path, turnover path, position history path, `oracle_used=false`, `shadow_only=true`, `promotion_allowed=false`, and `active_execution_strategy_expected_diff=none`.

