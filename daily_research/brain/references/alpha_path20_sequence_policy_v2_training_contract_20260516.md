# alpha_path20_sequence_policy_v2 Training Contract 2026-05-16

## Purpose
- Upgrade path20 sequence/RL from smoke training to an auditable annual episode loop.
- Keep `alpha_path20_sequence_policy_v1` as the policy version while using v2 training semantics.
- Do not modify live/default or `daily_research/output/active_execution_strategy.json`.
- Scope note after 2026-05-17: v2 is the episode training contract. v3 is the evidence matrix contract layered on top of v2.

## Dataset Contract
- `market_episode` artifacts contain historical market state, tradable mask, next-open returns, benchmark returns, and explicit reward columns.
- Model input features exclude reward/return columns and continue to reject `future_*`, `oracle_path20_*`, `path_q*`, and `path_mu_*`.
- `rollout_current_weight` is a dynamic training feature generated during rollout, not a fixed dataset label.
- Normalization statistics are fitted from train years only.
- Annual windows below 180 trading days are `incomplete` and cannot enter verdict aggregates.

## Training Contract
- Train years: `2019`, `2020`.
- Validation/model-selection year: `2022`.
- Final shadow test year: `2024`.
- Episode training uses a torch safety projection and computes loss from projected target weights, not raw target weights.
- The rollout updates current weights, previous rewards, equity, and drawdown proxy through the episode.
- Decision Transformer runs must receive previous weight and previous reward context from rollout, not all-zero defaults.

## Replay Contract
- Final evidence uses exact replay through `build_path_policy_frame` and `PortfolioState.step`.
- Replay summaries must report returns, turnover, position history, projection diagnostics, `oracle_used=false`, `shadow_only=true`, `promotion_allowed=false`, and `active_execution_strategy_expected_diff=none`.
- Train replay is diagnostic. Main verdicts must distinguish train, validation, and test metrics.
- v3 renames surrogate validation rollout fields to `validation_surrogate_metrics`; exact validation/test fields must come from replay summaries only.

## v3 Handoff
- v2 does not define the GRU vs Decision Transformer matrix verdict.
- v2 does not treat torch projection parity as sufficient evidence by itself.
- Use `alpha_path20_sequence_policy_v3_walkforward_matrix_contract_20260517.md` for evidence-grade matrix interpretation.

## Boundaries
- Oracle remains diagnostic upper-bound only.
- Legacy neural v1 stages remain diagnostic-only and require `--allow-legacy-neural-policy`.
- No loose `latest_*` or `default` lake ids are allowed as evidence.
