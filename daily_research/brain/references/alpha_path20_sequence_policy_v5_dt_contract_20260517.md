# alpha_path20_sequence_policy_v5 DT Contract 2026-05-17

## Summary
- Status: `research / shadow-only / path20 v5 paper-grade Decision Transformer contract`.
- v5 adds `decision_transformer_v2` as an architecture and evidence contract; it does not replace the v4 `sequence_gru` validation-first baseline.
- The v5 policy is pure utility training: no oracle imitation, no path-label imitation, and no future-feature input.
- `active_execution_strategy.json remains unchanged`; live/default promotion is forbidden.

## Fixed Split
- Train years: `2019`, `2020`.
- Validation/model selection year: `2022`.
- Final shadow test year: `2024`.
- Checkpoint selection may read only 2022 exact validation replay.
- 2024 test replay is interpretable only when validation first passes.

## Architecture Contract
- Main model family: `decision_transformer_v2`.
- Model class: `PortfolioDecisionTransformerPolicy`.
- The architecture factorizes sequence and cross-section structure:
  - per-stock causal temporal Transformer over market state, rolling previous weights, rolling previous rewards, and portfolio context;
  - cross-sectional Transformer over the latest per-stock temporal latents;
  - learned time positional embeddings;
  - learned stock-slot embeddings;
  - shared context embedding.
- Output contract remains `raw_target_weight`, `cash_logit`, `score_logits`, and `policy_aux`.
- Default v5 study settings are hidden dim `64`, temporal layers `2`, cross layers `1`, heads `4`, dropout `0.1`, AdamW lr `3e-4`, weight decay `1e-4`, and grad clip `1.0`.

## Training And Evidence Contract
- Stage: `rl-v5-dt-validation-study`.
- Default context grid: `20,60`; `60` is the primary v5 evidence context when present.
- `20` is a required short-context control used to answer whether the short window is limiting Transformer value.
- Training loss is computed from projected weights and includes next-open excess utility, costs, turnover, concentration, projection distance, tail mass, entropy, and drawdown-style penalties.
- Exact evidence still comes from deterministic projection plus `PortfolioState.step` replay; surrogate training metrics are diagnostic only.
- The summary must include `architecture_contract`, `context_length_comparison`, `checkpoint_exact_validation_replay`, `selected_checkpoint`, `baseline_comparison`, `multi_seed_summary`, `temporal_context_diagnostics`, `validation_passed`, `test_interpretable`, `evidence_verdict`, `oracle_used=false`, `shadow_only=true`, and `promotion_allowed=false`.

## Baselines
- v5 inherits the v4 baseline suite:
  - `cash_no_trade`
  - `liquidity_equal_top30`
  - `score_blend_top30`
  - `alpha_prior_target_weight`
  - `v3_final_checkpoint`
- v5 adds `v4_gru_selected_checkpoint` as an optional explicit baseline.
- Checkpoint baselines require explicit model paths and are skipped when the path is absent; they must not be faked.
- A v5 run can be `validation_promising` only if 2022 exact validation return and sharpe are positive and validation return beats the baseline suite.

## Verdict Rules
- `insufficient_or_incomplete`: data, primary context, artifact, or projection parity is broken.
- `contract_passed`: v5 architecture and evidence loop ran, but validation did not prove usefulness.
- `diagnosed_failure`: checkpoint selection completed but validation or baseline comparison failed.
- `validation_promising`: exact 2022 validation replay is positive, sharpe is positive, and baselines are beaten.
- `test_promising`: only allowed after validation is already promising; 2024 alone cannot create success evidence.
- Every verdict remains `promotion_allowed=false`.

## Boundaries
- Oracle is diagnostic upper-bound only and is not a teacher for v5.
- Sequence/RL inputs must reject oracle, future, `path_q*`, and `path_mu_*` label fields.
- Returns are reward targets only and must not enter model input features.
- Do not create loose `latest_*` evidence aliases.
- Do not run expensive full-year v5 studies unless explicitly approved for that run.
- Do not modify live/default or `daily_research/output/active_execution_strategy.json`.

## Next Allowed Actions
- Run fixture and small smoke tests for `rl-v5-dt-validation-study`.
- Use the default `20,60` context grid for approved full-year v5 evidence runs.
- Compare v5 only through exact 2022 validation replay and the shared baseline suite.
- Keep v4 GRU as the stable validation-first baseline until v5 produces stronger validation evidence.

## Blockers
- No full-year v5 evidence run has been approved or completed in this contract update.
- v5 is an architecture/evidence upgrade, not alpha success evidence.
- Prior v3/v4 evidence means 2024 test strength cannot be interpreted without 2022 validation success.

## Source
- Fixed near-term lake source: `policy_input_bundle__0f116a9b78c92ff045a6853d`.
