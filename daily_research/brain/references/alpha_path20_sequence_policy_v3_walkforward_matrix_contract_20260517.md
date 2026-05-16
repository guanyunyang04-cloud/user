# alpha_path20_sequence_policy_v3 Walk-Forward Matrix Contract 2026-05-17

## Purpose
- Promote the v2 episode loop to evidence-grade walk-forward interpretation.
- Keep `alpha_path20_sequence_policy_v1` as the policy version; v3 is the evidence and training/replay consistency contract.
- Produce shadow-only research evidence only. No live/default mutation and no `daily_research/output/active_execution_strategy.json` write are allowed.

## Fixed Split
- Train: `2019`, `2020`.
- Validation/model selection: `2022`.
- Final shadow test: `2024`.
- Train-year replay is diagnostic only.
- 2024 test may be interpreted only after complete 2022 validation evidence.

## Artifact Contract
- `market_episode` writes CSV plus tensor-friendly `.npz` arrays.
- Dataset manifest, array manifest, normalization manifest, model manifest, and replay manifest include explicit tag/year/role/source lake id where applicable.
- Episode artifacts are reused only when the stable manifest hash matches. A hash mismatch must fail loudly and must not silently overwrite evidence.
- `normalization_scope=train_years_only`; validation/test data must not alter train mean/std.
- No `latest_*` or `default` evidence aliases are created.

## Training And Replay Contract
- Training may use `rollout_grad_mode=detached` by default or `truncated` with bounded chunk BPTT.
- Training loss uses torch projected weights as a surrogate, but final evidence comes from exact `build_path_policy_frame` plus `PortfolioState.step` replay.
- `validation_surrogate_metrics` is not exact replay evidence.
- Exact replay metrics must be reported separately as `validation_exact_replay_metrics` and `test_exact_replay_metrics`.
- Summaries must include surrogate-vs-exact gap, projection diagnostics, context coverage, `oracle_used=false`, `shadow_only=true`, and `promotion_allowed=false`.

## Projection Parity
- Torch projection and pandas exact projection are compared on fixture and sampled episode states.
- `projection_parity_max_l1` defaults to `0.02`.
- If parity exceeds the threshold, summaries must mark `projection_mismatch_warning=true`.
- Projection mismatch downgrades verdict to `insufficient_or_incomplete`; it cannot be written as a success conclusion.

## Matrix Verdict
- `rl-walkforward-matrix` runs fixed-family comparison for `sequence_gru` and `decision_transformer`.
- Verdict levels:
  - `contract_passed`
  - `validation_promising`
  - `test_promising`
  - `insufficient_or_incomplete`
- Validation completeness is required before any test success interpretation.
- All matrix verdicts remain `promotion_allowed=false`.

## Boundaries
- Oracle, future labels, `path_q*`, and `path_mu_*` are forbidden as sequence/RL inputs.
- Oracle remains diagnostic upper-bound only, not a teacher.
- Full-year matrix execution is expensive and requires a separate run approval; v3 first milestone is implementing the contract and fixture smoke coverage.
