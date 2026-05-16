# alpha_path20_sequence_policy_v1 Status 2026-05-16

## Verdict
- Status: `main path20 research line / research / shadow-only / sequence-policy RL loop`.
- `alpha_path20_sequence_policy_v1` is the only active path20 main research route after the 2026-05-16 route consolidation.
- The first milestone is not promotion. The first milestone is a reproducible annual dataset -> train -> replay -> report loop for the fixed full-year windows.
- It does not replace live/default execution and must not modify `daily_research/output/active_execution_strategy.json`.
- Oracle is diagnostic upper-bound evidence only. It is not the default teacher for this line.

## Facts
- Added `alpha_path20_sequence_policy_v1` alongside the existing `alpha_path20_neural_policy_v1`.
- Added trajectory dataset construction under `daily_research/path_policy/rl_dataset.py`.
- Added sequence-policy model primitives under `daily_research/path_policy/rl_models.py`.
- Added sequence-policy replay through the existing target-weight adapter and `PortfolioState.step`.
- Added protocol stages:
  - `rl-dataset-smoke`
  - `rl-train-smoke`
  - `rl-replay-smoke`
  - `rl-multiyear-smoke`
- These `rl-*` stages are now the default path20 protocol route.
- Legacy `dataset-smoke`, `oracle-smoke`, and `tiny-smoke` are gated behind `--allow-legacy-neural-policy` and are not mainline evidence.
- Fixed full-year windows:
  - `2019`: `20190101 -> 20191231`
  - `2020`: `20200101 -> 20201231`
  - `2022`: `20220101 -> 20221231`
  - `2024`: `20240101 -> 20241231`
- Annual trajectory manifests record requested range, actual signal range, trading-day count, feature columns, portfolio feature columns, reward profile, leakage guard, dataset id, and fixed source lake id.
- Annual windows below 180 trading days must be marked `incomplete` and excluded from aggregate verdict metrics.

## Boundaries
- No oracle or future path labels may be used as sequence-policy inputs.
- Forbidden input prefixes include `future_`, `oracle_path20_`, `path_q`, and `path_mu_`.
- `portfolio_daily_target_weight` remains the simulator execution truth after deterministic safety projection.
- Source/receiver fields are derived diagnostics only.
- All evidence is `research / shadow-only`; no promotion is allowed from v1 sequence smokes.
- No loose latest references: use explicit study tag and fixed lake dataset id.
- Combined multiyear summaries must report completed years, incomplete years with reasons, aggregate metrics, `oracle_used=false`, `shadow_only=true`, `promotion_allowed=false`, and `active_execution_strategy_expected_diff=none`.

## Default Evidence Source
- Near-term lake source: `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- Default reward profile: `net_excess_turnover_drawdown_v1`.
- Default sequence policy version/profile: `alpha_path20_sequence_policy_v1`.

## Recommended Verification
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
- `git diff -- daily_research/output/active_execution_strategy.json`

## Next Allowed Actions
- Run `rl-dataset-smoke` on a small fixed lake slice.
- Run `rl-train-smoke` and `rl-replay-smoke` only as shadow research plumbing.
- Run `rl-multiyear-smoke` over 2019, 2020, 2022, and 2024 after the small smoke passes. Treat low-coverage years as incomplete rather than forcing them into the aggregate.
- Compare sequence GRU and decision-transformer variants on projected replay metrics, projection distance, turnover, drawdown, and net return.
