# Daily Research V2 High Return Normal-Sample Trial Plan

Date: `2026-06-06`

## Summary

- Status: `normal_sample_single_seed_objective_trial_launched / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Evidence grade target: `scout_only` after normal-sample forecast and quick bridge complete.

## Decision

- Datecap2 is no longer a model-direction gate.
- Datecap2 evidence is reclassified as `thin_sample_smoke_only`: useful for code path, metadata, and bridge sanity; insufficient to accept or reject objective quality.
- Normal-sample single-seed trials become the next gate for high-return objective/loss directions.
- Do not spend three seeds until normal samples show at least two positive-transfer split / pool-view settings.

## Trial Protocol

- Dataset key: `long_history_augmented`.
- Dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Pool view id: `policy_pool_view__eb690dd0becc330f029c21bd`.
- Model family: `gru_sequence_static_context`.
- Seed: `7`.
- Forecast sample cap:
  - `--forecast-max-samples-per-role 0`;
  - `--forecast-max-samples-per-date-per-role 0`.
- First split set: `a,b,long`.
- Execution shell for quick bridge and later matrix: `small_capital_balanced_return_v1`.
- First objective: `topn_excess_rank_v1`.
- First run tag: `v2_hrd_topn_normal_long_mainboard_wide_gru_seed7_scout_20260606_01`.
- First agent run id: `v2_hrd_topn_normal_long_mainboard_seed7_20260606_01`.
- Launch status: `launched`, PID `20652`, under `daily_research/output/agent_runs/v2_hrd_topn_normal_long_mainboard_seed7_20260606_01/`.

## Run Tags

- `v2_hrd_topn_normal_long_mainboard_wide_gru_seed7_scout_20260606_01`.
- `v2_hrd_topn_normal_long_mainboard_seed7_20260606_01`.

## Monitor

- Agent run status:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.agent_run status --project-id daily_research --run-id v2_hrd_topn_normal_long_mainboard_seed7_20260606_01 --json`
- Expected study report after completion:
  `daily_research/output/path_policy/studies/v2_hrd_topn_normal_long_mainboard_wide_gru_seed7_scout_20260606_01/v2_high_return_model_discovery_report.json`

## Gate

- If normal-sample `topn_excess_rank_v1` does not complete cleanly, fix the execution/data issue before trying other objectives.
- If only one split transfers, keep it as diagnostic and try the next objective normally; do not do deep root-cause analysis from one normal trial.
- If at least two of `a/b/long` show positive excess return and positive excess Sharpe, expand to full long-history splits and then same-period control.
- Only after at least two split / pool-view settings transfer should seeds `7,11,19` be considered.

## Boundaries

- `research_only=true`.
- `shadow_only=true`.
- `promotion_allowed=false`.
- `active_execution_strategy_expected_diff=none`.
- `paper_live_broker_allowed=false`.
- `active_artifact_impact=unchanged`.
- Do not tune execution shell to rescue one split.
- Do not treat datecap2 negative evidence as model rejection.
