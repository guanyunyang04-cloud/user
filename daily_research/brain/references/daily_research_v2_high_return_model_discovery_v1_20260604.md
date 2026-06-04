# Daily Research V2 High Return Model Discovery V1

Date: `2026-06-04`

## Summary

- Status: `high_return_model_discovery_v1 / orchestration_implemented / smoke32_pipeline_ok / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- This is not model-quality evidence. It implements the high-return scout orchestration and validates forecast -> quick bridge -> report on a 32-symbol smoke substrate only.

## Implementation

- Added research-only CLI: `daily_research.path_policy.v2_high_return_model_discovery`.
- Supported actions:
  - `--write-task-list`;
  - `--run-forecast`;
  - `--run-quick-bridge`;
  - `--run-candidate-matrix`;
  - `--collect-report`.
- Added early-stage discovery semantics:
  - single-seed runs are marked `scout_only`;
  - 32-symbol runs are marked `smoke_only`;
  - smoke-only rows may appear in the leaderboard but are excluded from formal shortlist and completed model-quality evidence;
  - three-seed confirmation remains required before finalist model-quality evidence.
- Added high-return report outputs:
  - `high_return_score`;
  - `return_potential_score`;
  - `forecast_strength_score`;
  - `transfer_score`;
  - `instability_penalty`;
  - `repairability_score`;
  - leaderboard, split consistency, bad-month preview, and shortlist.
- Added a prebuilt smoke32 augmented dataset mode:
  - dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`;
  - pool id: `policy_pool_view__04f11f51d7c0c7ae32038147`;
  - feature profile: `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`;
  - memmap manifest: `daily_research/output/path_policy/data_expansion/traditional_baostock_v2_1_augmented_memmap_smoke32_20260604_01/forecast_dataset_manifest.json`.

## Smoke Evidence

- Discovery run tag: `v2_high_return_model_discovery_smoke32_20260604_01`.
- Forecast run tag: `mh_v2_hrd_smoke32_same_gru_seed7_20260604_01`.
- Quick bridge run tag: `v2_hr_bridge_mh_v2_hrd_smoke32_same_gru_seed7_20260604_01`.
- Report JSON: `daily_research/output/path_policy/studies/v2_high_return_model_discovery_smoke32_20260604_01/v2_high_return_model_discovery_report.json`.
- Report Markdown: `daily_research/output/path_policy/studies/v2_high_return_model_discovery_smoke32_20260604_01/v2_high_return_model_discovery_report.md`.
- Forecast status: `completed`.
- Forecast evidence verdict: `forecast_failed`.
- Bridge status: `completed`.
- Completed forecast count: `1`.
- Completed backtest count: `1`.
- Smoke completed forecast count: `1`.
- Shortlist count: `0`.
- Smoke leaderboard row:
  - dataset: `smoke32_augmented`;
  - model: `gru_sequence_static_context`;
  - seed: `7`;
  - rank IC: `-0.469819`;
  - top-bottom spread: `-0.066454`;
  - hit lift: `-0.236111`;
  - excess annual return: `0.423159`;
  - excess Sharpe: `26.939966`;
  - high return score: `14.969538`;
  - evidence grade: `smoke_only`.

## Interpretation

- The smoke32 run proves the orchestration path can reuse the augmented memmap, train a forecast model, run quick score-backtest, and collect a report.
- The smoke32 run does not prove model strength. The forecast metrics are negative, and the bridge backtest covers only a 32-symbol, 3-date test panel, so the very high transfer score is a pipeline diagnostic artifact rather than durable alpha evidence.
- No candidate is promoted to shortlist. No completed model-quality evidence is registered from this run.

## Verification

- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_v2_high_return_model_discovery.py daily_research/path_policy/tests/test_v2_score_backtest_bridge.py daily_research/path_policy/tests/test_v2_candidate_review_matrix.py daily_research/path_policy/tests/test_v2_bad_month_attribution.py -q`: `21 passed`.
- `git diff -- daily_research/output/active_execution_strategy.json`: clean.
- `git diff --check`: clean.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json` before writeback reported the new smoke run as unregistered, as expected.

## Verdict

- `v2_high_return_model_discovery` is ready as a research-only orchestration layer for efficient high-return model scouting.
- The current smoke evidence is pipeline evidence only, not alpha/model evidence.
- Current blocker is `full_augmented_memmap_throughput_and_formal_single_seed_multi_split_not_completed`.

## Next Allowed Actions

- Build or reuse full same-period and long-history augmented memmaps with resource-aware staging; avoid rerunning a blind full same-period build that already showed CPU-bound throughput pain.
- Run Stage 1 as single-seed, multi-split scout on full augmented data, not on the smoke32 pool.
- Use quick bridge for early high-return ranking, then run candidate matrix only for shortlist rows.
- Reserve seeds `7,11,19` for finalist confirmation after at least two time splits show positive transfer.
- Keep all results research-only until a finalist passes forecast, transfer, bad-month, and multi-seed confirmation.
- Do not modify active execution strategy, production root, paper/live, broker, or promotion artifacts.
