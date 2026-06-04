# Daily Research V2 High Return Augmented Date-Capped Scout

Date: `2026-06-05`

## Summary

- Status: `high_return_model_discovery / augmented_same_period_datecap2_scout_completed / quick_bridge_completed / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Evidence grade: `scout_only`, single seed. This is not completed model-quality evidence and not promotion evidence.

## Implementation

- Added deterministic date-wise forecast sampling via `--forecast-max-samples-per-date-per-role`.
- Preserved the old total role cap behavior: `--forecast-max-samples-per-role 0` keeps total role sampling uncapped.
- Added date-wise stock rotation so a per-date cap does not always select the first symbols.
- Updated the high-return discovery CLI so forecast tasks record the per-date cap in task lists and summaries.
- Updated `run_forecast_tasks()` to write the matching task list before launching forecast tasks; this makes reboot-safe split execution compatible with later `--run-quick-bridge` and `--run-candidate-matrix`.

## Scout Evidence

- Discovery run tag: `v2_high_return_model_discovery_augmented_same_gru_datecap2_20260605_01`.
- Forecast run tag: `mh_v2_hrd_same_same_gru_seed7_20260604_01`.
- Bridge run tag: `v2_hr_bridge_mh_v2_hrd_same_same_gru_seed7_20260604_01`.
- Dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Pool view id: `policy_pool_view__d7a56d5164470b590e4f5a40`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`.
- Model family: `gru_sequence_static_context`.
- Seed: `7`.
- Split: train `2019-2022`, validation `2023`, test `2024`.
- Forecast caps:
  - `--forecast-max-samples-per-role 0`;
  - `--forecast-max-samples-per-date-per-role 2`.
- Forecast status: `completed`.
- Forecast evidence verdict: `forecast_failed`.
- Feature store shape: `1699 x 2693 x 190`.
- Feature nan ratio: `0.056453552426564114`.
- Sample coverage:
  - train: `1866` rows, `933` dates, `383` symbols, exactly `2` rows per date;
  - validation: `422` rows, `211` dates, `82` symbols, exactly `2` rows per date;
  - test: `422` rows, `211` dates, `77` symbols, exactly `2` rows per date.

## Forecast And Transfer Result

- Test decision rank IC: `0.07109004739336489`.
- Test top-bottom spread: `-0.0000828381949237699`.
- Test hit lift: `0.026066350710900472`.
- Quick bridge status: `completed`.
- Shared backtest status: `completed`.
- Annual return: `0.0513594286490735`.
- Excess annual return: `-0.1372823337272422`.
- Excess Sharpe: `-0.5999025289136765`.
- Max drawdown: `-0.05181477725748762`.
- Annual return cost drag: `0.031811790866875134`.
- Positive month ratio: `0.5454545454545454`.
- Negative month count: `5`.
- Worst monthly excess return: `-0.19357990489750188`.
- Monthly issue flags:
  - `low_positive_month_ratio`;
  - `deep_bad_month`;
  - `concentrated_positive_months`;
  - `multi_month_drawdown_streak`.

## Verdict

- The low-budget augmented scout path now has enough date coverage for bridge diagnostics.
- The task-list continuity blocker from split forecast/bridge execution is repaired.
- This specific same-period augmented GRU single-seed scout is weak: forecast gate failed and bridge transfer is negative versus benchmark.
- The result does not justify three-seed confirmation, candidate matrix expansion, active artifact changes, paper/live execution, or production rebuild.
- Interpretation: `datewise_sampling_bridge_path_repaired_but_same_period_gru_datecap2_candidate_weak`.

## Next Allowed Actions

- Continue high-return discovery with single-seed, multi-split scouts before spending three seeds:
  - run long-history augmented splits with the same per-date cap protocol;
  - compare same-period and long-history split consistency;
  - only promote a candidate to seeds `7,11,19` if at least two splits show useful forecast-to-portfolio transfer.
- Consider raising the per-date cap after resource checks if daily cross-section breadth becomes the suspected bottleneck.
- Prefer architecture/data scouting before local execution repair for this weak candidate; do not run production execution rebuild.

## Verification

- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_forecast_memmap_dataset.py::test_forecast_memmap_dataset_can_cap_samples_per_date daily_research/path_policy/tests/test_forecast_memmap_dataset.py::test_forecast_memmap_dataset_writes_progress_files daily_research/path_policy/tests/test_v2_high_return_model_discovery.py daily_research/path_policy/tests/test_rl_protocol.py::test_forecast_walkforward_study_memmap_contract_fixture -q`: `14 passed`.
- `git diff -- daily_research/output/active_execution_strategy.json`: clean.
