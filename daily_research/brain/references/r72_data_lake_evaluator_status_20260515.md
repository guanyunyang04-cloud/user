# r72 Data Lake Evaluator Status

Date: `2026-05-15`

## Summary
- Status: `research / shadow-only / infrastructure progress`.
- Scope completed: `evaluate_policy.py`, protocol shadow, export, behavior audit, and related helper CLIs can use `data_source=lake`.
- Default market lake dataset: `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- Training remains strict Gold by default for portfolio-set v5: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- This is not promotion evidence, not confirmatory evidence, and not live evidence.

## Implementation Facts
- `prepare_policy_inputs(..., data_source="lake")` now routes through `load_policy_inputs_from_lake(...)` and returns the standard `PreparedPolicyInputs` contract.
- Lake mode does not call TDX universe resolution; it derives symbols from the audited lake market bundle, with optional local pool-file filtering, extras, and `max_universe_size`.
- Lake coverage preflight reports `lake_coverage_blocker` for missing market fields, benchmark, membership, or insufficient trading days.
- Evaluate/shadow keep the multi-day rollout requirement; export can request `lake_min_trading_days=1` for a single signal date.
- `evaluate_policy.py`, `run_continuous_policy_protocol.py`, `export_action_panel.py`, `train_policy.py`, `run_execution_counterfactuals.py`, `check_budget_objective_contract.py`, `build_research_database.py`, and `build_gold_training_dataset.py` accept or propagate lake arguments.
- Behavior audit preserves `lake_dataset_id` and `data_lake_root` from evaluation summaries.

## Evidence
- Real lake preflight for `2019-04-01 -> 2019-04-30`:
  - `data_source=lake`;
  - `universe_size=3070`;
  - `date_count=21`;
  - `benchmark_rows=21`;
  - `membership_true_rows=21`;
  - `lake_coverage_status=ok`.
- Failed diagnostic smoke:
  - `protocol_r71_multistage_regret_v5_behavior_lake_smoke_20260515_01`;
  - train/evaluate/shadow completed through lake;
  - export failed because the first r72 coverage rule incorrectly required two trading days for single-day export;
  - diagnostic only.
- Completed infrastructure smoke:
  - `protocol_r71_multistage_regret_v5_behavior_lake_smoke_20260515_02`;
  - train/evaluate/shadow/export/behavior-audit/conclusion-ledger completed;
  - no TDX singleton empty blocker occurred;
  - `data_source=lake`;
  - `lake_dataset_id=policy_input_bundle__0f116a9b78c92ff045a6853d`;
  - shadow `cashflow_decision_contract_valid_rate=1.0`;
  - shadow `intent_translation_conflict_rate=0.0`.

## Behavior Result
- r72 unblocks reproducible lake evaluation, but it does not prove r71 behavior acceptance.
- In completed smoke `_02`, training evidence remains `insufficient`:
  - `teacher_action_rows=0`;
  - `best_epoch=2`;
  - `completed_epochs=2`.
- In completed smoke `_02`, shadow source/receiver behavior is still not accepted:
  - `portfolio_daily_source_target_count=0.0`;
  - `portfolio_daily_receiver_target_count=0.0`;
  - `cash_timing_quality_1d=-0.0`;
  - `immediate_reversal_rate_3d=0.0`.
- Promotion gate remains `shadow_only`.

## Tests
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests -q`
  - Result before final summary-field patch: `243 passed, 24 warnings`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests -q`
  - Result before export min-day patch: `14 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests/test_research_data_lake.py daily_research/continuous_policy/tests/test_lake_evaluator_cli_contract.py -q`
  - Result after export min-day patch: `13 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_lake_evaluator_cli_contract.py daily_research/continuous_policy/tests/test_protocol_profile_binding.py daily_research/continuous_policy/tests/test_behavior_audit_prepare_contract.py daily_research/data_lake/tests/test_research_data_lake.py -q`
  - Result after final protocol summary-field patch: `22 passed`.
- `git diff --check`
  - Result: clean.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
  - Result before writeback: passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
  - Result before writeback: `status=ok`.
- `git diff -- daily_research/output/active_execution_strategy.json`
  - Result: empty.

## Facts
- r72 solves the TDX singleton evaluate/shadow/export blocker by adding a generic lake evaluator path.
- r72 does not change model targets, simulator cashflow semantics, promotion gates, or live/default execution.
- r71 now has one completed lake-backed tiny protocol, but its behavior acceptance failed because source/receiver targets remained zero in eval/shadow.

## Inferences
- The current r71 blocker has moved from provider availability to model/oracle/head behavior quality.
- The next r71 work should focus on restoring nonzero source/receiver targets under lake evaluation before strict resume.
- A strict resume is not justified by `_02` because the first-stage behavior gate did not pass.

## Boundaries
- Do not treat r72 as promotion, confirmatory, or live evidence.
- Do not treat `protocol_r71_multistage_regret_v5_behavior_lake_smoke_20260515_02` as behavior acceptance.
- Do not change `daily_research/output/active_execution_strategy.json`.

## Next Allowed Actions
- Use `--data-source lake --lake-dataset-id policy_input_bundle__0f116a9b78c92ff045a6853d` for future r71 evaluate/shadow/export smoke runs.
- Repair r71 prediction/oracle behavior until source and receiver target counts are nonzero under the lake smoke.
- Only consider strict resume after translation remains closed and at least two behavior metrics improve versus r70 without source/receiver collapse.
