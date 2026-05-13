# r60 Profile-Bound Core V4 Screening Status 2026-05-13

## Scope
- Status: research / shadow-only / profile-bound core-v4 readiness plus 1-trial safe screening.
- Goal: make direct protocol consume active r59 profile defaults and surface release-first diagnostics before interpreting behavior.
- Boundary: no live/default/promotion change, no confirmatory run, and no write to `daily_research/output/active_execution_strategy.json`.

## Implementation Facts
- Direct protocol now applies active search profile defaults from `research_profile_registry.get_search_profile_config(...)`.
- Precedence is explicit CLI value > profile base trial > parser default.
- Protocol summary now writes `profile_binding` with requested profile, applied status, explicit overrides, and effective base trial.
- Non-active legacy profile names are rejected for new direct protocol runs.
- Protocol evaluation and shadow summaries now enrich continuity metrics from `daily_turnover.csv` with release-first/source/cash/intent diagnostics.
- `behavior_bottleneck_report` now uses enriched continuity metrics and does not report missing intent translation diagnostics as clean.
- Core-v4 prediction now reuses `derive_release_first_intent(...)` for release score, action hint, intent delta, and block reason.

## Test Evidence
- Command:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_protocol_profile_binding.py daily_research/continuous_policy/tests/test_protocol_release_first_diagnostics.py daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py daily_research/continuous_policy/tests/test_core_v4_training_contract.py daily_research/continuous_policy/tests/test_core_v4_artifact_contract.py daily_research/continuous_policy/tests/test_core_v4_release_first_loss.py daily_research/continuous_policy/tests/test_research_registry_simplification.py daily_research/continuous_policy/tests/test_training_runtime_acceleration.py daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py -q`
- Result: `140 passed, 24 warnings`.
- Warnings are existing cvxpylayers / NumPy deprecation warnings in portfolio strategy contract tests.

## Runtime Evidence
- Direct smoke completed:
  - Tag: `protocol_r60_profile_bound_core_v4_smoke_20260513_01`.
  - `profile_binding.profile_applied=true`.
  - Effective backend/loss/profile semantics: `formal_torch_core_v4`, `alpha_result_value_budget_split_v46`, `allocation_layer_v1`, `end_to_end_allocation_layer_v1`, `result_value_v10`, `active_execution_strategy`.
  - Training evidence: `insufficient`, `best_epoch=1`, `completed_epochs=1`.
  - Evaluation continuity diagnostics were present: release-first mode used `1.0`, source intent count `0.0`, source target count `0.0`, target sum gap about `0.5792`, actual cash about `0.8528`, intent translation conflict about `0.8916`.
- Study dry run completed:
  - Tag: `self_opt_study_r60_profile_bound_core_v4_dryrun_20260513_01`.
  - Selected trial remained r59 core-v4/v46 with allocation-layer/result-value/active-prior semantics.
- Safe screening completed:
  - Tag: `self_opt_study_r60_profile_bound_core_v4_screening_safe_20260513_01`.
  - Completed trials: `1`; failed trials: `0`; confirmatory disabled.
  - Trial 01 protocol summary is parseable and profile-bound to r59 core-v4.
  - Training evidence became `sufficient`, with `best_epoch=6`, `completed_epochs=12`.
  - Composite score: `-36.684195`.
  - Evaluation: annual return `-0.044853`, max drawdown `-0.071089`, cash timing quality `0.026076`, reduce success `0.0`, exit timeliness `0.0`.
  - Release/source diagnostics: release-first mode used `1.0`, but release-first source intent count `0.0`, source target count `0.0`, release rotation `0.0`.
  - Closure diagnostics: target sum gap about `0.6148`, actual cash about `0.8468`, exposure utilization about `0.2334`, intent translation conflict about `0.8434`.
  - Resource gate triggered and stopped after the single requested screening trial. Failed checks: `release_first_source_intent_dead`, `receiver_deploy_not_clean`, `economic_signal_too_weak`, `exposure_utilization_low`, `actual_cash_idle_high`, `actual_cash_weight_high`, `cash_funded_deployment_failed`, `target_sum_underdeployed`.

## Guard Evidence
- `git diff -- daily_research/output/active_execution_strategy.json`: no output.

## Interpretation
- Fact: r60 fixed the profile-binding bug; direct protocol no longer silently runs r59 smoke with legacy budget/objective/prior semantics.
- Fact: r60 fixed diagnostic surfacing; release-first/source/cash/intent metrics are now visible in protocol summaries and bottleneck reports.
- Fact: r60 safe screening produced completed evidence, and that evidence is negative.
- Inference: current core-v4 release-first behavior is not connected enough for r61 resume or confirmatory. The model enters release-first mode but does not produce source intent/source targets, remains highly underdeployed, and carries large intent translation conflict.
- Rule: do not package this as "needs more epoch/loss." The next fix must target core-v4 release/source target generation and allocator-consumable target-delta wiring.

## Next Gate
- r61 should not start with confirmatory or long resume.
- Minimum next work: inspect why core-v4 target-delta/release intent yields zero `release_first_source_intent_count` and zero `portfolio_daily_source_target_count` under r59 profile-bound safe screening.
- Any future safe screening must keep explicit tags, profile binding diagnostics, enriched release-first continuity metrics, and active artifact guard.
