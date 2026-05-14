# r70 V5 Version Boundary And Oracle Repair Status

Date: `2026-05-14`

## Summary
- Status: `research / shadow-only / version-boundary and mechanism-repair evidence`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- Live default remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- r70 restores the base portfolio-set v5 internal version to `portfolio_set_v5_dfl_pg_v1` and keeps `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration` behind explicit loss/profile or artifact metadata.
- r70 repairs the r69 oracle constraint violation and allows the tiny strict-Gold behavior smoke to complete past the previous TDX empty-batch blocker.
- This is not promotion evidence, not confirmatory evidence, not live evidence, and not a strategy-success verdict.

## Implementation Facts
- Updated `daily_research/continuous_policy/model_portfolio_set_v5.py`.
  - `PORTFOLIO_SET_V5_INTERNAL_VERSION` is back to `portfolio_set_v5_dfl_pg_v1`.
  - Added explicit behavior-mode resolution: `dfl_pg_v1` vs `r69_value_arbitration`.
  - New artifacts write `portfolio_set_v5_internal_version`, `portfolio_set_v5_behavior_mode`, and `portfolio_set_v5_loss_profile_requested`.
  - Legacy `alpha_result_value_budget_split_v48` and `portfolio_set_release_first_decision_v1` aliases resolve to the base DFL-PG internal version while retaining the requested loss profile in metadata.
  - Base target/loss/predict paths do not consume r69 deploy/release/defense/reversal signals; compatibility columns remain present in target width with neutral defaults.
  - r69 value-arbitration inputs only activate when explicit r69 mode is requested or loaded from artifact metadata.
  - Oracle diagnostics now split `cash_floor_violation`, `turnover_violation`, `source_capacity_violation`, and `receiver_capacity_violation`.
  - Oracle budget order now ranks source/receiver masks first, then computes reachable cash floor from sparse source capacity.
- Updated `daily_research/continuous_policy/portfolio_simulator.py`.
  - Action rows now carry `portfolio_set_v5_value_arbitration_mode` so downstream trace/metrics can gate r69-only fields.
  - r68 `portfolio_cashflow_decision_v1` remains the simulator contract; cashflow mode still does not call release-first v3 to recompute v5 targets.
- Updated `daily_research/continuous_policy/release_flow_trace.py` and `daily_research/continuous_policy/pipeline_utils.py`.
  - r69 metrics are gated by `portfolio_set_v5_value_arbitration_mode`.
- Updated `daily_research/baseline/data_provider.py`.
  - `_fetch_tq_data` retries empty TDX batches, recursively splits empty multi-stock batches, and raises a singleton stock/date/batch error only when the provider still returns empty data for one symbol.

## Test Evidence
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests -q`
  - Result: `229 passed, 24 warnings`.
  - Warnings are existing cvxpylayers / numpy deprecation warnings.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/baseline/tests/test_data_provider_tq_fetch.py -q`
  - Result: `2 passed`.
- `git diff --check`
  - Result: clean.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
  - Result: passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
  - Result: `status=ok`, `error_count=0`, `warning_count=0`.
- `git diff -- daily_research/output/active_execution_strategy.json`
  - Result: empty.

## Smoke Evidence
- Protocol tag: `protocol_r70_v5_version_boundary_oracle_repair_smoke_20260514_01`.
- Dataset: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Backend/loss: `formal_torch_portfolio_set_v5` / `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration`.
- Profile: explicit non-active r69 profile, `active_profile=false`.
- Completed stages: train, evaluate, shadow, export, behavior audit, conclusion ledger.
- Promotion gate remains `shadow_only`.
- Training evidence remains `insufficient` because `teacher_action_rows=0` and `best_epoch=2` equals completed epochs.

## r70 Metrics
- Artifact metadata:
  - `portfolio_set_v5_internal_version=portfolio_set_v5_dfl_pg_v1_r69_value_arbitration`.
  - `portfolio_set_v5_behavior_mode=r69_value_arbitration`.
  - `portfolio_set_v5_loss_profile_requested=portfolio_set_v5_dfl_pg_v1_r69_value_arbitration`.
- Oracle / target:
  - `decision_target_source_count=1185.0`.
  - `decision_target_receiver_count=160.0`.
  - `decision_target_constraint_violation_mean=2.0787514869876915e-10`.
  - validation `decision_oracle_constraint_violation_mean=0.0`.
  - validation cash-floor / turnover / source-capacity / receiver-capacity violation means all `0.0`.
  - `value_arbitration_source_wrong_side_sell_share=0.0`.
- Evaluation continuity:
  - `cashflow_decision_contract_valid_rate=1.0`.
  - `intent_translation_conflict_rate=0.0`.
  - `portfolio_daily_source_target_count=3.8095238095238093`.
  - `portfolio_daily_receiver_target_count=168.0`.
  - `release_flow_primary_blocker=none`.
  - `portfolio_daily_source_positive_forward_sell_share=0.24`.
  - `cash_timing_quality_1d=-0.2656747822950293`.
  - `immediate_reversal_rate_3d=0.29435483870967744`.
- Shadow continuity:
  - `cashflow_decision_contract_valid_rate=1.0`.
  - source and receiver alignment rates both `1.0`.
  - `intent_translation_conflict_rate=0.0`.
  - `portfolio_daily_source_target_count=4.380952380952381`.
  - `portfolio_daily_receiver_target_count=133.0`.
  - `release_flow_primary_blocker=none`.
  - `portfolio_daily_source_positive_forward_sell_share=0.26785714285714285`.
  - `portfolio_daily_receiver_minus_source_forward_excess_5d=-0.0013810522147760084`.
  - `cash_timing_quality_1d=-0.26880048476847535`.
  - `immediate_reversal_rate_3d=0.3333333333333333`.

## Comparison To r69 Baseline
- r69 traincheck `protocol_r69_value_arbitration_v5_behavior_traincheck_20260514_03` had:
  - target constraint violation about `0.022252461536274436`.
  - validation oracle violation about `0.03294592542994407`.
  - receiver target count `160.0`.
  - source wrong-side sell share `0.0`.
- r70 improves oracle constraint violation by more than 80 percent, effectively to zero, while preserving source wrong-side sell share at `0.0`.
- r70 does not improve receiver target coverage over r69; receiver target count remains `160.0`, below r68 target count `2048.0`.
- r70 completes the full tiny smoke where r69 `_01/_02` stopped during evaluate because TDX returned empty data.

## Facts
- Base `portfolio_set_v5_dfl_pg_v1` is again the default internal version and default training mode.
- r69 value arbitration is isolated behind explicit profile/loss or artifact metadata.
- Old/missing metadata artifacts default to base mode prediction.
- r68 cashflow translation closure is not regressed in the r70 tiny smoke.
- TDX empty-batch handling is improved and the r70 smoke passed the previous evaluate blocker.
- Active execution artifact diff remains empty.

## Inferences
- The r69 high oracle-violation blocker was mainly an oracle budgeting/order issue, not an unavoidable value-arbitration objective property.
- The remaining r69/r70 blocker is behavior quality, not translation closure or oracle feasibility.
- Receiver coverage, cash timing, reversal, and source selection quality still need objective-level work before any strict resume or confirmatory discussion.

## Assumptions
- The strict Gold dataset remains the correct training-safe source for this line.
- The r70 smoke is tiny and research/shadow-only; it should not be treated as stable behavior evidence.
- DFL decision evidence is diagnostic only and does not relax promotion training-evidence gates.

## Boundaries
- Do not treat `protocol_r70_v5_version_boundary_oracle_repair_smoke_20260514_01` as promotion, confirmatory, live, or strategy-success evidence.
- Do not modify `daily_research/output/active_execution_strategy.json`.
- Do not make r69 value arbitration an active/default profile.
- Do not run strict resume solely because oracle feasibility is repaired; first address receiver coverage, cash timing, reversal, and sell/source quality.

## Next Allowed Actions
- Keep r68 as translation-closure baseline and r70 as version-boundary/oracle-repair evidence.
- Improve r69/r70 receiver coverage without reintroducing source wrong-side sells or oracle violations.
- Replace or supplement the r69 cash timing / defense objective so `cash_timing_quality_1d` improves in evaluation and shadow.
- Add source opportunity-cost and reversal diagnostics that reduce positive-forward sell share and immediate reversal without breaking cashflow validity.
- Only after a tiny behavior smoke improves at least two behavior metrics should a longer strict resume be considered.
