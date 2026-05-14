# r68 Cashflow Decision V1 Close Portfolio-Set V5 Status

Date: `2026-05-14`

## Summary
- Status: `research / shadow-only / source-translation closure evidence`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- Live default remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- r68 adds a shared `portfolio_cashflow_decision_v1` contract so v5 prediction, simulator, release trace, and continuity metrics use the same source / receiver / cash semantics.
- First-stage behavior evidence is positive for translation closure: synthetic tests and tiny strict-Gold shadow smoke now show nonzero source targets, nonzero receiver targets, valid cashflow contracts, and `intent_translation_conflict_rate=0.0`.
- r68 is not promotion evidence, not confirmatory evidence, and not a strategy-success verdict.

## Implementation Facts
- New file: `daily_research/continuous_policy/portfolio_cashflow_decision.py`.
- New tests: `daily_research/continuous_policy/tests/test_portfolio_cashflow_decision.py`.
- Updated files:
  - `daily_research/continuous_policy/model_portfolio_set_v5.py`
  - `daily_research/continuous_policy/portfolio_simulator.py`
  - `daily_research/continuous_policy/release_flow_trace.py`
  - `daily_research/continuous_policy/pipeline_utils.py`
  - `daily_research/continuous_policy/tests/test_portfolio_set_v5_artifact_contract.py`
  - `daily_research/continuous_policy/tests/test_portfolio_set_v5_release_first_loss.py`
- The shared contract standardizes:
  - `current_weight`
  - `portfolio_daily_target_weight_intent`
  - `portfolio_daily_target_delta_intent`
  - `portfolio_set_v5_source_supply`
  - `portfolio_set_v5_receiver_demand`
  - `portfolio_set_v5_cash_buffer_score`
  - source / receiver intent masks
  - source / receiver executable candidates
  - coherent `action_label`
  - explicit invalid reasons and fail-closed diagnostics
- `predict_policy_portfolio_set_v5(...)` now emits `portfolio_cashflow_decision_v1_mode=1.0`, v5 turnover-budget diagnostics, source/receiver/cashflow fields, and normalized target fields from the oracle-compatible output.
- `portfolio_simulator.py` now gives cashflow mode priority over release-first v3 re-solving, uses explicit v5 target weights when the contract is valid, and fail-closes on invalid cashflow contracts.
- Simulator action rows now include cashflow contract fields so `pipeline_utils` continuity metrics can measure actual model intent and realized targets.
- The v5 torch oracle now uses sparse source/receiver top-k allocation and a conservative cashflow turnover budget so predict-time targets stay within simulator budget semantics.

## Diagnostic Runs
- `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_01`:
  - Completed all stages, diagnostic only.
  - Translation conflict was fixed, but source/receiver targets remained zero.
  - Main finding: cashflow mode was active and valid, but v5 prediction/oracle produced no nonzero cashflow deltas.
- `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_02`:
  - Completed all stages, diagnostic only.
  - The oracle produced cashflow targets, but simulator cashflow normalization fail-closed with `turnover_violation`.
  - Main finding: predict-time oracle budget and simulator calibrated turnover budget were not aligned.
- `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_03`:
  - Completed train, evaluate, shadow, export, behavior-audit, and conclusion-ledger.
  - This is the r68 closure evidence tag.

## Protocol Evidence
- Protocol tag: `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_03`.
- Protocol summary: `daily_research/output/continuous_policy/protocols/protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_03/protocol_summary.json`.
- Dataset: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Backend: `formal_torch_portfolio_set_v5`.
- Loss profile: `portfolio_set_v5_dfl_pg_v1`.
- Internal version: `portfolio_set_v5_dfl_pg_v1`.
- Promotion gate: `shadow_only`.
- Failed promotion checks include `contract_promotable`, `training_evidence_sufficient`, `open_win_rate_5d`, `exit_timeliness_rate_5d`, `cash_timing_quality_1d`, `max_drawdown`, `shadow_reversal`, `annual_return_vs_active`, and `sharpe_vs_active`.

## Key Metrics From `_03`
- Training / oracle:
  - `decision_target_source_count=793.0`.
  - `decision_target_receiver_count=2048.0`.
  - `target_intent_translation_conflict_count=0.0`.
  - `release_flow_source_target_count=6.0`.
  - `release_flow_receiver_target_count=8.0`.
  - `decision_oracle_constraint_violation_mean=0.0` in validation diagnostics.
- Shadow daily turnover:
  - `portfolio_cashflow_decision_v1_mode_used` sum: `21.0`.
  - `cashflow_decision_used` sum: `21.0`.
  - `cashflow_decision_failed_closed` sum: `0.0`.
  - `cashflow_decision_valid` sum: `21.0`.
  - `cashflow_decision_invalid_reason`: `none` on all 21 days.
  - `cashflow_decision_source_intent_count` sum: `97.0`.
  - `cashflow_decision_receiver_intent_count` sum: `139.0`.
  - `portfolio_daily_source_target_count` sum: `97.0`.
  - `portfolio_daily_receiver_target_count` sum: `139.0`.
  - `portfolio_daily_source_realized_reduction_weight` sum: `0.53308624660098`.
  - `portfolio_daily_receiver_realized_deploy_count` sum: `139.0`.
  - `intent_translation_conflict_rate` mean: `0.0`.
  - `release_flow_primary_blocker`: `none` on all 21 days.
  - `cashflow_decision_cash_conservation_gap` max: `2.7939677238464355e-09`.
  - `realized_turnover` sum: `1.6377625034885952`.
- Shadow action outcomes:
  - Action counts: `hold=1122`, `reduce=97`, `open=80`, `add=59`.
  - `portfolio_daily_source_target_intent` sum: `97.0`.
  - `portfolio_daily_receiver_target_intent` sum: `139.0`.
  - `portfolio_set_v5_source_supply` sum: `0.5330862535629375`.
  - `portfolio_set_v5_receiver_demand` sum: `1.1046762592159143`.

## Test Evidence
- Focused cashflow/v5/release-flow tests:
  - Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_set_v5_release_first_loss.py daily_research/continuous_policy/tests/test_portfolio_set_v5_artifact_contract.py daily_research/continuous_policy/tests/test_portfolio_cashflow_decision.py daily_research/continuous_policy/tests/test_release_flow_trace.py -q`
  - Result: `17 passed`.
- Full continuous-policy suite:
  - Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests -q`
  - Result: `220 passed, 24 warnings`.
  - Warnings are existing cvxpylayers / numpy deprecation warnings, not r68 failures.
- Guards:
  - `git diff --check`: passed.
  - `git diff -- daily_research/output/active_execution_strategy.json`: empty.

## Facts
- r68 closes the v5 prediction-to-simulator cashflow translation layer for the tiny strict-Gold shadow smoke.
- The simulator no longer re-solves v5 oracle output through release-first v3 when `portfolio_cashflow_decision_v1_mode` is active and valid.
- Source/receiver/cashflow intent, target, action rows, release trace, and continuity metrics now agree on the same cashflow decision object.
- Active execution artifact is unchanged.
- continuous_policy remains `research / shadow_only`.

## Inferences
- The r67 blocker was not only model no-op behavior; it also included budget semantics drift between v5 predict/oracle and simulator cashflow validation.
- A strict fail-closed contract is still useful: `_02` correctly exposed turnover-budget mismatch instead of silently falling back to legacy release-first paths.
- r68 turns the next blocker from translation closure into quality and evidence: cash timing, source selection quality, drawdown, reversal, and sufficient training evidence are still unresolved.

## Assumptions
- The strict Gold dataset `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1` remains the current training-safe dataset source.
- The tiny strict-Gold smoke is sufficient for mechanism and wiring evidence, but not for formal strategy judgment.
- GPU availability was present during `_03`; CPU-only environments should run synthetic tests first and postpone wider smoke without lowering evidence standards.

## Boundaries
- Do not treat `_01` or `_02` as r68 success evidence; they are diagnostic only.
- Do not treat `_03` as promotion, confirmatory, formal evidence, live evidence, or strategy-success evidence.
- Do not modify `daily_research/output/active_execution_strategy.json` from this evidence.
- Do not use loose `latest_*` as truth; use explicit tag `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_03`.

## Next Allowed Actions
- Continue research/shadow-only on `portfolio_set_v5_dfl_pg_v1`.
- Run a longer strict resume or wider strict-Gold smoke only after deciding the evidence question; training evidence remains insufficient in the tiny smoke.
- Improve source selection quality, cash timing, drawdown, and reversal behavior now that source/receiver translation is no longer the primary blocker.
- Keep r67 and r68 as mechanism/translation closure evidence, not live or promotion evidence.
