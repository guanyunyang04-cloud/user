# r71 Multi-Stage Regret V5 Status

Date: `2026-05-14`

## Summary
- Status: `research / shadow-only / multi-stage behavior-quality mechanism progress`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- Live default remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- r71 adds `portfolio_set_v5_dfl_pg_v1_r71_multistage_regret` as an explicit internal loss/profile line for multi-stage regret and ordered decision-quality objectives.
- r71 preserves the r70 version boundary: base v5 still defaults to `portfolio_set_v5_dfl_pg_v1`; r69 and r71 only activate through explicit loss/profile or artifact metadata.
- r71 preserves the r68 `portfolio_cashflow_decision_v1` prediction -> simulator contract; it does not add a second simulator solver or guard away source/receiver behavior.
- r71 has passed unit/contract/regression checks after implementation, but no completed tiny strict-Gold behavior smoke exists yet after the latest receiver-head calibration.
- This is not promotion evidence, not confirmatory evidence, not live evidence, and not a strategy-success verdict.

## Implementation Facts
- Updated `daily_research/continuous_policy/model_portfolio_set_v5.py`.
  - Added internal version `portfolio_set_v5_dfl_pg_v1_r71_multistage_regret`.
  - Added behavior mode `r71_multistage_regret` and prediction column `portfolio_set_v5_multistage_regret_mode`.
  - Added multi-stage target columns: `source_hold_regret_3d`, `source_hold_regret_5d`, `receiver_deploy_regret_3d`, `receiver_deploy_regret_5d`, `cash_defense_regret_1d`, `cash_defense_regret_3d`, `rotation_spread_regret_5d`, `reversal_action_regret_3d`, and `crowding_penalty`.
  - Added multi-stage regret and ordered decision-quality loss weights only for the explicit r71 profile.
  - Kept base loss insensitive to r69/r71 extension columns, and kept r69 loss insensitive to r71 extension columns.
  - Extended the torch projection oracle so r71 can penalize source rebound, weak receiver deployment, risk-on cash drag, risk-off cash defense, reversal action regret, and crowding.
  - Added a positive-spread / source-funded receiver coverage floor so r71 target construction does not collapse receiver examples when deploy spread is available.
  - Added receiver-head calibration for r71: stronger receiver recall, separate positive receiver normalization, negative receiver downweighting, raw receiver-logit margin, and high-regret propensity penalty.
- Updated `daily_research/continuous_policy/research_profile_registry.py`.
  - Registered explicit r71 profile `split_heads_portfolio_daily_multistage_regret_portfolio_set_v5_r71`.
  - Kept r71 out of active/default search profiles.
- Updated `daily_research/continuous_policy/portfolio_simulator.py`.
  - Propagates r71 regret diagnostics through action rows when present.
  - Does not change the r68 cashflow-mode priority or reintroduce release-first v3 recomputation for v5 cashflow mode.
- Updated `daily_research/continuous_policy/release_flow_trace.py` and `daily_research/continuous_policy/pipeline_utils.py`.
  - Added r71 evidence fields gated by `portfolio_set_v5_multistage_regret_mode`.
  - Kept `intent_translation_conflict_rate` as an intent-vs-final-target metric, not a bucket for r71 quality failures.
- Updated `daily_research/baseline/data_provider.py`.
  - `_fetch_tq_data` now retries and splits empty batches; singleton empties report stock/date/batch explicitly.

## Test Evidence
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests -q`
  - Result: `238 passed, 24 warnings`.
  - Warnings are existing cvxpylayers / numpy deprecation warnings.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/baseline/tests/test_data_provider_tq_fetch.py -q`
  - Result: `3 passed`.
- `git diff --check`
  - Result: clean; Git printed the known CRLF normalization warning for `daily_research/continuous_policy/research_profile_registry.py`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
  - Result: passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
  - Result: `status=ok`.
- `git diff -- daily_research/output/active_execution_strategy.json`
  - Result: empty.

## Diagnostic Smoke Attempts
- `protocol_r71_multistage_regret_v5_behavior_smoke_20260514_01`
  - Train stage completed 2 epochs.
  - Evaluate failed with `TDX returned empty data for singleton batch stock=000001.SZ start_date=20190401 end_date=20190430 batch_start=0`.
  - Diagnostic only; no completed protocol summary.
- `protocol_r71_multistage_regret_v5_behavior_smoke_20260514_02`
  - Train stage completed 2 epochs.
  - Evaluate failed with the same TDX singleton error.
  - Diagnostic only; no completed protocol summary.
- `protocol_r71_multistage_regret_v5_behavior_smoke_20260514_03`
  - Train stage completed 2 epochs.
  - Evaluate failed with the same TDX singleton error.
  - Diagnostic only; no completed protocol summary.
  - Last train diagnostics before evaluation failure showed oracle constraint violation near zero and source targets nonzero, but receiver targets were still zero before the latest receiver-head calibration.
- `protocol_r71_multistage_regret_v5_behavior_smoke_20260514_04`
  - Train failed before epoch completion with `CUDA-capable device(s) is/are busy or unavailable`.
  - Diagnostic only; no completed protocol summary.

## Mechanism Evidence
- The r71 explicit profile and artifact metadata path are wired and test-covered.
- Synthetic/unit fixtures cover:
  - source rebound after sell is penalized;
  - receiver deployment is reduced when source/cash dominates;
  - risk-off days can prefer cash defense;
  - positive-spread source-funded rotation seeds receiver demand;
  - oracle constraint violation remains below `1e-6` in fixtures.
- Strict-Gold train-stage diagnostics from `_01/_02/_03` show the r71 oracle remains feasible and source targets are nonzero, but these runs are not behavior evidence because evaluation/shadow did not complete.
- The latest code change after `_03` improved target-builder receiver coverage and receiver-head calibration, but that change has not yet been validated by a completed tiny strict-Gold behavior smoke.

## Baseline And Remaining Blockers
- r70 baseline tag: `protocol_r70_v5_version_boundary_oracle_repair_smoke_20260514_01`.
- r70 completed tiny smoke established:
  - `cashflow_decision_contract_valid_rate=1.0`;
  - `intent_translation_conflict_rate=0.0`;
  - oracle violation near zero;
  - remaining failures in `training_evidence`, `cash_timing_quality_1d`, `immediate_reversal_rate_3d`, and `source_positive_forward_sell_share`.
- r71 has not yet proved behavior improvement over r70 because all r71 smoke attempts are failed/incomplete.
- Current blockers:
  - TDX provider singleton empty data for `000001.SZ` over `20190401 -> 20190430` during evaluate.
  - CUDA busy/unavailable during `_04`.
  - No completed post-calibration r71 behavior smoke yet.
  - Training evidence is still expected to be `insufficient` in tiny 2-epoch smoke until a longer, accepted strict resume is justified by behavior metrics.

## Facts
- r71 is explicit-only and does not change the base v5 default.
- r69 remains explicit-only and is not silently folded into base v5.
- r68 cashflow contract remains the only v5 prediction -> simulator capital-flow contract.
- Active execution artifact diff remains empty.
- Failed / interrupted r71 protocol attempts are diagnostics only.

## Inferences
- The current blocker is not version boundary, simulator translation, or oracle feasibility.
- The next useful evidence is a completed r71 tiny strict-Gold behavior smoke after receiver-head calibration and with the provider/GPU blockers resolved.
- If r71 still yields zero receiver targets after a completed smoke, the next fix should stay in target/oracle/head calibration, not in simulator fallback or promotion gates.

## Assumptions
- The strict Gold dataset remains the correct training-safe source: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- The default interpreter remains `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`.
- GPU/provider availability may change; a failed provider/GPU run should be recorded as a blocker, not downgraded into weaker evidence standards.

## Boundaries
- Do not treat r71 `_01/_02/_03/_04` as completed behavior acceptance.
- Do not run strict resume until a completed tiny r71 smoke preserves translation closure and improves at least two behavior metrics versus r70.
- Do not make r71 active/default.
- Do not promotion, confirmatory, live/default, or edit `daily_research/output/active_execution_strategy.json` from r71 mechanism evidence.

## Next Allowed Actions
- Resolve the TDX singleton provider blocker or use a verified complete cached/csv evaluation path without lowering evidence standards.
- Rerun a tiny r71 behavior smoke with the next unused explicit r71 smoke tag only when GPU/provider are available.
- Inspect whether post-calibration train/evaluate diagnostics keep source targets nonzero and restore receiver target counts above zero.
- Accept r71 first-stage progress only if a completed tiny smoke keeps `cashflow_decision_contract_valid_rate=1.0`, keeps `intent_translation_conflict_rate=0.0`, keeps oracle violation near zero, and improves at least two behavior metrics versus r70.
- Keep r70 as version-boundary/oracle-feasibility evidence and r68 as translation-closure evidence until r71 has completed behavior evidence.
