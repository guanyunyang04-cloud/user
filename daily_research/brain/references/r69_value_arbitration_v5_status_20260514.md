# r69 Value Arbitration V5 Status

Date: `2026-05-14`

## Summary
- Status: `research / shadow-only / behavior-quality mechanism progress`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- Live default remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- r69 adds `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration` as an explicit internal loss/profile line for deploy / release / defense value arbitration.
- r69 does not pass first-stage behavior acceptance yet. It improves target/source quality mechanics and diagnostics, but full tiny strict-Gold protocol evaluation/shadow did not complete because TDX returned an empty data batch during evaluate.
- This is not promotion evidence, not confirmatory evidence, not live evidence, and not a strategy-success verdict.

## Implementation Facts
- Updated `daily_research/continuous_policy/model_portfolio_set_v5.py`.
  - Added r69 value fields to the v5 decision target contract.
  - Added value arbitration objective inputs: `deploy_value`, `release_value`, `defense_value`, `cash_timing_value`, `source_opportunity_cost`, `receiver_source_spread_value`, `reversal_risk_penalty`, and `source_wrong_side_sell_penalty`.
  - Reworked the torch cashflow oracle so value-adjusted source/receiver utilities drive capacity, top-k selection, and allocation.
  - Kept r68 `portfolio_cashflow_decision_v1` as the prediction -> simulator contract.
  - Added reachable cash-floor handling and source-funded-rotation budget competition so defense does not silently consume all receiver funding.
  - Added deterministic scenario day sampling for held source, flat receiver, cash-only, risk-off, and source-funded rotation coverage.
  - Added aggregated train/validation decision diagnostics instead of relying on the last batch only.
- Updated `daily_research/continuous_policy/research_profile_registry.py`.
  - Registered explicit r69 profile `split_heads_portfolio_daily_value_arbitration_portfolio_set_v5_r69`.
  - Kept r69 out of `ACTIVE_SEARCH_PROFILE_NAMES`.
- Updated `daily_research/continuous_policy/run_continuous_policy_protocol.py`.
  - Allows registered non-active research profiles to run explicit protocols while recording `active_profile=false`.
- Updated `daily_research/continuous_policy/train_policy.py`.
  - Keeps the backend default loss at base `portfolio_set_v5_dfl_pg_v1`; r69 requires an explicit loss/profile.
- Updated tests for r69 oracle behavior, artifact contract, dataset target width, protocol profile binding, and legacy/default compatibility.

## Diagnostic Runs
- `protocol_r69_value_arbitration_v5_behavior_smoke_20260514_01`:
  - Train stage completed.
  - Evaluate failed with `RuntimeError: TDX returned empty data for batch starting at 1344`.
  - Diagnostic only. Training showed receiver collapse in validation diagnostics before subsequent fixes.
- `protocol_r69_value_arbitration_v5_behavior_smoke_20260514_02`:
  - Train stage completed after oracle/cash-floor balancing fixes.
  - Evaluate failed with the same TDX empty batch error.
  - Diagnostic only. Target constraint violation improved versus `_01`, but full protocol evidence was still unavailable.
- `protocol_r69_value_arbitration_v5_behavior_traincheck_20260514_03`:
  - Train-only strict-Gold check completed successfully.
  - This is target/training mechanism evidence only, not evaluation/shadow behavior evidence.

## Traincheck Evidence
- Dataset: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Backend/loss: `formal_torch_portfolio_set_v5` / `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration`.
- Internal version: `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration`.
- Completed epochs: `2`; best epoch: `2`.
- Train day count: `256`.
- Scenario sampling counts:
  - held-source days: `207`.
  - flat-receiver days: `203`.
  - cash-only days: `126`.
  - risk-off days: `49`.
  - source-funded-rotation days: `205`.
- Target counts:
  - `decision_target_source_count=1185.0`.
  - `decision_target_receiver_count=160.0`.
  - `decision_target_intent_translation_conflict_count=0.0`.
  - `decision_target_constraint_violation_mean=0.022252461536274436`.
- Aggregated train diagnostics:
  - `release_flow_source_target_count=1041.0`.
  - `release_flow_receiver_target_count=392.0`.
  - `release_flow_intent_translation_conflict_count=0.0`.
  - `value_arbitration_source_wrong_side_sell_share=0.0`.
  - `decision_oracle_constraint_violation_mean=0.036376141011714935`.
- Aggregated validation diagnostics:
  - `release_flow_source_target_count=141.0`.
  - `release_flow_receiver_target_count=56.0`.
  - `release_flow_intent_translation_conflict_count=0.0`.
  - `value_arbitration_source_wrong_side_sell_share=0.0`.
  - `decision_oracle_constraint_violation_mean=0.03294592542994407`.

## Comparison To r68 Baseline
- r68 evidence tag: `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_03`.
- r68 target counts:
  - `decision_target_source_count=793.0`.
  - `decision_target_receiver_count=2048.0`.
  - `decision_target_constraint_violation_mean=1.93708651542423e-09`.
  - shadow source target=`97`, receiver target=`139`, cashflow valid=`21/21`, `intent_translation_conflict_rate=0.0`.
- r69 traincheck target changes:
  - Source target coverage increased from `793.0` to `1185.0`.
  - Receiver target coverage decreased from `2048.0` to `160.0`.
  - Wrong-side source sell share is explicitly measured and currently `0.0`.
  - Constraint violation regressed from effectively zero to about `0.0223` target mean / `0.0329` validation oracle mean.

## Facts
- r69 implementation preserves artifact type, backend name, loader/predict API, and r68 cashflow contract compatibility.
- Unit and contract tests pass after r69 changes.
- r69 explicit profile is registered but is not an active/default search entrypoint.
- Full r69 behavior smoke did not complete evaluation/shadow because TDX returned empty data.
- The active execution artifact diff is empty.

## Inferences
- r69 successfully moved source quality from "release capacity only" toward value arbitration: wrong-side sell pressure and reversal penalties now participate in target/oracle construction.
- The current r69 objective is still too defensive/credit-constrained for receiver coverage compared with r68.
- Constraint violation is now the main mechanism blocker before r69 can be treated as behavior-quality evidence.
- The TDX empty-batch failure is an infrastructure/data access blocker for protocol completion, separate from model/oracle behavior.

## Assumptions
- The strict Gold dataset remains the correct training-safe source for this line.
- Train-only evidence is useful for mechanism debugging but insufficient for r69 acceptance.
- A future full smoke should avoid live TDX fetch fragility, either by using a known-good cached/csv evaluation path or by fixing the TDX empty-batch handling.

## Boundaries
- Do not treat `_01`, `_02`, or `traincheck_03` as promotion, confirmatory, live, or completed behavior acceptance evidence.
- Do not modify `daily_research/output/active_execution_strategy.json`.
- Do not make r69 an active/default profile.
- Do not run strict resume to 18 epochs until tiny behavior smoke completes and shows behavior improvement without breaking r68 translation closure.

## Next Allowed Actions
- Reduce r69 oracle constraint violation while preserving `source_wrong_side_sell_share=0.0`.
- Rebalance deploy/release/defense so receiver coverage does not collapse relative to r68.
- Fix or bypass the TDX empty-batch evaluate blocker with an explicit cached/csv evaluation source.
- Rerun a full r69 tiny strict-Gold behavior smoke only after the mechanism target violation is near zero and receiver coverage is materially healthier.
- Keep r68 as the current translation-closure evidence baseline.
