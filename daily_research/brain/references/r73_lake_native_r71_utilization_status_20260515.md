# r73 Lake Native R71 Utilization Status

Date: `2026-05-15`

## Summary
- Status: `research / shadow-only / lake-native utilization and r71 collapse-repair evidence`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- Live default remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- Training dataset remains strict Gold: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Lake evaluation dataset remains `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- Completed protocol tag: `protocol_r73_lake_native_r71_collapse_repair_smoke_20260515_02`.
- r73 repairs the r72 lake-smoke source/receiver target collapse and adds data-lake feature utilization evidence, but it is not promotion evidence, confirmatory evidence, or live evidence.

## Implementation Facts
- Added `portfolio_decision_feature_bundle_v1` in `daily_research/continuous_policy/portfolio_decision_features.py`.
- r71/v5 prediction now attaches a shared portfolio decision feature bundle before oracle translation, so lake evaluation and strict-Gold training use aligned decision-field names for price, return, volatility, liquidity, membership, alpha prior, source/receiver spread, cash defense, and multistage regret inputs.
- Added `lake_feature_utilization_report` to evaluate/protocol summaries. It reports available panels, used panels, missing/low-variance fields, and unused high-value features.
- Added r73 collapse diagnostics around v5 prediction and simulator:
  - pre-oracle source/receiver candidate counts;
  - source capacity and receiver headroom;
  - cash dominance;
  - oracle target delta and post-cashflow target delta;
  - collapse layer attribution.
- Repaired r71 source collapse in lake prediction by changing weak held-source quality and positive-forward sell risk semantics; source-funded rotation is now preserved when held source is weak, receiver spread is positive, and risk/reversal penalties are low.
- Updated behavior audit release-translation health so cashflow-contract source targets count as funding-release evidence under r68/r73 semantics. This prevents stale `funding_release_not_observed` conclusions when `portfolio_cashflow_decision_v1` is active and healthy.
- Repaired r71 target-builder `r71_receiver_coverage_floor` attribution: the diagnostic now records positive-spread receiver coverage already produced by the main oracle, not only fallback-added receiver demand. This does not change the oracle allocation itself.

## Evidence
- Protocol `protocol_r73_lake_native_r71_collapse_repair_smoke_20260515_02` completed train/evaluate/shadow/export/audit through lake evaluation.
- Training diagnostics:
  - `completed_epochs=2`;
  - `best_epoch=2`;
  - `decision_target_source_count=1281`;
  - `decision_target_receiver_count=879`;
  - `decision_target_constraint_violation_mean=3.5448524466230294e-10`;
  - `decision_target_intent_translation_conflict_count=0`.
- Evaluation continuity metrics:
  - `cashflow_decision_contract_valid_rate=1.0`;
  - `intent_translation_conflict_rate=0.0`;
  - `portfolio_daily_source_target_count=1.1428571428571428`;
  - cashflow source intent count in behavior audit: `24`;
  - `portfolio_daily_receiver_target_count=159`;
  - `r73_collapse_layer_none_share=0.9865107913669064`;
  - `r73_collapse_layer_oracle_count=15`;
  - `portfolio_daily_source_positive_forward_sell_share=0.2`;
  - `portfolio_daily_receiver_minus_source_forward_excess_5d=0.006017551215227488`;
  - `immediate_reversal_rate_3d=0.00546448087431694`;
  - `cash_timing_quality_1d=-0.26114614493645566`.
- Shadow continuity metrics:
  - `cashflow_decision_contract_valid_rate=1.0`;
  - `intent_translation_conflict_rate=0.0`;
  - `portfolio_daily_source_target_count=2.6666666666666665`;
  - cashflow source intent count: `56`;
  - `portfolio_daily_receiver_target_count=168`;
  - `r73_collapse_layer_none_share=1.0`;
  - `portfolio_daily_source_positive_forward_sell_share=0.38095238095238093`;
  - `portfolio_daily_receiver_minus_source_forward_excess_5d=-0.006483055107116782`;
  - `immediate_reversal_rate_3d=0.008928571428571428`;
  - `cash_timing_quality_1d=-0.27509539900014823`.
- Behavior audit `release_translation_deploy_failure_mode=healthy` after cashflow-source health recomputation.
- Lake feature utilization report:
  - `feature_panel_count=115`;
  - `available_feature_count=115`;
  - `actually_used_feature_count=88`;
  - `unused_high_value_feature_count=10`;
  - first unused high-value features: `adv20`, `amount`, `volume`, `z_breakout_volume`, `z_drawdown_20`, `z_price_volume_divergence`, `z_vol_ratio_5_20`, `z_volatility_20`, `z_volatility_contraction`, `z_volume_contraction`.

## Facts
- r73 fixes the r72 completed lake-smoke blocker where source/receiver targets collapsed to zero.
- r73 keeps the r68 `portfolio_cashflow_decision_v1` prediction -> simulator contract as the only v5 cashflow contract.
- r73 does not reintroduce release-first v3 target recomputation for v5 cashflow mode.
- r73 preserves r70/r71 version boundaries: base v5 remains `portfolio_set_v5_dfl_pg_v1`; r69 and r71 remain explicit-only modes.
- r73 turns the data lake from a mere TDX replacement into a measured decision-feature source, with explicit utilization reporting.
- Active execution artifact diff remains empty.

## Inferences
- The current r71 blocker has moved from source/receiver translation collapse to behavior quality and training evidence sufficiency.
- The lake data itself is strong enough for reproducible evaluation, but the model still underuses several high-value liquidity, volume, volatility, and drawdown feature panels.
- The evaluation window shows improved translation and source/receiver activity, but shadow behavior still has weak cash timing and mixed source quality.
- A strict resume is not justified solely by this smoke because `best_epoch=2`, training evidence is still insufficient, and behavior metrics are mixed.

## Remaining Blockers
- `training_evidence.status=insufficient`.
- Failed training evidence checks include `teacher_action_rows` and `best_epoch_not_at_edge`.
- Promotion gate remains `shadow_only`.
- Promotion checks still fail on `contract_promotable`, `training_evidence_sufficient`, `open_win_rate_5d`, `exit_timeliness_rate_5d`, `cash_timing_quality_1d`, `annual_return_vs_active`, and `sharpe_vs_active`.
- Shadow `cash_timing_quality_1d=-0.27509539900014823` remains negative.
- Shadow `portfolio_daily_source_positive_forward_sell_share=0.38095238095238093` is still too high.
- Shadow `portfolio_daily_receiver_minus_source_forward_excess_5d=-0.006483055107116782` is not yet a healthy rotation-spread result.

## Tests
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_set_v5_release_first_loss.py::PortfolioSetV5ReleaseFirstLossTest::test_r71_target_builder_seeds_positive_spread_rotation_receiver -q`
  - Result: `1 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests -q`
  - Result: `248 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests -q`
  - Result: `15 passed`.
- `git diff --check`
  - Result: clean.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
  - Result before writeback: passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
  - Result before writeback: `status=ok`.
- `git diff -- daily_research/output/active_execution_strategy.json`
  - Result: empty.

## Assumptions
- The strict Gold dataset remains the correct training-safe source.
- The r73 lake smoke is a tiny research/shadow protocol, not confirmatory evidence.
- The lake evaluation bundle remains appropriate for reproducible research evaluation over `2019-04-01 -> 2019-04-30`.

## Boundaries
- Do not treat r73 as promotion, confirmatory, or live evidence.
- Do not treat lake evaluation or realtime/tail labels as completed training evidence.
- Do not change live/default or `daily_research/output/active_execution_strategy.json`.
- Do not use simulator guards or release-first recomputation to fabricate source/receiver targets.

## Next Allowed Actions
- Continue r71/r73 research on behavior quality: cash timing, source positive-forward sell share, receiver-source spread, and reversal quality.
- Use the r73 utilization report to decide which currently unused high-value lake features should feed the decision bundle next.
- Only consider strict resume after a fresh tiny lake smoke preserves cashflow closure, keeps source/receiver nonzero, and improves at least two behavior metrics versus r70/r72 without increasing oracle violations.
