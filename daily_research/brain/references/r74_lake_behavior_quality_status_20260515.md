# r74 Lake Behavior Quality Status

Date: `2026-05-15`

## Summary
- Status: `research / shadow-only / lake-native behavior-quality evidence`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- Live default remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- Training dataset remains strict Gold: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Lake evaluation dataset remains `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- Completed usable protocol tag: `protocol_r74_lake_behavior_quality_v5_smoke_20260515_03`.
- r74 is not promotion evidence, confirmatory evidence, or live/default evidence.

## Implementation Facts
- Added explicit V5 internal version / loss profile `portfolio_set_v5_dfl_pg_v1_r74_lake_behavior_quality`.
- Added explicit profile `split_heads_portfolio_daily_lake_behavior_quality_portfolio_set_v5_r74`.
- r74 only activates from explicit loss/profile or artifact metadata; base v5, r69, and r71 modes remain isolated.
- Extended `portfolio_decision_feature_bundle_v1` with high-value lake fields including `adv20`, `amount`, `volume`, `z_drawdown_20`, `z_volatility_20`, `z_vol_ratio_5_20`, `z_price_volume_divergence`, `z_breakout_volume`, `z_volatility_contraction`, and `z_volume_contraction`.
- Added feature contract severity and evidence fields: `ok / neutral_default / degraded / blocker`, degraded/blocker counts, and high-value feature usage diagnostics.
- Added r74 behavior-quality oracle inputs and diagnostics for cash timing, source quality, receiver-source spread, liquidity/crowding, and lake-native behavior mode.
- Kept r68 `portfolio_cashflow_decision_v1` as the only v5 prediction -> simulator funds-flow contract; r74 does not add a second solver or use simulator guards to fabricate source/receiver targets.
- Simulator action rows now preserve r74 behavior-quality fields so continuity metrics and release traces can see the model's r74 mode.
- Live write scripts now require explicit active-manifest confirmation:
  - `update_default_candidate_production.py` requires `--activate-strategy --confirm-active-manifest-write`;
  - `activate_execution_single_mapping_candidate.py` defaults to dry-run and requires `--write-active-manifest --confirm-active-manifest-write`.

## Smoke Evidence
- `_01` failed during train diagnostics with `NameError: cash is not defined`; this is an implementation blocker, not strategy evidence.
- `_02` completed train/evaluate/shadow/export, but action rows did not preserve r74 mode fields, so r74 continuity evidence was incomplete.
- `_03` completed train/evaluate/shadow/export after simulator passthrough repair and is the usable r74 tiny lake smoke.
- `_03` train diagnostics:
  - `completed_epochs=2`;
  - `best_epoch=2`;
  - `portfolio_set_v5_behavior_mode=r74_lake_behavior_quality`;
  - `decision_target_source_count=468`;
  - `decision_target_receiver_count=10`;
  - `decision_target_constraint_violation_mean=1.4355378947787577e-11`;
  - `r74_target_cash_timing_alignment_1d=0.03585006138573106`;
  - `r74_target_source_quality_score=0.5042674159994347`;
  - `r74_target_receiver_source_spread_quality=0.1516427841901735`.

## Evaluation Metrics
- r73 evaluation baseline from `protocol_r73_lake_native_r71_collapse_repair_smoke_20260515_02`:
  - `cashflow_decision_contract_valid_rate=1.0`;
  - `intent_translation_conflict_rate=0.0`;
  - `portfolio_daily_source_target_count=1.1428571428571428`;
  - `portfolio_daily_receiver_target_count=159`;
  - `portfolio_daily_source_positive_forward_sell_share=0.2`;
  - `portfolio_daily_receiver_minus_source_forward_excess_5d=0.006017551215227488`;
  - `immediate_reversal_rate_3d=0.00546448087431694`;
  - `cash_timing_quality_1d=-0.26114614493645566`.
- r74 `_03` evaluation:
  - `cashflow_decision_contract_valid_rate=1.0`;
  - `intent_translation_conflict_rate=0.0`;
  - `portfolio_daily_source_target_count=0.047619047619047616`;
  - `portfolio_daily_receiver_target_count=30`;
  - `portfolio_daily_source_positive_forward_sell_share=0.0`;
  - `portfolio_daily_receiver_minus_source_forward_excess_5d=NaN`;
  - `immediate_reversal_rate_3d=0.0`;
  - `cash_timing_quality_1d=0.007381697051177278`;
  - `r74_lake_behavior_quality_mode_count=252`;
  - `r74_feature_contract_degraded_rate=1.0`;
  - `r74_feature_contract_blocker_rate=0.0`.

## Shadow Metrics
- r73 shadow baseline:
  - `cashflow_decision_contract_valid_rate=1.0`;
  - `intent_translation_conflict_rate=0.0`;
  - `portfolio_daily_source_target_count=2.6666666666666665`;
  - `portfolio_daily_receiver_target_count=168`;
  - `portfolio_daily_source_positive_forward_sell_share=0.38095238095238093`;
  - `portfolio_daily_receiver_minus_source_forward_excess_5d=-0.006483055107116782`;
  - `immediate_reversal_rate_3d=0.008928571428571428`;
  - `cash_timing_quality_1d=-0.27509539900014823`.
- r74 `_03` shadow:
  - `cashflow_decision_contract_valid_rate=1.0`;
  - `intent_translation_conflict_rate=0.0`;
  - `portfolio_daily_source_target_count=0.8571428571428571`;
  - `portfolio_daily_receiver_target_count=99`;
  - `portfolio_daily_source_positive_forward_sell_share=0.0`;
  - `portfolio_daily_receiver_minus_source_forward_excess_5d=0.024442776344632607`;
  - `immediate_reversal_rate_3d=0.0`;
  - `cash_timing_quality_1d=-0.1696032883685711`;
  - `r74_lake_behavior_quality_mode_count=853`;
  - `r74_feature_contract_degraded_rate=1.0`;
  - `r74_feature_contract_blocker_rate=0.0`.

## Interpretation
- Fact: r74 preserves r68/r73 translation closure in the usable smoke: cashflow valid remains `1.0`, intent conflict remains `0.0`, and source/receiver targets remain nonzero.
- Fact: r74 improves at least two behavior metrics versus r73 in both evaluation and shadow: positive-forward source sell share improves to `0.0`, immediate reversal improves to `0.0`, and cash timing improves materially.
- Fact: shadow receiver-source spread improves from negative to positive.
- Fact: source/receiver activity shrinks versus r73, especially evaluation source count and receiver count.
- Fact: feature contract degraded rate is `1.0` with blocker rate `0.0`, so r74 has usable but degraded feature coverage; this must not be written as a fully healthy feature contract.
- Inference: r74 is credible behavior-quality research progress, but not a strategy-success verdict.
- Inference: the next blocker is no longer TDX, oracle feasibility, cashflow translation, or source/receiver collapse; it is feature-contract degradation, reduced trading coverage, training evidence sufficiency, and longer-run behavior stability.

## Remaining Blockers
- `training_evidence.status=insufficient`.
- `promotion_gate.status=shadow_only`.
- `best_epoch=2` equals `completed_epochs=2`.
- `teacher_action_rows` remains insufficient for promotion-style evidence.
- r74 source/receiver coverage is lower than r73.
- r74 feature contract degraded rate is `1.0`.
- Lake evaluation remains research/shadow evidence and must not be counted as completed training evidence.

## Tests
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests -q`
  - Result: `256 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests -q`
  - Result: `15 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/tools/tests -q`
  - Result: `37 passed`.
- `git diff -- daily_research/output/active_execution_strategy.json`
  - Result before writeback: empty.

## Boundaries
- Do not treat r74 as promotion, confirmatory, or live evidence.
- Do not change live/default or `daily_research/output/active_execution_strategy.json`.
- Do not treat lake evaluation or realtime/tail labels as completed training evidence.
- Do not use simulator guards or release-first recomputation to fabricate source/receiver targets.

## Next Allowed Actions
- Repair feature-contract degradation so high-value lake fields are used without forcing every action row into degraded severity.
- Recover source/receiver coverage while preserving zero intent conflict, cashflow validity, and lower wrong-side sell share.
- Only consider strict resume after a fresh tiny lake smoke keeps r74 translation closed, keeps oracle violation near zero, and preserves at least two behavior improvements without severe source/receiver coverage shrinkage.
