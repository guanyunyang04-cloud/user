# r54 Semantic Budget Controller Status

Date: `2026-05-12`

## Scope
- New research line: `r54 semantic-budget controller`.
- Branch: `codex/r54-semantic-budget-controller`.
- Production boundary: no live/default/promotion switch and no change to `daily_research/output/active_execution_strategy.json`.
- Execution boundary: no true solver, no confirmatory, no broker/live path.

## Code Contract
- r54 profile: `split_heads_portfolio_daily_semantic_budget_controller_r54`.
- r54 loss profile: `alpha_result_value_budget_split_v44`.
- r54 path changes the main policy semantics from lifecycle `action_label` to target-weight intent:
  - model export includes `allocation_intent_v2_mode`, `portfolio_daily_target_weight_intent`, and `portfolio_daily_target_delta_intent`;
  - simulator uses the r53 cash-funded allocator as final budget solver;
  - execution action is derived from final target-weight delta;
  - lifecycle action remains a hint/reporting surface, not the final portfolio target path.
- r53 scorer/gate semantics were corrected so budget-closed days do not require fresh receiver activity.
- intent translation audit is deadband-aware: micro target-weight deltas below execution deadband are not counted as translation conflicts.

## Dry-Run Evidence
- Dry-run `self_opt_study_r54_semantic_budget_controller_dryrun_20260512_02` passed.
- Dry-run selected 3 screening trials using `alpha_result_value_budget_split_v44`.
- `confirmatory_enabled=false`.
- `full_universe_train_solver_effective=false`.
- Resource plan used safe-compatible profile fields: epochs `8`, min epochs `6`, day-set batch sizes `1/2`.

## Safe Screening Evidence
- Safe screening tag: `self_opt_study_r54_semantic_budget_controller_screening_safe_20260512_02`.
- Run completed at `2026-05-12T20:30:32+08:00`.
- Study summary: `daily_research/output/continuous_policy/studies/self_opt_study_r54_semantic_budget_controller_screening_safe_20260512_02/study_summary.json`.
- Completed screening trials: `2`.
- Failed screening trials: `1`.
- Confirmatory remained disabled; confirmatory completed count was `0`.
- Resource gate evaluated trial 02 and did not stop screening early; remaining resource-gate failure was `cash_timing_bad`.

Best completed trial:
- Trial tag: `self_opt_study_r54_semantic_budget_controller_screening_safe_20260512_02__trial_02`.
- `composite_score=4.80437`.
- `annual_return=0.153993`.
- `monthly_return_mean=0.010115`.
- `max_drawdown=-0.106873`.
- `cash_timing_quality_1d=-0.168248`.
- `portfolio_daily_actual_cash_weight_mean=0.188532`.
- `portfolio_daily_actual_gross_exposure_mean=0.811468`.
- `portfolio_daily_exposure_utilization=1.082835`.
- `portfolio_daily_target_sum_gap=0.004812`.
- `portfolio_daily_cash_semantics_mismatch=0.0`.
- `allocation_layer_native_fallback_used=0.0`.
- `allocation_intent_v2_mode_share=1.0`.
- `intent_translation_conflict_rate=0.0`.

Completed-trial blockers:
- `training_evidence_status=insufficient`.
- `best_epoch=8` equals `completed_epochs=8`, failing `best_epoch_not_at_edge`.
- `cash_timing_quality_1d` remains negative.
- promotion gate still fails reduce success, exit timeliness, cash timing, max drawdown, and shadow reversal checks.

Trial 03 note:
- `self_opt_study_r54_semantic_budget_controller_screening_safe_20260512_02__trial_03` wrote `protocol_summary.json`, but the study runner received exit code `3221226505` and marked it `trial_execution_failed`.
- Trial 03 artifact may be used only for diagnostics.
- Trial 03 must not count as completed screening evidence until reproduced with a clean exit.

## Verdict
- r54 fixes the r53 audit/scoring false-negative around budget-closed days and makes intent translation conflict measurement deadband-aware.
- r54 materially improves score shape versus r53: best completed screening composite is positive while cash/exposure closure remains clean.
- r54 is still not a strategy-success verdict.
- r54 must not enter confirmatory, promotion, live/default, or active artifact changes.

Blocking reasons:
- training evidence is still insufficient;
- cash timing remains negative around `-0.168`;
- reduce/exit lifecycle quality remains weak;
- one safe-screening trial exited abnormally even though it wrote a protocol summary.

## Next Decision
- Do not run confirmatory from `self_opt_study_r54_semantic_budget_controller_screening_safe_20260512_02`.
- A strict resume is only reasonable if it keeps the same explicit r54 tag lineage, first investigates the trial 03 abnormal exit, and targets the known `best_epoch_at_edge` / cash-timing blockers.
- The next code work should focus on cash-timing and risk-reduction intent quality, not allocator closure or receiver-headroom gates.
