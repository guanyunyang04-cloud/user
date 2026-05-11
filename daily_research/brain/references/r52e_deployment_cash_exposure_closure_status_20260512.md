# r52e Deployment/Cash/Exposure Closure Status Capsule - 2026-05-12

## Scope
- Study tag: `self_opt_study_r52e_deployment_cash_exposure_closure_screening_safe_20260511_01`.
- Profile: `split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e`.
- Loss profile: `alpha_result_value_budget_split_v41`.
- Resource profile: `safe`; confirmatory disabled; true solver disabled; no live/default/promotion/active artifact change.

## Screening Result
- Executed at: `2026-05-12T00:14:42+08:00`.
- Completed screening trials: `1/3`; failed trials: `0`.
- Resource gate triggered after trial 01 and saved 2 screening trials.
- Original resource gate failures: `source_release_dead`, `economic_signal_too_weak`, `exposure_utilization_low`, `actual_cash_weight_high`.
- Original trial 01 score: `composite_score=-45.938456`, `annual_return=-0.268027`, `cash_timing_quality_1d=-0.053696`, `portfolio_daily_exposure_utilization=0.331584`, `training_evidence_status=insufficient`.
- Native validity stayed mostly healthy: `native_target_valid=0.987654`, `allocation_layer_native_fallback_used=0.012346`; fallback was not the main path.

## Closure Diagnostics
- Independent allocation closure diagnostic on the trial turnover export found the same r52d-like failure shape:
  - `actual_cash_weight_mean=0.719893`.
  - `actual_gross_exposure_mean=0.280107`.
  - `avg_gross_exposure_target=0.845967`.
  - `deployable_idle_cash_mean=0.565860`.
  - `cash_semantics_mismatch=true`.
  - `receiver_candidate_without_target_day_share=1.0`.
- A behavior audit export bug was found during this run: the audit chain computed closure from aggregated `day_merge` instead of the raw turnover export, so the original study summary did not surface `deployable_idle_cash` and `cash_semantics_mismatch`.
- Fix applied after the run: behavior audit now computes closure from turnover export columns and the diagnostic accepts simulator-native column names.
- Recomputed audit tag: `self_opt_study_r52e_deployment_cash_exposure_closure_screening_safe_20260511_01__trial_01__audit_recomputed_after_closure_fix_20260512_01`.
- Recomputed audit metrics:
  - `portfolio_daily_actual_cash_weight_mean=0.735699`.
  - `portfolio_daily_actual_gross_exposure_mean=0.264301`.
  - `portfolio_daily_deployable_idle_cash_mean=0.540194`.
  - `portfolio_daily_cash_semantics_mismatch=1.0`.
  - `portfolio_daily_receiver_candidate_without_target_day_share=0.925926`.
  - `portfolio_daily_exposure_utilization=0.331584`.
- Recomputed score/gate check would add `actual_cash_idle_high` and `cash_semantics_mismatch` to the resource failures.

## Verdict
- r52e safe screening failed the strict resume gate and failed confirmatory eligibility.
- Do not run confirmatory, strict resume, promotion, live/default, or active artifact changes from this tag.
- The issue remains deployment/cash/exposure closure and source release, not lack of confirmatory compute.
- Next work should inspect why target weight sum remains near `0.28` while stock budget / gross target is near `0.83-0.85`, and why source release stays dead despite receiver candidates.
