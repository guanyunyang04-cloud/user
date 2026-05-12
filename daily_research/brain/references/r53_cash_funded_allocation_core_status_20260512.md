# r53 Cash-Funded Allocation Core Status

Date: `2026-05-12`

## Scope
- New research line: `r53 cash-funded allocation core rebuild`.
- Branch: `codex/r53-allocation-core-rebuild`.
- Production boundary: no live/default/promotion switch and no change to `daily_research/output/active_execution_strategy.json`.
- Execution boundary: no true solver, no confirmatory, no broker/live path.

## Code Contract
- New pure allocator module: `daily_research/continuous_policy/allocation_core_v2.py`.
- New profile: `split_heads_portfolio_daily_cash_funded_allocation_core_r53`.
- New loss profile: `alpha_result_value_budget_split_v43`.
- r53 path is cash-first:
  1. compute stock budget from gross/cash reserve targets,
  2. deploy available cash into receiver headroom,
  3. release source only when cash is insufficient and receiver demand remains,
  4. return explicit closure diagnostics instead of silently falling back.
- r52/r52e native allocation paths remain unchanged.

## Screening Evidence
- Dry-run `self_opt_study_r53_cash_funded_allocation_core_dryrun_20260512_03` passed.
- Safe screening `self_opt_study_r53_cash_funded_allocation_core_screening_safe_20260512_03` completed `3/3` screening trials, `0` failed trials, `confirmatory_enabled=false`, `confirmatory_completed_trial_count=0`.
- Best screening trial by ranking: `self_opt_study_r53_cash_funded_allocation_core_screening_safe_20260512_03__trial_03`.

Key trial 03 facts:
- `composite_score=-1310.310271`.
- `training_evidence_status=insufficient`.
- `cash_timing_quality_1d=-0.16812973310159365`.
- `annual_return=0.19472030420062403`.
- `monthly_return_mean=0.012242807098976117`.
- `max_drawdown=-0.10642644735509244`.
- `avg_gross_exposure_target=0.7466423881202774`.
- `portfolio_daily_actual_gross_exposure_mean=0.8111197481053539`.
- `portfolio_daily_actual_cash_weight_mean=0.18888025189464594`.
- `portfolio_daily_exposure_utilization=1.0863564150803207`.
- `portfolio_daily_target_sum_gap=0.004874496665371866`.
- `portfolio_daily_deployable_idle_cash_mean=0.004874496665371867`.
- `portfolio_daily_cash_semantics_mismatch=0.0`.
- `allocation_layer_native_fallback_used=0.0`.
- `native_target_valid=1.0`.

## Fixes During r53
- First screening exposed that receiver support was still gated by old executable-candidate masks; r53 now broadens receiver/source support from score/executability/headroom signals before v2 allocation.
- Second screening exposed that low learned `gross_exposure_target` could shrink r53 stock budget to about `0.20`; r53 now emits an explicit `allocation_core_v2_stock_budget_floor` and simulator v2 honors that floor.
- Third screening showed the budget/cash/exposure closure works materially better: cash fell from the previous high-cash failure shape to about `0.19`, target gap was near zero, and fallback was not used.
- Resource gate semantics were corrected after trial 03: when budget is already closed, the gate no longer requires fresh cash-funded deployment or receiver target counts. Re-evaluating trial 03 with the patched gate leaves only `cash_timing_bad`.

## Verdict
- r53 has a valid code contract and safe-screening evidence that the high-cash / low-budget deployment bug is materially improved.
- r53 is not a strategy-success verdict.
- r53 must not enter confirmatory, strict resume, promotion, live/default, or active artifact changes yet.

Blocking reasons:
- all screening trials still have `training_evidence_status=insufficient`;
- `cash_timing_quality_1d` remains negative;
- `composite_score` remains strongly negative under the current objective;
- action semantics and order translation quality still require review before any longer run is useful.

## Next Decision
- Do not run confirmatory from `self_opt_study_r53_cash_funded_allocation_core_screening_safe_20260512_03`.
- Do not simply add epochs as the next move.
- Next work should inspect why cash timing stays negative and why scoring remains deeply negative despite cash/exposure closure improvement.
- A strict resume is only reasonable after cash timing/objective semantics are addressed and the same r53 gate remains clean on a fresh safe screening.
