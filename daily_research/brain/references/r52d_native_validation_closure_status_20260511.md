# r52d Native Validation Closure Status Capsule

Date: `2026-05-11`

## Source
- Study tag: `self_opt_study_r52d_native_validation_closure_screening_safe_20260511_01`.
- Command used: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -X utf8 -m daily_research.tools.brain_workflow status --workflow continuous_policy --study-tag self_opt_study_r52d_native_validation_closure_screening_safe_20260511_01 --json`.
- Source files: explicit study summary, trial ranking, protocol summaries, training diagnostics, and evaluation summaries under `daily_research/output/continuous_policy/`.
- Freshness note: loose `latest_*` pointers are still mixed-source; latest study is r52d, but latest protocol/audit/ledger still point to `cp_v3_portfolio_daily_allocation_breadth_r34_bounded_20260430__confirm_01`.

## Facts
- Search profile: `split_heads_portfolio_daily_day_set_native_validation_closure_r52d`.
- Loss profile: `alpha_result_value_budget_split_v40`.
- Resource profile: `safe`; `thread_limit = 4`; `cpu_affinity_count = 4`; `process_priority = below_normal`.
- Study completed `3/3` screening trials with `0` failed.
- `confirmatory_enabled = false`; `confirmatory_completed_trial_count = 0`.
- `native_validation_closure_support = true`.
- `full_universe_train_solver_effective = false`.

## Trial Evidence
| Trial | Composite | Annual Return | Sharpe | Cash Timing 1d | Exposure Utilization | Source Targets | Source Sell Rate | Native Target Valid | Native Fallback | Training Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | -14.285045 | 0.027635 | 0.284789 | -0.141275 | 0.334744 | 3 | 1.000000 | 1.000000 | 0.000000 | insufficient |
| 02 | -24.938072 | 0.153216 | 1.299246 | -0.116505 | 0.342143 | 9 | 1.000000 | 1.000000 | 0.000000 | insufficient |
| 03 | -44.706098 | -0.266680 | -2.023256 | -0.052119 | 0.331616 | 0 | 0.000000 | 0.987654 | 0.012346 | insufficient |

## Confirmatory Gate
Current conservative gate requires at least one trial to satisfy all of:
- `composite_score > 0`.
- `training_evidence_status = sufficient`.
- no v2/stability hard gate failure relevant to strategy acceptance.
- `cash_timing_quality_1d >= 0`.
- `portfolio_daily_exposure_utilization >= 0.60`.
- source/deploy evidence is non-empty and realized.
- native fallback is not the main path.

No r52d screening trial passes this gate. The common blockers are negative composite score, insufficient training evidence, negative cash timing, and exposure utilization near `0.33`; trial 03 also has `source_target_count = 0` and negative return.

## Decision
- Verdict: `r52d screening-only failed confirmatory eligibility`.
- Do not start confirmatory, strict resume, promotion, live/default switch, or active artifact update from this tag.
- Keep r39 as the current effective continuous_policy evidence baseline.
- Next allowed work should target deployment/cash timing/exposure utilization/training evidence closure at the research-code objective or feedback level, not receiver mask repair or extra epochs on the same r52d evidence.
