# r56 Release-First Constrained Decoder Status 2026-05-13

## Scope
- Status: research / shadow-only.
- Branch: `main`.
- Production anchor: `daily_research/output/active_execution_strategy.json` remained unchanged.
- Active label remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.

## Fact / Inference / Assumption Split
- Fact: r56 code contract exists and focused unit/contract tests pass.
- Fact: dry run `self_opt_study_r56_release_first_constrained_decoder_dryrun_20260513_01` completed with confirmatory disabled and true solver disabled.
- Fact: safe screening tags `..._01`, `..._02`, and `..._03` failed before any completed protocol summary; tag `..._04` timed out and was stopped with no completed protocol summary.
- Fact: completed r56 safe-screening evidence is `0`.
- Fact: failed and timed-out attempts are diagnostics only and have `completed_evidence=0.0`.
- Inference: r56 allocator/source semantics cannot yet be judged from behavior metrics because no completed r56 protocol exists.
- Inference: the immediate bottleneck is r56 training runtime completion under AMP/day-set v46, not another loss-weight increase.
- Assumption: the next attempt remains research / shadow-only and must not touch live/default/promotion/active artifacts.

## Implemented Code Contract
- Added `daily_research/continuous_policy/allocation_core_v3.py`.
- Added release-first allocator API:
  - `ReleaseFirstAllocationConstraints`
  - `ReleaseFirstAllocationResult`
  - `solve_release_first_allocation_v3(...)`
- Added release-first semantic intent in `semantic_budget_intent.py`:
  - `derive_release_first_intent(...)`
- Added r56 profile:
  - `split_heads_portfolio_daily_release_first_constrained_decoder_r56`
  - `alpha_result_value_budget_split_v46`
- Added simulator mode:
  - `release_first_allocation_v3_mode`
  - `release_first_source_intent_count`
  - `release_first_source_realized_count`
  - `release_first_rotation_amount`
  - `release_first_cash_buffer_amount`
  - `release_first_block_reason`
- Kept r53-r55 v2 allocator path compatible; r56 uses v3 before v2 when explicitly enabled.
- Added AMP safety fixes discovered during screening:
  - dtype-safe mask fill values for half precision in day-set attention/native allocation.
  - classification losses compute from float32 logits.
  - validation loss is no longer wrapped in a broad autocast context.

## Verification
- Required focused test command passed after implementation:
  - `136 passed, 24 warnings`
- Active artifact guard passed:
  - `git diff -- daily_research/output/active_execution_strategy.json` produced no output.
- Doc guard passed:
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`

## Runs
- Dry run passed:
  - `self_opt_study_r56_release_first_constrained_decoder_dryrun_20260513_01`
  - confirmatory disabled; true solver disabled; safe resource profile resolved.
- Safe screening exact plan tag failed diagnostically:
  - `self_opt_study_r56_release_first_constrained_decoder_screening_safe_20260513_01`
  - completed trials: 0; failed trials: 1.
  - failure: AMP half mask overflow, `value cannot be converted to type at::Half without overflow`.
- Retry tag failed diagnostically:
  - `self_opt_study_r56_release_first_constrained_decoder_screening_safe_20260513_02`
  - completed trials: 0; failed trials: 1.
  - failure: AMP classification loss dtype mismatch, `expected scalar type Half but found Float`.
- Retry tag failed diagnostically:
  - `self_opt_study_r56_release_first_constrained_decoder_screening_safe_20260513_03`
  - completed trials: 0; failed trials: 1.
  - failure: validation loss under broad autocast, `binary_cross_entropy ... unsafe to autocast`.
- Retry tag timed out and was stopped:
  - `self_opt_study_r56_release_first_constrained_decoder_screening_safe_20260513_04`
  - no `protocol_summary.json`; no completed evidence.
  - process `10816` was terminated after exceeding 1 hour with heartbeat-only progress.

## Verdict
- r56 code contract is implemented and unit/contract tests pass.
- r56 has not produced completed safe-screening evidence.
- Failed and timed-out screening attempts must be treated as diagnostics only; `completed_evidence=0.0`.
- Do not enter strict resume, confirmatory, promotion, live/default, or active artifact changes.

## Next Diagnostic Focus
- The next blocker is training runtime completion under r56 day-set v46, not allocator/source semantics yet.
- Do not claim source/reduce/exit success or failure from r56, because no completed r56 protocol exists.
- Before another full safe screening, run a shorter r56 protocol or add progress/epoch-level diagnostics that prove AMP loss boundaries and runtime throughput are healthy.
