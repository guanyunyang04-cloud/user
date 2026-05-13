# r58 Framework Simplification Status 2026-05-13

## Facts
- Branch: `main`.
- Scope: continuous_policy research framework simplification only.
- Production anchor guard: `daily_research/output/active_execution_strategy.json` is not part of this change and must remain diff-free.
- The r57 parent/child protocol runner, subprocess watchdog, wall-time timeout, stale-progress timeout, and watchdog test entry were removed from the study runner path.
- Low-complexity diagnostics were retained: `runtime_progress.py`, `--protocol-progress-jsonl`, training progress events, in-process study progress events, and explicit study-tag collision protection.
- Search profile registry is now separated into `daily_research/continuous_policy/research_profile_registry.py`.
- New study CLI choices are restricted to active profiles: `focused_seq_v1`, r53, r54, r55, and r56.
- Legacy r19-r52 profile definitions remain available as archived compatibility data, but they are not default new-study entrypoints.

## Verification
- Required focused regression passed:
  `136 passed, 24 warnings`.
- Command:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_runtime_progress.py daily_research/continuous_policy/tests/test_training_runtime_acceleration.py daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py daily_research/continuous_policy/tests/test_research_registry_simplification.py -q`

## Interpretation
- The project had entered a framework-complexity loop: adding more runner layers and loss/profile variants was increasing operational surface without producing completed r56 behavior evidence.
- r58 deliberately does not judge strategy quality. It restores a smaller operating loop so r56/r59 research can be evaluated through completed protocol summaries rather than orchestration machinery.

## Boundaries
- r58 is `research / framework_simplification`.
- It is not a promotion, live/default change, confirmatory result, or active artifact change.
- Future runtime debugging should use direct foreground protocol smoke first, then dry-run study, then safe screening.
- Do not reintroduce parent/child subprocess watchdog behavior as the default study path without a new explicit design decision.
