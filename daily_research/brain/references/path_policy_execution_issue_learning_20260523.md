# PathPolicy Execution Issue Learning - 2026-05-23

## Scope
- Status: `brain / path_policy execution reliability / self-evolution`.
- Mainline context: `alpha_multi_horizon_utility_policy_v1`.
- This note records an execution lesson, not model evidence, promotion evidence, live/default authority, or active artifact authority.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Facts
- `test_forecast_dataset.py` is slow because its synthetic fixtures use 700, 820, or 900 business days and call `build_forecast_sequence_dataset()`, which builds forecast features, labels, cumulative horizon labels, and horizon-specific risk labels before sample caps are applied.
- The dynamic-horizon dataset contract test passed individually, but full-file execution can exceed short 5-minute timeouts.
- Parallel pytest timeout left residual pytest processes during the earlier run; stacking new pytest runs on top of old ones can create self-inflicted resource pressure.
- Direct script execution of `daily_research/path_policy/run_alpha_path20_protocol.py --help` failed before the fix because package imports could not resolve `daily_research`; module execution with `-m daily_research.path_policy.run_alpha_path20_protocol --help` worked.
- The standard project Python remains `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`.

## Inferences
- Treating timeout as a test failure without checking process state, logs, and per-test runtime can create false negatives and unnecessary code churn.
- Writing only a brain note would not prevent recurrence: the entrypoint failure and verification over-selection were both fixable in code.
- Selective verification should recommend fast forecast contract tests by default and keep full slow dataset/training tests in `deferred_long_commands`.

## Assumptions
- This round fixes execution reliability only; it does not start a real study, training, allocator, replay, live/default, promotion, or active artifact change.
- Slow tests remain valuable and should not be deleted; they should be routed as long validation when risk or release scope requires them.
- Direct script support is compatibility hardening; package `-m` execution remains the documented standard.

## Root Causes
- `test_forecast_dataset.py` is an integration-style dataset builder test despite using small stock counts; date span and full feature/label construction dominate runtime.
- Residual pytest processes were not a product bug, but an execution hygiene problem after timeout.
- Direct script import failed because the repo root was not inserted into `sys.path` when `__package__` was empty.

## Fixes
- `tools.brain.selective_verification` now maps forecast file changes to fast forecast contract tests and defers full `test_forecast_dataset.py` / `test_forecast_training.py`.
- `tools.brain.selective_verification` now includes `test_run_alpha_path20_protocol_entrypoint.py` when `run_alpha_path20_protocol.py` changes.
- `daily_research/path_policy/run_alpha_path20_protocol.py` now bootstraps the repo root for direct script execution.
- `tools.brain.adapters.daily_research_evidence` now indexes `path_policy_*.md` reference names so this note is machine-queryable.
- `daily_research/brain/operations_center.md` records safe residual pytest inspection and cleanup boundaries.

## Standard Commands
- Recommended run protocol help:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol --help`
- Direct script smoke:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/path_policy/run_alpha_path20_protocol.py --help`
- Selective verification:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow verify-plan --json`
- Safe inspect-only residual pytest query:
  `Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*pytest*daily_research/path_policy/tests/test_forecast_dataset.py*' }`

## Next Allowed Actions
- Use selective verification fast contract tests before long PathPolicy validation.
- Run deferred long forecast tests only when the change touches dataset/training behavior deeply, before release-style completion, or when the user explicitly asks for full validation.
- If another timeout occurs, first inspect residual processes and per-test runtime; do not change code until the slow or failing node is isolated.
- Keep standard command docs on the `-m daily_research.path_policy.run_alpha_path20_protocol` entrypoint.

## Validation
- Add/keep smoke coverage for both module and direct script `--help`.
- Add/keep selective verification coverage so forecast changes select fast tests and full slow tests move to `deferred_long_commands`.
- Rebuild `daily_research/brain/references/evidence_registry.json` after this reference is added.
