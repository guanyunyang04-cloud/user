# Alpha Multi-Horizon Output/Aux/Grid Tooling - 2026-05-25

## Verdict

- Status: `research tooling implemented / shadow-only / no training launched`.
- Verdict: `alpha_multi_horizon_utility_policy_v1` now has code-level support for output/loss profile comparison, auxiliary utility tasks, explicit 45d horizon-grid feasibility, and read-only comparison artifacts.
- Boundary: `daily_research/output/active_execution_strategy.json`, production root, live/default, trade-plan, paper account, allocator/replay, and broker behavior are unchanged.

## Implemented Facts

- Added formal forecast loss profiles:
  - `forecast_path_v1_baseline`
  - `decision_utility_v1_baseline`
  - `decision_utility_path_aux_v1`
  - `decision_utility_hit_risk_aux_v1`
  - `decision_utility_rank_aux_v1`
- Added `forecast_loss_profile_contract(...)` so every profile records:
  - required output profile
  - primary objective
  - auxiliary objectives
  - explicit loss component weights
  - cumulative horizon grid
  - `shadow_only=true`
  - `promotion_allowed=false`
- Kept decision-utility profiles as utility-primary; auxiliary weights are lower than the main `decision_utility` component.
- Extended CLI validation so decision auxiliary loss profiles automatically require `decision_utility_v1` output.
- Added an explicit guard: horizon grids above 30d require explicit `--forecast-horizon 45`.
- Fixed a dynamic horizon prediction contract bug: `LinearLastDayPath20Forecaster` and `MLPLastDayPath20Forecaster` now expose `horizon`, so prediction export no longer falls back to 20d for 45d grids.
- Added `daily_research/path_policy/output_aux_profile_comparison.py` to generate:
  - `output_profile_comparison.csv`
  - `horizon_grid_comparison.csv`
  - `architecture_input_interaction_report.json`
  - `research_verdict.md`

## Scope Boundary

No real shadow training matrix was launched in this change. The code path is prepared for later long-task execution using the brain long-task contract and ETA-capable progress files.

`daily_grid 1..45` remains feasibility-only. A successful feasibility contract test means labels, model outputs, loss, and prediction columns can be produced; it does not mean daily horizon grid is superior.

## Verification

- `pytest daily_research/path_policy/tests/test_models.py daily_research/path_policy/tests/test_decision_score_diagnostics.py daily_research/path_policy/tests/test_target_calibration_audit.py daily_research/path_policy/tests/test_horizon_root_cause_audit.py daily_research/path_policy/tests/test_output_aux_profile_comparison.py -q`
  - Result: `30 passed`
- `pytest daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_decision_utility_forecast_contract daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_auxiliary_decision_loss_profiles_and_sets_decision_output daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_daily_grid_requires_explicit_45d_forecast_horizon -q`
  - Result: `3 passed`
- `pytest daily_research/path_policy/tests/test_forecast_training.py -q`
  - Result: `22 passed`
  - PID-bound wait used: PID `4456`, elapsed about `13m18s`
- `pytest daily_research/path_policy/tests/test_forecast_dataset.py daily_research/path_policy/tests/test_forecast_memmap_dataset.py -q`
  - Result: `12 passed`
  - PID-bound wait used: PID `25864`, elapsed about `10m51s`
- `git diff -- daily_research/output/active_execution_strategy.json`
  - Result: empty
- `git diff --check`
  - Result: clean
- `python -m tools.brain.doc_guard check`
  - Result: errors `0`
- `python -m tools.brain.integrity_check --json`
  - Result: errors `0`, warnings are acknowledged non-truth catalog entries.

## Next Research Action

Use the implemented profiles and comparison tool to run staged shadow-only experiments. The first executable stage should compare the output/loss profiles on the fixed dataset and full horizon grid, using explicit tags and long-task monitoring. Do not start architecture or input-feature comparisons until output/profile evidence passes the multi-seed gate.
