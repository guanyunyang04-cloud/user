# Alpha Multi-Horizon Output/Aux/Grid Stage 1 Shadow - 2026-05-25

## Verdict

- Status: `stage1_completed / shadow-only / research-only`.
- Verdict: full-grid output/loss comparison completed for 5 profiles x 3 separate seeds. No profile passed the multi-seed continuation gate; Stage 2 horizon-grid expansion was not launched.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Boundary: `daily_research/output/active_execution_strategy.json`, production root, live/default, trade-plan, paper account, allocator/replay, and broker behavior are unchanged.

## Evidence

- Comparison root: `daily_research/output/path_policy/studies/mh_output_aux_grid_full_comparison_20260525_01/`.
- Comparison artifacts:
  - `output_profile_comparison.csv`
  - `horizon_grid_comparison.csv`
  - `architecture_input_interaction_report.json`
  - `research_verdict.md`
  - `target_calibration_audit.json`
  - `decision_score_diagnostics.json`
- Run log: `daily_research/output/path_policy/studies/mh_output_aux_grid_full_comparison_20260525_01_stage1_run_log.jsonl`.
- Study tags:
  - `mh_out_forecast_path_v1_baseline_fullgrid_seed7_20260525_01`
  - `mh_out_forecast_path_v1_baseline_fullgrid_seed11_20260525_01`
  - `mh_out_forecast_path_v1_baseline_fullgrid_seed19_20260525_01`
  - `mh_out_decision_utility_v1_baseline_fullgrid_seed7_20260525_01`
  - `mh_out_decision_utility_v1_baseline_fullgrid_seed11_20260525_01`
  - `mh_out_decision_utility_v1_baseline_fullgrid_seed19_20260525_01`
  - `mh_out_decision_utility_path_aux_v1_fullgrid_seed7_20260525_01`
  - `mh_out_decision_utility_path_aux_v1_fullgrid_seed11_20260525_01`
  - `mh_out_decision_utility_path_aux_v1_fullgrid_seed19_20260525_01`
  - `mh_out_decision_utility_hit_risk_aux_v1_fullgrid_seed7_20260525_01`
  - `mh_out_decision_utility_hit_risk_aux_v1_fullgrid_seed11_20260525_01`
  - `mh_out_decision_utility_hit_risk_aux_v1_fullgrid_seed19_20260525_01`
  - `mh_out_decision_utility_rank_aux_v1_fullgrid_seed7_20260525_01`
  - `mh_out_decision_utility_rank_aux_v1_fullgrid_seed11_20260525_01`
  - `mh_out_decision_utility_rank_aux_v1_fullgrid_seed19_20260525_01`

## Results

All 15 studies finished with `forecast_training_summary.status=completed`.

| Loss profile | Mean test rank IC | Mean test spread | Mean hit lift | Mean monthly positive rate | Mean negative months | Gate-pass seeds |
|---|---:|---:|---:|---:|---:|---:|
| `decision_utility_path_aux_v1` | `0.054987` | `0.017470` | `0.017615` | `0.606061` | `4.33` | `0` |
| `decision_utility_hit_risk_aux_v1` | `0.046616` | `0.009422` | `0.023045` | `0.545455` | `5.00` | `0` |
| `decision_utility_rank_aux_v1` | `0.033037` | `0.011119` | `0.012776` | `0.575758` | `4.67` | `0` |
| `decision_utility_v1_baseline` | `0.026608` | `0.009294` | `0.015936` | `0.484848` | `5.67` | `0` |
| `forecast_path_v1_baseline` | `-0.047931` | `-0.028419` | `0.005371` | `0.212121` | `8.67` | `0` |

Facts:
- `decision_utility_path_aux_v1` had the best mean test rank IC and spread, but did not pass monthly stability.
- `forecast_path_v1_baseline` required `decision_score_source=path_proxy` in comparison artifacts and remained clearly weaker than utility-native profiles.
- `architecture_input_interaction_report.json` reported research verdict `utility_baseline_keep` only; it did not report aux continuation or architecture compare allowance.
- `target_calibration_audit.json` completed for all 15 studies with `gate_a.passed=true` and conclusion hint `target_calibration_gate_passed`.
- `decision_score_diagnostics.json` completed for all 15 studies with conclusion hint `selection_calibration_candidate_found`.

## Next Action

Do not launch Stage 2 horizon-grid expansion from this evidence alone. The next useful research step is input/target redesign or a narrower stability-focused utility variant, because the current output/loss profiles preserve positive average signal in several utility-native variants but fail the monthly stability gate.
