# Alpha Multi-Horizon Stage 2 Calibration Status - 2026-05-25

## Verdict

- Status: `stage2_completed / shadow-only / research-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Verdict: Stage 2 horizon-grid calibration completed for `decision_utility_path_aux_v1` and `decision_utility_v1_baseline`. No non-daily multi-seed grid passed the Stage 3 architecture gate.
- Decision: do not launch `patch_transformer` / `stock_mixer_sequence` architecture expansion from this evidence. Continue score/loss/horizon calibration first.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Boundary: no production root, live/default, trade-plan, paper account, allocator/replay, or broker integration changed.

## Evidence

- Stage 1 calibration review root: `daily_research/output/path_policy/studies/mh_stage1_calibration_review_20260525_02/`.
- Stage 2 comparison root: `daily_research/output/path_policy/studies/mh_horizon_grid_calibration_20260525_02/`.
- Stage 2 driver: `daily_research/output/path_policy/studies/mh_horizon_grid_calibration_20260525_02/stage2_driver.py`.
- Progress: `daily_research/output/path_policy/studies/mh_horizon_grid_calibration_20260525_02/stage2_progress.json` reported `completed`, `task_count=14`, `failed_tags=[]`.
- Comparison artifacts:
  - `output_profile_comparison.csv`
  - `horizon_grid_comparison.csv`
  - `profile_aggregate.csv`
  - `score_variant_comparison.csv`
  - `next_stage_decision.json`
  - `calibration_review.md`
  - `architecture_input_interaction_report.json`
  - `research_verdict.md`

## Run Tags

Reused full-grid Stage 1 studies:

- `mh_out_decision_utility_v1_baseline_fullgrid_seed7_20260525_01`
- `mh_out_decision_utility_v1_baseline_fullgrid_seed11_20260525_01`
- `mh_out_decision_utility_v1_baseline_fullgrid_seed19_20260525_01`
- `mh_out_decision_utility_path_aux_v1_fullgrid_seed7_20260525_01`
- `mh_out_decision_utility_path_aux_v1_fullgrid_seed11_20260525_01`
- `mh_out_decision_utility_path_aux_v1_fullgrid_seed19_20260525_01`

New Stage 2 studies:

- `mh_grid_decision_utility_path_aux_v1_sparse_long_seed7_20260525_02`
- `mh_grid_decision_utility_path_aux_v1_sparse_long_seed11_20260525_02`
- `mh_grid_decision_utility_path_aux_v1_sparse_long_seed19_20260525_02`
- `mh_grid_decision_utility_path_aux_v1_dense_short_mid_seed7_20260525_02`
- `mh_grid_decision_utility_path_aux_v1_dense_short_mid_seed11_20260525_02`
- `mh_grid_decision_utility_path_aux_v1_dense_short_mid_seed19_20260525_02`
- `mh_grid_decision_utility_path_aux_v1_daily1_45_feas_seed7_20260525_02`
- `mh_grid_decision_utility_v1_baseline_sparse_long_seed7_20260525_02`
- `mh_grid_decision_utility_v1_baseline_sparse_long_seed11_20260525_02`
- `mh_grid_decision_utility_v1_baseline_sparse_long_seed19_20260525_02`
- `mh_grid_decision_utility_v1_baseline_dense_short_mid_seed7_20260525_02`
- `mh_grid_decision_utility_v1_baseline_dense_short_mid_seed11_20260525_02`
- `mh_grid_decision_utility_v1_baseline_dense_short_mid_seed19_20260525_02`
- `mh_grid_decision_utility_v1_baseline_daily1_45_feas_seed7_20260525_02`

## Results

`profile_aggregate.csv` test / `pred_decision_score` rows:

| Loss profile | Grid | Seeds | Daily feasibility | Mean rank IC | Min rank IC | Mean spread | Min spread | Mean hit lift | Min hit lift | Mean monthly positive rate | Max negative months | Stage 3 weak gate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `decision_utility_path_aux_v1` | `1..45` | 1 | true | `0.087711` | `0.087711` | `0.043338` | `0.043338` | `0.009885` | `0.009885` | `0.800000` | 2 | false |
| `decision_utility_path_aux_v1` | `1,2,3,5,8,10,15,20,30` | 3 | false | `0.054987` | `0.020521` | `0.017470` | `0.012713` | `0.017615` | `0.014159` | `0.606061` | 5 | false |
| `decision_utility_path_aux_v1` | `1,3,5,10,15,20,30,45` | 3 | false | `0.032397` | `0.021955` | `-0.014401` | `-0.019112` | `0.022917` | `0.015689` | `0.333333` | 7 | false |
| `decision_utility_v1_baseline` | `1,2,3,5,8,10,15,20,30` | 3 | false | `0.026608` | `-0.041701` | `0.009294` | `-0.016828` | `0.015936` | `0.010900` | `0.484848` | 7 | false |
| `decision_utility_path_aux_v1` | `1,2,3,4,5,8,10,15,20,30` | 3 | false | `0.022468` | `-0.020793` | `0.002568` | `-0.015461` | `0.015778` | `0.005509` | `0.484848` | 7 | false |
| `decision_utility_v1_baseline` | `1,2,3,4,5,8,10,15,20,30` | 3 | false | `0.021540` | `-0.015988` | `0.003050` | `-0.012776` | `0.015778` | `0.004325` | `0.454545` | 7 | false |
| `decision_utility_v1_baseline` | `1,3,5,10,15,20,30,45` | 3 | false | `0.017862` | `-0.002389` | `-0.014079` | `-0.021863` | `0.017815` | `0.009311` | `0.366667` | 7 | false |
| `decision_utility_v1_baseline` | `1..45` | 1 | true | `0.014753` | `0.014753` | `0.004085` | `0.004085` | `0.014349` | `0.014349` | `0.400000` | 6 | false |

Facts:

- `decision_utility_path_aux_v1` remains the strongest non-daily multi-seed candidate on average.
- Full grid `decision_utility_path_aux_v1` is the best multi-seed row, with all-seed positive rank IC / spread / hit lift, but monthly stability is below the weak gate: mean monthly positive rate `0.606061`, max negative months `5`.
- Sparse long grid degrades spread and monthly stability despite positive rank IC / hit lift.
- Dense short-mid grid is not an improvement; it introduces negative min rank IC and negative min spread.
- Daily `1..45` path-aux seed7 is promising as feasibility evidence only: test rank IC `0.087711`, spread `0.043338`, monthly positive rate `0.800000`, negative months `2`, long horizon share `0.653253`.
- Daily `1..45` must not unlock architecture or input expansion by itself because it is single-seed feasibility, not a multi-seed gate.
- `next_stage_decision.json` reports `stage3_architecture_allowed=false` and `stage3_candidates=[]`.

## Tooling Fix Recorded With This Evidence

- `output_aux_profile_comparison.py` now writes `daily_grid_feasibility` in `profile_aggregate.csv`.
- Stage 3 weak gate now requires `seed_count >= 3` and `daily_grid_feasibility=false`.
- Daily feasibility rows can continue as feasibility evidence but cannot populate `stage3_candidates`.

## Next Action

Continue calibration before architecture/input expansion:

- Prefer focused score/loss/horizon calibration around `decision_utility_path_aux_v1`.
- Investigate why full grid has positive all-seed rank/spread/hit but weak monthly stability.
- Treat daily `1..45` as a promising information-density clue; run multi-seed daily only if the next objective is explicitly feasibility-to-stability conversion.
- Do not promote or connect to active strategy from this evidence.

## Verification

- `test_output_aux_profile_comparison.py`: added coverage for path-only proxy comparison, score variants, aggregate gates, grid-separated aggregation, and daily feasibility not unlocking Stage 3.
- Stage 2 comparison regenerated after the gate fix.
- `daily_research/output/active_execution_strategy.json` expected diff: none.
