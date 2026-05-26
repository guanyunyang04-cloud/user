# Alpha Multi-Horizon Stage 2.5 Stability Calibration - 2026-05-26

## Verdict

- Status: `stage25_completed / evidence-grade calibration / shadow-only / research-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Verdict: Stage 2.5 completed 9/9 runs with seeds `7,11,19`; no candidate passed the Stage 2.5 stability gate.
- Decision: do not launch `patch_transformer`, `stock_mixer_sequence`, allocator, replay, live/default, paper, or promotion from this evidence.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: this is stability calibration evidence only; it confirms utility-native signal but does not prove enough monthly stability for architecture expansion.

## Evidence

- Study root: `daily_research/output/path_policy/studies/mh_stage25_stability_calibration_20260526_01/`.
- Driver: `daily_research/output/path_policy/studies/mh_stage25_stability_calibration_20260526_01/stage25_driver.py`.
- Driver summary: `daily_research/output/path_policy/studies/mh_stage25_stability_calibration_20260526_01/stage25_driver_summary.json`.
- Progress: `stage25_progress.json` reported `completed`, `task_count=9`, `failed_tags=[]`.
- Dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Pool: `rolling_liquid500`.
- Fixed configuration: `gru_sequence_static_context`, `raw_kline_context_no_alpha_prior_v1`, `decision_utility_v1`, `decision_utility_path_aux_v1`.
- Budget: `forecast_epochs=24`, `forecast_min_epochs=8`, `early_stop_patience=6`, `checkpoint_every_n_epochs=4`, CUDA.
- Comparison artifacts:
  - `profile_aggregate.csv`
  - `score_variant_comparison.csv`
  - `horizon_grid_comparison.csv`
  - `research_verdict.md`
  - `calibration_review.md`
  - `next_stage_decision.json`

## Study Tags

- `mh25_path_aux_fullgrid_rebudget_seed7_20260526_01`
- `mh25_path_aux_fullgrid_rebudget_seed11_20260526_01`
- `mh25_path_aux_fullgrid_rebudget_seed19_20260526_01`
- `mh25_path_aux_daily1_45_multiseed_seed7_20260526_01`
- `mh25_path_aux_daily1_45_multiseed_seed11_20260526_01`
- `mh25_path_aux_daily1_45_multiseed_seed19_20260526_01`
- `mh25_path_aux_long_family_simplified_seed7_20260526_01`
- `mh25_path_aux_long_family_simplified_seed11_20260526_01`
- `mh25_path_aux_long_family_simplified_seed19_20260526_01`

## Budget And Completion

All runs completed with valid elevated budget, and none were 2-epoch smoke/scout runs.

| Candidate | Seeds completed | Epochs ran | Best epochs | Stop reason |
|---|---:|---|---|---|
| `fullgrid_rebudget` | 3/3 | `8,9,10` | `2,3,4` | `early_stopping_patience_exhausted` |
| `daily1_45_multiseed` | 3/3 | `8,8,8` | `2,2,2` | `early_stopping_patience_exhausted` |
| `long_family_simplified` | 3/3 | `8,8,11` | `2,1,5` | `early_stopping_patience_exhausted` |

## Results

`profile_aggregate.csv` test / `pred_decision_score` rows:

| Candidate | Grid | Seeds | Mean rank IC | Min rank IC | Mean spread | Min spread | Mean hit lift | Min hit lift | Mean monthly positive rate | Min monthly positive rate | Max negative months | All rank/spread/hit positive | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `daily1_45_multiseed` | `1..45` | 3 | `0.063869` | `0.042002` | `0.025983` | `0.014961` | `0.009460` | `0.006059` | `0.633333` | `0.500000` | `5` | true | fail |
| `fullgrid_rebudget` | `1,2,3,5,8,10,15,20,30` | 3 | `0.069457` | `0.054849` | `0.014567` | `0.012139` | `0.018108` | `0.010900` | `0.636364` | `0.545455` | `5` | true | fail |
| `long_family_simplified` | `10,15,20,30,45` | 3 | `0.057170` | `0.035014` | `-0.006879` | `-0.018298` | `0.020642` | `0.015115` | `0.400000` | `0.400000` | `6` | false | fail |

Facts:

- `daily1_45_multiseed` and `fullgrid_rebudget` both keep all-seed positive test rank IC, top-bottom spread, and hit lift, so the utility-native signal is real.
- Neither candidate meets monthly stability: required mean monthly positive rate is `>=0.75`, but both are about `0.63`; required max negative months is `<=2`, but both reach `5`.
- `daily1_45_multiseed` has the strongest mean spread, but this converts the prior single-seed feasibility clue into multi-seed "promising but unstable" evidence, not a pass.
- `fullgrid_rebudget` has the strongest mean rank IC and hit lift, but it still fails monthly stability.
- `long_family_simplified` is rejected for the next stage because test spread is negative and monthly stability is worse.
- `research_verdict.md` reports `daily_grid_continue_feasibility_only` and `daily_grid_rejected`; `next_stage_decision.json` keeps `stage3_architecture_allowed=false` and `stage3_candidates=[]`.

## Horizon Calibration

Post-hoc horizon calibration was run for the top candidate `daily1_45_multiseed` seeds `7,11,19`.

| Seed | Baseline test rank IC | Baseline test spread | Baseline monthly positive rate | Baseline negative months | Calibration gate |
|---:|---:|---:|---:|---:|---|
| 7 | `0.087711` | `0.043338` | `0.800000` | `2` | `failed_regression` |
| 11 | `0.042002` | `0.019650` | `0.500000` | `5` | `failed_regression` |
| 19 | `0.061895` | `0.014961` | `0.600000` | `4` | `failed_regression` |

Facts:

- No post-hoc calibration profile preserved baseline validation/test quality across the candidate.
- The best-looking seed7 daily grid remains useful as a clue, but the multi-seed evidence says the broad daily horizon selector is not stable enough.
- Test should only confirm; these calibration results must not be used for repeated test-tuned scoring.

## Interpretation

- Stage 2.5 answers the intended question: stability did not improve enough under elevated budget.
- The bottleneck is no longer "insufficient experiment budget"; it is target/loss/monthly-stability behavior.
- The model is past smoke/scout for signal existence, but not past evidence-grade stability gate for architecture expansion.
- Current state: `research / shadow-only / no architecture / no promotion`.

## Next Action

- Do not expand model architecture or connect allocator/replay/live from this evidence.
- Next research should audit monthly failure modes by month/market regime and compare validation-to-test sign changes without tuning on test.
- Focus on target/loss/monthly stability root cause: utility target normalization, drawdown/risk shaping, horizon-selection regularization, and regime-robust score aggregation.
- Treat `daily1_45_multiseed` as feasibility-to-stability conversion material only, not a production or architecture gate pass.
- Keep `fullgrid_rebudget` as the cleaner multi-seed baseline for future stability diagnostics.
