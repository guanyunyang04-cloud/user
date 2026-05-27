# Alpha Multi-Horizon Stage 2.6 Stability Root-Cause Calibration - 2026-05-27

## Verdict

- Status: `stage26_completed / cap80 diagnostic calibration / shadow-only / research-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Study family: `stage26_stability_root_cause`.
- Verdict: Stage 2.6 completed the root-cause audit and 9/9 calibration runs, but no candidate passed the Stage 2.6 stability gate.
- Decision: do not launch `patch_transformer`, `stock_mixer_sequence`, liquid800, allocator, replay, paper, live/default, or promotion from this evidence.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: this is cap80 fixed-GRU / fixed-input target-loss-score diagnostic evidence only; it does not prove full rolling_liquid500 monthly stability and cannot unlock architecture expansion.

## Evidence

- Study root: `daily_research/output/path_policy/studies/mh_stage26_stability_root_cause_20260527_01/`.
- Dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Intended pool: `rolling_liquid500`.
- Universe scope correction, 2026-05-27: Stage 2.6 training artifacts are `cap80_diagnostic`, not full rolling_liquid500 evidence. Example artifact `mh26_score_monthly_robust_v1_fullgrid_seed7_20260527_01` has `prepared_summary.universe_size=80`, `source_pool_view_id=""`, `feature_store_shape=[1699,80,156]`, and `train_rows=74640`.
- Fixed configuration: `gru_sequence_static_context`, `raw_kline_context_no_alpha_prior_v1`, `decision_utility_v1`, fullgrid horizons `1,2,3,5,8,10,15,20,30`.
- Root-cause audit inputs: Stage 2.5 `fullgrid_rebudget` and `daily1_45_multiseed`, seeds `7,11,19`.
- Stage 2.6 calibration candidates: `score_monthly_robust_v1`, `horizon_entropy_regularized_v1`, `risk_drawdown_reweighted_v1`.
- Budget: `forecast_epochs=24`, `forecast_min_epochs=8`, `early_stop_patience=6`, `checkpoint_every_n_epochs=4`, CUDA.
- Training summary: `stage26_training_summary.json` status `completed`, completed tags `9`, failed tags `0`.
- Comparison summary: `stage26_comparison_summary.json` status `completed`, `stage3_architecture_allowed=false`, `stage3_candidates=[]`.
- Key artifacts:
  - `stage26_root_cause_summary.json`
  - `stage26_root_cause_matrix.csv`
  - `stage26_root_cause_verdict.md`
  - `profile_aggregate.csv`
  - `score_variant_comparison.csv`
  - `horizon_grid_comparison.csv`
  - `stage26_research_verdict.md`

## Run Tags

- `mh26_score_monthly_robust_v1_fullgrid_seed7_20260527_01`
- `mh26_score_monthly_robust_v1_fullgrid_seed11_20260527_01`
- `mh26_score_monthly_robust_v1_fullgrid_seed19_20260527_01`
- `mh26_horizon_entropy_regularized_v1_fullgrid_seed7_20260527_01`
- `mh26_horizon_entropy_regularized_v1_fullgrid_seed11_20260527_01`
- `mh26_horizon_entropy_regularized_v1_fullgrid_seed19_20260527_01`
- `mh26_risk_drawdown_reweighted_v1_fullgrid_seed7_20260527_01`
- `mh26_risk_drawdown_reweighted_v1_fullgrid_seed11_20260527_01`
- `mh26_risk_drawdown_reweighted_v1_fullgrid_seed19_20260527_01`

## Budget And Completion

All 9 calibration runs completed with elevated budget and learning-curve evidence. None were 2-epoch smoke/scout runs, and none ended with best epoch贴近最后一轮.

| Candidate | Seeds completed | Epochs ran | Best epochs | Checkpoints / curve |
|---|---:|---|---|---|
| `score_monthly_robust_v1` | 3/3 | `15,8,10` | `9,1,4` | best + last checkpoint, learning curve |
| `horizon_entropy_regularized_v1` | 3/3 | `8,8,10` | `2,1,4` | best + last checkpoint, learning curve |
| `risk_drawdown_reweighted_v1` | 3/3 | `8,9,8` | `1,3,2` | best + last checkpoint, learning curve |

## Root-Cause Audit

Stage 2.5 read-only audit completed 6/6 retained runs.

| Candidate | Seeds | Mean test monthly positive | Max negative months | Mean predicted long share | Recommended path |
|---|---:|---:|---:|---:|---|
| `fullgrid_rebudget` | 3 | `0.582492` | `9` | `0.933294` | `loss_regularization_stabilization` |
| `daily1_45_multiseed` | 3 | `0.460000` | `7` | `0.792262` | `loss_regularization_stabilization` |

Facts:

- No common test negative month survived across all seeds; negative months are seed-specific.
- Fullgrid shows `true_long_horizon_edge`, `target_bias_to_long`, `horizon_head_collapse`, `short_feature_insufficient`, and `training_instability`.
- Daily grid reduces 30d-only collapse but still has long-horizon bias and weak monthly stability.
- The root cause is not just score aggregation; it remains target/loss/horizon-head stability under seed variation.

## Calibration Results

`profile_aggregate.csv` test / `pred_decision_score` rows:

| Candidate | Seeds | Mean rank IC | Min rank IC | Mean spread | Min spread | Mean hit lift | Min hit lift | Mean monthly positive rate | Min monthly positive rate | Max negative months | Mean long horizon share | All rank/spread/hit positive | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `risk_drawdown_reweighted_v1` | 3 | `0.068361` | `0.029313` | `0.014945` | `-0.004303` | `0.026698` | `0.020379` | `0.666667` | `0.636364` | `4` | `0.930766` | false | fail |
| `score_monthly_robust_v1` | 3 | `0.060450` | `0.037862` | `0.014519` | `0.010174` | `0.011295` | `0.005273` | `0.606061` | `0.545455` | `5` | `0.930174` | true | fail |
| `horizon_entropy_regularized_v1` | 3 | `0.059783` | `0.017666` | `0.013447` | `0.009000` | `0.015640` | `0.011197` | `0.575758` | `0.545455` | `5` | `0.951086` | true | fail |

Validation looked stronger than test for all three candidates. Test confirms utility signal remains positive in rank/spread/hit for `score_monthly_robust_v1` and `horizon_entropy_regularized_v1`, but monthly stability and horizon concentration still fail. `risk_drawdown_reweighted_v1` has the best mean rank/spread/hit profile, but seed7 test spread is negative, so it cannot pass.

## Interpretation

- Stage 2.6 answers a cap80 diagnostic question only: simple loss/score calibration did not stabilize the default capped fixed-GRU multi-horizon utility model.
- The cap80 diagnostic artifacts retain utility-native signal, but long-horizon concentration remains around `0.93-0.95` for fullgrid calibration candidates.
- The current bottleneck is still target/loss/horizon-head stability, not input availability or basic architecture capacity.
- The correct next move is not architecture expansion. It is a smaller root-cause loop around target normalization, horizon-head constraint, seed instability, and validation-test mismatch.

## Next Action

- Keep `fullgrid_rebudget` as the Stage 2.5 cap80 baseline and `score_monthly_robust_v1` / `horizon_entropy_regularized_v1` as signal-preserving but unstable cap80 calibration references.
- Do not copy these fullgrid candidates to daily `1..45`, because fullgrid did not pass.
- Before Stage 3 architecture review, require a full rolling_liquid500 calibration candidate with all three seeds positive on rank/spread/hit, mean monthly positive rate `>=0.75`, max negative months `<=2`, and no worsened horizon concentration.
- Continue with constrained target/loss work: reduce target bias to long, make horizon selection non-collapsed without forcing uniformity, and verify validation/test behavior before using test as confirmation.
