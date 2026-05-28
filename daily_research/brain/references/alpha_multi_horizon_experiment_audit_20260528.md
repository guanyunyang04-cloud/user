# Alpha Multi-Horizon Experiment Audit - 2026-05-28

## Verdict

- Status: `multi_horizon_evidence_audit / shadow-only / no active change`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Audit target: all retained multi-horizon research artifacts from the 2026-05-23 full-pool discovery through Stage 3A-3C architecture/input scout.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Executive Corrections

1. Early `mh_out_*`, `mh_grid_*`, Stage 2.5, Stage 2.6, and Stage 2.7 artifacts are all `cap80_diagnostic`, not full rolling_liquid500 evidence.
   - Current audit count: `mh_out_*` 15 + `mh_grid_*` 14 + Stage 2.5 9 + Stage 2.6 9 + Stage 2.7 9 = `56` cap80 individual training artifacts.
   - Typical scope: `universe_size=80`, `train_rows=73440/74640`, empty pool view, `feature_store_shape=[1699,80,156]`.
   - These results remain useful for plumbing and relative diagnostics, but cannot support full-pool stability, monthly gate, or architecture-readiness conclusions.
2. The 2026-05-27 universe-scope correction correctly identified the cap80 mechanism, but its written affected-study list was too narrow: it named Stage 2.5-2.7, while `mh_out_*` and `mh_grid_*` are also cap80.
3. The 2026-05-24 target-family single-seed note used `forecast_test_confirmed` language for mid/long target families. That is valid only as an internal single-seed forecast verdict. The comparison CSV shows `gate_strength=failed_or_incomplete` for short, mid, and long, so none of the three target-family runs should be read as a stability/gate pass.
4. No retained evidence proves that `30d` is globally optimal or that horizons above `30d` are worse on full rolling_liquid500. The safe conclusion is: within the tested full-pool grids, the signal tilts strongly toward `20d/30d`; old 30d concentration was partly target/loss/head-pressure bias; `target_norm_head_constraint_v1` repairs concentration enough to pass Stage 2.8.
5. The current final research conclusion is still valid: keep the Stage 3 baseline as `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1`; do not upgrade to patch/stock/slot architecture and do not promote anything to execution.

## Scope Inventory

| Artifact family | Count | Scope | Universe / rows | Interpretation |
|---|---:|---|---|---|
| `path20_horizon_discovery...20260523_01` | 1 | `full_rolling_liquid500` | `2430`, `452034` train rows | First full-pool no-alpha GRU discovery |
| `mh_short/mid/long_utility_*` | 3 | `full_rolling_liquid500` | `2430`, `452034-464418` train rows | Single-seed target-family controls |
| `mh_out_*` | 15 | `cap80_diagnostic` | `80`, `74640` train rows | Output/loss cap80 diagnostic only |
| `mh_grid_*` | 14 | `cap80_diagnostic` | `80`, `73440/74640` train rows | Horizon-grid cap80 diagnostic only |
| `mh25_*` | 9 | `cap80_diagnostic` | `80`, `73440/74640` train rows | Stage 2.5 stability diagnostic only |
| `mh26_*` | 9 | `cap80_diagnostic` | `80`, `74640` train rows | Stage 2.6 target/loss diagnostic only |
| `mh27_*` | 9 | `cap80_diagnostic` | `80`, `74640` train rows | Stage 2.7 target/head diagnostic only |
| `mh28_*` | 9 | `full_rolling_liquid500` | `2430`, `452034` train rows | Evidence-grade full-pool target/loss revalidation |
| `mh30/mh31/mh32_*` | 7 retained summaries | `full_rolling_liquid500` | `2430`, `452034` train rows | Stage 3 architecture/input scout and confirmation |

## Full-Pool Discovery And Diagnostics

### Initial Full-Pool GRU Discovery

- Study tag: `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`.
- Configuration: `raw_kline_context_no_alpha_prior_v1`, `gru_sequence_static_context`, `decision_utility_v1`, horizons `1,2,3,5,8,10,15,20,30`, seed `7`, CUDA AMP, `epochs=16`, `min_epochs=8`.
- Completion: `epochs_ran=8`, `best_epoch=2`, `evidence_verdict=forecast_test_confirmed`.
- Validation `trade_utility_score`: rank IC `0.113993`, spread `0.028361`, hit lift `0.017385`, monthly positive rate `0.818182`, negative months `2`.
- Test `trade_utility_score`: rank IC `0.099703`, spread `0.040183`, hit lift `0.022097`, monthly positive rate `1.000000`, negative months `0`.
- Test horizon diagnostics strengthened with longer horizons: `5d` rank/spread `0.030911/0.007263`, `10d` `0.045247/0.013858`, `20d` `0.064482/0.028690`, `30d` `0.087132/0.045881`.
- Audit judgment: valid evidence for a full-pool long-horizon utility signal. Not valid evidence for a balanced horizon chooser, because predicted best horizon collapsed heavily toward `30d`.

### Post-Hoc Horizon Calibration

- Study root: `mh_utility_horizon_calibration_posthoc_20260524_01`.
- Baseline test: rank IC `0.099703`, spread `0.040183`, hit lift `0.022097`, monthly positive rate `1.000000`, long horizon share `0.962485`, 30d concentration `0.957246`.
- `top2_blend` reduced test 30d concentration to `0.589367` but rank/spread fell to `0.099012/0.038974`.
- Softmax and long-blend variants reduced concentration but failed validation or regression constraints.
- Audit judgment: the `failed_regression` conclusion is correct; post-hoc score calibration did not safely repair horizon collapse.

### Root-Cause Audit

- Study root: `mh_utility_horizon_root_cause_audit_20260524_01`.
- Full-pool test single-horizon diagnostics showed `30d` strongest among tested horizons: rank IC `0.087132`, spread `0.045881`, hit lift `0.033476`, monthly positive rate `0.727273`, negative months `3`.
- `1d` was weak/negative in comparison: validation rank IC `-0.031868`, test rank IC `-0.029043`.
- Audit judgment: `true_long_horizon_edge + target_bias_to_long + horizon_head_collapse` is directionally supported. The phrase "true long-horizon edge" should be read as "strongest within this tested target/grid", not as proof that 30d is globally optimal.

## Target-Family Shadow Controls

- Study tags: `mh_short_utility_1_3_5d_v1`, `mh_mid_utility_5_10_20d_v1`, `mh_long_utility_15_20_30d_v1`.
- Scope: all three are `full_rolling_liquid500`.
- Configuration: `raw_kline_context_no_alpha_prior_v1`, `gru_sequence_static_context`, `decision_utility_v1`, seed `7`, CUDA AMP, `epochs=16`, `min_epochs=8`.

| Target family | Epochs / best | Validation rank / spread / hit | Test rank / spread / hit | Test monthly / negatives | CSV gate |
|---|---:|---:|---:|---:|---|
| `1,3,5` short | `8 / 1` | `0.013986 / 0.005116 / 0.000865` | `0.029569 / 0.005598 / -0.001870` | `0.750000 / 3` | `failed_or_incomplete` |
| `5,10,20` mid | `8 / 2` | `0.086660 / 0.010785 / 0.040938` | `0.036099 / 0.010266 / 0.005346` | `0.750000 / 3` | `failed_or_incomplete` |
| `15,20,30` long | `9 / 3` | `0.094242 / 0.025083 / 0.038625` | `0.095089 / 0.041296 / 0.021996` | `0.727273 / 3` | `failed_or_incomplete` |

Audit judgment:

- Correct: short target is weak; long target is the strongest single-seed candidate.
- Needs qualification: mid/long `forecast_test_confirmed` is not a stability pass. All three target-family controls failed the comparison gate and remained single-seed evidence.
- Correct next-step logic at the time: multi-seed stability was required before architecture expansion.

## Cap80 Diagnostic Stages

### Stage 1 Output/Aux Grid

- Study root: `mh_output_aux_grid_full_comparison_20260525_01`.
- Scope correction: 15 individual `mh_out_*` runs are `cap80_diagnostic`.
- Configuration: 5 loss/output profiles x seeds `7,11,19`, fixed GRU/raw input/fullgrid, default capped universe.

| Loss profile | Mean test rank IC | Mean test spread | Mean hit lift | Mean monthly positive | Max negative months | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| `decision_utility_path_aux_v1` | `0.054987` | `0.017470` | `0.017615` | `0.606061` | `5` | best cap80 average, unstable |
| `decision_utility_hit_risk_aux_v1` | `0.046616` | `0.009422` | `0.023045` | `0.545455` | `7` | unstable |
| `decision_utility_rank_aux_v1` | `0.033037` | `0.011119` | `0.012776` | `0.575758` | `6` | unstable |
| `decision_utility_v1_baseline` | `0.026608` | `0.009294` | `0.015936` | `0.484848` | `7` | unstable |
| `forecast_path_v1_baseline` | `-0.047931` | `-0.028419` | `0.005371` | `0.212121` | `10` | rejected |

Audit judgment: no continuation-gate pass is correct. Any wording implying full-pool output/loss evidence is incorrect; this stage is cap80 diagnostic only.

### Stage 2 Horizon Grid

- Study root: `mh_horizon_grid_calibration_20260525_02`.
- Scope correction: 14 new `mh_grid_*` runs plus reused `mh_out_*` runs are `cap80_diagnostic`.
- Key `pred_decision_score` rows:

| Candidate | Grid | Seeds | Rank IC mean/min | Spread mean/min | Hit mean/min | Monthly mean/min | Max negative | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `path_aux` | `1..45` | 1 | `0.087711 / 0.087711` | `0.043338 / 0.043338` | `0.009885 / 0.009885` | `0.800000 / 0.800000` | `2` | fail, single-seed feasibility only |
| `path_aux` | `1,2,3,5,8,10,15,20,30` | 3 | `0.054987 / 0.020521` | `0.017470 / 0.012713` | `0.017615 / 0.014159` | `0.606061 / 0.545455` | `5` | fail |
| `decision_utility_v1` | `1,2,3,5,8,10,15,20,30` | 3 | `0.026608 / -0.041701` | `0.009294 / -0.016828` | `0.015936 / 0.010900` | `0.484848 / 0.363636` | `7` | fail |

Audit judgment: "no Stage 3 architecture from Stage 2" is correct. The `1..45` result is only a cap80 single-seed feasibility clue, not evidence that horizons above 30d were properly tested on full-pool.

### Stage 2.5 Stability Calibration

- Scope: `cap80_diagnostic`, 9/9 completed.
- Budget: `forecast_epochs=24`, `forecast_min_epochs=8`, `early_stop_patience=6`, `checkpoint_every_n_epochs=4`, CUDA.

| Candidate | Grid | Rank IC mean/min | Spread mean/min | Hit mean/min | Monthly mean/min | Max negative | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `daily1_45_multiseed` | `1..45` | `0.063869 / 0.042002` | `0.025983 / 0.014961` | `0.009460 / 0.006059` | `0.633333 / 0.500000` | `5` | fail |
| `fullgrid_rebudget` | `1,2,3,5,8,10,15,20,30` | `0.069457 / 0.054849` | `0.014567 / 0.012139` | `0.018108 / 0.010900` | `0.636364 / 0.545455` | `5` | fail |
| `long_family_simplified` | `10,15,20,30,45` | `0.057170 / 0.035014` | `-0.006879 / -0.018298` | `0.020642 / 0.015115` | `0.400000 / 0.400000` | `6` | fail |

Audit judgment: existing Stage 2.5 conclusion is correct after its own scope correction: cap80 signal exists, monthly stability fails, no architecture unlock.

### Stage 2.6 Stability Root-Cause Calibration

- Scope: `cap80_diagnostic`, 9/9 completed.
- Budget: `24 / min 8 / patience 6 / checkpoint 4 / CUDA`.

| Candidate | Rank IC mean/min | Spread mean/min | Hit mean/min | Monthly mean/min | Max negative | Long share | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `risk_drawdown_reweighted_v1` | `0.068361 / 0.029313` | `0.014945 / -0.004303` | `0.026698 / 0.020379` | `0.666667 / 0.636364` | `4` | `0.930766` | fail |
| `score_monthly_robust_v1` | `0.060450 / 0.037862` | `0.014519 / 0.010174` | `0.011295 / 0.005273` | `0.606061 / 0.545455` | `5` | `0.930174` | fail |
| `horizon_entropy_regularized_v1` | `0.059783 / 0.017666` | `0.013447 / 0.009000` | `0.015640 / 0.011197` | `0.575758 / 0.545455` | `5` | `0.951086` | fail |

Audit judgment: existing conclusion is correct: simple score/loss calibration did not fix monthly stability or long-horizon concentration on cap80.

### Stage 2.7 Target/Head Stability

- Scope: `cap80_diagnostic`, 9/9 completed.
- Budget: `24 / min 8 / patience 6 / checkpoint 4 / CUDA`.
- Gate score: `pred_decision_score`.

| Candidate | Rank IC mean/min | Spread mean/min | Hit mean/min | Monthly mean/min | Max negative | Long share | 30d concentration | Gap | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `horizon_target_normalized_v1` | `0.089155 / 0.042855` | `0.034566 / 0.020533` | `0.012283 / 0.003791` | `0.757576 / 0.636364` | `4` | `0.617081` | `0.604542` | `14.853949` | fail |
| `target_norm_head_constraint_v1` | `0.091762 / 0.051075` | `0.036049 / 0.024198` | `0.013369 / 0.007642` | `0.757576 / 0.636364` | `4` | `0.656694` | `0.637559` | `15.049013` | fail |
| `horizon_head_soft_constraint_v1` | `0.074777 / 0.043408` | `0.016388 / 0.004748` | `0.014554 / 0.003791` | `0.606061 / 0.545455` | `5` | `0.938152` | `0.906181` | `16.056359` | fail |

Audit judgment: existing conclusion is correct. Target normalization reduced concentration and preserved positive rank/spread/hit, but monthly stability failed on cap80; full-pool revalidation was required.

## Full-Pool Revalidation And Stage 3

### Stage 2.8 Full Rolling-Liquid500 Revalidation

- Study root: `mh_stage28_full_pool_revalidation_20260527_01`.
- Scope: `full_rolling_liquid500`, 9/9 completed.
- Configuration: full-pool memmap, `raw_kline_context_no_alpha_prior_v1`, `gru_sequence_static_context`, `decision_utility_v1`, horizons `1,2,3,5,8,10,15,20,30`, seeds `7,11,19`.
- Budget: `24 / min 8 / patience 6 / checkpoint 4 / CUDA`.
- Gate score: `pred_decision_score`.

| Candidate | Rank IC mean/min | Spread mean/min | Hit mean/min | Monthly mean/min | Max negative | Long share | 30d concentration | Gap | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `decision_utility_v1` | `0.085812 / 0.083041` | `0.031648 / 0.026515` | `0.016155 / 0.004281` | `0.787879 / 0.727273` | `3` | `0.912850` | `0.884804` | `16.221907` | fail |
| `horizon_target_normalized_v1` | `0.106268 / 0.101335` | `0.040972 / 0.035339` | `0.016523 / 0.009562` | `0.939394 / 0.818182` | `2` | `0.825395` | `0.819793` | `15.816018` | pass |
| `target_norm_head_constraint_v1` | `0.097677 / 0.083206` | `0.037694 / 0.028098` | `0.016526 / 0.010771` | `0.878788 / 0.818182` | `2` | `0.815462` | `0.776934` | `15.618355` | pass |

Audit judgment:

- Existing Stage 2.8 conclusion is correct.
- `horizon_target_normalized_v1` passed with the highest rank/spread; `target_norm_head_constraint_v1` was the correct Stage 3 baseline because it reduced 30d concentration and pred/future gap more while preserving the gate.
- The result supports a real `20d/30d` utility family plus a repaired target/loss surface. It does not prove that a broad horizon chooser is solved, and it does not authorize execution promotion.

### Stage 3A-3C Architecture/Input Scout

- Study root: `mh_stage30_arch_input_scout_20260528_01`.
- Scope: `full_rolling_liquid500`.
- Fixed target/loss: `target_norm_head_constraint_v1`; output `decision_utility_v1`; horizons `1,2,3,5,8,10,15,20,30`.
- Stage 3A budget: `16 / min 6 / patience 4 / checkpoint 4 / CUDA AMP`; seed `7`.
- Stage 3B/3C budget: `24 / min 8 / patience 6`; seeds `11` then `19`.
- Sector branch blocker: `raw_kline_context_sector_v1` full-pool anchor had `sector_context_feature_count=0`; sector input was blocked and is not evidence against sector information.

Stage 3A scout leaderboard:

| Candidate | Composite | Validation gate | Test veto | Canonical Stage 3B |
|---|---:|---|---|---|
| `patch_transformer_static_context + raw` | `1.283495` | pass | no | yes |
| `sector_slot_mixer_sequence + raw` | `0.957194` | pass | no | no after corrected selection |
| `stock_mixer_sequence + raw` | `0.697762` | pass | no | no |

Final Stage 3C comparison, `pred_decision_score`:

| Metric | Stage 2.8 GRU baseline | Patch candidate | Audit |
|---|---:|---:|---|
| Test rank IC mean | `0.097677` | `0.099253` | slight patch gain |
| Test spread mean | `0.037694` | `0.049589` | patch gain |
| Test hit lift mean | `0.016526` | `0.002137` | patch worse |
| Test hit lift min | `0.010771` | `-0.000133` | patch fails all-seed positivity |
| Test monthly positive mean | `0.878788` | `0.787879` | patch still above mean threshold |
| Max negative months | `2` | `3` | patch fails final gate |
| Long horizon share | `0.815462` | `0.916702` | patch worse |
| 30d concentration | `0.776934` | `0.916702` | patch worse |
| Pred/future gap | `15.618355` | `16.871921` | patch worse |

Audit judgment: existing Stage 3 conclusion is correct. Patch was a good scout but failed final evidence-grade upgrade. Keep GRU/raw/target-normalized-head-constrained baseline.

## Conclusion Integrity Check

| Prior / possible conclusion | Audit status | Correct wording |
|---|---|---|
| "Initial full-pool no-alpha GRU discovered useful multi-horizon utility signal." | valid | Keep. It was full-pool and test-confirmed on seed7. |
| "30d is the strongest tested horizon." | valid only within tested full-pool grids | Say "strongest among tested 1-30d full-pool diagnostics", not globally optimal. |
| "30d is proven best; shorter and longer horizons are worse." | invalid | Longer-than-30d full-pool evidence is insufficient; cap80 45d diagnostics cannot settle this. |
| "Mid/long target-family controls were test-confirmed." | needs qualification | They were single-seed forecast-test-confirmed but all had `gate_strength=failed_or_incomplete`; no stability pass. |
| "Stage 1/2 output/loss/horizon-grid results are multi-seed evidence." | valid only as cap80 | They are multi-seed cap80 diagnostics, not full-pool evidence. |
| "Stage 2.5-2.7 failed and should not unlock architecture." | valid | Keep. Also classify as cap80 diagnostic only. |
| "Stage 2.7 target normalization reduced concentration but failed monthly stability." | valid | Keep, but cap80 only. |
| "Stage 2.8 passed full-pool target/loss gate with `target_norm_head_constraint_v1`." | valid | Keep. This is the current evidence-grade baseline. |
| "Stage 3A single-seed scout can upgrade architecture." | invalid and not retained | Stage 3A is scout only; Stage 3C rejected patch upgrade. |
| "Sector input is useless." | invalid | Sector branch was blocked because sector context count was zero. |

## Current Safe Research State

- Evidence-grade fixed target/loss baseline: `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1`.
- Evidence-grade pass source: Stage 2.8 full rolling_liquid500, 3 seeds.
- Architecture upgrade: not allowed after Stage 3C; patch failed final gate.
- Input conclusion: raw no-alpha remains the only clean full-pool input baseline; sector input still needs construction repair before evaluation.
- Horizon conclusion: use `20d/30d` tilt as a signal-family fact; do not claim 30d global optimality or broad horizon chooser maturity.
- Boundary: no allocator, replay, paper, live/default, broker, production root, active artifact, or promotion action is authorized by any multi-horizon evidence reviewed here.

## Verification Performed

- Recomputed scope with `daily_research.path_policy.stage_universe_scope.study_scope_from_summary_path`.
- Parsed comparison CSV/JSON for:
  - `mh_target_function_shadow_seed7_comparison_20260524_01`
  - `mh_output_aux_grid_full_comparison_20260525_01`
  - `mh_horizon_grid_calibration_20260525_02`
  - `mh_stage25_stability_calibration_20260526_01`
  - `mh_stage26_stability_root_cause_20260527_01`
  - `mh_stage27_target_head_stability_20260527_01`
  - `mh_stage28_full_pool_revalidation_20260527_01`
  - `mh_stage30_arch_input_scout_20260528_01`
- Confirmed `daily_research/output/active_execution_strategy.json` is not part of this evidence update.
