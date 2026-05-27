# Alpha Multi-Horizon Stage 2.7 Target/Head Stability - 2026-05-27

## Verdict

- Status: `stage27_completed / evidence-grade target-loss stability / shadow-only / research-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Study family: `stage27_target_head_stability`.
- Verdict: Stage 2.7 completed the 9/9 target-normalization and horizon-head stability runs, but no candidate passed the Stage 2.7 stability gate.
- Decision: do not launch `patch_transformer`, `stock_mixer_sequence`, liquid800, allocator, replay, paper, live/default, or promotion from this evidence.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: this is fixed-GRU / fixed-input target-loss evidence only; it unlocks neither Stage 3 architecture review nor active execution changes.

## Evidence

- Study root: `daily_research/output/path_policy/studies/mh_stage27_target_head_stability_20260527_01/`.
- Dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Pool: `rolling_liquid500`.
- Fixed configuration: `gru_sequence_static_context`, `raw_kline_context_no_alpha_prior_v1`, `decision_utility_v1`, fullgrid horizons `1,2,3,5,8,10,15,20,30`.
- Stage 2.7 candidates: `horizon_target_normalized_v1`, `horizon_head_soft_constraint_v1`, `target_norm_head_constraint_v1`.
- Budget: `forecast_epochs=24`, `forecast_min_epochs=8`, `early_stop_patience=6`, `checkpoint_every_n_epochs=4`, CUDA.
- Training summary: `stage27_training_summary.json` status `completed`, completed tags `9`, failed tags `0`.
- Comparison summary: `stage27_comparison_summary.json` status `completed`, `stage3_architecture_allowed=false`, `stage3_candidates=[]`.
- Key artifacts:
  - `stage27_task_list.json`
  - `stage27_training_summary.json`
  - `stage27_comparison_summary.json`
  - `stage27_research_verdict.md`
  - `profile_aggregate.csv`
  - `score_variant_comparison.csv`
  - `output_profile_comparison.csv`
  - `architecture_input_interaction_report.json`

## Run Tags

- `mh27_horizon_target_normalized_v1_fullgrid_seed7_20260527_01`
- `mh27_horizon_target_normalized_v1_fullgrid_seed11_20260527_01`
- `mh27_horizon_target_normalized_v1_fullgrid_seed19_20260527_01`
- `mh27_horizon_head_soft_constraint_v1_fullgrid_seed7_20260527_01`
- `mh27_horizon_head_soft_constraint_v1_fullgrid_seed11_20260527_01`
- `mh27_horizon_head_soft_constraint_v1_fullgrid_seed19_20260527_01`
- `mh27_target_norm_head_constraint_v1_fullgrid_seed7_20260527_01`
- `mh27_target_norm_head_constraint_v1_fullgrid_seed11_20260527_01`
- `mh27_target_norm_head_constraint_v1_fullgrid_seed19_20260527_01`

## Results

`profile_aggregate.csv` test / `pred_decision_score` rows:

| Candidate | Seeds | Min rank IC | Min spread | Min hit lift | Mean monthly positive rate | Max negative months | Mean long horizon share | Mean 30d concentration | Mean future long share | Mean pred/future horizon gap | Validation-test rank gap | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `target_norm_head_constraint_v1` | 3 | `0.051075` | `0.024198` | `0.007642` | `0.757576` | `4` | `0.656694` | `0.637559` | `0.459360` | `15.049013` | `-0.038506` | fail |
| `horizon_target_normalized_v1` | 3 | `0.042855` | `0.020533` | `0.003791` | `0.757576` | `4` | `0.617081` | `0.604542` | `0.459360` | `14.853949` | `-0.042207` | fail |
| `horizon_head_soft_constraint_v1` | 3 | `0.043408` | `0.004748` | `0.003791` | `0.606061` | `5` | `0.938152` | `0.906181` | `0.459360` | `16.056359` | `0.005345` | fail |

Gate requirements were: 3 seeds all positive on rank IC, top-bottom spread, and hit lift; mean monthly positive rate `>=0.75`; max negative months `<=2`; long horizon share not above the Stage 2.6 baseline; 30d concentration materially reduced without rank/spread/hit collapse.

## Interpretation

- Target normalization and the combined target-normalization/head-constraint candidates both reduced 30d concentration sharply while preserving positive rank/spread/hit across all three seeds.
- They still failed on monthly stability: `max_negative_months=4`, above the gate limit `<=2`.
- The head soft constraint alone did not repair collapse: mean 30d concentration stayed `0.906181`, long horizon share rose above the Stage 2.6 baseline, and monthly stability worsened.
- Stage 2.7 therefore confirms that the bottleneck is not solved by simply normalizing horizon utility scale or softly constraining the head.
- The evidence suggests the current signal is real but still behaves like an unstable long-horizon utility family. Stage 3 architecture expansion would risk amplifying unresolved target/loss instability.

## Next Action

- Keep Stage 3 architecture review locked.
- Treat `target_norm_head_constraint_v1` and `horizon_target_normalized_v1` as useful diagnostic evidence, not deployable candidates.
- Next research should narrow or redesign the target family around `15/20/30d` long-horizon utility, with monthly regime diagnostics and seed-specific negative month analysis before any larger architecture.
- Continue research-only / shadow-only boundaries; no active artifact, allocator, replay, paper, live/default, or promotion action is authorized.
