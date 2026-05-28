# Alpha Multi-Horizon Stage 2.8 Full Rolling-Liquid500 Revalidation - 2026-05-28

## Verdict

- Status: `stage28_completed / full_rolling_liquid500 revalidation / shadow-only / research-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Study family: `stage28_full_pool_revalidation`.
- Verdict: Stage 2.8 completed the 9/9 full rolling_liquid500 revalidation runs. `target_norm_head_constraint_v1` passed the full-pool gate.
- Decision: Stage 3 architecture review is now allowed as a planning step, still shadow-only. This does not authorize allocator, replay, paper, liquid800, live/default, active artifact changes, or promotion.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: this is full-pool research evidence for the fixed GRU/raw-input multi-horizon utility family.

## Evidence

- Study root: `daily_research/output/path_policy/studies/mh_stage28_full_pool_revalidation_20260527_01/`.
- Dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Universe scope: `full_rolling_liquid500`.
- Reused memmap manifest: `daily_research/output/path_policy/studies/path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01/forecast_dataset_manifest.json`.
- Full-pool manifest properties: `universe_size=2430`, `train_rows=452034`, `feature_store_shape=[1699,2430,156]`.
- Fixed configuration: `gru_sequence_static_context`, `raw_kline_context_no_alpha_prior_v1`, `decision_utility_v1`, fullgrid horizons `1,2,3,5,8,10,15,20,30`.
- Stage 2.8 candidates: `decision_utility_v1`, `horizon_target_normalized_v1`, `target_norm_head_constraint_v1`.
- Budget: `forecast_epochs=24`, `forecast_min_epochs=8`, `early_stop_patience=6`, `checkpoint_every_n_epochs=4`, CUDA.
- Training summary: `stage28_training_summary.json` status `completed`, completed tags `9`, failed tags `0`.
- Comparison summary: `stage28_comparison_summary.json` status `completed`, `stage3_architecture_allowed=true`, `stage3_candidates=[target_norm_head_constraint_v1]`.
- Scope guard: comparison was allowed only after all study summaries passed `full_rolling_liquid500` scope checks.
- Key artifacts:
  - `stage28_task_list.json`
  - `stage28_training_summary.json`
  - `stage28_comparison_summary.json`
  - `stage28_research_verdict.md`
  - `profile_aggregate.csv`
  - `score_variant_comparison.csv`
  - `output_profile_comparison.csv`
  - `architecture_input_interaction_report.json`

## Run Tags

- `mh28_fullpool_decision_utility_v1_fullgrid_seed7_20260527_01`
- `mh28_fullpool_decision_utility_v1_fullgrid_seed11_20260527_01`
- `mh28_fullpool_decision_utility_v1_fullgrid_seed19_20260527_01`
- `mh28_fullpool_horizon_target_normalized_v1_fullgrid_seed7_20260527_01`
- `mh28_fullpool_horizon_target_normalized_v1_fullgrid_seed11_20260527_01`
- `mh28_fullpool_horizon_target_normalized_v1_fullgrid_seed19_20260527_01`
- `mh28_fullpool_target_norm_head_constraint_v1_fullgrid_seed7_20260527_01`
- `mh28_fullpool_target_norm_head_constraint_v1_fullgrid_seed11_20260527_01`
- `mh28_fullpool_target_norm_head_constraint_v1_fullgrid_seed19_20260527_01`

## Results

`profile_aggregate.csv` test / `pred_decision_score` rows:

| Candidate | Seeds | Min rank IC | Min spread | Min hit lift | Mean monthly positive rate | Max negative months | Mean long horizon share | Mean 30d concentration | Mean future long share | Mean pred/future horizon gap | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `decision_utility_v1` | 3 | `0.083041` | `0.026515` | `0.004281` | `0.787879` | `3` | `0.912850` | `0.884804` | `0.436126` | `16.221907` | fail |
| `horizon_target_normalized_v1` | 3 | `0.101335` | `0.035339` | `0.009562` | `0.939394` | `2` | `0.825395` | `0.819793` | `0.436126` | `15.816018` | pass |
| `target_norm_head_constraint_v1` | 3 | `0.083206` | `0.028098` | `0.010771` | `0.878788` | `2` | `0.815462` | `0.776934` | `0.436126` | `15.618355` | pass |

Gate requirements were: 3 seeds all positive on rank IC, top-bottom spread, and hit lift; mean monthly positive rate `>=0.75`; max negative months `<=2`; long horizon share not above the full-pool baseline; 30d concentration materially reduced without rank/spread/hit collapse.

## Interpretation

- The original full-pool baseline remains predictive across all seeds, but it fails the strict Stage 2.8 gate because max negative months is `3` and 30d concentration is still high at `0.884804`.
- Per-horizon target normalization is not merely a cap80 artifact. On the full rolling_liquid500 pool it reduces 30d concentration from the baseline `0.884804` to `0.819793` while improving rank/spread and meeting monthly stability.
- The combined `target_norm_head_constraint_v1` candidate reduces concentration further to `0.776934`, keeps all-seed rank/spread/hit positive, and meets the monthly stability gate.
- This supports the interpretation that the 20/30d family contains real edge, but the old horizon chooser was amplified by target scale / head-collapse pressure.
- Stage 3 architecture review may now start, but only as a shadow-only research plan. A stronger architecture should test whether the repaired target/loss surface is robust; it must not be treated as promotion evidence.

## Next Action

- Keep active execution unchanged.
- Write a Stage 3 architecture review plan using `target_norm_head_constraint_v1` as the primary fixed target/loss baseline.
- Candidate architectures may include `patch_transformer_static_context` and `stock_mixer_sequence`, but all runs must remain research / shadow-only.
- Do not mechanically rerun every old cap80 grid. If more evidence is needed, add only targeted full-pool controls around target normalization, head concentration, and monthly regime stability.
