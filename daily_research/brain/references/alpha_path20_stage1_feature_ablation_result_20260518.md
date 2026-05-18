# alpha_path20 Stage 1 Feature Ablation Result 2026-05-18

## Verdict
- Status: `completed feature ablation evidence / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Shared window: train `2019-2022`, validation `2023`, test `2024`.
- Shared universe cap: `max_universe_size=80`.
- Shared samples: `8000` per role.
- Shared model families: `linear_last_day`, `mlp_last_day`, `gru_sequence`, `patch_transformer`.
- Shared seeds: `7,11`.
- Shared training controls: `epochs=60`, `min_epochs=12`, `early_stop_patience=8`, `batch_size=256`, CUDA, AMP.
- All four feature profiles completed and wrote dataset manifests, learning curves, checkpoints, validation/test predictions, and training summaries.
- All four profiles produced `forecast_test_confirmed`, but their signal profiles and opportunity behavior differ materially.
- `shadow_only=true`, `promotion_allowed=false`, `active_execution_strategy_expected_diff=none`.

## Feature Profiles
- `state_v1`: tag `path20_stage1_ablation_state_v1_cap80_repaired_20260518_01`, feature cap `96`, selected `patch_transformer` seed `7`, selected signal profile `trend_20d`.
- `raw_kline_v1`: tag `path20_stage1_ablation_raw_kline_v1_cap80_repaired_20260518_01`, feature cap `192`, selected `patch_transformer` seed `11`, selected signal profile `multiscale`.
- `raw_kline_context_v1`: tag `path20_stage1_ablation_raw_kline_context_v1_cap80_repaired_20260518_01`, feature cap `192`, selected `gru_sequence` seed `7`, selected signal profile `multiscale`.
- `raw_kline_context_no_alpha_prior_v1`: tag `path20_stage1_ablation_raw_kline_context_no_alpha_prior_v1_cap80_repaired_20260518_01`, feature cap `192`, selected `patch_transformer` seed `11`, selected signal profile `multiscale`.

## Dataset Manifest Comparison
- `state_v1`: train/validation/test `8000/8000/8000`, feature count `96`, raw `0`, market `0`, peer `0`, alpha-prior `25`, role purge `21`.
- `raw_kline_v1`: train/validation/test `8000/8000/8000`, feature count `164`, raw `15`, market `0`, peer `0`, alpha-prior `25`, role purge `21`.
- `raw_kline_context_v1`: train/validation/test `8000/8000/8000`, feature count `192`, raw `15`, market `26`, peer `12`, alpha-prior `25`, role purge `21`.
- `raw_kline_context_no_alpha_prior_v1`: train/validation/test `8000/8000/8000`, feature count `152`, raw `15`, market `26`, peer `12`, alpha-prior `0`, role purge `21`.

## Validation Metrics
- `state_v1`:
  - `rank_ic_1d/3d/5d/10d/20d = 0.013662/0.011940/0.015681/0.044958/0.052529`
  - `top_bottom_spread_1d/3d/5d/10d/20d = -0.000144/-0.000685/-0.002487/0.000921/0.000463`
  - `rank_ic_upside_20d=-0.035122`, `top_bottom_spread_upside_20d=-0.022949`
  - `q10/q90 coverage = 0.915381/0.878400`
- `raw_kline_v1`:
  - `rank_ic_1d/3d/5d/10d/20d = 0.015741/0.071302/0.084635/0.124211/0.099878`
  - `top_bottom_spread_1d/3d/5d/10d/20d = 0.000086/0.003857/0.005728/0.015182/0.011222`
  - `rank_ic_upside_20d=-0.006853`, `top_bottom_spread_upside_20d=-0.003853`
  - `q10/q90 coverage = 0.787900/0.744138`
- `raw_kline_context_v1`:
  - `rank_ic_1d/3d/5d/10d/20d = 0.040505/0.062401/0.069521/0.081199/0.067361`
  - `top_bottom_spread_1d/3d/5d/10d/20d = 0.001181/0.000583/0.000695/0.000029/0.004410`
  - `rank_ic_upside_20d=0.120046`, `top_bottom_spread_upside_20d=0.030344`
  - `q10/q90 coverage = 0.949244/0.908131`
- `raw_kline_context_no_alpha_prior_v1`:
  - `rank_ic_1d/3d/5d/10d/20d = 0.052640/0.072098/0.091486/0.078123/0.086905`
  - `top_bottom_spread_1d/3d/5d/10d/20d = 0.001544/0.005140/0.007549/0.004123/0.015142`
  - `rank_ic_upside_20d=-0.050637`, `top_bottom_spread_upside_20d=-0.022390`
  - `q10/q90 coverage = 0.932625/0.898994`

## Inferences
- Raw K-line shape adds material information over the old state baseline for 3d/5d/10d/20d ranking and spreads.
- The old `state_v1` baseline is not useless, but it mainly behaves like a weak 20d trend signal and does not capture upside opportunity well.
- `raw_kline_v1` is the strongest selected-profile validation result by multiscale score in this ablation batch, but its upside opportunity metrics remain slightly negative.
- `raw_kline_context_v1` is the only ablation profile whose selected run has strongly positive upside opportunity metrics, suggesting market/benchmark/peer context mainly improves opportunity-window capture rather than only 20d cumulative return.
- `raw_kline_context_no_alpha_prior_v1` remains strongly positive on validation 1d/3d/5d/10d/20d rank and spreads after removing alpha-prior fields. This supports the conclusion that Stage 1 is not merely replaying legacy alpha-prior features.
- Removing alpha-prior weakens upside opportunity metrics in this batch, so alpha-prior or old score context may still carry useful opportunity-window information.
- The feature-ablation evidence supports continuing with multi-profile comparisons rather than treating a single default input profile as final.

## Boundaries
- This remains capped evidence, not full-universe evidence.
- This remains forecast evidence, not allocator/replay/live evidence.
- No active execution artifact changed.
- No allocator, oracle, replay, or production promotion is implied.

## Next Allowed Actions
- Inspect and possibly revise forecast model-selection policy before larger experiments, because selected family/seed can differ from the family that is strongest on 20d robustness.
- Run a lower-cost repeat ablation or bootstrap over different cap80 symbol slices if robustness across universe slice is required.
- Design Stage 2 allocator/oracle/replay as a shadow-only follow-up, with separate tracks for:
  - multiscale trend/path allocation;
  - short-burst/opportunity capture;
  - alpha-prior-free allocation sanity checks.
- Implement memory-safe sequence loading before full-universe `max_universe_size=0`.

## Guards
- Post-ablation active execution diff: empty.
- Post-ablation brain integrity: status `ok`.
