# daily_research v2 Local-State Input Scout

Date: 2026-06-02

## Verdict

`raw_kline_context_v2_tradeable_local_state_v1` completed a 3-seed research-only scout on the v2 strict tradeable mainboard contract, but the aggregate gate is `near_pass`, not `pass`.

The profile preserves positive 3-seed test rank IC, spread, and hit lift, and improves mean hit lift versus the current v2 strict baseline. It does not replace the current pass-grade baseline because monthly stability degrades: `negative_month_count_max=4`.

Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Artifacts

- Anchor: `mh_v2_local_state_input_scout_anchor_20260602_01`
- Seed runs:
  - `mh_v2_local_state_input_scout_seed7_20260602_01`
  - `mh_v2_local_state_input_scout_seed11_20260602_01`
  - `mh_v2_local_state_input_scout_seed19_20260602_01`
- Dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`
- Pool: `policy_pool_view__925e8604a91a9c07a5387fb1`
- Feature profile: `raw_kline_context_v2_tradeable_local_state_v1`
- Baseline anchor: `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`
- Baseline feature profile: `raw_kline_context_v2_tradeable_amount_checked`

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_local_state_input_scout --run-training --seeds 7 --no-skip-existing --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_local_state_input_scout --run-comparison --seeds 7 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_local_state_input_scout --run-training --seeds 11,19 --no-skip-existing --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_local_state_input_scout --run-comparison --seeds 7,11,19 --json
```

## Manifest Evidence

Seed7 built and validated the shared memmap manifest:

- `feature_store_shape=[1699,2602,144]`
- `feature_count=144`
- `local_state_context_feature_count=20`
- `alpha_like_feature_count=0`
- `amount_unit_policy=as_is`
- `amount_consistency_median=1.000075266090736`
- `source_pool_view_id=policy_pool_view__925e8604a91a9c07a5387fb1`
- `label_semantics=next_open_entry_to_future_open`

Seed11 and seed19 reused the seed7 memmap manifest.

## 3-Seed Gate

From `daily_research/output/path_policy/studies/mh_v2_local_state_input_scout_anchor_20260602_01/v2_local_state_input_scout_summary.json`:

- Gate: `near_pass`
- `seed_count=3`
- `rank_ic_mean=0.1061998621132488`
- `rank_ic_min=0.0881543205611268`
- `spread_mean=0.0370699463245396`
- `spread_min=0.0303419278731737`
- `hit_lift_mean=0.0188572516186681`
- `hit_lift_min=0.0143019720318046`
- `monthly_positive_rate_mean=0.7878787878787877`
- `monthly_positive_rate_min=0.6363636363636364`
- `negative_month_count_mean=2.333333333333333`
- `negative_month_count_max=4`
- `worst_month_spread_min=-0.0494888111307637`
- `thirty_d_concentration_mean=0.759643026932514`
- `validation_test_rank_ic_gap=-0.0395422834264249`
- `validation_test_monthly_positive_rate_gap=-0.2121212121212122`

Checks:

- `seed_count_ge_3=true`
- `rank_ic_min_positive=true`
- `spread_min_positive=true`
- `hit_lift_min_positive=true`
- `monthly_positive_rate_mean_ge_075=true`
- `negative_month_count_max_le_2=false`
- `thirty_d_concentration_not_worse_than_stage28=true`

## Comparison To Current v2 Strict Baseline

Current pass-grade strict baseline `mh_v2_reset_tradeable_mainboard_anchor_20260601_01` has:

- `rank_ic_mean=0.1097597160364623`
- `spread_mean=0.0395815603562681`
- `hit_lift_mean=0.0154853717382623`
- `monthly_positive_rate_mean=0.8787878787878789`
- `negative_month_count_max=2`
- `thirty_d_concentration_mean=0.6153860880052838`

Local-state scout has better hit lift but slightly weaker rank/spread, worse monthly stability, and higher 30d concentration. Therefore it is useful model-input evidence, but not a baseline replacement and not execution-candidate evidence.

## Interpretation

Facts:

- The local-state input contract is file-backed and leak-guarded.
- The 3-seed model-quality evidence is positive on rank/spread/hit.
- The gate fails because one seed has too many negative months; the aggregate `negative_month_count_max` is `4`.
- The profile remains research-only and shadow-only.

Inferences:

- Moving local volatility/reversal/runup features into model input helps hit-lift, but does not yet teach the model enough month-state stability.
- Since post-hoc overlays and fixed score penalties also failed monthly stability, the next work should target loss/output calibration or horizon/month-state constraints rather than more simple overlay stacking.

Boundaries:

- Do not promote this profile.
- Do not overwrite `raw_kline_context_v2_tradeable_amount_checked`.
- Do not write or promote `daily_research/output/active_execution_strategy.json`.
- Do not treat this as evidence that execution can be unfrozen.

## Next Actions

1. Design a v2 loss/output scout that penalizes adverse local-state concentration or monthly instability directly.
2. Add horizon concentration diagnostics for local-state profile, especially `30d` over-selection.
3. If a score/backtest bridge is run for local-state later, treat it as a separate candidate-review experiment, not as automatic promotion.
4. Keep the current v2 strict baseline as `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
