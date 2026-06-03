# Daily Research V2 Local-State Feature Profile - 2026-06-02

## Summary

- Status: `v2_local_state_feature_profile / input-contract-upgrade / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_local_state_feature_profile`.
- New feature profile: `raw_kline_context_v2_tradeable_local_state_v1`.
- Parent feature profile: `raw_kline_context_v2_tradeable_amount_checked`.
- Dataset id for real-lake smoke: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Strict pool id for real-lake smoke: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Motivation

- v2 bad-month attribution, selective throttle, local cap, targeted local cap, and validation-selected score-state calibration all point to the same blocker:
  - high-score local-volatility / adverse-local-state exposure remains a source of unstable months.
- Simple target-weight overlays and fixed score penalties did not close the promotion-review monthly gate.
- Therefore the next useful move is to expose local state as a first-class model input contract, without changing the existing v2 amount-checked baseline profile.

## Implementation

- Added `raw_kline_context_v2_tradeable_local_state_v1` to forecast feature profile registration.
- The new profile inherits:
  - raw kline features;
  - context features;
  - history quality features;
  - regime context features;
  - amount-unit checked behavior;
  - no-alpha/no-score input contract.
- Added a new feature group:
  - `local_state_context`.
- Added local-state features:
  - `local_ret_1d`;
  - `local_ret_5d`;
  - `local_ret_20d`;
  - `local_vol_5d`;
  - `local_vol_20d`;
  - `local_vol_ratio_5_20`;
  - `local_drawdown_20d`;
  - `local_distance_to_low_20d`;
  - `cs_rank_local_ret_5d`;
  - `cs_rank_local_vol_20d`;
  - `cs_rank_local_reversal_1d`;
  - `cs_z_local_ret_5d`;
  - `cs_z_local_vol_20d`;
  - `cs_z_local_reversal_1d`;
  - `local_high_volatility_flag`;
  - `local_high_runup_flag`;
  - `local_high_reversal_flag`;
  - `local_high_volatility_x_runup`;
  - `local_high_volatility_x_reversal`;
  - `local_low_volatility_x_runup`.

## Contract Boundaries

- The existing `raw_kline_context_v2_tradeable_amount_checked` profile is unchanged and does not gain local-state features.
- The new profile is an input-contract candidate, not a trained model result.
- All local-state features use current or past close-derived windows only.
- The profile remains no-alpha / no-score:
  - no `alpha_prior_*`;
  - no `score*`;
  - no `z_score*`.
- The profile is not a promotion, live/default change, production root rebuild, or active execution artifact.

## Test Evidence

- Focused tests:
  - `python -m pytest daily_research/path_policy/tests/test_v2_feature_profile_amount_checked.py -q`: `9 passed`.
- Broader related tests:
  - `python -m pytest daily_research/path_policy/tests/test_forecast_features.py daily_research/path_policy/tests/test_v2_feature_profile_amount_checked.py daily_research/path_policy/tests/test_v2_research_reset_baseline.py -q`: `28 passed`.
- Tests cover:
  - new profile registration;
  - local-state feature presence;
  - local-state group counts and store audit;
  - amount-unit audit inheritance;
  - original amount-checked profile not being modified;
  - no alpha/score input leakage;
  - future-close perturbation does not change same-date local-state features.

## Real-Lake Smoke

- Data source:
  - `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool:
  - `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Date:
  - `2024-03-29`.
- Result:
  - universe size loaded: `861`;
  - total feature count: `179`;
  - local-state feature count: `20`;
  - local-state finite ratio: `0.999942`;
  - amount unit policy: `as_is`;
  - amount consistency median: `0.999929`;
  - alpha/score leakage: `false`.

## Interpretation

- The input contract upgrade is ready for a narrow training scout.
- This is not evidence that the model is better. It only proves the feature profile exists, is auditable, works on the explicit v2 lake and strict pool, and preserves v2 boundaries.
- The next meaningful experiment should compare:
  - current pass-grade baseline `raw_kline_context_v2_tradeable_amount_checked`;
  - new local-state profile `raw_kline_context_v2_tradeable_local_state_v1`;
  - same dataset, strict pool, model family, loss profile, horizon grid, date split, and seeds.

## Next Allowed Actions

- Run a narrow v2 local-state input scout, preferably starting with seed7 and reusing the v2 strict baseline orchestration pattern.
- If seed7 is promising, continue with seed11/19 and aggregate gate.
- Compare against `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`, not against old short_v5b.
- Do not promote to live/default from this input-contract change.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
