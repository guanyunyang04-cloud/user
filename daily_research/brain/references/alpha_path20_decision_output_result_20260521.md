# alpha_path20 Decision Output Result 2026-05-21

## Verdict
- Status: `forecast decision-output study / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Study tag: `path20_decision_output_liquid500_sector_v1_20260521_01`.
- Evidence verdict: `forecast_promising`.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Inputs
- Lake dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Pool view id: `policy_pool_view__c11400fa72ad263f3d1eecfa` (`rolling_liquid500`).
- Sector/board view id: `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`.
- Feature profile: `raw_kline_context_sector_v1`.
- Static fields: `symbol,exchange,industry,board,liquidity_bucket,price_bucket`.
- Model family: `gru_sequence_static_context`.
- Output profile: `decision_utility_v1`.
- Loss profile: `decision_utility_v1`.
- Selection profile: `decision_utility`.
- Decision parameters: cost `20.0` bps, hit threshold `20.0` bps, drawdown penalty `0.25`.
- Seed: `7`.
- Device: `cuda` with AMP.

## Result
- Status: completed `forecast_walkforward_study`.
- Training stopped by `early_stopping_patience_exhausted`.
- Epochs ran: `8`; best epoch: `1`.
- Validation rows: `108920`; test rows: `109972`.

| Split | decision_score_rank_ic | decision_score_top_bottom_spread | decision_hit_lift_top20_mean | decision_best_horizon_accuracy | decision profile |
|---|---:|---:|---:|---:|---|
| validation | 0.093516 | 0.012156 | 0.066078 | 0.343215 | passed |
| test | 0.020946 | -0.004432 | 0.011445 | 0.307942 | failed |

## Forecast Path Context
| Split | rank_ic_20d | top_bottom_spread_20d |
|---|---:|---:|
| validation | 0.164175 | 0.038678 |
| test | 0.103587 | 0.018474 |

## Interpretation
- The `decision_utility_v1` head learned a validation-positive decision signal on the liquid500 sector/static GRU backbone.
- Test transfer is not confirmed: `decision_score_rank_ic` stayed positive, but `decision_score_top_bottom_spread` turned negative on test.
- This supports keeping the output contract for further research, but it does not justify liquid800 expansion, allocator/replay work, or promotion.
- The existing forecast path metrics remain positive, so the new decision head did not break the underlying Path20 forecast task in this run.

## Boundaries
- This is forecast-stage output-contract evidence only.
- No allocator, replay, target-weight head, differentiable optimizer, live/default, or promotion claim is made.
- Do not modify `daily_research/output/active_execution_strategy.json` based on this result.
- Do not run liquid800 decision-output expansion unless a later liquid500 run confirms positive validation and test decision utility gates.

## Next Allowed Actions
- Inspect why validation decision spread did not transfer to test before increasing universe size or seeds.
- Compare alternative decision score definitions, especially expected utility aggregation versus horizon-logit selected utility.
- Review whether the drawdown penalty or hit threshold is too aggressive for 2024 test distribution.
- If continuing, prefer narrow output-target ablations on liquid500 before moving to liquid800.
