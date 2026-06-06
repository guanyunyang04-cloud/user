# Daily Research V2 High Return TopN Normal Same-Period And Matrix Result

Date: `2026-06-06`

## Summary

- Status: `normal_sample_topn_same_period_control_and_long_matrix_completed / finalist_three_seed_allowed / research-only`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Loss profile: `topn_excess_rank_v1`.
- Evidence grade: `scout_only`.
- Active artifact impact: `unchanged`.
- Three seed confirmation: `allowed`.

## Facts

- Run tags:
  - `mh_v2_hrd_topn_long_a_gru_seed7_20260606_01`
  - `mh_v2_hrd_topn_long_b_gru_seed7_20260606_01`
  - `mh_v2_hrd_topn_long_long_gru_seed7_20260606_01`
  - `mh_v2_hrd_topn_same_same_gru_seed7_20260606_01`
- Long-history scout run tag: `v2_hrd_topn_normal_long_mainboard_wide_gru_seed7_scout_20260606_01`.
- Same-period control run tag: `v2_hrd_topn_normal_same_mainboard_wide_gru_seed7_scout_20260606_01`.
- Same-period agent run id: `v2_hrd_topn_normal_same_mainboard_seed7_20260606_01`.
- Long-history small-capital matrix agent run id: `v2_hrd_topn_normal_long_matrix_20260606_01`.
- Long-history matrix run tag: `v2_hr_matrix_mh_v2_hrd_topn_long_long_gru_seed7_20260606_01`.
- Dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Long-history pool view id: `policy_pool_view__eb690dd0becc330f029c21bd`.
- Same-period pool view id: `policy_pool_view__d7a56d5164470b590e4f5a40`.
- Model family: `gru_sequence_static_context`.
- Seed: `7`.
- Forecast sample cap: `--forecast-max-samples-per-role 0 --forecast-max-samples-per-date-per-role 0`.
- Same-period quick bridge completed with report `daily_research/output/path_policy/studies/v2_hrd_topn_normal_same_mainboard_wide_gru_seed7_scout_20260606_01/v2_high_return_model_discovery_report.json`.
- Long-history small-capital matrix completed with report `daily_research/output/path_policy/studies/v2_hrd_topn_normal_long_mainboard_wide_gru_seed7_scout_20260606_01/matrices/v2_hr_matrix_mh_v2_hrd_topn_long_long_gru_seed7_20260606_01/v2_candidate_review_matrix_report.json`.

## Results

- Long-history split `a`: excess annual return `0.1381527084176608`, excess Sharpe `0.863342839655209`, positive month ratio `0.5454545454545454`, positive transfer `true`.
- Long-history split `b`: excess annual return `-0.009128600851242785`, excess Sharpe `-0.0409923039145095`, positive month ratio `0.6363636363636364`, positive transfer `false`.
- Long-history split `long`: excess annual return `0.3721040271530076`, excess Sharpe `1.423482278126685`, positive month ratio `0.7272727272727273`, worst monthly return `-0.09288538572416127`, matrix entry `true`.
- Same-period split `same`: excess annual return `0.2572616856952217`, excess Sharpe `1.029989368963496`, positive month ratio `0.5454545454545454`, worst monthly return `-0.06405622679433465`, positive transfer `true`.
- Long-history small-capital matrix: `27/27` variants positive transfer and `15/27` variants research-grade under `small_capital_balanced_return_v1`.
- Best small-capital variant: `h30_mw080_rb10d_all_c10_5_10_regime_off`.
- Best small-capital metrics: annual return `0.7407818837373514`, excess annual return `0.42843945020543517`, excess Sharpe `1.7463396422791433`, positive month ratio `0.7272727272727273`, worst monthly return `-0.07756054662202583`, max drawdown `-0.3038733906784732`.

## Inferences

- The datecap2 pessimistic conclusion is now superseded for `topn_excess_rank_v1`; normal-sample evidence shows real transfer on multiple settings.
- `topn_excess_rank_v1` now satisfies finalist-confirmation prerequisites: at least two split or pool-view settings have positive transfer, at least one split has excess Sharpe above `1.2`, same-period is not negative, and the fixed small-capital matrix contains research-grade variants.
- This is still single-seed scout evidence. It is strong enough to run seeds `7,11,19`, but not strong enough for promotion, live, paper, broker, or active artifact changes.
- Split `b` remains a warning: the candidate is promising, not solved.
- Same-period monthly positive ratio is just below the matrix-entry threshold (`0.5454545454545454` versus `0.55`), so same-period is a positive control, not a matrix-grade confirmation by itself.

## Assumptions

- `000300.SH` remains the benchmark.
- The small-capital research shell is `holding_count=10,20,30`, `max_weight=0.08,0.12,0.16`, `rebalance_freq=5d,10d,20d`, transaction cost `10bps`, slippage `5bps`, sell tax `10bps`.
- Seed `7` from the long-history normal-sample run may be reused as part of the three-seed confirmation if the orchestrator skips existing matching study tags.
- Finalist confirmation should focus first on `long_history_augmented / split long / topn_excess_rank_v1 / gru_sequence_static_context`, then only expand same-period or execution adaptation if seed stability holds.

## Boundaries

- `research_only=true`.
- `shadow_only=true`.
- `promotion_allowed=false`.
- `active_artifact_impact=unchanged`.
- `paper_live_broker_allowed=false`.
- Do not modify `daily_research/output/active_execution_strategy.json`.
- Do not mark this as production, promotion, or live-ready.

## Next

- Run finalist confirmation for `topn_excess_rank_v1` with seeds `7,11,19`; reuse seed `7` when possible and train only missing seeds.
- Confirmation pass requires at least two seeds with positive excess return and positive excess Sharpe.
- After seed confirmation, run ensemble or seed-aggregate bridge/matrix only if the tooling supports explicit seed aggregation; otherwise write seed-level evidence and implement a minimal explicit ensemble collector before claiming ensemble superiority.
- If seed stability fails, diagnose split `b`, month concentration, and seed sensitivity before trying execution adaptation.
