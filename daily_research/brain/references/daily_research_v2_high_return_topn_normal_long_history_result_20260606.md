# Daily Research V2 High Return TopN Normal Long-History Result

Date: `2026-06-06`

## Summary

- Status: `normal_sample_single_seed_topn_long_history_scout_completed / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Loss profile: `topn_excess_rank_v1`.
- Evidence grade: `scout_only`.
- Active artifact impact: `unchanged`.

## Facts

- Run tag: `v2_hrd_topn_normal_long_mainboard_wide_gru_seed7_scout_20260606_01`.
- Agent run id: `v2_hrd_topn_normal_long_mainboard_seed7_20260606_01`.
- Dataset key: `long_history_augmented`.
- Dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Pool view id: `policy_pool_view__eb690dd0becc330f029c21bd`.
- Model family: `gru_sequence_static_context`.
- Seed: `7`.
- Forecast sample cap: `--forecast-max-samples-per-role 0 --forecast-max-samples-per-date-per-role 0`.
- Splits completed: `a`, `b`, `long`.
- Quick bridge status: completed for all three splits, returncode `0`.
- Outer discovery report was collected after completion at `daily_research/output/path_policy/studies/v2_hrd_topn_normal_long_mainboard_wide_gru_seed7_scout_20260606_01/v2_high_return_model_discovery_report.json`.

## Bridge Results

- Split `a`: excess annual return `0.1381527084176608`, excess Sharpe `0.863342839655209`, monthly positive ratio `0.5454545454545454`, max drawdown `-0.24516376138828777`.
- Split `b`: excess annual return `-0.009128600851242785`, excess Sharpe `-0.0409923039145095`, monthly positive ratio `0.6363636363636364`, max drawdown `-0.2373816148284037`.
- Split `long`: excess annual return `0.3721040271530076`, excess Sharpe `1.423482278126685`, monthly positive ratio `0.7272727272727273`, max drawdown `-0.3385461463281403`.

## Inferences

- Normal-sample evidence differs materially from datecap2: `topn_excess_rank_v1` is not rejected by thin scout results.
- The loss has at least two positive-transfer long-history settings (`a` and `long`), so same-period control is now allowed.
- This is not yet a finalist because split `b` remains slightly negative and same-period transfer is untested.
- Three-seed confirmation is still not allowed.

## Assumptions

- The default quick bridge shell is treated as the fixed research bridge for this normal-sample scout: holding count `20`, max weight `0.12`, rebalance `10d`, transaction cost `10bps`, slippage `5bps`, sell tax `10bps`.
- `000300.SH` remains the benchmark.
- The outer discovery report is a post-run collector artifact; forecast summaries, quick bridge run summary, and all three bridge reports were already complete before collection.

## Boundaries

- `research_only=true`.
- `shadow_only=true`.
- `promotion_allowed=false`.
- `active_artifact_impact=unchanged`.
- `paper_live_broker_allowed=false`.
- Do not modify `daily_research/output/active_execution_strategy.json`.
- Do not run seeds `7,11,19` until same-period control is non-negative and finalist criteria are met.

## Next

- Run same-period normal-sample control for `topn_excess_rank_v1` before trying three seed.
- If same-period is positive or at least not clearly negative, run small-capital candidate matrix on the best long-history and same-period candidates.
- If same-period is negative, record this as long-history-specific transfer and try the next normal-sample objective.
