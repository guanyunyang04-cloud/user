# Daily Research V2 High Return TopN Finalist Three-Seed And Ensemble Bridge

Date: `2026-06-06`

## Summary

- Status: `normal_sample_topn_three_seed_completed / seed_level_confirmation_passed / all_seed_ensemble_not_confirmed / research-only`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Loss profile: `topn_excess_rank_v1`.
- Evidence grade: `research_grade_candidate_partial`.
- Active artifact impact: `unchanged`.
- Three seed confirmation: `partial_pass_seed_level / ensemble_superiority_failed`.

## Facts

- Run tags:
  - `v2_hrd_topn_normal_long_mainboard_wide_gru_seeds7_11_19_finalist_20260606_01`
  - `mh_v2_hrd_topn_long_long_gru_seed7_20260606_01`
  - `mh_v2_hrd_topn_long_long_gru_seed11_20260606_01`
  - `mh_v2_hrd_topn_long_long_gru_seed19_20260606_01`
  - `v2_hr_matrix_mh_v2_hrd_topn_long_long_gru_seed7_20260606_01`
  - `v2_hr_matrix_mh_v2_hrd_topn_long_long_gru_seed19_20260606_01`
  - `v2_score_bridge_topn_finalist_seed_ensemble_20260606_01`
- Finalist three-seed agent run id: `v2_hrd_topn_normal_long_finalist_seeds_20260606_01`.
- Finalist matrix agent run id: `v2_hrd_topn_normal_long_finalist_matrix_20260606_01`.
- Ensemble bridge agent run id: `v2_score_bridge_topn_finalist_seed_ensemble_20260606_01`.
- Dataset key: `long_history_augmented`.
- Dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Pool view id: `policy_pool_view__eb690dd0becc330f029c21bd`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`.
- Model family: `gru_sequence_static_context`.
- Seeds: `7,11,19`.
- Forecast sample cap: `--forecast-max-samples-per-role 0 --forecast-max-samples-per-date-per-role 0`.
- Finalist discovery report: `daily_research/output/path_policy/studies/v2_hrd_topn_normal_long_mainboard_wide_gru_seeds7_11_19_finalist_20260606_01/v2_high_return_model_discovery_report.json`.
- Ensemble bridge report: `daily_research/output/path_policy/studies/v2_score_bridge_topn_finalist_seed_ensemble_20260606_01/v2_score_backtest_bridge_report.json`.

## Results

- Three seed quick bridge positive transfer count: `3/3`.
- Matrix entry candidate count: `2/3`.
- Small-capital research-grade matrix count: `1`.
- Seed `7`: excess annual return `0.3721040271530076`, excess Sharpe `1.423482278126685`, positive month ratio `0.7272727272727273`, worst monthly return `-0.09288538572416127`, max drawdown `-0.3385461463281403`, matrix entry `true`.
- Seed `19`: excess annual return `0.27170179096011604`, excess Sharpe `1.0638148467301434`, positive month ratio `0.7272727272727273`, worst monthly return `-0.08492485405980532`, max drawdown `-0.3116169148345489`, matrix entry `true`.
- Seed `11`: excess annual return `0.05158371296158859`, excess Sharpe `0.20022232999536796`, positive month ratio `0.5454545454545454`, worst monthly return `-0.07598469184633283`, max drawdown `-0.30117677315782265`, matrix entry `false`.
- Seed `7` small-capital matrix: `27/27` variants positive transfer and `15/27` variants research-grade.
- Seed `7` best matrix variant: `h30_mw080_rb10d_all_c10_5_10_regime_off`.
- Seed `7` best matrix metrics: excess annual return `0.42843945020543517`, excess Sharpe `1.7463396422791433`, positive month ratio `0.7272727272727273`, worst monthly return `-0.07756054662202583`, max drawdown `-0.3038733906784732`.
- Seed `19` small-capital matrix: `27/27` variants positive transfer and `0/27` variants research-grade.
- Seed `19` best matrix variant: `h20_mw120_rb10d_all_c10_5_10_regime_off`.
- Seed `19` best matrix metrics: excess annual return `0.27170179096011604`, excess Sharpe `1.0638148467301434`, positive month ratio `0.7272727272727273`, worst monthly return `-0.08492485405980532`.
- All-seed ensemble bridge completed with annual return `0.3621422617433985`, excess annual return `0.11773781749666945`, excess Sharpe `0.4028515166232868`, positive month ratio `0.6363636363636364`, worst monthly return `-0.08946119583758427`, max drawdown `-0.2925926852916775`.
- All-seed ensemble bridge is weaker than single-seed median: median seed excess annual return is `0.27170179096011604` and median seed excess Sharpe is `1.0638148467301434`.
- Seed score Spearman correlation mean is `0.6856349009480321`; seed top-20 pairwise overlap mean is `0.30868878357030016`.

## Inferences

- `topn_excess_rank_v1` normal-sample evidence is materially stronger than earlier datecap2 smoke evidence: all three finalist seeds have positive excess return and positive excess Sharpe on the long-history `long` split.
- The candidate is not a production, promotion, paper, live, or active-artifact candidate. It is a research-grade candidate only at seed-level / partial-finalist level.
- The result is still seed-sensitive: seed `7` is strong, seed `19` is positive but below research-grade, and seed `11` is weak.
- The fixed all-seed mean ensemble does not improve over the single-seed median. Therefore the planned "ensemble bridge better than single-seed median" confirmation failed.
- The ensemble weakness is likely caused by averaging a weak seed into the score panel and by limited top-20 name overlap across seeds. This is an inference from the seed-level metrics and ensemble diagnostics, not a proven root cause.

## Assumptions

- `000300.SH` remains the benchmark.
- The all-seed ensemble bridge used the research-only small-capital shell: holding count `20`, max weight `0.08`, rebalance `5d`, all offset sleeves, transaction cost `10bps`, slippage `5bps`, sell tax `10bps`, no market regime filter.
- Seed aggregation should be judged by explicit seed-aggregate or ensemble bridge evidence, not by assuming that multi-seed is automatically better.

## Boundaries

- `research_only=true`.
- `shadow_only=true`.
- `promotion_allowed=false`.
- `active_artifact_impact=unchanged`.
- `paper_live_broker_allowed=false`.
- Do not modify `daily_research/output/active_execution_strategy.json`.
- Do not mark this as production-ready, promotion-ready, or live-ready.
- Do not start execution adaptation as if three-seed finalist confirmation fully passed; ensemble / seed aggregation remains unresolved.

## Next

- Treat `topn_excess_rank_v1` as the current best high-return model candidate, but with `seed_aggregation_blocker`.
- Diagnose seed aggregation before execution adaptation: compare validation-selected seed weighting, rank aggregation, topN vote aggregation, and same-period multi-seed stability.
- Do not exclude seed `11` by hindsight without a validation-side selection rule.
- If seed aggregation remains weak, return to objective/loss refinement or split/regime diagnostics rather than trying to rescue the candidate with production execution changes.
