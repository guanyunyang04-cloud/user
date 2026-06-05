# Daily Research V2 High Return Long-History Scout

Date: `2026-06-05`

## Summary

- Status: `high_return_model_discovery / long_history_single_seed_multi_split_completed / same_period_control_completed / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Evidence grade: `scout_only`, single seed. This is not completed model-quality evidence, not promotion evidence, and not a three-seed finalist.

## Facts

- Long-history discovery run tag: `v2_hrd_long_mainboard_wide_gru_seed7_scout_20260605_01`.
- Same-period control run tag: `v2_hrd_same_mainboard_wide_gru_seed7_scout_20260605_01`.
- Dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Long-history pool view id: `policy_pool_view__eb690dd0becc330f029c21bd`.
- Same-period pool view id: `policy_pool_view__d7a56d5164470b590e4f5a40`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`.
- Model family: `gru_sequence_static_context`.
- Seed: `7`.
- Forecast cap: `--forecast-max-samples-per-role 0` and `--forecast-max-samples-per-date-per-role 2`.
- Long-history forecast tags:
  - `mh_v2_hrd_long_a_gru_seed7_20260604_01`;
  - `mh_v2_hrd_long_b_gru_seed7_20260604_01`;
  - `mh_v2_hrd_long_c_gru_seed7_20260604_01`;
  - `mh_v2_hrd_long_long_gru_seed7_20260604_01`.
- Long-history completed forecast count: `4`.
- Long-history completed quick bridge count: `4`.
- Long-history small-capital gate id: `small_capital_balanced_return_v1`.
- Long-history positive transfer split count: `1 / 4`.
- Long-history matrix-entry split count: `1 / 4`.
- Long-history small-capital research-grade split count: `1 / 4`.
- Long-history `long_a` quick bridge:
  - excess annual return: `0.21993258747711364`;
  - excess Sharpe: `1.2890077222278065`;
  - positive month ratio: `0.7272727272727273`;
  - worst monthly return: `-0.09559172840421681`.
- Long-history `long_a` candidate matrix tag: `v2_hr_matrix_mh_v2_hrd_long_a_gru_seed7_20260604_01`.
- Long-history `long_a` matrix variants: `27 / 27` completed.
- Long-history `long_a` small-capital positive transfer variants: `27 / 27`.
- Long-history `long_a` small-capital research-grade variants: `9 / 27`.
- Best long-history `long_a` small-capital variant:
  - variant id: `h10_mw080_rb20d_all_c10_5_10_regime_off`;
  - holding count: `10`;
  - max weight: `0.08`;
  - rebalance frequency: `20d`;
  - transaction cost / slippage / sell tax bps: `10 / 5 / 10`;
  - excess annual return: `0.26694328273590995`;
  - excess Sharpe: `1.5368942376582724`;
  - positive month ratio: `0.7272727272727273`;
  - worst monthly return: `-0.08654751730684906`;
  - promotion review eligible: `false`.
- Long-history non-transfer splits:
  - `long_b`: excess annual return `-0.05559422712912809`, excess Sharpe `-0.40847805877439614`;
  - `long_c`: excess annual return `-0.12720931902378374`, excess Sharpe `-0.5606155079781685`;
  - `long_long`: excess annual return `-0.1891443933852902`, excess Sharpe `-0.7835002444851403`.
- Same-period control forecast tag: `mh_v2_hrd_same_same_gru_seed7_20260604_01`.
- Same-period control quick bridge tag: `v2_hr_bridge_mh_v2_hrd_same_same_gru_seed7_20260604_01`.
- Same-period control completed forecast count: `1`.
- Same-period control completed quick bridge count: `1`.
- Same-period control positive transfer split count: `0 / 1`.
- Same-period control metrics:
  - annual return: `0.023507226028246953`;
  - excess annual return: `-0.16013711258861663`;
  - excess Sharpe: `-0.6746248899485193`;
  - max drawdown: `-0.07516659464458408`;
  - positive month ratio: `0.5454545454545454`;
  - worst monthly return: `-0.21097857209196647`.

## Inferences

- The fixed small-capital execution shell can extract a strong-looking result from one early long-history split, especially in concentrated `h10 / mw0.08 / rb20d` form.
- The signal is not yet durable across time splits: only `long_a` transfers, while `long_b`, `long_c`, `long_long`, and the same-period control fail bridge transfer.
- The strong `long_a` matrix result is useful as an alpha-mining clue, not as finalist evidence.
- The next bottleneck is more likely objective / label / loss alignment than execution-shell tuning or model size.

## Assumptions

- Current goal remains high-return research-grade discovery for small-capital assumptions, not live readiness.
- Single-seed evidence is acceptable only for scout triage.
- Same-period control is the correct first guard against split luck after a positive long-history scout.
- PIT / no-leakage / explicit dataset id / explicit pool view id / clean OOS remain minimum requirements.

## Boundaries

- `research_only=true`.
- `shadow_only=true`.
- `promotion_allowed=false`.
- `active_execution_strategy_expected_diff=none`.
- `paper_live_broker_allowed=false`.
- `active_artifact_impact=unchanged`.
- Do not run seeds `7,11,19` from this candidate.
- Do not rebuild production root, trade plan, paper/live, broker, or active execution artifact.

## Verdict

- The long-history GRU seed7 scout is `weak_positive_single_split_transfer`.
- It is not a finalist because fewer than two split / pool-view settings show positive transfer.
- Same-period control failed transfer, so the apparent `long_a` strength should be treated as a useful objective/label/loss design clue rather than a model-quality pass.
- Three-seed confirmation is not allowed from this evidence.

## Next Allowed Actions

- Pivot to direct return objective / label / loss scouts, still single-seed and research-only:
  - top-decile / topN excess-return objective;
  - score-to-weight proxy loss;
  - bad-month-aware auxiliary loss;
  - local-volatility / reversal-state penalty;
  - high-return label variants that reduce forecast-metric-to-portfolio-return mismatch.
- Use `long_a` best execution shell as a diagnostic reference, but require at least two positive transfer split / pool-view settings before any three-seed confirmation.
- Do not expand to larger Tier 2 models as the next default action.

## Output Artifacts

- Long-history report JSON: `daily_research/output/path_policy/studies/v2_hrd_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_report.json`.
- Long-history report Markdown: `daily_research/output/path_policy/studies/v2_hrd_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_report.md`.
- Long-history `long_a` matrix report JSON: `daily_research/output/path_policy/studies/v2_hrd_long_mainboard_wide_gru_seed7_scout_20260605_01/matrices/v2_hr_matrix_mh_v2_hrd_long_a_gru_seed7_20260604_01/v2_candidate_review_matrix_report.json`.
- Same-period report JSON: `daily_research/output/path_policy/studies/v2_hrd_same_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_report.json`.
- Same-period report Markdown: `daily_research/output/path_policy/studies/v2_hrd_same_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_report.md`.

## Verification

- Forecast and bridge runs completed with explicit run tags.
- Long-history shortlist matrix completed `27 / 27` variants.
- `git diff -- daily_research/output/active_execution_strategy.json`: clean before writeback.
