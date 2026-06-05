# Daily Research V2 High Return Objective Loss Scout

Date: `2026-06-05`

## Summary

- Status: `objective_loss_scout_completed / single_seed_multi_split / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_high_return_model_discovery`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Evidence grade: `scout_only`, single seed. This is not completed model-quality evidence, not promotion evidence, and not a three-seed finalist.

## Facts

- Existing-loss scout run tag: `v2_hrd_existing_loss_long_mainboard_wide_gru_seed7_scout_20260605_01`.
- Proxy-loss scout run tag: `v2_hrd_proxy_loss_long_mainboard_wide_gru_seed7_scout_20260605_01`.
- Dataset key: `long_history_augmented`.
- Dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Pool view id: `policy_pool_view__eb690dd0becc330f029c21bd`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`.
- Model family: `gru_sequence_static_context`.
- Seed: `7`.
- Forecast cap: `--forecast-max-samples-per-role 0` and `--forecast-max-samples-per-date-per-role 2`.
- Split keys: `a`, `b`, `c`, `long`.
- Existing loss profiles tested:
  - `horizon_30d_soft_penalty_v1`;
  - `score_monthly_robust_v1`;
  - `risk_drawdown_reweighted_v1`;
  - `horizon_entropy_regularized_v1`.
- Existing-loss scout completed `16` forecasts and `16` quick bridges.
- Existing-loss scout ran `4` split-a matrix-entry candidates through `small_capital_balanced_return_v1`; all four loss profiles still had only `1 / 4` positive-transfer split.
- Existing-loss split-a small-capital matrix repeated the same best variant shape:
  - variant id: `h10_mw080_rb20d_all_c10_5_10_regime_off`;
  - excess annual return: `0.26694328273590995`;
  - excess Sharpe: `1.5368942376582724`;
  - positive month ratio: `0.7272727272727273`;
  - worst month: `-0.08654751730684906`;
  - research-grade small-capital variants: `9 / 27`.
- New proxy loss profiles implemented and tested:
  - `topn_excess_rank_v1`;
  - `score_to_weight_proxy_v1`;
  - `bad_month_aware_v1`.
- Proxy-loss scout completed `12` forecasts and `12` quick bridges.
- Proxy-loss split consistency:
  - `topn_excess_rank_v1`: `1 / 4` positive-transfer split, best split `a`, worst split `long`;
  - `score_to_weight_proxy_v1`: `1 / 4` positive-transfer split, best split `a`, worst split `b`;
  - `bad_month_aware_v1`: `1 / 4` positive-transfer split, best split `a`, worst split `c`.
- Proxy-loss split-a quick bridge metrics were identical across the three proxy losses:
  - excess annual return: `0.21993258747711364`;
  - excess Sharpe: `1.2890077222278065`;
  - positive month ratio: `0.7272727272727273`;
  - worst monthly return: `-0.09559172840421681`;
  - max drawdown: `-0.09681524777489703`.
- Proxy-loss non-transfer examples:
  - split `b`: excess annual return `-0.05559422712912809`, excess Sharpe `-0.40847805877439614`;
  - split `c`: excess annual return `-0.12720931902378374`, excess Sharpe `-0.5606155079781685`;
  - split `long`: excess annual return `-0.1891443933852902`, excess Sharpe `-0.7835002444851403`.
- No proxy loss reached two positive-transfer splits or same-period eligibility.

## Run Tags

- Discovery/report tags:
  - `v2_hrd_existing_loss_long_mainboard_wide_gru_seed7_scout_20260605_01`;
  - `v2_hrd_existing_loss_long_mainboard_matrix_entry_20260605_01`;
  - `v2_hrd_proxy_loss_long_mainboard_wide_gru_seed7_scout_20260605_01`;
  - `v2_hrd_proxy_loss_long_mainboard_gru_seed7_datecap2_20260605_01`.
- Existing-loss forecast tags:
  - `mh_v2_hrd_h30soft_long_a_gru_seed7_20260605_01`;
  - `mh_v2_hrd_h30soft_long_b_gru_seed7_20260605_01`;
  - `mh_v2_hrd_h30soft_long_c_gru_seed7_20260605_01`;
  - `mh_v2_hrd_h30soft_long_long_gru_seed7_20260605_01`;
  - `mh_v2_hrd_monthly_robust_long_a_gru_seed7_20260605_01`;
  - `mh_v2_hrd_monthly_robust_long_b_gru_seed7_20260605_01`;
  - `mh_v2_hrd_monthly_robust_long_c_gru_seed7_20260605_01`;
  - `mh_v2_hrd_monthly_robust_long_long_gru_seed7_20260605_01`;
  - `mh_v2_hrd_risk_dd_long_a_gru_seed7_20260605_01`;
  - `mh_v2_hrd_risk_dd_long_b_gru_seed7_20260605_01`;
  - `mh_v2_hrd_risk_dd_long_c_gru_seed7_20260605_01`;
  - `mh_v2_hrd_risk_dd_long_long_gru_seed7_20260605_01`;
  - `mh_v2_hrd_hentropy_long_a_gru_seed7_20260605_01`;
  - `mh_v2_hrd_hentropy_long_b_gru_seed7_20260605_01`;
  - `mh_v2_hrd_hentropy_long_c_gru_seed7_20260605_01`;
  - `mh_v2_hrd_hentropy_long_long_gru_seed7_20260605_01`.
- Proxy-loss forecast tags:
  - `mh_v2_hrd_topn_long_a_gru_seed7_20260605_01`;
  - `mh_v2_hrd_topn_long_b_gru_seed7_20260605_01`;
  - `mh_v2_hrd_topn_long_c_gru_seed7_20260605_01`;
  - `mh_v2_hrd_topn_long_long_gru_seed7_20260605_01`;
  - `mh_v2_hrd_s2w_long_a_gru_seed7_20260605_01`;
  - `mh_v2_hrd_s2w_long_b_gru_seed7_20260605_01`;
  - `mh_v2_hrd_s2w_long_c_gru_seed7_20260605_01`;
  - `mh_v2_hrd_s2w_long_long_gru_seed7_20260605_01`;
  - `mh_v2_hrd_badmonth_long_a_gru_seed7_20260605_01`;
  - `mh_v2_hrd_badmonth_long_b_gru_seed7_20260605_01`;
  - `mh_v2_hrd_badmonth_long_c_gru_seed7_20260605_01`;
  - `mh_v2_hrd_badmonth_long_long_gru_seed7_20260605_01`.

## Implementation Notes

- `v2_high_return_model_discovery` now accepts explicit `--loss-profiles` and records loss profile, output profile, selection profile, dataset id, pool view id, split key, run tag, and sample cap in task/report artifacts.
- Candidate ids and tags include loss aliases so per-loss run artifacts are separable.
- Split consistency groups by `(dataset_key, model_family, loss_profile)`.
- The candidate-matrix runner now executes all `matrix_entry_candidate` rows by default, not just the top-three shortlist.
- `forecast_training` now exposes the three high-return proxy loss profiles as research losses. They do not read or modify active execution artifacts.
- `continuous_policy.runtime` received a Windows JSON atomic-write retry repair for progress files encountered during this implementation; this is a runtime robustness fix, not a strategy change.

## Inferences

- The current bottleneck is not the fixed small-capital execution shell. The shell can expose a strong split-a result, but the model signal does not transfer across adjacent long-history splits.
- The first direct objective/loss upgrade did not solve cross-split instability. It improved or preserved split-a diagnostics, but did not create a finalist.
- Because all tested loss families converge to the same split-a-only transfer shape, the next useful work is root-cause diagnosis of split/regime/pool/label alignment rather than same-period control or three-seed confirmation.
- The datecap2 scout remains useful for fast triage, but it is not enough evidence for model quality when the positive transfer is isolated to one split.

## Assumptions

- Current goal remains small-capital high-return research-grade candidate discovery, not promotion-grade or live-ready execution.
- Single-seed evidence is allowed only for scout triage.
- At least two split / pool-view settings with positive excess return and positive excess Sharpe are still required before same-period continuation or three-seed confirmation.
- PIT / no-leakage / explicit dataset id / explicit pool view id / clean OOS remain minimum requirements.

## Boundaries

- `research_only=true`.
- `shadow_only=true`.
- `promotion_allowed=false`.
- `active_execution_strategy_expected_diff=none`.
- `paper_live_broker_allowed=false`.
- `active_artifact_impact=unchanged`.
- Do not run same-period control from the proxy-loss scout.
- Do not run seeds `7,11,19` from these candidates.
- Do not rebuild production root, trade plan, paper/live, broker, or active execution artifact.

## Verdict

- Existing-loss scout verdict: `weak_positive_single_split_transfer`.
- Proxy-loss scout verdict: `objective_loss_proxy_scout_failed_cross_split_transfer`.
- Finalist status: `not_allowed`.
- Three-seed confirmation: `not_allowed`.
- The correct current research state is: implementation support for configurable high-return loss profiles exists, and the first objective/loss scout has completed, but no objective has passed the cross-split transfer gate.

## Next Allowed Actions

- Do not proceed to same-period control or three-seed for the tested objectives.
- Run focused root-cause diagnostics before adding more model capacity:
  - compare split `a` versus `b/c/long` by market regime, volatility, reversal, liquidity, and benchmark-relative return distribution;
  - audit whether datecap2 symbol sampling creates a thin-scout artifact, then retest only the best diagnostic objective at datecap `5` if the root-cause read supports it;
  - inspect label/selection alignment for topN excess return, decision utility, and bridge score-to-weight semantics;
  - test local-volatility/reversal-state label or sample weighting only after the split-regime diagnosis identifies a stable hypothesis.
- Keep `small_capital_balanced_return_v1` as the fixed diagnostic shell; do not tune execution to rescue a one-split signal.

## Output Artifacts

- Existing-loss report JSON: `daily_research/output/path_policy/studies/v2_hrd_existing_loss_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_report.json`.
- Existing-loss task list JSON: `daily_research/output/path_policy/studies/v2_hrd_existing_loss_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_task_list.json`.
- Existing-loss matrix run summary JSON: `daily_research/output/path_policy/studies/v2_hrd_existing_loss_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_candidate_matrix_run_summary.json`.
- Proxy-loss report JSON: `daily_research/output/path_policy/studies/v2_hrd_proxy_loss_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_report.json`.
- Proxy-loss task list JSON: `daily_research/output/path_policy/studies/v2_hrd_proxy_loss_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_task_list.json`.
- Proxy-loss quick bridge run summary JSON: `daily_research/output/path_policy/studies/v2_hrd_proxy_loss_long_mainboard_wide_gru_seed7_scout_20260605_01/v2_high_return_model_discovery_quick_bridge_run_summary.json`.

## Verification

- Changed-surface tests for high-return discovery, forecast loss profiles, bridge/matrix, and runtime progress passed during implementation.
- A full combined `test_forecast_training.py + test_v2_high_return_model_discovery.py` run was deferred as a slow suite after exceeding the practical local loop budget.
- Final guards should confirm `git diff -- daily_research/output/active_execution_strategy.json` is clean before completion.
