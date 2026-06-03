# daily_research v2 local-state loss calibration scout

Date: `2026-06-03`

## Scope

This note records a research-only / shadow-only loss-output calibration scout for `daily_research_v2_research_reset`.

The scout tests whether the monthly-stability blocker observed in `raw_kline_context_v2_tradeable_local_state_v1` can be repaired by changing the training loss/output calibration while keeping the same explicit v2 dataset, strict pool, model family, feature profile, horizon grid, and date split.

## Facts

- Anchor run: `mh_v2_local_state_loss_calibration_anchor_20260603_01`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_local_state_loss_calibration_scout`.
- Dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool view: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_v1`.
- Model family: `gru_sequence_static_context`.
- Output profile: `decision_utility_v1`.
- Horizon grid: `1,2,3,5,8,10,15,20,30`.
- Source memmap manifest: `daily_research/output/path_policy/studies/mh_v2_local_state_input_scout_seed7_20260602_01/forecast_dataset_manifest.json`.
- Source manifest validation status: `ok`.
- Feature store shape: `[1699,2602,144]`.
- Local-state feature count: `20`.
- Alpha-like feature count: `0`.
- Amount unit policy: `as_is`.

Seed7 screening completed three loss profiles:

- `score_monthly_robust_v1`.
- `horizon_entropy_regularized_v1`.
- `risk_drawdown_reweighted_v1`.

The seed7 screening selected `horizon_entropy_regularized_v1` for seed11/19 confirmation because it had the strongest test rank/spread/hit combination while meeting single-seed monthly and negative-month checks.

The final 3-seed confirmation for `horizon_entropy_regularized_v1` completed seeds `7,11,19` with gate status `near_pass`:

- Test rank IC mean/min: `0.122041 / 0.113839`.
- Test top-bottom spread mean/min: `0.040043 / 0.035862`.
- Test hit lift mean/min: `0.030121 / 0.025153`.
- Mean monthly positive rate: `0.848485`.
- Max negative months: `2`.
- 30d concentration mean: `0.805152`.
- Long horizon share mean: `0.832882`.
- All seed rank/spread/hit positive: `true`.
- Stage3 weak gate pass: `true`.
- Full gate failed only on `thirty_d_concentration_not_worse_than_stage28=false`.

Boundary checks:

- `promotion_allowed=false`.
- Research-only / shadow-only.
- No active manifest, production root, trade plan, paper account, or broker integration is authorized.

## Inferences

- Loss/output calibration is a real lever for the local-state branch. Compared with the prior local-state input scout, `horizon_entropy_regularized_v1` repaired the negative-month blocker and strengthened hit lift.
- The same loss also reintroduces a horizon-concentration problem: 30d concentration `0.805152` is above the current strict v2 baseline and fails the not-worse-than-Stage-2.8 concentration check.
- The result is therefore not a replacement baseline. It is a strong candidate direction that needs horizon-concentration repair before any architecture, input, candidate backtest, or execution-candidate step.
- The current blocker has moved from "local-state input hurts monthly stability" to "local-state plus horizon-entropy loss improves rank/spread/hit/months but over-concentrates the predicted horizon toward long/30d."

## Assumptions

- The scout is evidence-grade for the tested 3-seed loss profile, but only within the local-state input branch.
- It does not prove that local-state input should replace `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
- It does not prove promotion-grade behavior, because no score-to-weight candidate review was run for this branch and the research gate is still `near_pass`.

## Next Work

- Design a narrow v2 horizon-concentration repair scout for `horizon_entropy_regularized_v1`, such as horizon share caps, horizon entropy target calibration, horizon-state penalties, or alternative horizon-grid/output calibration.
- Keep comparison anchored to:
  - current strict v2 baseline `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`;
  - local-state input scout `mh_v2_local_state_input_scout_anchor_20260602_01`;
  - this loss-calibration anchor `mh_v2_local_state_loss_calibration_anchor_20260603_01`.
- Do not move this branch to score-backtest bridge or execution-candidate review until the research gate is full pass or a clearly declared diagnostic exception is approved.

## Artifact Pointers

- Anchor directory: `daily_research/output/path_policy/studies/mh_v2_local_state_loss_calibration_anchor_20260603_01/`.
- 3-seed confirmed run directories:
  - `daily_research/output/path_policy/studies/mh_v2_local_state_loss_calibration_horizon_entropy_regularized_v1_seed7_20260603_01/`
  - `daily_research/output/path_policy/studies/mh_v2_local_state_loss_calibration_horizon_entropy_regularized_v1_seed11_20260603_01/`
  - `daily_research/output/path_policy/studies/mh_v2_local_state_loss_calibration_horizon_entropy_regularized_v1_seed19_20260603_01/`
- Main summary: `daily_research/output/path_policy/studies/mh_v2_local_state_loss_calibration_anchor_20260603_01/v2_local_state_loss_calibration_summary.json`.
- Profile aggregate: `daily_research/output/path_policy/studies/mh_v2_local_state_loss_calibration_anchor_20260603_01/profile_aggregate.csv`.
- Research verdict: `daily_research/output/path_policy/studies/mh_v2_local_state_loss_calibration_anchor_20260603_01/research_verdict.md`.
