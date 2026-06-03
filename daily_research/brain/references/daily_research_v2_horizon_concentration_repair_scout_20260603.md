# daily_research v2 horizon concentration repair scout

Date: `2026-06-03`

## Scope

This note records a research-only / shadow-only posthoc output calibration scout for `daily_research_v2_research_reset`.

The immediate blocker was created by the local-state loss calibration result: `horizon_entropy_regularized_v1` repaired the local-state monthly-stability problem and strengthened hit lift, but failed the research gate because predicted best horizon concentrated on `30d`.

This scout does not train a model. It reads completed `horizon_entropy_regularized_v1` prediction artifacts and tests whether simple, validation-aware output calibration can reduce 30d concentration while preserving rank IC, top-bottom spread, hit lift, monthly quality, and negative-month limits.

## Facts

- Repair anchor: `mh_v2_horizon_concentration_repair_anchor_20260603_01`.
- Source anchor: `mh_v2_local_state_loss_calibration_anchor_20260603_01`.
- Source loss profile: `horizon_entropy_regularized_v1`.
- Dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool view: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_v1`.
- Model family: `gru_sequence_static_context`.
- Output profile: `decision_utility_v1`.
- Horizon grid: `1,2,3,5,8,10,15,20,30`.
- Seeds: `7,11,19`.
- Baseline test metrics from original prediction semantics:
  - rank IC mean: `0.122041`;
  - spread mean: `0.040043`;
  - hit lift mean: `0.030121`;
  - monthly positive rate mean: `0.848485`;
  - negative month count max: `2`;
  - 30d concentration mean: `0.805152`.

Tested posthoc variants:

- `penalty_30d_0p001`.
- `penalty_30d_0p002`.
- `penalty_30d_0p005`.
- `penalty_long_linear_0p001`.
- `penalty_long_linear_0p002`.
- `validation_center_mean`.
- `validation_center_median`.
- `validation_zscore_rescaled`.
- `validation_zscore_blend_0p50`.

The gate status is `repair_pass`.

Selected variant by the report gate: `penalty_30d_0p005`.

Selected variant test metrics:

- rank IC mean/min: `0.121945 / 0.112109`.
- spread mean/min: `0.039844 / 0.035484`.
- hit lift mean/min: `0.029417 / 0.024863`.
- monthly positive rate mean/min: `0.848485 / 0.818182`.
- negative month count max: `2`.
- 30d concentration mean: `0.109297`.

Other notable diagnostic:

- `validation_zscore_rescaled` has stronger test rank/spread/month stability (`rank_ic_mean=0.155320`, `spread_mean=0.053692`, `monthly_positive_rate_mean=0.969697`, `negative_month_count_max=1`, `30d concentration=0.175759`) but failed the validation quality-floor selection because validation hit lift fell too far versus baseline. It remains a diagnostic direction, not the selected repair.

## Inferences

- The 30d concentration blocker is not inseparable from the signal. Simple output calibration can reduce 30d concentration below the historical Stage 2.8 limit while preserving the main research metrics.
- `penalty_30d_0p005` is the conservative selected posthoc repair because it passes validation and test quality floors with minimal rank/spread/hit damage.
- This suggests the next training-side work should encode a horizon concentration / 30d soft penalty or calibration head rather than abandoning the local-state + horizon-entropy branch.
- Because this is posthoc output calibration, it is not a model-training gate pass and does not by itself authorize score-backtest bridge, execution-candidate review, active promotion, paper trading, broker integration, or execution unfreeze.

## Assumptions

- The posthoc score is a research diagnostic for output semantics, not a production target-weight panel.
- Any future use in a score-backtest bridge must explicitly declare the calibrated score semantics and rerun candidate review under normal cost/rebalance/risk gates.
- A training-side repair is preferred before execution-candidate work, because it would make the horizon constraint part of the model/loss/output contract rather than an ad hoc score override.

## Next Work

- Add a train-side or output-contract scout that internalizes the successful posthoc shape, for example:
  - `horizon_30d_soft_penalty_v1`;
  - `horizon_concentration_cap_v1`;
  - `validation_centered_horizon_head_v1`;
  - a calibrated decision-score head with explicit horizon exposure metadata.
- Compare any train-side candidate against:
  - strict v2 baseline `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`;
  - local-state input scout `mh_v2_local_state_input_scout_anchor_20260602_01`;
  - local-state loss calibration `mh_v2_local_state_loss_calibration_anchor_20260603_01`;
  - this posthoc repair anchor `mh_v2_horizon_concentration_repair_anchor_20260603_01`.
- Only after a train-side or explicitly declared output-calibrated branch passes research gate should it enter score-backtest bridge or execution-candidate review.

## Artifact Pointers

- Implementation: `daily_research/path_policy/v2_horizon_concentration_repair_scout.py`.
- Tests: `daily_research/path_policy/tests/test_v2_horizon_concentration_repair_scout.py`.
- Anchor directory: `daily_research/output/path_policy/studies/mh_v2_horizon_concentration_repair_anchor_20260603_01/`.
- Summary: `daily_research/output/path_policy/studies/mh_v2_horizon_concentration_repair_anchor_20260603_01/v2_horizon_concentration_repair_summary.json`.
- Variant aggregate: `daily_research/output/path_policy/studies/mh_v2_horizon_concentration_repair_anchor_20260603_01/variant_aggregate.csv`.
- Research verdict: `daily_research/output/path_policy/studies/mh_v2_horizon_concentration_repair_anchor_20260603_01/research_verdict.md`.
