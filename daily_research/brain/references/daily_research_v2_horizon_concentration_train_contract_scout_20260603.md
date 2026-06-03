# daily_research v2 horizon concentration train-contract scout

Date: `2026-06-03`

## Scope

This note records the first train-side / output-contract internalization of the posthoc horizon concentration repair found in `mh_v2_horizon_concentration_repair_anchor_20260603_01`.

The branch is research-only / shadow-only. It does not authorize score-backtest bridge promotion, execution-candidate review, paper/live trading, broker integration, production-root rebuild, or active artifact promotion.

## Facts

- Anchor run: `mh_v2_horizon_concentration_train_contract_anchor_20260603_01`.
- Seed7 run: `mh_v2_horizon_30d_soft_penalty_seed7_20260603_01`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_horizon_concentration_train_contract_scout`.
- Dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool view: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_v1`.
- Model family: `gru_sequence_static_context`.
- Output profile: `decision_utility_v1`.
- New loss profile: `horizon_30d_soft_penalty_v1`.
- Source memmap manifest: `daily_research/output/path_policy/studies/mh_v2_local_state_input_scout_seed7_20260602_01/forecast_dataset_manifest.json`.
- Source manifest validation: `ok`; feature store shape `[1699,2602,144]`; local-state features `20`; alpha-like feature count `0`.

Implementation changes:

- Added `horizon_30d_soft_penalty_v1` to `daily_research.path_policy.forecast_training`.
- Added output calibration contract:
  - method: `max_horizon_utility_soft_penalty`;
  - penalized horizon: `30`;
  - utility penalty: `0.005`;
  - applies to `pred_decision_score`, `trade_utility_score`, and `pred_best_horizon`.
- Raw `pred_decision_utility_*` columns remain unmodified model outputs.
- Calibrated utilities are written separately as `calibrated_pred_decision_utility_*`.
- Added train-side calibrated ranking component `calibrated_decision_rank_aux`.
- Added scout CLI: `daily_research.path_policy.v2_horizon_concentration_train_contract_scout`.

Seed7 completed successfully:

- Training summary status: `completed`.
- Evidence verdict: `forecast_test_confirmed`.
- Comparison / gate status: `fail` only because `seed_count_ge_3=false`.
- Single-seed gate checks otherwise passed:
  - rank IC positive: `true`;
  - spread positive: `true`;
  - hit lift positive: `true`;
  - monthly positive rate `>=0.75`: `true`;
  - negative month count `<=2`: `true`;
  - 30d concentration not worse than Stage 2.8: `true`.

Primary `pred_decision_score` test aggregate:

- rank IC: `0.103705`.
- top-bottom spread: `0.038083`.
- hit lift: `0.015980`.
- monthly positive rate: `0.818182`.
- negative month count: `2`.
- 30d concentration: `0.024208`.
- long horizon share: `0.352873`.

Diagnostic score variant:

- `short_horizon_blend` test rank IC: `0.131716`.
- `short_horizon_blend` test spread: `0.043892`.
- `short_horizon_blend` test hit lift: `0.029485`.
- The same horizon concentration diagnostics apply because the score variants share the prediction frame horizon distribution.

## Inferences

- The posthoc `penalty_30d_0p005` repair shape can be internalized into a file-backed loss/output contract.
- The 30d concentration blocker is strongly reduced in the seed7 train-contract branch: `0.024208` versus `0.805152` in the uncalibrated local-state loss branch and `0.109297` in the prior posthoc repair.
- Seed7 alone is not full evidence. The branch remains `scout_only` until seed11/19 confirmation completes and the 3-seed aggregate passes the normal v2 research gate.
- The primary score has lower seed7 hit lift than the prior 3-seed posthoc repair aggregate, while the diagnostic `short_horizon_blend` is stronger. This is a clue for the next confirmation/comparison pass, not a permission to switch scoring semantics.

## Assumptions

- The seed7 scout is enough to justify seed11/19 confirmation if resources are available.
- It is not enough to replace `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
- It is not enough to enter score-backtest bridge or execution-candidate review.

## Boundaries

- `promotion_allowed=false`.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.
- `daily_research/output/active_execution_strategy.json` remains unchanged.
- No old `156` profile or `short_v5b` replay is involved.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_forecast_training.py::test_auxiliary_decision_loss_profiles_record_weight_contract_and_finite_loss daily_research/path_policy/tests/test_forecast_training.py::test_horizon_30d_soft_penalty_profile_calibrates_prediction_score_and_best_horizon daily_research/path_policy/tests/test_v2_horizon_concentration_train_contract_scout.py daily_research/path_policy/tests/test_v2_horizon_concentration_repair_scout.py daily_research/path_policy/tests/test_v2_local_state_loss_calibration_scout.py -q`: `18 passed`.
- `python -m daily_research.path_policy.v2_horizon_concentration_train_contract_scout --write-task-list --json`: completed.
- `python -m daily_research.path_policy.v2_horizon_concentration_train_contract_scout --run-training --seeds 7 --json`: completed.
- `python -m daily_research.path_policy.v2_horizon_concentration_train_contract_scout --run-comparison --seeds 7 --json`: completed.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.
- `git diff --check`: exit `0`, with unrelated `traditional_quant_research` CRLF warning.

## Next Work

- Run seed11/19 confirmation for `horizon_30d_soft_penalty_v1`.
- Rebuild the 3-seed aggregate and compare against:
  - strict v2 baseline `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`;
  - local-state input scout `mh_v2_local_state_input_scout_anchor_20260602_01`;
  - local-state loss calibration `mh_v2_local_state_loss_calibration_anchor_20260603_01`;
  - posthoc repair anchor `mh_v2_horizon_concentration_repair_anchor_20260603_01`.
- Only if the 3-seed branch passes the research gate should it enter score-backtest bridge as a declared research candidate.
