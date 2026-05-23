# Alpha Path20 Horizon Discovery Result - 2026-05-23

## Scope
- Status: `forecast horizon-discovery study / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Study tag: `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Run command used `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol` with `forecast-walkforward-study`, `raw_kline_context_no_alpha_prior_v1`, `gru_sequence_static_context`, liquid500, seed `7`, train `2019-2022`, validation `2023`, test `2024`, `decision_utility_v1`, and horizon grid `1,2,3,5,8,10,15,20,30`.

## Verdict
- The multi-horizon no-alpha GRU run completed: `status=completed`, `evidence_verdict=forecast_test_confirmed`.
- `trade_utility_score` passed the corrected selection calibration gate on validation and test.
- Target calibration Gate A passed for `cost20_hit10_dd0.1`; utility deciles were monotonic and hit label base rate was stable across validation/test.
- This is a research-stage forecast / selection candidate only; it does not authorize liquid800, multi-seed, allocator, replay, live/default, or active artifact promotion.

## Key Evidence
- Dataset ids:
  - `policy_input_bundle__7c8f58d851bce8179e1e9e2d`
  - `policy_pool_view__c11400fa72ad263f3d1eecfa`
  - `policy_sector_board_view__ed15b2873f544e9e9b24aae5`
- Output artifacts:
  - `daily_research/output/path_policy/studies/path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01/study_summary.json`
  - `daily_research/output/path_policy/studies/path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01/target_calibration_audit.json`
  - `daily_research/output/path_policy/studies/path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01/decision_score_diagnostics_with_target_audit.json`
- Training diagnostics: `device=cuda`, completed with seed `7`, `epochs_ran=8`, `best_epoch=2`.

## Metrics
- `trade_utility_score` validation:
  - rank IC `0.113993`
  - top-bottom spread `0.028361`
  - hit lift `0.017385`
  - monthly spread positive rate `81.8%`
  - negative months: `2023-01`, `2023-09`
- `trade_utility_score` test:
  - rank IC `0.099703`
  - top-bottom spread `0.040183`
  - hit lift `0.022097`
  - monthly spread positive rate `100.0%`
  - negative months: none
- Horizon discovery test signal strengthened with longer horizons:
  - `5d`: rank IC `0.030911`, spread `0.007263`
  - `10d`: rank IC `0.045247`, spread `0.013858`
  - `20d`: rank IC `0.064482`, spread `0.028690`
  - `30d`: rank IC `0.087132`, spread `0.045881`

## Interpretation
- Fact: no-alpha input remains valuable under dynamic horizon discovery.
- Fact: the 30d horizon is the strongest tested horizon on test rank/spread, while 1d remains weak or negative.
- Fact: the predicted best horizon distribution collapses heavily toward `30d` on test (`100484` of `109972` rows), while future best horizon remains distributed across the grid.
- Inference: the model is learning a strong longer-horizon utility ranking signal, not a balanced per-sample horizon chooser yet.
- Inference: the original Path20 direction is not invalid, but the better framing is now `multi-horizon utility ranking with strong 20-30d tilt`, not fixed 20d path accuracy.

## Blockers
- Best-horizon distribution collapse must be treated as a calibration blocker before claiming a true horizon chooser.
- This is single seed only; it is not yet robust enough for expansion.
- `forecast_5d_mu`, `forecast_20d_mu`, and `short_horizon_blend` did not pass the corrected gate because test negative months remained.
- The study is still shadow-only and does not include allocator/replay/live execution evidence.

## Next Allowed Actions
- Do not run liquid800, allocator, replay, live/default, or active promotion from this result.
- Next research should compare `trade_utility_score` against a constrained horizon-score variant that prevents near-total 30d collapse.
- If the constrained variant keeps validation/test spread, hit lift, and monthly stability, then run seeds `7,11,19`.
- If multi-seed is stable, then test whether `15/20/30d` can be simplified into a longer-horizon utility family instead of a broad horizon grid.

## Validation
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_target_calibration_audit.py -q`: `8 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_decision_score_diagnostics.py -q`: `8 passed`.
- `test_forecast_dataset.py` four single-test runs passed; full-file run exceeded the previous timeout because each fixture takes about one to two minutes.
- Key forecast training contract tests passed individually:
  - `test_train_forecast_models_accepts_dlinear_sequence_checkpoint_and_predictions`
  - `test_train_forecast_models_records_decision_output_contract_predictions_and_resume_mismatch`
  - `test_train_forecast_models_accepts_custom_horizon_decision_utility_contract`
  - `test_forecast_prediction_metrics_scores_decision_utility_columns`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`: passed with existing 3 brain-integrity warnings.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`: `status=ok`, existing warnings only.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.
- `git diff --check`: passed.
