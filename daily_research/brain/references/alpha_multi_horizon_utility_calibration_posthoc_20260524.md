# Alpha Multi-Horizon Utility Post-Hoc Calibration - 2026-05-24

## Scope
- Status: `research / post-hoc calibration / shadow-only / no retraining / no promotion`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Source run tag: `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`.
- Calibration artifact tag: `mh_utility_horizon_calibration_posthoc_20260524_01`.
- Dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Model family: `gru_sequence_static_context`; seed `7`; feature profile `raw_kline_context_no_alpha_prior_v1`.
- Role years: train `2019-2022`, validation `2023`, test `2024`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Verdict
- Fact: the post-hoc calibration run completed and wrote:
  - `daily_research/output/path_policy/studies/mh_utility_horizon_calibration_posthoc_20260524_01/horizon_calibration_report.json`
  - `daily_research/output/path_policy/studies/mh_utility_horizon_calibration_posthoc_20260524_01/horizon_score_comparison.csv`
- Fact: gate result is `failed_regression`; no tested post-hoc score profile both preserved baseline validation/test quality and reduced horizon concentration enough to pass.
- Inference: current evidence still supports a strong long-horizon utility ranking signal, but not a mature per-sample horizon chooser.
- Boundary: this result does not authorize multi-seed expansion, liquid800, allocator, replay, production root, live/default, trade-plan, paper account, or active manifest changes.

## Profiles Tested
- `baseline_trade_utility`: current `trade_utility_score`.
- `softmax_expected_utility_t015`: temperature-softmax expected utility over horizons.
- `long_blend_15_20_30`: `0.20*u15 + 0.35*u20 + 0.45*u30`.
- `long_blend_10_15_20_30`: `0.10*u10 + 0.20*u15 + 0.35*u20 + 0.35*u30`.
- `top2_blend`: `0.70*top1_utility + 0.30*top2_utility`.

## Key Evidence
- Baseline validation: rank IC `0.113993`, spread `0.028361`, monthly positive `81.8%`, negative months `2023-01`, `2023-09`, `30d` concentration `83.7%`.
- Baseline test: rank IC `0.099703`, spread `0.040183`, monthly positive `100.0%`, negative months none, `30d` concentration `95.7%`.
- `top2_blend` reduced concentration most usefully while staying near baseline:
  - validation rank IC `0.104640`, spread `0.023818`, monthly positive `81.8%`, `30d` concentration `39.1%`.
  - test rank IC `0.099012`, spread `0.038974`, monthly positive `100.0%`, `30d` concentration `58.9%`.
  - It still failed the gate because validation spread fell below the required `90%` baseline floor.
- `softmax_expected_utility_t015` sharply reduced long-horizon share but validation monthly positive rate fell to `72.7%` with three negative months.
- Both long blends preserved test stability but validation monthly positive rate fell to `63.6%` with four negative months.

## Interpretation
- Fact: simple post-hoc calibration is not enough to turn this single-seed model into a balanced horizon chooser.
- Fact: `top2_blend` is the closest near-miss and is worth keeping as a diagnostic comparator, not as an approved profile.
- Inference: the next useful experiments should validate whether the long-horizon ranking family is robust, rather than forcing a broad horizon chooser label.
- Inference: if a true horizon chooser remains the goal, it likely needs training objective or architecture changes, not only score-time blending.

## Next Allowed Actions
- Keep `trade_utility_score` as the current baseline research score for this source study.
- Do not proceed to multi-seed on any calibrated post-hoc profile from this result.
- A next research plan may compare:
  - a deliberately longer-horizon utility family using `15/20/30d` targets,
  - a training-time horizon diversity or entropy regularizer,
  - a two-stage model that separates utility ranking from horizon selection.
- Any future progression still requires validation/test stability first, then multi-seed, then only later allocator/replay/paper-shadow evidence.

## Validation
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_horizon_calibration.py -q`: `8 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_decision_score_diagnostics.py daily_research/path_policy/tests/test_target_calibration_audit.py daily_research/path_policy/tests/test_horizon_calibration.py -q`: `24 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_forecast_training.py::test_train_forecast_models_accepts_custom_horizon_decision_utility_contract daily_research/path_policy/tests/test_forecast_training.py::test_forecast_prediction_metrics_scores_decision_utility_columns -q`: `2 passed`.
