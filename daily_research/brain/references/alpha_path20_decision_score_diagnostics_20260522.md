# Alpha Path20 Decision Score Diagnostics - 2026-05-22

## Scope
- Status: `research / shadow-only / no promotion`.
- Active artifact: `daily_research/output/active_execution_strategy.json` unchanged.
- Objective: diagnose whether `decision_utility_v1` instability is caused by the current score aggregation / selection contract, target calibration, input contamination, or regime sensitivity.
- Audit artifact: `daily_research/output/path_policy/studies/path20_decision_score_diagnostics_20260522.json`.
- Tool: `daily_research/path_policy/decision_score_diagnostics.py`.

## Inputs
- Baseline GRU decision output: `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01`.
- Longer train window GRU: `path20_data_len_liquid500_du_train2018_2022_20260522_01`.
- Long train window GRU: `path20_data_len_liquid500_du_train2016_2022_20260522_01`.
- Architecture reference: `path20_decision_arch_stock_mixer_liquid500_du_cost20_hit10_dd010_20260521_01`.
- Score variants audited: `max_pred_utility`, `horizon_selected_utility`, `hit_weighted_utility`, `short_horizon_blend`, `forecast_5d_mu`, `forecast_20d_mu`, `decision_forecast_blend`.
- Gate: validation and test rank IC, spread, and hit lift must all be positive; test monthly spread positive rate must be at least `60%`; negative test month count must be lower than the study's current `max_pred_utility` baseline.

## Gate Results
| Study / score | Test rank IC | Test spread | Test hit lift | Test monthly spread positive rate | Negative test months | Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| GRU baseline / `max_pred_utility` | 0.040376 | 0.009600 | 0.015789 | 50.0% | 2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12 | failed |
| GRU baseline / `horizon_selected_utility` | 0.044307 | 0.010502 | 0.015744 | 50.0% | 2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12 | failed |
| GRU baseline / `forecast_5d_mu` | 0.043293 | 0.008193 | 0.021073 | 50.0% | 2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12 | failed |
| Train2018 GRU / best audited score | 0.027410 | -0.003053 | 0.016377 | 33.3% | 2024-01, 2024-05, 2024-06, 2024-07, 2024-09, 2024-10, 2024-11, 2024-12 | failed |
| Train2016 GRU / `forecast_5d_mu` | 0.030127 | 0.001313 | 0.025899 | 50.0% | 2024-01, 2024-06, 2024-07, 2024-10, 2024-11, 2024-12 | failed |
| Stock mixer / `max_pred_utility` | 0.067616 | 0.033849 | -0.001243 | 83.3% | 2024-01, 2024-06 | failed hit-lift gate |
| Stock mixer / `decision_forecast_blend` | 0.033667 | 0.020844 | -0.000300 | 75.0% | 2024-01, 2024-03, 2024-06 | failed hit-lift gate |

No audited score passed the complete selection calibration gate.

## Interpretation
- The current GRU signal is not zero: rank IC, spread, and hit lift are positive for several aggregate scores.
- Selection-only calibration did not solve monthly instability: the GRU baseline, longer train-window GRU runs, and simple score blends all keep recurring weak months, especially early-year and late-2024 months.
- Longer train windows did not fix the selection problem. The `2016-2022` run improved some hit-lift and monthly-rate signals, but its aggregate spread stayed weak or negative for most decision scores.
- Stock mixer is the most important exception: it has materially stronger test ranking / spread stability (`max_pred_utility` monthly positive rate `83.3%`) but negative hit lift. This suggests ranking signal exists, while the current hit / utility target contract may be miscalibrated.
- The bottleneck is more likely `hit_label` / utility target definition and regime-aware selection than model capacity alone.

## Next Allowed Actions
- Do not run liquid800 or allocator/replay/live from these results.
- First inspect target calibration: hit threshold, utility definition, and how `hit_lift_top20_mean` is computed relative to future utility.
- If target calibration remains coherent, run input contamination ablations on liquid500: `no_alpha_prior` and `no_sector_static`.
- Treat stock mixer as a research candidate for ranking stability only; it is not a promotion candidate because hit lift failed.

## Boundaries
- This reference records forecast-stage diagnostics only.
- No `daily_research/output/active_execution_strategy.json` change is authorized.
- Do not treat any aggregate-positive score as sufficient without monthly / regime stability.
