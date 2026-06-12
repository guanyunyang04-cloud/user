# Frontier ML Signal Rebuild

- run_id: `frontier_ml_signal_rebuild_20260604_150037`
- decision: `diagnostic_ml_signal_ready`
- ml_signal_name: `ml_lgbm_xsec_excess_score_h20_prior_fit`
- model_family: `lightgbm.LGBMRegressor`
- years: `[2026]`
- horizon: `20`
- factor_set: `expanded`
- target_col: `xsec_excess_ret_20d`
- feature_count: `14`
- ready_eval_year_count: `1`
- prediction_rows: `292135`
- fit_uses_eval_year_count: `0`
- candidate_count: `0`
- Artifacts: `traditional_quant_research\output\experiments\frontier_ml_signal_rebuild\frontier_ml_signal_rebuild_20260604_150037`

## Plan

|   eval_year |   horizon | ml_signal_name                          | train_start_date   | fit_end_date   | label_cutoff_date   | eval_start_date   | eval_end_date   | train_years              |   train_year_count |   train_row_count |   eval_feature_row_count |   feature_count | features                                                                                                                                                                                                                                                                   | target_col          | fit_uses_eval_year   | evidence_grade          | status   |
|------------:|----------:|:----------------------------------------|:-------------------|:---------------|:--------------------|:------------------|:----------------|:-------------------------|-------------------:|------------------:|-------------------------:|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:---------------------|:------------------------|:---------|
|        2026 |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit | 2021-01-01         | 2025-12-31     | 2025-12-03          | 2026-01-01        | 2026-06-01      | 2021,2022,2023,2024,2025 |                  5 |           3522747 |                   292135 |              14 | reversal_5d_z,momentum_20d_z,ma20_gap_z,neg_volatility_20d_z,log_amount_mean_20d_z,neg_amplitude_20d_z,reversal_1d_z,log_amount_mean_5d_z,log_amount_change_20d_z,range_position_20d_z,volume_price_corr_20d_z,neg_turn_mean_20d_z,turn_change_20d_z,pctchg_align_gap_1d_z | xsec_excess_ret_20d | False                | diagnostic_ml_prior_fit | ready    |

## Training Audit

|   eval_year |   horizon | ml_signal_name                          | train_start_date   | fit_end_date   | label_cutoff_date   | eval_start_date   | eval_end_date   | train_years              |   train_year_count |   train_row_count |   eval_feature_row_count |   feature_count | features                                                                                                                                                                                                                                                                   | target_col          | fit_uses_eval_year   | evidence_grade          | status   | model_class   |   prediction_row_count |   target_mean |   target_std |
|------------:|----------:|:----------------------------------------|:-------------------|:---------------|:--------------------|:------------------|:----------------|:-------------------------|-------------------:|------------------:|-------------------------:|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:---------------------|:------------------------|:---------|:--------------|-----------------------:|--------------:|-------------:|
|        2026 |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit | 2021-01-01         | 2025-12-31     | 2025-12-03          | 2026-01-01        | 2026-06-01      | 2021,2022,2023,2024,2025 |                  5 |           3522747 |                   292135 |              14 | reversal_5d_z,momentum_20d_z,ma20_gap_z,neg_volatility_20d_z,log_amount_mean_20d_z,neg_amplitude_20d_z,reversal_1d_z,log_amount_mean_5d_z,log_amount_change_20d_z,range_position_20d_z,volume_price_corr_20d_z,neg_turn_mean_20d_z,turn_change_20d_z,pctchg_align_gap_1d_z | xsec_excess_ret_20d | False                | diagnostic_ml_prior_fit | ready    | LGBMRegressor |                 292135 |       3.3e-05 |     0.120026 |

## Prediction Sample

|   eval_year | date                | code      |   horizon | ml_signal_name                          |     score | fit_uses_eval_year   |   feature_count | model_status   | evidence_grade          |
|------------:|:--------------------|:----------|----------:|:----------------------------------------|----------:|:---------------------|----------------:|:---------------|:------------------------|
|        2026 | 2026-01-05 00:00:00 | 000001.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit | -0.008374 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000002.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit | -0.006876 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000006.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit | -0.004654 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000007.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.008779 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000008.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.003741 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000009.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit | -0.002735 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000010.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.015416 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000011.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.013491 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000012.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.00484  | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000014.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.008277 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000016.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.004004 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000017.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.00211  | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000019.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.007237 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000020.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.015523 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000021.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit | -0.007664 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000025.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.000784 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000026.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.003434 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000027.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.00158  | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000028.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.007372 | False                |              14 | ready          | diagnostic_ml_prior_fit |
|        2026 | 2026-01-05 00:00:00 | 000029.SZ |        20 | ml_lgbm_xsec_excess_score_h20_prior_fit |  0.003222 | False                |              14 | ready          | diagnostic_ml_prior_fit |

## Interpretation

This experiment creates a prior-fit Baostock-only ML signal for downstream formal gates. It is not a promotion artifact by itself.
