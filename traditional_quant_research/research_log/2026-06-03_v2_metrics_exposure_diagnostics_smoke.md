# v2.1 Metrics Exposure Diagnostics

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_v2_metrics_exposure_diagnostics_smoke.md`.


- run_id: `v2_metrics_exposure_diagnostics_20260603_064456`
- snapshot_id: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- years: `[2026]`
- status: `metrics_exposure_diagnostics`
- max_mean_abs_signal_metric_corr: `0.698310`
- max_mean_abs_basket_metric_active_exposure: `1.092798`
- candidate_count: `0`

## Strongest Signal-Metric Correlations

|   eval_year | signal                                | exposure                     |   date_count |   mean_corr |   mean_abs_corr |   p95_abs_corr |   max_abs_corr |
|------------:|:--------------------------------------|:-----------------------------|-------------:|------------:|----------------:|---------------:|---------------:|
|        2026 | multifactor_ic_weighted_score         | momentum_20d_z_xsec_z        |           96 |   -0.69831  |        0.69831  |       0.758538 |       0.78089  |
|        2026 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z |           96 |   -0.678379 |        0.678379 |       0.743562 |       0.756845 |
|        2026 | multifactor_low_corr_rank_score       | momentum_20d_z_xsec_z        |           96 |   -0.663813 |        0.663813 |       0.728245 |       0.748177 |
|        2026 | multifactor_rolling_ic_weighted_score | momentum_20d_z_xsec_z        |           96 |   -0.658243 |        0.658243 |       0.718655 |       0.739586 |
|        2026 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  |           96 |    0.654908 |        0.654908 |       0.79564  |       0.816447 |
|        2026 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z |           96 |   -0.641648 |        0.641648 |       0.717991 |       0.750881 |
|        2026 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z |           96 |   -0.637318 |        0.637318 |       0.717539 |       0.743599 |
|        2026 | multifactor_low_corr_rank_score       | neg_volatility_20d_z_xsec_z  |           96 |    0.627565 |        0.627565 |       0.761484 |       0.777555 |
|        2026 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  |           96 |    0.604785 |        0.604785 |       0.743241 |       0.753306 |
|        2026 | multifactor_rolling_ic_weighted_score | turn_xsec_z                  |           96 |   -0.558102 |        0.558102 |       0.616614 |       0.643032 |
|        2026 | multifactor_low_corr_rank_score       | turn_xsec_z                  |           96 |   -0.550625 |        0.550625 |       0.606406 |       0.626862 |
|        2026 | multifactor_ic_weighted_score         | turn_xsec_z                  |           96 |   -0.546833 |        0.546833 |       0.602339 |       0.624449 |
|        2026 | multifactor_low_corr_rank_score       | pctChg_xsec_z                |           96 |   -0.254116 |        0.259037 |       0.476594 |       0.558548 |
|        2026 | multifactor_ic_weighted_score         | pctChg_xsec_z                |           96 |   -0.237503 |        0.243764 |       0.470206 |       0.550499 |
|        2026 | multifactor_rolling_ic_weighted_score | pctChg_xsec_z                |           96 |   -0.216828 |        0.229059 |       0.463717 |       0.529492 |
|        2026 | multifactor_rolling_ic_weighted_score | psTTM_lag1_xsec_z            |           96 |   -0.147633 |        0.147633 |       0.241894 |       0.246989 |
|        2026 | multifactor_low_corr_rank_score       | psTTM_lag1_xsec_z            |           96 |   -0.135963 |        0.135963 |       0.220459 |       0.242425 |
|        2026 | multifactor_ic_weighted_score         | psTTM_lag1_xsec_z            |           96 |   -0.13528  |        0.13528  |       0.228052 |       0.242614 |
|        2026 | multifactor_rolling_ic_weighted_score | pbMRQ_lag1_xsec_z            |           96 |   -0.028829 |        0.031216 |       0.062875 |       0.074538 |
|        2026 | multifactor_ic_weighted_score         | pbMRQ_lag1_xsec_z            |           96 |   -0.022237 |        0.026826 |       0.052731 |       0.070042 |
|        2026 | multifactor_low_corr_rank_score       | pbMRQ_lag1_xsec_z            |           96 |   -0.022688 |        0.026646 |       0.052773 |       0.070145 |
|        2026 | multifactor_rolling_ic_weighted_score | peTTM_lag1_xsec_z            |           96 |   -0.012684 |        0.01578  |       0.043376 |       0.055282 |
|        2026 | multifactor_ic_weighted_score         | peTTM_lag1_xsec_z            |           96 |   -0.012256 |        0.015287 |       0.041584 |       0.053688 |
|        2026 | multifactor_low_corr_rank_score       | peTTM_lag1_xsec_z            |           96 |   -0.012398 |        0.015223 |       0.040879 |       0.054078 |
|        2026 | multifactor_ic_weighted_score         | pcfNcfTTM_lag1_xsec_z        |           96 |   -0.005553 |        0.014616 |       0.028602 |       0.034684 |
|        2026 | multifactor_low_corr_rank_score       | pcfNcfTTM_lag1_xsec_z        |           96 |   -0.005356 |        0.014422 |       0.027951 |       0.035867 |
|        2026 | multifactor_rolling_ic_weighted_score | pcfNcfTTM_lag1_xsec_z        |           96 |   -0.007033 |        0.014276 |       0.02565  |       0.033216 |

## Strongest Basket Metric Active Exposures

|   eval_year | signal                                | factor                       | period_type   |   period_count |   mean_active_exposure |   mean_abs_active_exposure |   max_abs_active_exposure |
|------------:|:--------------------------------------|:-----------------------------|:--------------|---------------:|-----------------------:|---------------------------:|--------------------------:|
|        2026 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | monthly       |              2 |              -1.0928   |                   1.0928   |                  1.10418  |
|        2026 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | quarterly     |              2 |              -1.0928   |                   1.0928   |                  1.10418  |
|        2026 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | monthly       |              2 |              -1.01427  |                   1.01427  |                  1.01985  |
|        2026 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | quarterly     |              2 |              -1.01427  |                   1.01427  |                  1.01985  |
|        2026 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | quarterly     |              2 |              -0.976369 |                   0.976369 |                  1.02649  |
|        2026 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | monthly       |              2 |              -0.976369 |                   0.976369 |                  1.02649  |
|        2026 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  | quarterly     |              2 |               0.837693 |                   0.837693 |                  0.878247 |
|        2026 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  | monthly       |              2 |               0.837693 |                   0.837693 |                  0.878247 |
|        2026 | multifactor_low_corr_rank_score       | neg_volatility_20d_z_xsec_z  | quarterly     |              2 |               0.799141 |                   0.799141 |                  0.8012   |
|        2026 | multifactor_low_corr_rank_score       | neg_volatility_20d_z_xsec_z  | monthly       |              2 |               0.799141 |                   0.799141 |                  0.8012   |
|        2026 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  | quarterly     |              2 |               0.741035 |                   0.741035 |                  0.776451 |
|        2026 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  | monthly       |              2 |               0.741035 |                   0.741035 |                  0.776451 |
|        2026 | multifactor_ic_weighted_score         | momentum_20d_z_xsec_z        | monthly       |              2 |              -0.630873 |                   0.630873 |                  0.70036  |
|        2026 | multifactor_ic_weighted_score         | momentum_20d_z_xsec_z        | quarterly     |              2 |              -0.630873 |                   0.630873 |                  0.70036  |
|        2026 | multifactor_low_corr_rank_score       | momentum_20d_z_xsec_z        | monthly       |              2 |              -0.618734 |                   0.618734 |                  0.67713  |
|        2026 | multifactor_low_corr_rank_score       | momentum_20d_z_xsec_z        | quarterly     |              2 |              -0.618734 |                   0.618734 |                  0.67713  |
|        2026 | multifactor_rolling_ic_weighted_score | momentum_20d_z_xsec_z        | quarterly     |              2 |              -0.54206  |                   0.54206  |                  0.642906 |
|        2026 | multifactor_rolling_ic_weighted_score | momentum_20d_z_xsec_z        | monthly       |              2 |              -0.54206  |                   0.54206  |                  0.642906 |
|        2026 | multifactor_rolling_ic_weighted_score | turn_xsec_z                  | quarterly     |              2 |              -0.463565 |                   0.463565 |                  0.488436 |
|        2026 | multifactor_rolling_ic_weighted_score | turn_xsec_z                  | monthly       |              2 |              -0.463565 |                   0.463565 |                  0.488436 |
|        2026 | multifactor_ic_weighted_score         | turn_xsec_z                  | quarterly     |              2 |              -0.439451 |                   0.439451 |                  0.464294 |
|        2026 | multifactor_ic_weighted_score         | turn_xsec_z                  | monthly       |              2 |              -0.439451 |                   0.439451 |                  0.464294 |
|        2026 | multifactor_low_corr_rank_score       | turn_xsec_z                  | quarterly     |              2 |              -0.436872 |                   0.436872 |                  0.44939  |
|        2026 | multifactor_low_corr_rank_score       | turn_xsec_z                  | monthly       |              2 |              -0.436872 |                   0.436872 |                  0.44939  |
|        2026 | multifactor_low_corr_rank_score       | peTTM_lag1_xsec_z            | monthly       |              2 |              -0.067832 |                   0.095875 |                  0.163707 |
|        2026 | multifactor_low_corr_rank_score       | peTTM_lag1_xsec_z            | quarterly     |              2 |              -0.067832 |                   0.095875 |                  0.163707 |
|        2026 | multifactor_rolling_ic_weighted_score | psTTM_lag1_xsec_z            | monthly       |              2 |              -0.093912 |                   0.093912 |                  0.097036 |
|        2026 | multifactor_rolling_ic_weighted_score | psTTM_lag1_xsec_z            | quarterly     |              2 |              -0.093912 |                   0.093912 |                  0.097036 |
|        2026 | multifactor_low_corr_rank_score       | psTTM_lag1_xsec_z            | monthly       |              2 |              -0.070769 |                   0.070769 |                  0.07581  |
|        2026 | multifactor_low_corr_rank_score       | psTTM_lag1_xsec_z            | quarterly     |              2 |              -0.070769 |                   0.070769 |                  0.07581  |

## Interpretation

This diagnostic uses `turn`, `pctChg`, liquidity/volatility proxies, and one-day-lagged valuation fields. Lagged valuation fields are exposure diagnostics only because valuation publication timing is not proven. Large signal-metric correlations or basket active exposures identify promotion gates, not strategy candidates.

Artifacts: `traditional_quant_research\output\experiments\v2_metrics_exposure_diagnostics\v2_metrics_exposure_diagnostics_20260603_064456`
