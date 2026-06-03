# v2.1 Metrics Exposure Diagnostics

- run_id: `v2_metrics_exposure_diagnostics_20260603_064734`
- snapshot_id: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- years: `[2024, 2025, 2026]`
- status: `metrics_exposure_diagnostics`
- max_mean_abs_signal_metric_corr: `0.760402`
- max_mean_abs_basket_metric_active_exposure: `1.170983`
- candidate_count: `0`

## Strongest Signal-Metric Correlations

|   eval_year | signal                                | exposure                     |   date_count |   mean_corr |   mean_abs_corr |   p95_abs_corr |   max_abs_corr |
|------------:|:--------------------------------------|:-----------------------------|-------------:|------------:|----------------:|---------------:|---------------:|
|        2024 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  |          242 |    0.760402 |        0.760402 |       0.867042 |       0.882034 |
|        2024 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  |          242 |    0.738429 |        0.738429 |       0.849573 |       0.892702 |
|        2025 | multifactor_ic_weighted_score         | momentum_20d_z_xsec_z        |          243 |   -0.71264  |        0.71264  |       0.785248 |       0.803796 |
|        2026 | multifactor_ic_weighted_score         | momentum_20d_z_xsec_z        |           96 |   -0.69831  |        0.69831  |       0.758538 |       0.78089  |
|        2025 | multifactor_rolling_ic_weighted_score | momentum_20d_z_xsec_z        |          243 |   -0.693996 |        0.693996 |       0.778378 |       0.806795 |
|        2026 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z |           96 |   -0.678379 |        0.678379 |       0.743562 |       0.756845 |
|        2026 | multifactor_low_corr_rank_score       | momentum_20d_z_xsec_z        |           96 |   -0.663813 |        0.663813 |       0.728245 |       0.748177 |
|        2026 | multifactor_rolling_ic_weighted_score | momentum_20d_z_xsec_z        |           96 |   -0.658243 |        0.658243 |       0.718655 |       0.739586 |
|        2024 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z |          242 |   -0.656447 |        0.656447 |       0.753875 |       0.779325 |
|        2026 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  |           96 |    0.654908 |        0.654908 |       0.79564  |       0.816447 |
|        2025 | multifactor_low_corr_rank_score       | momentum_20d_z_xsec_z        |          243 |   -0.648118 |        0.648118 |       0.726927 |       0.76777  |
|        2026 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z |           96 |   -0.641648 |        0.641648 |       0.717991 |       0.750881 |
|        2026 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z |           96 |   -0.637318 |        0.637318 |       0.717539 |       0.743599 |
|        2026 | multifactor_low_corr_rank_score       | neg_volatility_20d_z_xsec_z  |           96 |    0.627565 |        0.627565 |       0.761484 |       0.777555 |
|        2026 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  |           96 |    0.604785 |        0.604785 |       0.743241 |       0.753306 |
|        2025 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  |          243 |    0.592305 |        0.592305 |       0.734967 |       0.779293 |
|        2025 | multifactor_low_corr_rank_score       | neg_volatility_20d_z_xsec_z  |          243 |    0.583006 |        0.583006 |       0.721609 |       0.77669  |
|        2025 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z |          243 |   -0.580037 |        0.580037 |       0.702494 |       0.745534 |
|        2024 | multifactor_low_corr_rank_score       | momentum_20d_z_xsec_z        |          242 |   -0.560209 |        0.560209 |       0.700734 |       0.777932 |
|        2026 | multifactor_rolling_ic_weighted_score | turn_xsec_z                  |           96 |   -0.558102 |        0.558102 |       0.616614 |       0.643032 |
|        2026 | multifactor_low_corr_rank_score       | turn_xsec_z                  |           96 |   -0.550625 |        0.550625 |       0.606406 |       0.626862 |
|        2026 | multifactor_ic_weighted_score         | turn_xsec_z                  |           96 |   -0.546833 |        0.546833 |       0.602339 |       0.624449 |
|        2025 | multifactor_ic_weighted_score         | turn_xsec_z                  |          243 |   -0.546145 |        0.546145 |       0.60349  |       0.633544 |
|        2025 | multifactor_low_corr_rank_score       | turn_xsec_z                  |          243 |   -0.539761 |        0.539761 |       0.608829 |       0.636933 |
|        2025 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  |          243 |    0.53485  |        0.53485  |       0.706204 |       0.745507 |
|        2025 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z |          243 |   -0.531491 |        0.531491 |       0.690773 |       0.736017 |
|        2024 | multifactor_rolling_ic_weighted_score | turn_xsec_z                  |          242 |   -0.528622 |        0.528622 |       0.604607 |       0.668865 |
|        2025 | multifactor_rolling_ic_weighted_score | turn_xsec_z                  |          243 |   -0.524324 |        0.524324 |       0.58953  |       0.61603  |
|        2024 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z |          242 |   -0.518812 |        0.518812 |       0.732705 |       0.752468 |
|        2024 | multifactor_ic_weighted_score         | turn_xsec_z                  |          242 |   -0.503281 |        0.503281 |       0.591166 |       0.61087  |

## Strongest Basket Metric Active Exposures

|   eval_year | signal                                | factor                       | period_type   |   period_count |   mean_active_exposure |   mean_abs_active_exposure |   max_abs_active_exposure |
|------------:|:--------------------------------------|:-----------------------------|:--------------|---------------:|-----------------------:|---------------------------:|--------------------------:|
|        2024 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | monthly       |              8 |              -1.17098  |                   1.17098  |                  1.24551  |
|        2024 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | quarterly     |              4 |              -1.16926  |                   1.16926  |                  1.23591  |
|        2026 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | monthly       |              2 |              -1.0928   |                   1.0928   |                  1.10418  |
|        2026 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | quarterly     |              2 |              -1.0928   |                   1.0928   |                  1.10418  |
|        2025 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | monthly       |              8 |              -1.07733  |                   1.07733  |                  1.25074  |
|        2025 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | quarterly     |              4 |              -1.04544  |                   1.04544  |                  1.14152  |
|        2024 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | quarterly     |              4 |              -1.02771  |                   1.02771  |                  1.12407  |
|        2026 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | monthly       |              2 |              -1.01427  |                   1.01427  |                  1.01985  |
|        2026 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | quarterly     |              2 |              -1.01427  |                   1.01427  |                  1.01985  |
|        2024 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | monthly       |              7 |              -1.01395  |                   1.01395  |                  1.12407  |
|        2024 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  | monthly       |              7 |               0.985543 |                   0.985543 |                  1.16055  |
|        2024 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | quarterly     |              4 |              -0.983164 |                   0.983164 |                  1.10046  |
|        2026 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | quarterly     |              2 |              -0.976369 |                   0.976369 |                  1.02649  |
|        2026 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | monthly       |              2 |              -0.976369 |                   0.976369 |                  1.02649  |
|        2024 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  | quarterly     |              4 |               0.974086 |                   0.974086 |                  1.07047  |
|        2024 | multifactor_low_corr_rank_score       | log_amount_mean_20d_z_xsec_z | monthly       |              7 |              -0.966408 |                   0.966408 |                  1.10046  |
|        2025 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | monthly       |              7 |              -0.963859 |                   0.963859 |                  1.15232  |
|        2024 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  | monthly       |              8 |               0.96323  |                   0.96323  |                  1.11176  |
|        2024 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  | quarterly     |              4 |               0.937093 |                   0.937093 |                  1.0374   |
|        2025 | multifactor_rolling_ic_weighted_score | log_amount_mean_20d_z_xsec_z | quarterly     |              4 |              -0.903447 |                   0.903447 |                  1.14224  |
|        2026 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  | quarterly     |              2 |               0.837693 |                   0.837693 |                  0.878247 |
|        2026 | multifactor_rolling_ic_weighted_score | neg_volatility_20d_z_xsec_z  | monthly       |              2 |               0.837693 |                   0.837693 |                  0.878247 |
|        2026 | multifactor_low_corr_rank_score       | neg_volatility_20d_z_xsec_z  | quarterly     |              2 |               0.799141 |                   0.799141 |                  0.8012   |
|        2026 | multifactor_low_corr_rank_score       | neg_volatility_20d_z_xsec_z  | monthly       |              2 |               0.799141 |                   0.799141 |                  0.8012   |
|        2025 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | monthly       |              8 |              -0.795414 |                   0.795414 |                  1.09007  |
|        2025 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  | monthly       |              8 |               0.755638 |                   0.755638 |                  1.03228  |
|        2025 | multifactor_ic_weighted_score         | log_amount_mean_20d_z_xsec_z | quarterly     |              4 |              -0.749764 |                   0.749764 |                  0.910215 |
|        2025 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  | quarterly     |              4 |               0.747499 |                   0.747499 |                  0.890349 |
|        2025 | multifactor_rolling_ic_weighted_score | momentum_20d_z_xsec_z        | quarterly     |              4 |              -0.747062 |                   0.747062 |                  1.12467  |
|        2026 | multifactor_ic_weighted_score         | neg_volatility_20d_z_xsec_z  | quarterly     |              2 |               0.741035 |                   0.741035 |                  0.776451 |

## Interpretation

This diagnostic uses `turn`, `pctChg`, liquidity/volatility proxies, and one-day-lagged valuation fields. Lagged valuation fields are exposure diagnostics only because valuation publication timing is not proven. Large signal-metric correlations or basket active exposures identify promotion gates, not strategy candidates.

Artifacts: `traditional_quant_research\output\experiments\v2_metrics_exposure_diagnostics\v2_metrics_exposure_diagnostics_20260603_064734`
