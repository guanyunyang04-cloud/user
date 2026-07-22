# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_025539`
- snapshot_id: `baostock_v2_2023_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `771488`
- tradeable_rows: `744550`
- metrics_rows: `771488`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |         mean |               min |         max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|-------------:|------------------:|------------:|
| all       | turn      |          771488 |                      1 |          769678 |        0.997654 |        769678 |           0 |     2.15873  |       0.0006      |     85.0058 |
| all       | pctChg    |          771488 |                      1 |          769678 |        0.997654 |        769678 |       30549 |     0.024295 |     -89.1827      |    753.07   |
| all       | peTTM     |          771488 |                      1 |          771488 |        1        |        771488 |           0 |     3.73657  | -333021           |  60884.5    |
| all       | pbMRQ     |          771488 |                      1 |          771488 |        1        |        771488 |           0 |     4.09787  |    -652.367       |   4531      |
| all       | psTTM     |          771488 |                      1 |          771488 |        1        |        771488 |           0 |     4.52043  |    -923.7         |   9523.28   |
| all       | pcfNcfTTM |          771488 |                      1 |          771488 |        1        |        771488 |           0 | -2421.36     |      -2.72991e+07 | 328420      |
| tradeable | turn      |          744550 |                      1 |          744550 |        1        |        744550 |           0 |     2.18276  |       0.0174      |     85.0058 |
| tradeable | pctChg    |          744550 |                      1 |          744550 |        1        |        744550 |       28521 |     0.029079 |     -89.1827      |    753.07   |
| tradeable | peTTM     |          744550 |                      1 |          744550 |        1        |        744550 |           0 |     5.06275  | -333021           |  60884.5    |
| tradeable | pbMRQ     |          744550 |                      1 |          744550 |        1        |        744550 |           0 |     4.11367  |    -652.367       |   4531      |
| tradeable | psTTM     |          744550 |                      1 |          744550 |        1        |        744550 |           0 |     3.84267  |    -923.7         |    514.477  |
| tradeable | pcfNcfTTM |          744550 |                      1 |          744550 |        1        |        744550 |           0 | -2520.56     |      -2.72991e+07 | 105004      |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2023 | all       |          771488 |                      1 | turn      |        0.997654 |
|   2023 | all       |          771488 |                      1 | pctChg    |        0.997654 |
|   2023 | all       |          771488 |                      1 | peTTM     |        1        |
|   2023 | all       |          771488 |                      1 | pbMRQ     |        1        |
|   2023 | all       |          771488 |                      1 | psTTM     |        1        |
|   2023 | all       |          771488 |                      1 | pcfNcfTTM |        1        |
|   2023 | tradeable |          744550 |                      1 | turn      |        1        |
|   2023 | tradeable |          744550 |                      1 | pctChg    |        1        |
|   2023 | tradeable |          744550 |                      1 | peTTM     |        1        |
|   2023 | tradeable |          744550 |                      1 | pbMRQ     |        1        |
|   2023 | tradeable |          744550 |                      1 | psTTM     |        1        |
|   2023 | tradeable |          744550 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
