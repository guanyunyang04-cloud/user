# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_014212`
- snapshot_id: `baostock_v2_2025_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `772799`
- tradeable_rows: `742360`
- metrics_rows: `772799`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |        mean |              min |         max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|------------:|-----------------:|------------:|
| all       | turn      |          772799 |                      1 |          770721 |        0.997311 |        770721 |           0 |    3.22291  |      0.0015      |     88.7098 |
| all       | pctChg    |          772799 |                      1 |          770721 |        0.997311 |        770721 |       26813 |    0.131616 |    -89.3258      |    606.831  |
| all       | peTTM     |          772799 |                      1 |          772799 |        1        |        772799 |           0 |   38.2565   | -90126.6         | 148628      |
| all       | pbMRQ     |          772799 |                      1 |          772799 |        1        |        772799 |           0 |    3.86202  |  -2878.38        |   2959.86   |
| all       | psTTM     |          772799 |                      1 |          772799 |        1        |        772799 |           0 |    4.13448  |   -329.233       |    219.288  |
| all       | pcfNcfTTM |          772799 |                      1 |          772799 |        1        |        772799 |           0 | -126.066    |     -1.49093e+06 | 121937      |
| tradeable | turn      |          742360 |                      1 |          742360 |        1        |        742360 |           0 |    3.25918  |      0.0256      |     88.7098 |
| tradeable | pctChg    |          742360 |                      1 |          742360 |        1        |        742360 |       25260 |    0.130845 |    -89.3258      |    606.831  |
| tradeable | peTTM     |          742360 |                      1 |          742360 |        1        |        742360 |           0 |   40.2244   | -90126.6         | 148628      |
| tradeable | pbMRQ     |          742360 |                      1 |          742360 |        1        |        742360 |           0 |    3.96543  |   -544.9         |   2959.86   |
| tradeable | psTTM     |          742360 |                      1 |          742360 |        1        |        742360 |           0 |    3.96321  |      0.018237    |    219.288  |
| tradeable | pcfNcfTTM |          742360 |                      1 |          742360 |        1        |        742360 |           0 | -125.995    |     -1.49093e+06 | 121937      |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2025 | all       |          772799 |                      1 | turn      |        0.997311 |
|   2025 | all       |          772799 |                      1 | pctChg    |        0.997311 |
|   2025 | all       |          772799 |                      1 | peTTM     |        1        |
|   2025 | all       |          772799 |                      1 | pbMRQ     |        1        |
|   2025 | all       |          772799 |                      1 | psTTM     |        1        |
|   2025 | all       |          772799 |                      1 | pcfNcfTTM |        1        |
|   2025 | tradeable |          742360 |                      1 | turn      |        1        |
|   2025 | tradeable |          742360 |                      1 | pctChg    |        1        |
|   2025 | tradeable |          742360 |                      1 | peTTM     |        1        |
|   2025 | tradeable |          742360 |                      1 | pbMRQ     |        1        |
|   2025 | tradeable |          742360 |                      1 | psTTM     |        1        |
|   2025 | tradeable |          742360 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
