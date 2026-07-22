# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_041445`
- snapshot_id: `baostock_v2_2020_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `717104`
- tradeable_rows: `669952`
- metrics_rows: `717104`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |      mean |          min |        max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|----------:|-------------:|-----------:|
| all       | turn      |          717104 |                      1 |          712129 |        0.993062 |        712129 |           6 |  2.45186  |       0      |    106.483 |
| all       | pctChg    |          717104 |                      1 |          717102 |        0.999997 |        717102 |       30009 |  0.07734  |     -11.5385 |    314.993 |
| all       | peTTM     |          717104 |                      1 |          717104 |        1        |        717104 |           0 | 36.1782   | -413347      | 236810     |
| all       | pbMRQ     |          717104 |                      1 |          717104 |        1        |        717104 |           0 |  4.66159  |    -823.002  |  18394.5   |
| all       | psTTM     |          717104 |                      1 |          717104 |        1        |        717104 |           6 |  5.54148  |    -136.804  |   1775     |
| all       | pcfNcfTTM |          717104 |                      1 |          717104 |        1        |        717104 |           0 | 54.9958   | -236729      | 405731     |
| tradeable | turn      |          669952 |                      1 |          669952 |        1        |        669952 |           6 |  2.53165  |       0      |    106.483 |
| tradeable | pctChg    |          669952 |                      1 |          669952 |        1        |        669952 |       21484 |  0.084756 |     -11.5385 |    314.993 |
| tradeable | peTTM     |          669952 |                      1 |          669952 |        1        |        669952 |           0 | 38.2793   | -413347      | 236810     |
| tradeable | pbMRQ     |          669952 |                      1 |          669952 |        1        |        669952 |           0 |  3.35519  |    -297.808  |    643.872 |
| tradeable | psTTM     |          669952 |                      1 |          669952 |        1        |        669952 |           0 |  4.64957  |     -15.2381 |    495.98  |
| tradeable | pcfNcfTTM |          669952 |                      1 |          669952 |        1        |        669952 |           0 | 34.8225   | -236729      | 405731     |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2020 | all       |          717104 |                      1 | turn      |        0.993062 |
|   2020 | all       |          717104 |                      1 | pctChg    |        0.999997 |
|   2020 | all       |          717104 |                      1 | peTTM     |        1        |
|   2020 | all       |          717104 |                      1 | pbMRQ     |        1        |
|   2020 | all       |          717104 |                      1 | psTTM     |        1        |
|   2020 | all       |          717104 |                      1 | pcfNcfTTM |        1        |
|   2020 | tradeable |          669952 |                      1 | turn      |        1        |
|   2020 | tradeable |          669952 |                      1 | pctChg    |        1        |
|   2020 | tradeable |          669952 |                      1 | peTTM     |        1        |
|   2020 | tradeable |          669952 |                      1 | pbMRQ     |        1        |
|   2020 | tradeable |          669952 |                      1 | psTTM     |        1        |
|   2020 | tradeable |          669952 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
