# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_032246`
- snapshot_id: `baostock_v2_2022_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `762466`
- tradeable_rows: `728743`
- metrics_rows: `762466`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |         mean |             min |         max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|-------------:|----------------:|------------:|
| all       | turn      |          762466 |                      1 |          760083 |        0.996875 |        760083 |           0 |     2.5702   |      0.0001     |     83.3311 |
| all       | pctChg    |          762466 |                      1 |          760083 |        0.996875 |        760083 |       22986 |    -0.007115 |    -89.5775     |    488.444  |
| all       | peTTM     |          762466 |                      1 |          762466 |        1        |        762466 |           0 |    40.7891   | -31549.9        |  75339.5    |
| all       | pbMRQ     |          762466 |                      1 |          762466 |        1        |        762466 |           0 |     4.15407  |  -1339.1        |   4487.31   |
| all       | psTTM     |          762466 |                      1 |          762466 |        1        |        762466 |           0 |     4.21996  |  -7858.75       |  16377.2    |
| all       | pcfNcfTTM |          762466 |                      1 |          762466 |        1        |        762466 |        3146 | -1204.26     |     -2.4508e+07 | 619046      |
| tradeable | turn      |          728743 |                      1 |          728743 |        1        |        728743 |           0 |     2.61245  |      0.0001     |     83.3311 |
| tradeable | pctChg    |          728743 |                      1 |          728743 |        1        |        728743 |       20880 |    -0.005025 |    -89.5775     |    488.444  |
| tradeable | peTTM     |          728743 |                      1 |          728743 |        1        |        728743 |           0 |    38.359    | -31549.9        |  74419.1    |
| tradeable | pbMRQ     |          728743 |                      1 |          728743 |        1        |        728743 |           0 |     3.81591  |   -133.751      |   3075.97   |
| tradeable | psTTM     |          728743 |                      1 |          728743 |        1        |        728743 |           0 |     3.79888  |   -815.776      |    438.314  |
| tradeable | pcfNcfTTM |          728743 |                      1 |          728743 |        1        |        728743 |        2985 | -1277.08     |     -2.4508e+07 | 619046      |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2022 | all       |          762466 |                      1 | turn      |        0.996875 |
|   2022 | all       |          762466 |                      1 | pctChg    |        0.996875 |
|   2022 | all       |          762466 |                      1 | peTTM     |        1        |
|   2022 | all       |          762466 |                      1 | pbMRQ     |        1        |
|   2022 | all       |          762466 |                      1 | psTTM     |        1        |
|   2022 | all       |          762466 |                      1 | pcfNcfTTM |        1        |
|   2022 | tradeable |          728743 |                      1 | turn      |        1        |
|   2022 | tradeable |          728743 |                      1 | pctChg    |        1        |
|   2022 | tradeable |          728743 |                      1 | peTTM     |        1        |
|   2022 | tradeable |          728743 |                      1 | pbMRQ     |        1        |
|   2022 | tradeable |          728743 |                      1 | psTTM     |        1        |
|   2022 | tradeable |          728743 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
