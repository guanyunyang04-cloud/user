# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_035218`
- snapshot_id: `baostock_v2_2021_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `750932`
- tradeable_rows: `706460`
- metrics_rows: `750932`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |      mean |          min |         max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|----------:|-------------:|------------:|
| all       | turn      |          750932 |                      1 |          747893 |        0.995953 |        747893 |           2 |  2.29463  |       0      |     86.5514 |
| all       | pctChg    |          750932 |                      1 |          750577 |        0.999527 |        750577 |       29662 |  0.109204 |     -11.5385 |    306.109  |
| all       | peTTM     |          750932 |                      1 |          750932 |        1        |        750932 |           0 | 44.4545   |  -25930.2    |  66535.2    |
| all       | pbMRQ     |          750932 |                      1 |          750932 |        1        |        750932 |           0 |  6.37683  |    -706.006  |  18394.5    |
| all       | psTTM     |          750932 |                      1 |          750932 |        1        |        750932 |           0 |  3.26995  |  -27979.5    |   2019.99   |
| all       | pcfNcfTTM |          750932 |                      1 |          750932 |        1        |        750932 |           0 | 18.0248   | -855577      | 726541      |
| tradeable | turn      |          706460 |                      1 |          706460 |        1        |        706460 |           2 |  2.3407   |       0      |     86.5514 |
| tradeable | pctChg    |          706460 |                      1 |          706460 |        1        |        706460 |       24274 |  0.103684 |     -11.5385 |    306.109  |
| tradeable | peTTM     |          706460 |                      1 |          706460 |        1        |        706460 |           0 | 42.2104   |  -14072.1    |  60125.3    |
| tradeable | pbMRQ     |          706460 |                      1 |          706460 |        1        |        706460 |           0 |  4.17168  |     -97.8176 |   9429.34   |
| tradeable | psTTM     |          706460 |                      1 |          706460 |        1        |        706460 |           0 |  4.2653   |     -19.7643 |    240.55   |
| tradeable | pcfNcfTTM |          706460 |                      1 |          706460 |        1        |        706460 |           0 | 27.6395   | -855577      | 726541      |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2021 | all       |          750932 |                      1 | turn      |        0.995953 |
|   2021 | all       |          750932 |                      1 | pctChg    |        0.999527 |
|   2021 | all       |          750932 |                      1 | peTTM     |        1        |
|   2021 | all       |          750932 |                      1 | pbMRQ     |        1        |
|   2021 | all       |          750932 |                      1 | psTTM     |        1        |
|   2021 | all       |          750932 |                      1 | pcfNcfTTM |        1        |
|   2021 | tradeable |          706460 |                      1 | turn      |        1        |
|   2021 | tradeable |          706460 |                      1 | pctChg    |        1        |
|   2021 | tradeable |          706460 |                      1 | peTTM     |        1        |
|   2021 | tradeable |          706460 |                      1 | pbMRQ     |        1        |
|   2021 | tradeable |          706460 |                      1 | psTTM     |        1        |
|   2021 | tradeable |          706460 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
