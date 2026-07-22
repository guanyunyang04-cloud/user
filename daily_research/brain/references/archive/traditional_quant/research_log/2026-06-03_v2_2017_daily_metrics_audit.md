# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_053848`
- snapshot_id: `baostock_v2_2017_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `640611`
- tradeable_rows: `584379`
- metrics_rows: `640611`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |       mean |           min |              max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|-----------:|--------------:|-----------------:|
| all       | turn      |          640611 |                      1 |          596876 |        0.931729 |        596876 |           0 |   2.44569  |       2.3e-05 |     87.9464      |
| all       | pctChg    |          640611 |                      1 |          640453 |        0.999753 |        640453 |       62862 |   0.000579 |     -28.9192  |     44.1176      |
| all       | peTTM     |          640611 |                      1 |          640611 |        1        |        640611 |           0 | 242.925    | -125058       |      2.49457e+06 |
| all       | pbMRQ     |          640611 |                      1 |          640611 |        1        |        640611 |          95 |  11.0108   |   -1179.61    |  19077.3         |
| all       | psTTM     |          640611 |                      1 |          640611 |        1        |        640611 |         641 |   8.73323  |   -1050.45    |   1847.17        |
| all       | pcfNcfTTM |          640611 |                      1 |          640611 |        1        |        640611 |           0 | -53.7106   | -572891       | 178303           |
| tradeable | turn      |          584379 |                      1 |          584379 |        1        |        584379 |           0 |   2.47193  |       2.3e-05 |     87.9464      |
| tradeable | pctChg    |          584379 |                      1 |          584379 |        1        |        584379 |       18731 |   0.002848 |     -28.9192  |     44.1176      |
| tradeable | peTTM     |          584379 |                      1 |          584379 |        1        |        584379 |           0 |  74.8158   | -125058       | 138793           |
| tradeable | pbMRQ     |          584379 |                      1 |          584379 |        1        |        584379 |          93 |   9.95387  |   -1179.61    |  19077.3         |
| tradeable | psTTM     |          584379 |                      1 |          584379 |        1        |        584379 |         227 |   7.22268  |       0       |   1037.23        |
| tradeable | pcfNcfTTM |          584379 |                      1 |          584379 |        1        |        584379 |           0 | -61.233    | -572891       | 178303           |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2017 | all       |          640611 |                      1 | turn      |        0.931729 |
|   2017 | all       |          640611 |                      1 | pctChg    |        0.999753 |
|   2017 | all       |          640611 |                      1 | peTTM     |        1        |
|   2017 | all       |          640611 |                      1 | pbMRQ     |        1        |
|   2017 | all       |          640611 |                      1 | psTTM     |        1        |
|   2017 | all       |          640611 |                      1 | pcfNcfTTM |        1        |
|   2017 | tradeable |          584379 |                      1 | turn      |        1        |
|   2017 | tradeable |          584379 |                      1 | pctChg    |        1        |
|   2017 | tradeable |          584379 |                      1 | peTTM     |        1        |
|   2017 | tradeable |          584379 |                      1 | pbMRQ     |        1        |
|   2017 | tradeable |          584379 |                      1 | psTTM     |        1        |
|   2017 | tradeable |          584379 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
