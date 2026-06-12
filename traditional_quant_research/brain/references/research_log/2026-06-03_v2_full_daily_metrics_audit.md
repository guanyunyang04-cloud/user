# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_060435`
- snapshot_id: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `7451610`
- tradeable_rows: `7031085`
- metrics_rows: `7451610`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |        mean |               min |              max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|------------:|------------------:|-----------------:|
| all       | turn      |         7451610 |                      1 |         7304272 |        0.980227 |       7304272 |           8 |    2.5366   |       0           |    106.483       |
| all       | pctChg    |         7451610 |                      1 |         7440688 |        0.998534 |       7440688 |      381656 |    0.040104 |     -96.4431      |   1917.42        |
| all       | peTTM     |         7451610 |                      1 |         7451610 |        1        |       7451610 |           0 |  112        | -474317           |      3.05844e+06 |
| all       | pbMRQ     |         7451610 |                      1 |         7451610 |        1        |       7451610 |         156 |    5.41122  |   -8566.06        |  20702.3         |
| all       | psTTM     |         7451610 |                      1 |         7451610 |        1        |       7451610 |        2181 |    6.12754  |  -27979.5         | 135709           |
| all       | pcfNcfTTM |         7451610 |                      1 |         7451610 |        1        |       7451610 |        3278 | -226.063    |      -2.72991e+07 |      3.70757e+07 |
| tradeable | turn      |         7031085 |                      1 |         7031085 |        1        |       7031085 |           8 |    2.57541  |       0           |    106.483       |
| tradeable | pctChg    |         7031085 |                      1 |         7031085 |        1        |       7031085 |      227356 |    0.043071 |     -96.4431      |   1917.42        |
| tradeable | peTTM     |         7031085 |                      1 |         7031085 |        1        |       7031085 |           0 |   76.2845   | -474317           |      3.05844e+06 |
| tradeable | pbMRQ     |         7031085 |                      1 |         7031085 |        1        |       7031085 |         154 |    4.64527  |   -1581.99        |  19077.3         |
| tradeable | psTTM     |         7031085 |                      1 |         7031085 |        1        |       7031085 |         983 |    5.15976  |    -923.7         | 122165           |
| tradeable | pcfNcfTTM |         7031085 |                      1 |         7031085 |        1        |       7031085 |        3117 | -242.099    |      -2.72991e+07 |      3.70757e+07 |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2016 | all       |          577686 |                      1 | turn      |        0.915643 |
|   2016 | all       |          577686 |                      1 | pctChg    |        0.998906 |
|   2016 | all       |          577686 |                      1 | peTTM     |        1        |
|   2016 | all       |          577686 |                      1 | pbMRQ     |        1        |
|   2016 | all       |          577686 |                      1 | psTTM     |        1        |
|   2016 | all       |          577686 |                      1 | pcfNcfTTM |        1        |
|   2016 | tradeable |          518103 |                      1 | turn      |        1        |
|   2016 | tradeable |          518103 |                      1 | pctChg    |        1        |
|   2016 | tradeable |          518103 |                      1 | peTTM     |        1        |
|   2016 | tradeable |          518103 |                      1 | pbMRQ     |        1        |
|   2016 | tradeable |          518103 |                      1 | psTTM     |        1        |
|   2016 | tradeable |          518103 |                      1 | pcfNcfTTM |        1        |
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
|   2018 | all       |          680626 |                      1 | turn      |        0.951722 |
|   2018 | all       |          680626 |                      1 | pctChg    |        0.999808 |
|   2018 | all       |          680626 |                      1 | peTTM     |        1        |
|   2018 | all       |          680626 |                      1 | pbMRQ     |        1        |
|   2018 | all       |          680626 |                      1 | psTTM     |        1        |
|   2018 | all       |          680626 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
