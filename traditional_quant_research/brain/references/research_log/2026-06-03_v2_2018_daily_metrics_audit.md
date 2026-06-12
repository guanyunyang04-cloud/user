# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_051728`
- snapshot_id: `baostock_v2_2018_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `680626`
- tradeable_rows: `631122`
- metrics_rows: `680626`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |       mean |           min |              max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|-----------:|--------------:|-----------------:|
| all       | turn      |          680626 |                      1 |          647767 |        0.951722 |        647767 |           0 |   2.06464  |       1.9e-05 |     82.1754      |
| all       | pctChg    |          680626 |                      1 |          680495 |        0.999808 |        680495 |       55830 |  -0.131255 |     -36.8771  |     44.1442      |
| all       | peTTM     |          680626 |                      1 |          680626 |        1        |        680626 |           0 | 302.614    | -114502       |      2.65383e+06 |
| all       | pbMRQ     |          680626 |                      1 |          680626 |        1        |        680626 |          61 |   5.21974  |   -7855.92    |   9763.41        |
| all       | psTTM     |          680626 |                      1 |          680626 |        1        |        680626 |         523 |   6.79251  |     -36.1753  |  10639.1         |
| all       | pcfNcfTTM |          680626 |                      1 |          680626 |        1        |        680626 |           4 | -66.1313   | -358360       | 107273           |
| tradeable | turn      |          631122 |                      1 |          631122 |        1        |        631122 |           0 |   2.09446  |       1.9e-05 |     82.1754      |
| tradeable | pctChg    |          631122 |                      1 |          631122 |        1        |        631122 |       22103 |  -0.134766 |     -36.8771  |     44.1442      |
| tradeable | peTTM     |          631122 |                      1 |          631122 |        1        |        631122 |           0 |  46.3188   | -114502       |  41913.7         |
| tradeable | pbMRQ     |          631122 |                      1 |          631122 |        1        |        631122 |          61 |   3.94411  |    -537.927   |   9498.94        |
| tradeable | psTTM     |          631122 |                      1 |          631122 |        1        |        631122 |         182 |   5.62045  |      -1.44509 |   9963.34        |
| tradeable | pcfNcfTTM |          631122 |                      1 |          631122 |        1        |        631122 |           4 | -74.0311   | -358360       | 107273           |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2018 | all       |          680626 |                      1 | turn      |        0.951722 |
|   2018 | all       |          680626 |                      1 | pctChg    |        0.999808 |
|   2018 | all       |          680626 |                      1 | peTTM     |        1        |
|   2018 | all       |          680626 |                      1 | pbMRQ     |        1        |
|   2018 | all       |          680626 |                      1 | psTTM     |        1        |
|   2018 | all       |          680626 |                      1 | pcfNcfTTM |        1        |
|   2018 | tradeable |          631122 |                      1 | turn      |        1        |
|   2018 | tradeable |          631122 |                      1 | pctChg    |        1        |
|   2018 | tradeable |          631122 |                      1 | peTTM     |        1        |
|   2018 | tradeable |          631122 |                      1 | pbMRQ     |        1        |
|   2018 | tradeable |          631122 |                      1 | psTTM     |        1        |
|   2018 | tradeable |          631122 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
