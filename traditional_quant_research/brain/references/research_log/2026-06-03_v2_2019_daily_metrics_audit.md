# v2.1 Daily Metrics Audit

- run_id: `v2_daily_metrics_audit_20260603_044440`
- snapshot_id: `baostock_v2_2019_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `699798`
- tradeable_rows: `667879`
- metrics_rows: `699798`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |       mean |           min |         max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|-----------:|--------------:|------------:|
| all       | turn      |          699798 |                      1 |          695433 |        0.993762 |        695433 |           0 |   2.32254  |       5.2e-05 |     84.1869 |
| all       | pctChg    |          699798 |                      1 |          699787 |        0.999984 |        699787 |       31427 |   0.110044 |     -23.2019  |     44.1767 |
| all       | peTTM     |          699798 |                      1 |          699798 |        1        |        699798 |           0 |  43.1622   | -117663       | 236810      |
| all       | pbMRQ     |          699798 |                      1 |          699798 |        1        |        699798 |           0 |   3.77492  |   -8566.06    |   3064.01   |
| all       | psTTM     |          699798 |                      1 |          699798 |        1        |        699798 |         126 |   5.91474  |    -215.451   |   9504.23   |
| all       | pcfNcfTTM |          699798 |                      1 |          699798 |        1        |        699798 |         128 |  11.4555   | -422068       | 231745      |
| tradeable | turn      |          667879 |                      1 |          667879 |        1        |        667879 |           0 |   2.36661  |       5.2e-05 |     84.1869 |
| tradeable | pctChg    |          667879 |                      1 |          667879 |        1        |        667879 |       24858 |   0.117995 |     -12.5     |     44.1767 |
| tradeable | peTTM     |          667879 |                      1 |          667879 |        1        |        667879 |           0 |  50.1799   | -117663       | 236810      |
| tradeable | pbMRQ     |          667879 |                      1 |          667879 |        1        |        667879 |           0 |   3.44051  |    -104.528   |   3064.01   |
| tradeable | psTTM     |          667879 |                      1 |          667879 |        1        |        667879 |           0 |   4.36026  |     -37.042   |    630.796  |
| tradeable | pcfNcfTTM |          667879 |                      1 |          667879 |        1        |        667879 |         128 | -15.0794   | -422068       | 153626      |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2019 | all       |          699798 |                      1 | turn      |        0.993762 |
|   2019 | all       |          699798 |                      1 | pctChg    |        0.999984 |
|   2019 | all       |          699798 |                      1 | peTTM     |        1        |
|   2019 | all       |          699798 |                      1 | pbMRQ     |        1        |
|   2019 | all       |          699798 |                      1 | psTTM     |        1        |
|   2019 | all       |          699798 |                      1 | pcfNcfTTM |        1        |
|   2019 | tradeable |          667879 |                      1 | turn      |        1        |
|   2019 | tradeable |          667879 |                      1 | pctChg    |        1        |
|   2019 | tradeable |          667879 |                      1 | peTTM     |        1        |
|   2019 | tradeable |          667879 |                      1 | pbMRQ     |        1        |
|   2019 | tradeable |          667879 |                      1 | psTTM     |        1        |
|   2019 | tradeable |          667879 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
