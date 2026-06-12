# v2.1 Daily Metrics Audit

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_v2_2026_daily_metrics_audit.md`.


- run_id: `v2_daily_metrics_audit_20260603_010425`
- snapshot_id: `baostock_v2_2026_to_0601_industry_metrics_smoke_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `306630`
- tradeable_rows: `292483`
- metrics_rows: `306630`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |       mean |            min |             max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|-----------:|---------------:|----------------:|
| all       | turn      |          306630 |                      1 |          305737 |        0.997088 |        305737 |           0 |   3.34132  |      0.0032    |    83.0027      |
| all       | pctChg    |          306630 |                      1 |          305737 |        0.997088 |        305737 |        7260 |   0.046255 |    -90.4412    |   710.152       |
| all       | peTTM     |          306630 |                      1 |          306630 |        1        |        306630 |           0 |   6.76966  | -69424.8       | 42399.1         |
| all       | pbMRQ     |          306630 |                      1 |          306630 |        1        |        306630 |           0 |   4.95816  |  -1822.4       |  6684           |
| all       | psTTM     |          306630 |                      1 |          306630 |        1        |        306630 |           0 |   4.84046  |   -377.95      |   349.726       |
| all       | pcfNcfTTM |          306630 |                      1 |          306630 |        1        |        306630 |           0 | -83.395    |     -1.516e+06 |     3.41711e+06 |
| tradeable | turn      |          292483 |                      1 |          292483 |        1        |        292483 |           0 |   3.39239  |      0.0485    |    83.0027      |
| tradeable | pctChg    |          292483 |                      1 |          292483 |        1        |        292483 |        6748 |   0.049245 |    -90.4412    |   710.152       |
| tradeable | peTTM     |          292483 |                      1 |          292483 |        1        |        292483 |           0 |   7.9303   | -69424.8       | 42399.1         |
| tradeable | pbMRQ     |          292483 |                      1 |          292483 |        1        |        292483 |           0 |   4.40101  |  -1332.73      |  1473.14        |
| tradeable | psTTM     |          292483 |                      1 |          292483 |        1        |        292483 |           0 |   4.71588  |      0.016788  |   349.726       |
| tradeable | pcfNcfTTM |          292483 |                      1 |          292483 |        1        |        292483 |           0 | -86.1748   |     -1.516e+06 |     3.41711e+06 |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2026 | all       |          306630 |                      1 | turn      |        0.997088 |
|   2026 | all       |          306630 |                      1 | pctChg    |        0.997088 |
|   2026 | all       |          306630 |                      1 | peTTM     |        1        |
|   2026 | all       |          306630 |                      1 | pbMRQ     |        1        |
|   2026 | all       |          306630 |                      1 | psTTM     |        1        |
|   2026 | all       |          306630 |                      1 | pcfNcfTTM |        1        |
|   2026 | tradeable |          292483 |                      1 | turn      |        1        |
|   2026 | tradeable |          292483 |                      1 | pctChg    |        1        |
|   2026 | tradeable |          292483 |                      1 | peTTM     |        1        |
|   2026 | tradeable |          292483 |                      1 | pbMRQ     |        1        |
|   2026 | tradeable |          292483 |                      1 | psTTM     |        1        |
|   2026 | tradeable |          292483 |                      1 | pcfNcfTTM |        1        |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
