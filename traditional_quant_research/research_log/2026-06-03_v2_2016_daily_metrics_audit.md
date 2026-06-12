# v2.1 Daily Metrics Audit

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_v2_2016_daily_metrics_audit.md`.


- run_id: `v2_daily_metrics_audit_20260603_055630`
- snapshot_id: `baostock_v2_2016_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `577686`
- tradeable_rows: `518103`
- metrics_rows: `577686`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |        mean |          min |              max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|------------:|-------------:|-----------------:|
| all       | turn      |          577686 |                      1 |          528954 |        0.915643 |        528954 |           0 |    2.83175  |       0.0012 |     84.6956      |
| all       | pctChg    |          577686 |                      1 |          577054 |        0.998906 |        577054 |       61425 |    0.026554 |     -10.0985 |     44.0922      |
| all       | peTTM     |          577686 |                      1 |          577686 |        1        |        577686 |           0 |  478.961    | -474317      |      3.05844e+06 |
| all       | pbMRQ     |          577686 |                      1 |          577686 |        1        |        577686 |           0 |   10.1875   |   -1581.99   |  20702.3         |
| all       | psTTM     |          577686 |                      1 |          577686 |        1        |        577686 |         885 |   18.5876   |   -1389.45   | 135709           |
| all       | pcfNcfTTM |          577686 |                      1 |          577686 |        1        |        577686 |           0 | 2166.05     | -341442      |      3.70757e+07 |
| tradeable | turn      |          518103 |                      1 |          518103 |        1        |        518103 |           0 |    2.8573   |       0.0012 |     84.6956      |
| tradeable | pctChg    |          518103 |                      1 |          518103 |        1        |        518103 |       12994 |    0.02859  |     -10.0985 |     44.0922      |
| tradeable | peTTM     |          518103 |                      1 |          518103 |        1        |        518103 |           0 |  520.826    | -474317      |      3.05844e+06 |
| tradeable | pbMRQ     |          518103 |                      1 |          518103 |        1        |        518103 |           0 |    8.89733  |   -1581.99   |  15057.1         |
| tradeable | psTTM     |          518103 |                      1 |          518103 |        1        |        518103 |         574 |   13.5428   |    -500.723  | 122165           |
| tradeable | pcfNcfTTM |          518103 |                      1 |          518103 |        1        |        518103 |           0 | 2471.47     | -341442      |      3.70757e+07 |

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

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
