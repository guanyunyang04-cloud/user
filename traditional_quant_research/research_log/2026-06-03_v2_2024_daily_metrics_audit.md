# v2.1 Daily Metrics Audit

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_v2_2024_daily_metrics_audit.md`.


- run_id: `v2_daily_metrics_audit_20260603_022337`
- snapshot_id: `baostock_v2_2024_industry_metrics_audit_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `771470`
- tradeable_rows: `745054`
- metrics_rows: `771470`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |       mean |          min |         max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|-----------:|-------------:|------------:|
| all       | turn      |          771470 |                      1 |          769001 |          0.9968 |        769001 |           0 |   2.64631  |       0.0002 |     91.6783 |
| all       | pctChg    |          771470 |                      1 |          769001 |          0.9968 |        769001 |       22833 |   0.037348 |     -96.4431 |   1917.42   |
| all       | peTTM     |          771470 |                      1 |          771470 |          1      |        771470 |           0 |  53.3427   |  -44364.7    |  92746.4    |
| all       | pbMRQ     |          771470 |                      1 |          771470 |          1      |        771470 |           0 |   2.88275  |   -1370.63   |   1255.47   |
| all       | psTTM     |          771470 |                      1 |          771470 |          1      |        771470 |           0 |   3.56673  |     -25.1966 |   1367.89   |
| all       | pcfNcfTTM |          771470 |                      1 |          771470 |          1      |        771470 |           0 | -10.5625   | -124660      | 161419      |
| tradeable | turn      |          745054 |                      1 |          745054 |          1      |        745054 |           0 |   2.67119  |       0.0224 |     91.6783 |
| tradeable | pctChg    |          745054 |                      1 |          745054 |          1      |        745054 |       21503 |   0.044359 |     -96.4431 |   1917.42   |
| tradeable | peTTM     |          745054 |                      1 |          745054 |          1      |        745054 |           0 |  54.6066   |  -44364.7    |  92746.4    |
| tradeable | pbMRQ     |          745054 |                      1 |          745054 |          1      |        745054 |           0 |   2.92336  |    -637.82   |   1255.47   |
| tradeable | psTTM     |          745054 |                      1 |          745054 |          1      |        745054 |           0 |   3.35936  |     -23.482  |    686.692  |
| tradeable | pcfNcfTTM |          745054 |                      1 |          745054 |          1      |        745054 |           0 |  -9.24973  | -124660      | 161419      |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2024 | all       |          771470 |                      1 | turn      |          0.9968 |
|   2024 | all       |          771470 |                      1 | pctChg    |          0.9968 |
|   2024 | all       |          771470 |                      1 | peTTM     |          1      |
|   2024 | all       |          771470 |                      1 | pbMRQ     |          1      |
|   2024 | all       |          771470 |                      1 | psTTM     |          1      |
|   2024 | all       |          771470 |                      1 | pcfNcfTTM |          1      |
|   2024 | tradeable |          745054 |                      1 | turn      |          1      |
|   2024 | tradeable |          745054 |                      1 | pctChg    |          1      |
|   2024 | tradeable |          745054 |                      1 | peTTM     |          1      |
|   2024 | tradeable |          745054 |                      1 | pbMRQ     |          1      |
|   2024 | tradeable |          745054 |                      1 | psTTM     |          1      |
|   2024 | tradeable |          745054 |                      1 | pcfNcfTTM |          1      |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
