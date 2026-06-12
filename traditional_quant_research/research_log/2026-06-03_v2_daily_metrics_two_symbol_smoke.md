# v2.1 Daily Metrics Audit

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_v2_daily_metrics_two_symbol_smoke.md`.


- run_id: `v2_daily_metrics_audit_20260603_002131`
- snapshot_id: `baostock_v2_metrics_two_symbol_smoke_20260603`
- status: `metrics_missingness_audit`
- universe_rows: `12`
- tradeable_rows: `12`
- metrics_rows: `12`
- min_tradeable_coverage: `1.000000`
- full_metrics_ready_for_research: `True`
- valuation_fields_ready_for_research: `True`
- candidate_count: `0`

## Field Summary

| scope     | field     |   expected_rows |   metrics_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   zero_rows |      mean |       min |       max |
|:----------|:----------|----------------:|-----------------------:|----------------:|----------------:|--------------:|------------:|----------:|----------:|----------:|
| all       | turn      |              12 |                      1 |              12 |               1 |            12 |           0 |  0.3968   |  0.2254   |  0.7211   |
| all       | pctChg    |              12 |                      1 |              12 |               1 |            12 |           1 |  0.577725 | -2.333    |  2.5328   |
| all       | peTTM     |              12 |                      1 |              12 |               1 |            12 |           0 |  5.50757  |  4.80416  |  6.2465   |
| all       | pbMRQ     |              12 |                      1 |              12 |               1 |            12 |           0 |  0.430839 |  0.401162 |  0.459554 |
| all       | psTTM     |              12 |                      1 |              12 |               1 |            12 |           0 |  1.673    |  1.55527  |  1.79867  |
| all       | pcfNcfTTM |              12 |                      1 |              12 |               1 |            12 |           0 | 26.8427   |  2.90372  | 51.602    |
| tradeable | turn      |              12 |                      1 |              12 |               1 |            12 |           0 |  0.3968   |  0.2254   |  0.7211   |
| tradeable | pctChg    |              12 |                      1 |              12 |               1 |            12 |           1 |  0.577725 | -2.333    |  2.5328   |
| tradeable | peTTM     |              12 |                      1 |              12 |               1 |            12 |           0 |  5.50757  |  4.80416  |  6.2465   |
| tradeable | pbMRQ     |              12 |                      1 |              12 |               1 |            12 |           0 |  0.430839 |  0.401162 |  0.459554 |
| tradeable | psTTM     |              12 |                      1 |              12 |               1 |            12 |           0 |  1.673    |  1.55527  |  1.79867  |
| tradeable | pcfNcfTTM |              12 |                      1 |              12 |               1 |            12 |           0 | 26.8427   |  2.90372  | 51.602    |

## Yearly Summary

|   year | scope     |   expected_rows |   metrics_row_coverage | field     |   coverage_rate |
|-------:|:----------|----------------:|-----------------------:|:----------|----------------:|
|   2026 | all       |              12 |                      1 | turn      |               1 |
|   2026 | all       |              12 |                      1 | pctChg    |               1 |
|   2026 | all       |              12 |                      1 | peTTM     |               1 |
|   2026 | all       |              12 |                      1 | pbMRQ     |               1 |
|   2026 | all       |              12 |                      1 | psTTM     |               1 |
|   2026 | all       |              12 |                      1 | pcfNcfTTM |               1 |
|   2026 | tradeable |              12 |                      1 | turn      |               1 |
|   2026 | tradeable |              12 |                      1 | pctChg    |               1 |
|   2026 | tradeable |              12 |                      1 | peTTM     |               1 |
|   2026 | tradeable |              12 |                      1 | pbMRQ     |               1 |
|   2026 | tradeable |              12 |                      1 | psTTM     |               1 |
|   2026 | tradeable |              12 |                      1 | pcfNcfTTM |               1 |

## Interpretation

This audit checks whether the optional v2.1 daily metrics table has enough coverage to become a research input. It does not validate valuation PIT timing and does not promote any strategy candidate.
