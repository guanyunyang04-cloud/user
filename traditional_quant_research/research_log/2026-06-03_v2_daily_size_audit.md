# v2.2 Daily Size Audit

- run_id: `v2_daily_size_audit_20260603_195538`
- snapshot_id: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- status: `daily_size_absent`
- universe_rows: `7451610`
- tradeable_rows: `7031085`
- size_rows: `0`
- min_tradeable_coverage: `0.000000`
- min_required_tradeable_coverage: `0.000000`
- min_required_positive_rate: `0.000000`
- required_units_present: `False`
- source_grade_ok: `False`
- current_cross_check_ok: `False`
- current_cross_check_path: `traditional_quant_research/output/experiments/v2_free_size_current_cross_check/v2_free_size_current_cross_check_20260603_195515/daily_size_current_cross_check.csv`
- daily_size_ready_for_research: `False`
- candidate_count: `0`

## Field Summary

| scope     | field            |   expected_rows |   size_row_coverage |   non_null_rows |   coverage_rate |   finite_rows |   positive_rows |   positive_rate |   mean |   min |   max |
|:----------|:-----------------|----------------:|--------------------:|----------------:|----------------:|--------------:|----------------:|----------------:|-------:|------:|------:|
| all       | total_market_cap |         7451610 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| all       | float_market_cap |         7451610 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| all       | total_share      |         7451610 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| all       | float_share      |         7451610 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| all       | free_share       |         7451610 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| tradeable | total_market_cap |         7031085 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| tradeable | float_market_cap |         7031085 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| tradeable | total_share      |         7031085 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| tradeable | float_share      |         7031085 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |
| tradeable | free_share       |         7031085 |                   0 |               0 |               0 |             0 |               0 |               0 |    nan |   nan |   nan |

## Source/Unit Summary

_No rows._

## Current Cross-Check

_No rows._

## Yearly Summary

|   year | scope     |   expected_rows |   size_row_coverage | field            |   coverage_rate |
|-------:|:----------|----------------:|--------------------:|:-----------------|----------------:|
|   2016 | all       |          577686 |                   0 | total_market_cap |               0 |
|   2016 | all       |          577686 |                   0 | float_market_cap |               0 |
|   2016 | all       |          577686 |                   0 | total_share      |               0 |
|   2016 | all       |          577686 |                   0 | float_share      |               0 |
|   2016 | all       |          577686 |                   0 | free_share       |               0 |
|   2016 | tradeable |          518103 |                   0 | total_market_cap |               0 |
|   2016 | tradeable |          518103 |                   0 | float_market_cap |               0 |
|   2016 | tradeable |          518103 |                   0 | total_share      |               0 |
|   2016 | tradeable |          518103 |                   0 | float_share      |               0 |
|   2016 | tradeable |          518103 |                   0 | free_share       |               0 |
|   2017 | all       |          640611 |                   0 | total_market_cap |               0 |
|   2017 | all       |          640611 |                   0 | float_market_cap |               0 |
|   2017 | all       |          640611 |                   0 | total_share      |               0 |
|   2017 | all       |          640611 |                   0 | float_share      |               0 |
|   2017 | all       |          640611 |                   0 | free_share       |               0 |
|   2017 | tradeable |          584379 |                   0 | total_market_cap |               0 |
|   2017 | tradeable |          584379 |                   0 | float_market_cap |               0 |
|   2017 | tradeable |          584379 |                   0 | total_share      |               0 |
|   2017 | tradeable |          584379 |                   0 | float_share      |               0 |
|   2017 | tradeable |          584379 |                   0 | free_share       |               0 |
|   2018 | all       |          680626 |                   0 | total_market_cap |               0 |
|   2018 | all       |          680626 |                   0 | float_market_cap |               0 |
|   2018 | all       |          680626 |                   0 | total_share      |               0 |
|   2018 | all       |          680626 |                   0 | float_share      |               0 |
|   2018 | all       |          680626 |                   0 | free_share       |               0 |
|   2018 | tradeable |          631122 |                   0 | total_market_cap |               0 |
|   2018 | tradeable |          631122 |                   0 | float_market_cap |               0 |
|   2018 | tradeable |          631122 |                   0 | total_share      |               0 |
|   2018 | tradeable |          631122 |                   0 | float_share      |               0 |
|   2018 | tradeable |          631122 |                   0 | free_share       |               0 |

## Interpretation

This audit checks whether the optional v2.2 daily_size table has enough coverage and unit metadata to become a research input. It does not validate PIT timing, does not fetch external data, and does not promote any strategy candidate.
