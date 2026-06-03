# v2 Full Industry Month-Start Snapshot

- Date: 2026-06-02
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_month_start_20260602`
- Status: `data_snapshot_complete`

## Summary

Built the full Baostock v2 PIT snapshot with `stock_industry.parquet` for `2016-01-01` to `2026-06-01`.

Industry cache uses `--industry-frequency month-start`: the first trading day of each month is queried from Baostock and then forward-filled by `code` to the daily PIT stock list. This is suitable for long-sample industry exposure and neutralization audits, but it must not be described as exact daily industry-change tracking.

## Cache Coverage

| Year | Frequency | Query dates | Trade dates | Rows | Failures |
|---|---:|---:|---:|---:|---:|
| 2016 | month-start | 12 | 244 | 577686 | 0 |
| 2017 | month-start | 12 | 244 | 640611 | 0 |
| 2018 | month-start | 12 | 243 | 680626 | 0 |
| 2019 | month-start | 12 | 244 | 699798 | 0 |
| 2020 | month-start | 12 | 243 | 717104 | 0 |
| 2021 | month-start | 12 | 243 | 750932 | 0 |
| 2022 | month-start | 12 | 242 | 762466 | 0 |
| 2023 | month-start | 12 | 242 | 771488 | 0 |
| 2024 | month-start | 12 | 242 | 771470 | 0 |
| 2025 | month-start | 12 | 243 | 772799 | 0 |
| 2026 | month-start | 6 | 96 | 306630 | 0 |

## Snapshot Manifest

- `date_min`: `2016-01-04`
- `date_max`: `2026-06-01`
- `trade_date_count`: `2526`
- `security_count`: `3393`
- `daily_universe_rows`: `7451610`
- `daily_bar_rows`: `7451610`
- `daily_status_rows`: `7451610`
- `stock_industry_rows`: `7451610`
- `quality.failure_count`: `0`
- `quality.source_error_count`: `0`
- `quality.tradeable_rows`: `7031085`
- `quality.suspended_like_rows`: `147344`
- `quality.st_rows`: `298861`

## Loader Smoke

`load_tradeable_panel(..., include_industry=True)` was tested on `2026-01-05`.

- Tradeable rows: `3056`
- Non-empty industry rows: `3056`
- `industry_source` exists.
- `source_x` and `source_y` do not exist.

Sample rows:

| date | code | industry | industry_source |
|---|---|---|---|
| 2026-01-05 | 000001.SZ | J66货币金融服务 | baostock:month-start-ffill |
| 2026-01-05 | 000002.SZ | K70房地产业 | baostock:month-start-ffill |
| 2026-01-05 | 000006.SZ | K70房地产业 | baostock:month-start-ffill |

## Interpretation

The previous blocker for true industry exposure audit is removed at the dataset level. The frontier signals can now be re-audited against real Baostock industry labels rather than only `log_amount_mean_20d_z` proxy neutralization.

This does not solve market cap, float market cap, share base, or turnover. Those remain missing from v2 and must either be sourced separately or kept explicitly as proxy-only controls.

Strategy candidate count remains `0`.
