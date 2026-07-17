# QDP Current Data Store

Canonical brain source: `quant_data_platform/brain/state_center.md`.

QDP now contains one mutable research data store. The source of truth is:

1. `data/qdp_v2/active/active.json`
2. `data/qdp_v2/datasets/<domain>/<current_id>/dataset.json`
3. the Parquet shards referenced by that manifest

`qdp_v2` is retained only as a directory name. There is no v3 generation,
candidate, publish or rollback surface.

## Current Tables (2026-07-16)

| Domain | Rows | Range | Shards |
|---|---:|---|---:|
| `adjust_factor` | 9,644,664 | 2010-01-04..2026-07-16 | 5 |
| `corporate_actions` | 28,755 | 2011-11-28..2026-06-26 | 1 |
| `index_constituents` | 2,567,239 | 2011-11-30..2026-06-26 | 1 |
| `industry_concept` | 8,611,140 | 2011-11-22..2026-06-26 | 1 |
| `market_daily_raw` | 9,644,664 | 2010-01-04..2026-07-16 | 37 |
| `market_intraday_5m` | 462,928,800 | 2010-01-04..2026-07-16 | 17 |
| `name_change` | 2,229 | 2011-11-23..2026-06-26 | 1 |
| `security_identity` | 3,192 | current identity master | 1 |
| `security_status` | 9,927,671 | 2010-01-04..2026-07-16 | 34 |
| `share_capital` | 8,611,140 | 2011-11-22..2026-06-26 | 1 |
| `symbol_history` | 4,677 | current identities, including prior codes | 1 |
| `trading_calendar` | 6,041 | 2010-01-01..2026-07-16 | 4 |
| `universe_snapshot` | 9,927,671 | 2010-01-04..2026-07-16 | 36 |
| `valuation` | 8,611,140 | 2011-11-22..2026-06-26 | 1 |

## Data Source Policy

- Early 5m history comes primarily from the purchased local mootdx-derived ZIP.
- Accepted targeted Tushare repairs are already in the current table; their raw
  downloads are not retained. Tushare is an explicit, targeted historical-gap
  fallback for current-scope securities only, not a routine update source.
- Future updates use mootdx for speed and BaoStock as the free primary/fallback
  source. BaoStock owns daily/status/factor updates and defaults to four isolated
  process-local connections for 5m fallback.
- Provider values are trusted. QDP checks only conversion risks: schema, key,
  identity, OHLC legality, non-negative turnover, factor semantics and exact
  48-bar stock-days.
- Sources are never stitched within a stock-day and missing bars are not
  interpolated.
- The mutable price scope is the 3,192 currently listed Shanghai/Shenzhen
  main-board A shares. Current ST names remain. When a security formally leaves
  the listed universe, all of its history is removed from QDP by design.

## Known Boundary

The table contains 9,644,350 complete 48-bar stock-days. Against 9,644,351
positive-volume daily rows, exactly one complete 5m day is missing:
`600568.SH / 2011-11-21`. The local 1m/5m archive, BaoStock, mootdx, the earlier
Tushare call and the current Tushare-compatible proxy all return no complete
minute data for it. It remains daily-only and is excluded automatically from
complete-window research. Historical complete-day coverage is
99.99998963123594%; the latest date, `2026-07-16`, is 3,191/3,191 (100%).

The full audit has zero invalid 48-bar days, daily/5m OHLC or turnover
mismatches, 100x volume errors, or non-zero suspended bars. Daily/factor keys
match exactly; non-positive factors, uncompensated short-interval changes and
excessive-change symbol-years are all zero. Sixteen raw-price discontinuities
after 53—990 day suspension/restructuring gaps remain as training-window
warnings, not factor-contamination failures.

Historical status was also normalized after finding that the old table mixed
ST labels into `is_suspended` and missed 27 long-suspension ranges. QDP changed
10,338 rows to suspended and 5,190 rows to tradable. The current invariant is:
`is_suspended` iff the daily row is absent or its volume is non-positive;
`is_st` remains independent. All 9,927,671 status rows now pass this check.

## Storage and Cleanup

- 1m data, 1m-derived features, daily panel, limit-status copy, old dataset
  generations, qdp_v3 and obsolete raw/bootstrap artifacts were deleted.
- Approximately 52.2 GiB of allocated H-drive space was reclaimed.
- Parquet compression is retained because it reduces disk I/O and storage; it
  is not a second dataset. Validation runs on write batches, not repeated
  cross-source full-table copies.
- The 5m table is compacted by natural year into 17 files, currently about
  5.09 GiB for 462,928,800 rows.
- Training packs, memmaps, models and backtests belong to research projects,
  not QDP.

Latest formal full audit:
`data/qdp_v2/audits/database_audit_20260717T092836+0000.json`.

## Commands

```powershell
$env:PYTHONPATH = 'H:\quant_project\quant_data_platform\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli list --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date <date> --workers 4 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli compact --dry-run --json
```
