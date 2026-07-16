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
| `adjust_factor` | 10,203,195 | 2010-01-04..2026-07-16 | 5 |
| `corporate_actions` | 28,761 | 2011-11-28..2026-06-26 | 1 |
| `index_constituents` | 2,567,239 | 2011-11-30..2026-06-26 | 1 |
| `industry_concept` | 8,617,077 | 2011-11-22..2026-06-26 | 1 |
| `market_daily_raw` | 10,203,195 | 2010-01-04..2026-07-16 | 37 |
| `market_intraday_5m` | 472,590,576 | 2010-01-04..2026-07-16 | 17 |
| `name_change` | 2,240 | 2011-11-23..2026-06-26 | 1 |
| `security_identity` | 5,534 | identity master | 1 |
| `security_status` | 10,551,539 | 2010-01-04..2026-07-16 | 36 |
| `share_capital` | 8,617,077 | 2011-11-22..2026-06-26 | 1 |
| `symbol_history` | 7,232 | 1990-12-10..current identity | 1 |
| `trading_calendar` | 6,041 | 2010-01-01..2026-07-16 | 4 |
| `universe_snapshot` | 10,551,539 | 2010-01-04..2026-07-16 | 36 |
| `valuation` | 8,617,077 | 2011-11-22..2026-06-26 | 1 |

## Data Source Policy

- Early 5m history comes primarily from the purchased local mootdx-derived ZIP.
- Tushare was used once for early/delisted gaps. Its accepted rows are already
  in the current tables; raw downloads, runtime and provider code were deleted.
- Future updates use mootdx for speed and BaoStock as the free primary/fallback
  source. BaoStock defaults to four isolated process-local connections.
- Provider values are trusted. QDP checks only conversion risks: schema, key,
  identity, OHLC legality, non-negative turnover and exact 48-bar stock-days.
- Sources are never stitched within a stock-day and missing bars are not
  interpolated.
- The mutable price scope is Shanghai/Shenzhen main-board A shares. Historical
  securities remain present; current-name survivor filtering is not used.

## Known Boundary

The table contains 9,845,637 complete 48-bar stock-days. There are no complete
5m days without a daily row. Of the daily rows, 357,558 have no complete 5m
day; 357,550 are in the 2010—2019 early-source boundary and eight are in
2020—2023. The eight recent cases were unavailable as complete days from the
local archive, mootdx, BaoStock, Tushare and the AkShare probe, so they remain
daily-only and are naturally excluded from complete-window research.

The retired one-time Tushare residual run accepted 495 stock-days (23,760
rows). At that run's snapshot it reported 357,193 residual candidates, of
which 357,178 were explicit upstream empty responses. These are historical
execution metrics, not the current daily-minus-5m count; later BaoStock daily
repairs changed the daily inventory.

## Storage and Cleanup

- 1m data, 1m-derived features, daily panel, limit-status copy, old dataset
  generations, qdp_v3, raw/bootstrap/runtime/audit artifacts were deleted.
- Approximately 52.2 GiB of allocated H-drive space was reclaimed.
- Parquet compression is retained because it reduces disk I/O and storage; it
  is not a second dataset. Validation runs on write batches, not repeated
  cross-source full-table copies.
- The 5m table is compacted by natural year: 8,269 old shards became 17 files,
  reducing logical Parquet size from about 6.77 GiB to 5.21 GiB without changing
  its 472,590,576 rows.
- Training packs, memmaps, models and backtests belong to research projects,
  not QDP.

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli list --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date <date> --workers 4 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli compact --dry-run --json
```
