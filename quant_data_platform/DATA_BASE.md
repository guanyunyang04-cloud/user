# QDP Current Data Store

Canonical state: `quant_data_platform/brain/state_center.md`.

QDP now contains one mutable research data store. The source of truth is:

1. `data/qdp_v2/active/active.json`
2. `data/qdp_v2/datasets/<domain>/<current_id>/dataset.json`
3. the Parquet shards referenced by that manifest

`qdp_v2` is retained only as a directory name. There is no v3 generation,
candidate, publish or rollback surface.

## Current Tables (2026-07-16)

| Domain | Rows | Range | Shards |
|---|---:|---|---:|
| `adjust_factor` | 10,190,067 | 2010-01-04..2026-07-13 | 1 |
| `corporate_actions` | 28,761 | 2011-11-28..2026-06-26 | 1 |
| `index_constituents` | 2,567,239 | 2011-11-30..2026-06-26 | 1 |
| `industry_concept` | 8,617,077 | 2011-11-22..2026-06-26 | 1 |
| `market_daily_raw` | 10,193,252 | 2010-01-04..2026-07-13 | 34 |
| `market_intraday_5m` | 472,107,360 | 2010-01-04..2026-07-13 | 8,237 |
| `name_change` | 2,240 | 2011-11-23..2026-06-26 | 1 |
| `security_identity` | 5,534 | identity master | 1 |
| `security_status` | 10,541,959 | 2010-01-04..2026-07-13 | 33 |
| `share_capital` | 8,617,077 | 2011-11-22..2026-06-26 | 1 |
| `symbol_history` | 7,232 | 1990-12-10..current identity | 1 |
| `trading_calendar` | 6,038 | 2010-01-01..2026-07-13 | 3 |
| `universe_snapshot` | 10,541,959 | 2010-01-04..2026-07-13 | 33 |
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

## Known Boundary

After the 2020—2026 repair, eight traded daily rows still have no complete
48-bar day from local data, mootdx, BaoStock, Tushare or the AkShare probe.
They are retained as daily facts but absent from 5m research windows. This is
8 of 109,886 targeted days (`0.0073%`) and is not filled artificially.

## Storage and Cleanup

- 1m data, 1m-derived features, daily panel, limit-status copy, old dataset
  generations, qdp_v3, raw/bootstrap/runtime/audit artifacts were deleted.
- Approximately 52.2 GiB of allocated H-drive space was reclaimed.
- Parquet compression is retained because it reduces disk I/O and storage; it
  is not a second dataset. Validation runs on write batches, not repeated
  cross-source full-table copies.
- Training packs, memmaps, models and backtests belong to research projects,
  not QDP.

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli list --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.qdp_v2.recent_market_repair all --start-date <date> --end-date <date> --workers 4 --workspace-root H:\quant_project
```
