# QDP v2 Data Base

This file describes the current active local data base. The source of truth is:

1. `data/qdp_v2/active/active.json`
2. `data/qdp_v2/datasets/<domain>/<dataset_id>/dataset.json`
3. Parquet shards referenced by each `dataset.json`

No separate catalog is required to know what the active data base contains.

## Scope

- Market: Shanghai A-share main board plus Shenzhen A-share main board.
- Excluded: ChiNext, STAR Market, ST stocks, delisted stocks.
- Current names containing `退市` are excluded even when a provider status flag does not mark `is_delisted=true`.
- Active window: `2011-11-22` to `2026-06-26`.
- Special continuity boundary: `600036.SH` starts at `2016-07-25`.
- Active symbol scope count: `3037`.
- Primary use case: short-line price/volume research.
- Memmap is not part of the data base. It is a downstream research/training artifact and is not recorded in `active.json`.

## Active Tables

| Domain | Simple meaning | Layer | Frequency | Rows | Range |
|---|---|---|---:|---:|---|
| `market_daily_raw` | Real daily OHLCV bars only | raw | 1d | 8,392,689 | 2011-11-22..2026-06-26 |
| `market_intraday_1m` | 1-minute OHLCV bars, 240 bars/day | raw | 1m | 2,014,245,360 | 2011-11-22..2026-06-26 |
| `market_intraday_5m` | 5-minute OHLCV bars derived from 1m, 48 bars/day | raw-derived | 5m | 402,849,072 | 2011-11-22..2026-06-26 |
| `market_daily_panel` | Rectangular daily research panel with `has_bar` | cache | 1d | 10,758,956 | 2011-11-22..2026-06-26 |
| `trading_calendar` | Trading calendar | raw | calendar | 5,331 | 2011-11-22..2026-06-26 |
| `universe_snapshot` | PIT tradable universe scope | raw | 1d | 8,617,077 | 2011-11-22..2026-06-26 |
| `security_status` | PIT listing/ST/suspension/status fields | raw | 1d | 8,617,077 | 2011-11-22..2026-06-26 |
| `valuation` | Daily valuation fields with market-cap fields | raw | 1d | 8,617,077 | 2011-11-22..2026-06-26 |
| `adjust_factor` | Standard daily back-adjust factors aligned to daily raw keys | raw | 1d | 8,392,689 | 2011-11-22..2026-06-26 |
| `industry_concept` | Industry labels aligned to universe | raw | 1d | 8,617,077 | 2011-11-22..2026-06-26 |
| `index_constituents` | Daily PIT-style index membership facts | raw | 1d | 2,567,239 | 2011-11-30..2026-06-26 |
| `limit_status` | Daily limit-up/down close status | raw-derived | 1d | 205,694 | 2011-11-23..2026-06-26 |
| `corporate_actions` | Corporate action event facts | raw | event | 28,761 | 2011-11-28..2026-06-26 |
| `share_capital` | Daily PIT share-capital facts | raw | 1d | 8,617,077 | 2011-11-22..2026-06-26 |
| `name_change` | Name-change event facts | raw | event | 2,240 | 2011-11-23..2026-06-26 |
| `intraday_daily_features` | Daily features summarized from intraday bars | derived | 1d | 8,392,689 | 2011-11-22..2026-06-26 |
| `limit_intraday_features` | 1m-derived limit-board features | derived | 1d | 8,392,689 | 2011-11-22..2026-06-26 |

## Layer Rules

- `raw` means stored source facts or normalized source facts.
- `raw-derived` means the table is stored for speed but must be reproducible from a lower-level raw table. Current `market_intraday_5m` is derived from `market_intraday_1m`.
- `derived` means feature tables that can be rebuilt from raw tables.
- `cache` means a convenience table for research access. Current `market_daily_panel` should not replace `market_daily_raw` as the source fact table.
- `active.json` is intentionally flat: `datasets.<domain> = <dataset_id>`. Layer meaning lives in each `dataset.json`.
- `qdp describe <table>` prints a compact human summary by default.
- `qdp describe <table> --full --json` prints the complete `dataset.json`, including schema hashes and audit paths.

## Quality State

The current active data base has passed:

- `qdp status --json`
- `qdp check --quick --json`
- `qdp check meta --runtime fast --duckdb-memory-limit 12GB --threads 4 --writeback --json`
- Full primary-key proof for 1m and 5m using symbol-date-bar-time overlap checks.
- Exact primary-key checks for valuation and daily intraday feature tables.
- Cross-frequency audit: daily rows and 1m symbol-days are aligned over the active window.
- PIT/meta/factor/index proof: calendar continuity, universe/security_status open-date coverage, scope filtering, adjustment-factor alignment to `market_daily_raw`, industry alignment to `universe_snapshot`, and index-constituent date/scope checks all pass.
- `qdp check --full` is not a routine close-out command for this local setup because it repeats very large row-level scans; use the targeted proof commands above.

Known boundaries:

- Price OHLC cross-frequency differences still exist historically, but the close mismatch rate is near zero and large price differences are a small minority. These are treated as provider/source口径 differences, not active file corruption.
- `limit_intraday_features` uses only 1m OHLCV. It cannot contain L2-only fields such as order-book queue size or sealed order amount.
- Long-horizon disclosure/fundamental datasets are not part of this short-line active data base.
- `adjust_factor.adjust_factor` uses positive `back_adjust_factor` semantics. Source `fore_adjust_factor` is retained as evidence but may be non-positive and should not be used as a positive multiplicative factor.
- `industry_concept.industry` has no blank or `UNKNOWN` rows. The remaining new-stock gap for `001399.SZ` was filled from AkShare/CNInfo company profile industry.
- `industry_concept` intentionally stores only industry labels. Historical concept tags are not included because no reliable PIT concept-tag source is active.

## Common Commands

```bash
conda run -n yolos python -m quant_data_platform.cli status
conda run -n yolos python -m quant_data_platform.cli list
conda run -n yolos python -m quant_data_platform.cli describe market_intraday_1m
conda run -n yolos python -m quant_data_platform.cli describe market_intraday_1m --full --json
conda run -n yolos python -m quant_data_platform.cli check --quick --json
conda run -n yolos python -m quant_data_platform.cli check meta --runtime fast --writeback --json
conda run -n yolos python -m quant_data_platform.cli rebuild scope-active --runtime fast --workers 4 --activate --json
conda run -n yolos python -m quant_data_platform.cli rebuild limit-intraday --runtime fast --json
conda run -n yolos python -m quant_data_platform.cli gc --dry-run --with-size --json
```
