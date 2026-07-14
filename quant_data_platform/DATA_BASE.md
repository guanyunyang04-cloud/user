# QDP Data Base: v2 Active, v3 Rebuild

Canonical brain source: `quant_data_platform/brain/state_center.md`.

This file describes the current active local data base. The source of truth is:

1. `data/qdp_v2/active/active.json`
2. `data/qdp_v2/datasets/<domain>/<dataset_id>/dataset.json`
3. Parquet shards referenced by each `dataset.json`

No separate catalog is required to know what the active data base contains.

## Current Safety Status (2026-07-14)

- v2 remains the only active data base. Its active manifest SHA-256 is `e56f72a6cba8bcf86055817f6a0ec5e7391271fb3c27b4d628c3abc62944051e`.
- QDP v3 contract is `qdp_v3_20260714_bootstrap_5m` / schema `3.2.0` / manifest `3`. It has no 1-minute domain: `market_intraday_5m` is the sole intraday canonical fact, not a 1m-derived cache.
- QDP v3 已完成 2010—2012 共 729 个交易日的全 A 股日线、`query_all_stock` 与日期因子事件 strict raw 回灌；Tushare-compatible proxy 的参考域和历史 5m bootstrap 正在可恢复运行。旧三年 candidate 只证明 identity/PIT/manifest 链路，因子和新 5m-only 发布合同尚未闭合。
- M0 freeze found 17 missing v2 source ancestors referenced by the 17 active dataset manifests. The leaf manifests and parquet data remain readable, but lineage is incomplete；用户已明确授权以 39 个现存 dataset/23,777 个文件的逐文件 hash 证明替代不可恢复祖先，使重建可继续，但该合同不等于 `lineage_complete=true`。v3 发布与验证前禁止删除；发布后只有 5m coverage/hash、diff、下游切换、无消费者/job 和 retirement manifest 全通过，才可删除四条旧分钟链。
- The old v2 adjustment-factor checks proved key coverage, positivity and provenance only. They did not prove company-action semantics. `600076.SH/2024` is a fixed counterexample with non-event factor jumps, so old adjusted returns and dependent research remain provisional.
- BaoStock 0.9.3 full compatibility passed across 34 anchors with zero issues; the ordinary multi-page proof returned 5537 rows and the wheel hash matched the lock. Live strict smoke also passed for three A-share dates and one factor-event date; ETF remains provisional by design.
- 下载 runtime 已将 BaoStock 日期与 symbol-range 长任务改为任务内单 login 复用；Tushare 历史 5m 使用 8,000 行反向分页、页级断点和单证券 immutable raw；mootdx 节点集合来自 last-good、tdxpy 与 mootdx 自带列表，固定三证券的 48-bar/15:00 协议探针已通过，冷启动约 9.69 秒、热启动约 0.08 秒，失败时负缓存 300 秒并走 BaoStock 完整日 fallback。

Detailed implementation and audit state: `brain/references/qdp_v3_rebuild_20260713.md`.

## Scope

- Market: Shanghai A-share main board plus Shenzhen A-share main board.
- Excluded: ChiNext, STAR Market, ST stocks, delisted stocks.
- Current names containing `退市` are excluded even when a provider status flag does not mark `is_delisted=true`.
- Active window: `2011-11-22` to `2026-06-26`.
- Special continuity boundary: `600036.SH` starts at `2016-07-25`.
- Active symbol scope count: `3037`.
- This 3037-symbol core scope is conditioned on the 2026-06-26 survivor set. It must not be used as proof of a survivorship-free historical universe.
- A separate candidate research scope, `pit_mainboard_non_st_v1`, reconstructs date-local eligibility for 2016-01-04..2026-06-01 without the current-survivor filter. Its datasets are built and audited but are not present in the current `active.json`.
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
| `universe_snapshot` | Core 3037-symbol daily universe facts; current-survivor conditioned | raw | 1d | 8,617,077 | 2011-11-22..2026-06-26 |
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

Candidate research-scope datasets (not active):

| Domain | Contract | Layer | Frequency | Rows | Coverage |
|---|---|---:|---:|---:|---:|
| `pit_signal_universe` | Date-local main-board non-ST research/signal eligibility | derived | 1d | 7,451,610 | 2016-01-04..2026-06-01 |
| `pit_signal_universe_daily` | Daily eligibility counts and deterministic membership hash | derived | 1d | 2,526 | 2016-01-04..2026-06-01 |

## QDP v3 Target Contract

- Formal data begins at `2010-01-01`; later-listed securities begin on their actual listing date.
- Low-frequency raw preserves all A shares. Historical 5m covers the PIT superset of Shanghai/Shenzhen main-board identities, including delisted, historical ST and code-change securities.
- Through `2026-07-13`, a complete Tushare-proxy 5m day must reconcile to daily OHLCV/amount before becoming strict. Raw `vol` is shares and raw `amount` is CNY; both canonical scales are `1.0`.
- After `2026-07-13`, complete mootdx is preferred and complete BaoStock is fallback. Sources are never stitched, and a material conflict between two complete sources is quarantined.
- Quality is evidence-driven for every year; 2020 is not a tier boundary. Strict coverage from 2010 must be at least 99.95%, with exactly 48 right-closed bars per normal trading stock-day.
- The 5m watermark must equal the daily watermark. There is no 1m watermark, lag allowance, raw domain, build path or update task.
- Tushare proxy is a temporary bootstrap source with non-exposed upstream provenance, not an official truth source. Its Token exists only in `QDP_TUSHARE_PROXY_TOKEN` and is forbidden from repository files, logs, jobs, receipts and manifests.

## Layer Rules

- `raw` means stored source facts or normalized source facts.
- `raw-derived` in the table above describes the current v2 active only: its `market_intraday_5m` is derived from v2 `market_intraday_1m`. QDP v3 classifies 5m as a canonical fact and has no 1m table.
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
- Candidate `pit_signal_universe` proof: unique symbol-date keys, date-local eligibility, no current-survivor filter, no future name/`out_date` feature exposure, exact match to source tradeability semantics, and one deterministic membership hash per date. This proves the candidate data, not active-pointer promotion.
- `qdp check --full` is not a routine close-out command for this local setup because it repeats very large row-level scans; use the targeted proof commands above.

Known boundaries:

- Price OHLC cross-frequency differences still exist historically, but the close mismatch rate is near zero and large price differences are a small minority. These are treated as provider/source口径 differences, not active file corruption.
- `limit_intraday_features` uses only 1m OHLCV. It cannot contain L2-only fields such as order-book queue size or sealed order amount.
- Long-horizon disclosure/fundamental datasets are not part of this short-line active data base.
- `adjust_factor.adjust_factor` uses positive `back_adjust_factor` semantics. Source `fore_adjust_factor` is retained as evidence but may be non-positive and should not be used as a positive multiplicative factor.
- `industry_concept.industry` has no blank or `UNKNOWN` rows. The remaining new-stock gap for `001399.SZ` was filled from AkShare/CNInfo company profile industry.
- `industry_concept` intentionally stores only industry labels. Historical concept tags are not included because no reliable PIT concept-tag source is active.
- `pit_signal_universe` fixes the universe-definition boundary only. Core OHLCV/intraday/limit/auxiliary tables still cover the current 3037-symbol scope, so restored multi-channel history for the additional securities remains a separate requirement.
- The PIT research scope source ends on 2026-06-01, 18 open dates before the core QDP end date; it fully covers the planned 2018–2025 research windows.

## Common Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --verify-files
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli list
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check meta --runtime fast --writeback --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --provider mootdx --as-of-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --provider tushare-proxy --as-of-date 2026-07-13 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 update --bootstrap --historical-provider tushare-proxy --start-date 2010-01-01 --as-of-date 2026-07-13 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli retire-v2-intraday --expect-active-sha <v3-active-sha> --delete --yes --json
```
