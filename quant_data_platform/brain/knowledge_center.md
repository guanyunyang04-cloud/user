# Quant Data Platform 知识对象
快照日期：`2026-07-01`

## Object Classes
### class `manifest_first_data_base`
`definition`: 可审计、可更新的本地数据基底；事实源是 `parquet + dataset.json + active.json`。
`owner`: `quant_data_platform`
`consumers`: research projects consume explicit table/domain names, dataset ids, manifests or downstream packs.
`invariant`: active data base does not depend on archived v1 workflow artifacts or downstream training-artifact pointers.

### class `raw_fact_table`
`examples`: `market_daily_raw`、`market_intraday_1m`、PIT status/state tables、valuation、adjust_factor、event facts。
`invariant`: 不用研究矩形填充或派生特征替代原始事实。

### class `cache_or_derived_table`
`examples`: `market_intraday_5m`、`market_daily_panel`、`intraday_daily_features`、`limit_intraday_features`、`limit_status`。
`invariant`: 可从 raw facts 重建；便于研究读取，但不是第二份真相。

### class `provider_object`
`properties`: domain、endpoint、frequency、unit、adjustment semantics、coverage depth、server/source stability、PIT/available-time status。
`methods`: `probe()`；`archive_raw()`；`normalize()`；`validate()`；`write_manifest()`。

### class `adjust_factor_standard`
`definition`: QDP v2 active `adjust_factor` is a daily dense factor table aligned exactly to `market_daily_raw` keys.
`semantics`: `adjust_factor` equals positive `back_adjust_factor`; source `fore_adjust_factor` is retained as evidence only and may be non-positive.
`invariant`: one row per `(trade_date, symbol)` in `market_daily_raw`; `back_adjust_factor > 0`; `adjust_factor > 0`; missing prior factors are explicit default rows, not silent joins.

### class `industry_concept_complete`
`definition`: QDP v2 active `industry_concept` is aligned exactly to `universe_snapshot` keys.
`invariant`: one row per `(trade_date, symbol)` in universe; blank/missing labels must be filled from same-symbol history or reliable profile metadata when available; persistent `UNKNOWN` is allowed only as explicit unavailable evidence, not as a silent join failure.
`current_state`: active `industry` has `0` blank/UNKNOWN rows as of `2026-06-26`; `concept_tags` are intentionally blank because reliable historical concept tags are not present and are not fabricated.

### class `active_scope_mainboard_non_delisted`
`definition`: the active short-line data base covers Shanghai/Shenzhen A-share mainboard symbols after board/ST/delisting filters.
`invariant`: current active-date names containing `退市` are excluded even if provider status flags do not mark `is_delisted=true`.
`current_scope_name`: `mainboard_hs_a_ex_current_st_name_delisted_v2`
`current_symbol_count`: `3037`

### class `mootdx_online`
`domain`: fast daily/1m/5m market bars and quote-like market data.
`production_role`: preferred market bar source when coverage and unit audit pass.

### class `baostock_online`
`domain`: trading calendar, universe/listing status, security status, industry/concept, index constituents, turnover/valuation and structural daily fields.
`bar_role`: audit/backfill/fallback only unless a task selects it.

### class `cninfo_online`
`domain`: disclosure events and report metadata.
`shortline_default`: not part of current active short-line data base unless a future task explicitly reactivates disclosure research.

## Long-Term Lessons
- 数据基底和研究训练产物要分开：active data base 是源头，memmap/training pack 是下游研究加速产物。
- 1m 和 daily raw 是事实源；5m、daily panel、日内特征和涨跌停特征是缓存或派生。
- DuckDB/catalog 类索引可以是工具，但不能成为本地数据基底唯一事实源。
- 旧迁移、修复、兼容命令不应留在日常 CLI；保留到 archive/reference 即可。
- 质量证明字段如 schema hash 和 audit path 应保留在 manifest，但默认 `describe` 应显示人读摘要。
- 对复权因子不能把原始多来源 factor pool 直接当 active 真相；active 必须是标准化后的一键一行事实表。
- PIT/meta 域的质量证明应写进 manifest：主键唯一、交易日覆盖、scope 过滤、跨域 key 对齐和显式 unknown/default 标记。
- Scope filtering must not inherit stale nested audit blocks from source manifests; transformed datasets need fresh proof or clearly marked inherited proof.

## Pure Functions
- `classify_provider(source) -> market|structure|disclosure|exploration|local_qdp`
- `is_active_truth_source(table) -> bool`
- `is_rebuildable_cache(table) -> bool`
- `requires_pit_available_time(domain) -> bool`
- `can_delete_dataset_dir(path, active_graph) -> bool`
- `is_standard_adjust_factor(table) -> bool`
- `is_complete_universe_aligned_label_table(table) -> bool`
