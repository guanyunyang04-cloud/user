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

## Pure Functions
- `classify_provider(source) -> market|structure|disclosure|exploration|local_qdp`
- `is_active_truth_source(table) -> bool`
- `is_rebuildable_cache(table) -> bool`
- `requires_pit_available_time(domain) -> bool`
- `can_delete_dataset_dir(path, active_graph) -> bool`
