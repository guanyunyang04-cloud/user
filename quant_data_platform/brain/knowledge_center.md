# Quant Data Platform 知识对象
快照日期：`2026-07-14`

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
`semantics`: 旧表只证明 `adjust_factor` 等于正 `back_adjust_factor` 且键覆盖完整；它没有证明公司行动语义，现有值可在非事件日逐日跟随价格跳变。
`invariant`: 旧表和依赖它的调整收益只可作 provisional 历史证据；v3 必须从已核验事件正向构造日因子，并通过非事件日稳定、事件比率、除权参考价一 tick 与 2010 baseline 证明。

### class `qdp_v3_identity_first_data_base`
`definition`: 以稳定 `security_id`、PIT `symbol_history`、不可变日期 raw 分区和递归 manifest lineage 为核心的下一代数据基底。
`primary_keys`: 日线/状态/估值/因子为 `security_id + trade_date`；5m 为 `security_id + trade_date + bar_end`；provider code 永久保留为 `provider_symbol` 而不是历史身份。
`identity_rule`: 只有官方公告或可靠代码变更记录才能合并两个 symbol；相同 IPO 日期、相似名称或 provider 返回相同行情都不能自动合并。
`fixed_regression`: 中航电测/中航成飞、深赤湾A/招商港口、中航善达/招商积余分别使用稳定 security id；历史当前代码重述必须恢复为当日真实代码。security master 中新代码继承的原上市日不能替代官方 symbol 生效区间。

### class `baostock_date_partition_protocol`
`version`: `baostock==0.9.3`
`snapshot`: `query_daily_history_k_AStock(date)` 与 ETF adapter 返回单日全市场 snapshot；一次 A 股响应派生日线、状态和估值。
`events`: `query_daily_adjust_factor(date)` 只返回 `dividOperateDate == query_date` 的事件，不是全市场因子截面。
`parser_rule`: 批量响应直接读取一次 `fields/data`，验证 `per_page_count=20000`，禁止调用普通 `next()`；达到 20,000 行、字段宽度错误、解压/CRC/消息体错误都阻断。
`factor_aliases`: 只显式接受 `adjustFacto`、`adjustFactor`、`adjust_factor`，不按模糊位置映射。
`runtime_rule`: 日期回灌复用一个隔离子进程中的单一 BaoStock login；同日日线与 `query_all_stock` 原子获取。`max_workers=2` 只预取两个日期，本地处理可流水化，但网络 session 和同时在途请求始终为 1。
`rejected_route`: 不允许以历史低错误率自动开启第二个 BaoStock login；live 双 session 探针已证明登录状态会互相失效。速度来自消除逐日登录、覆盖日历复用和 O(N²) 扫描，而不是放宽质量闸门。

### class `qdp_v3_factor_event_model`
`definition`: `adjust_factor_event` 与 `adjust_factor_daily` 分离；batch date-events 与 legacy symbol-history 首次重建必须做事件键和值的双路径全集比较。
`arbitration`: 只在 mootdx xdxr 与带官方文档 hash 的公告证据能唯一解释冲突时接受 disputed 事件；单源 xdxr、单源 BaoStock 或无历史 baseline 都不得自动进入 strict。
`execution_boundary`: 执行价始终使用 raw；复权只用于通过语义闸门后的收益和特征。

### class `qdp_v3_intraday_5m`
`definition`: 标准右闭合 48 根 bar；完整 mootdx 优先，完整 BaoStock fallback，任何不完整来源都不得跨源拼接。
`tiers`: 2020 年以来逐证券探针决定 strict 覆盖；2011-11-22..2019-12-31 旧 TDX 只为 provisional；更早明确无覆盖；1m 退出 active 但不删除。
`release_gate`: 2020 年以来 PIT 有效交易股票日 strict 覆盖至少 99.95%，每个 strict 股票日必须恰好 48 根且来源冲突已解决。

### class `industry_concept_complete`
`definition`: QDP v2 active `industry_concept` is aligned exactly to `universe_snapshot` keys.
`invariant`: one row per `(trade_date, symbol)` in universe; blank/missing labels must be filled from same-symbol history or reliable profile metadata when available; persistent `UNKNOWN` is allowed only as explicit unavailable evidence, not as a silent join failure.
`current_state`: active `industry` has `0` blank/UNKNOWN rows as of `2026-06-26`; concept tags are not stored in the active short-line data base because no reliable PIT concept-tag source is active.

### class `active_scope_mainboard_non_delisted`
`definition`: the active short-line data base covers Shanghai/Shenzhen A-share mainboard symbols after board/ST/delisting filters.
`invariant`: current active-date names containing `退市` are excluded even if provider status flags do not mark `is_delisted=true`.
`current_scope_name`: `mainboard_hs_a_ex_current_st_name_delisted_v2`
`current_symbol_count`: `3037`

### class `mootdx_online`
`domain`: fast daily/1m/5m market bars and quote-like market data.
`production_role`: preferred market bar source when coverage and unit audit pass.

### class `baostock_online`
`domain`: trading calendar, universe/listing status, security status, industry labels, index constituents, turnover/valuation and structural daily fields.
`bar_role`: v2 中主要作 audit/backfill；v3 中承担批量日线、状态、估值、日历、因子事件、财务和 5m 历史回补。

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
- provider 的当前证券代码会重述历史；任何以 symbol 直接作为长期主键的设计都不能证明历史身份正确。
- 对有状态公共 provider，更多 login 不等于更高吞吐；先复用单 session、合并同日请求并流水化本地处理，同时用 OS job lock 防止重复进程浪费带宽和覆盖进度。
- 因子键覆盖、正值、来源字段齐全与 daily 对齐都只是结构证明，不等价于因子语义正确。
- 当前 active 可读不等于 lineage 完整；GC 必须递归保护 active/candidate/pin/rollback/audit 的全部 inputs，而不能只保留 active 叶子。

## Pure Functions
- `classify_provider(source) -> market|structure|disclosure|exploration|local_qdp`
- `is_active_truth_source(table) -> bool`
- `is_rebuildable_cache(table) -> bool`
- `requires_pit_available_time(domain) -> bool`
- `can_delete_dataset_dir(path, active_graph) -> bool`
- `is_standard_adjust_factor(table) -> bool`
- `is_complete_universe_aligned_label_table(table) -> bool`
