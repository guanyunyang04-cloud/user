# Quant Data Platform 知识对象

快照日期：`2026-07-15`

## Object Classes

### class `manifest_first_data_base`

`definition`: 可审计、可更新的本地数据基底；事实源是 `parquet + dataset.json + active.json`。
`owner`: `quant_data_platform`
`consumers`: research projects consume explicit table/domain names, dataset ids, manifests or downstream packs.
`invariant`: active data base does not depend on archived v1 workflow artifacts or downstream training-artifact pointers.

### class `raw_fact_table`

`examples`: v3 首发核心中的 `market_daily_raw`、`market_intraday_5m`、`security_status_daily`、`adjust_factor_daily` 与 PIT identity/state；valuation、event、financial 等为非阻塞 enrichment；v2 `market_intraday_1m` 只是退休前受保护的旧 active 事实。
`invariant`: 不用研究矩形填充或派生特征替代原始事实。

### class `cache_or_derived_table`

`examples`: `market_daily_panel`、`intraday_daily_features`、`limit_intraday_features`、`limit_status`；只有 v2 的旧 5m 是 1m 派生缓存，v3 5m 不是缓存。
`invariant`: 可从 raw facts 重建；便于研究读取，但不是第二份真相。

### class `provider_object`

`properties`: domain、endpoint、frequency、unit、adjustment semantics、coverage depth、server/source stability、PIT/available-time status。
`methods`: `probe()`；`archive_raw()`；`normalize()`；`validate()`；`write_manifest()`。

### class `adjust_factor_standard`

`definition`: QDP v2 active `adjust_factor` is a daily dense factor table aligned exactly to `market_daily_raw` keys.
`semantics`: 旧表只证明 `adjust_factor` 等于正 `back_adjust_factor` 且键覆盖完整；它没有证明公司行动语义，现有值可在非事件日逐日跟随价格跳变。
`invariant`: 旧表和依赖它的调整收益只可作 provisional 历史证据；v3 以指定单源合同重建：每证券首个 2010+ Tushare 值归一为 1 并保留相邻比率，cutoff 后把 BaoStock 新事件比率续接到已有序列。因子必须有限正值，已知例外通过显式 correction 修正；不再要求跨源或 pre-2010 官方 baseline 证明。

### class `qdp_v3_identity_first_data_base`

`definition`: 合同 `qdp_v3_20260715_trusted_source_5m`（schema `3.3.0`、manifest `4`）以稳定 `security_id`、PIT `symbol_history`、可信单源 raw、紧凑 bundle 和递归 manifest lineage 为核心。
`primary_keys`: 日线/状态/估值/因子为 `security_id + trade_date`；5m 为 `security_id + trade_date + bar_end`；provider code 永久保留为 `provider_symbol` 而不是历史身份。
`identity_rule`: provider symbol 不可自动成为长期身份；代码合并和已知例外必须出现在经版本控制、区间无冲突且 old value 可复核的 `configs/qdp_v3_corrections.json`。官方公告可作为 reason/evidence，但文档 hash 不再是发布前置。
`fixed_regression`: 中航电测/中航成飞、深赤湾A/招商港口、中航善达/招商积余分别使用稳定 security id；历史当前代码重述必须恢复为当日真实代码。security master 中新代码继承的原上市日不能替代官方 symbol 生效区间。

### class `baostock_date_partition_protocol`

`version`: `baostock==0.9.3`
`snapshot`: `query_daily_history_k_AStock(date)` 与 ETF adapter 返回单日全市场 snapshot；一次 A 股响应派生日线、状态和估值。
`events`: `query_daily_adjust_factor(date)` 只返回 `dividOperateDate == query_date` 的事件，不是全市场因子截面。
`parser_rule`: 批量响应直接读取一次 `fields/data`，验证 `per_page_count=20000`，禁止调用普通 `next()`；达到 20,000 行、字段宽度错误、解压/CRC/消息体错误都阻断。
`factor_aliases`: 只显式接受 `adjustFacto`、`adjustFactor`、`adjust_factor`，不按模糊位置映射。
`runtime_rule`: 日期回灌复用一个隔离子进程中的单一 BaoStock login；同日日线与 `query_all_stock` 原子获取。`max_workers=2` 只预取两个日期，本地处理可流水化，但网络 session 和同时在途请求始终为 1。
`rejected_route`: 不允许以历史低错误率自动开启第二个 BaoStock login；live 双 session 探针已证明登录状态会互相失效。速度来自消除逐日登录、覆盖日历复用和 O(N²) 扫描，而不是放宽质量闸门。
`production_role`: 已保存的 2010—2012 date-partition raw 作为 legacy 证据紧凑归档，但不进入 trusted historical canonical；生产路由只在 `2026-07-13` 后维护日线、状态、因子事件和 mootdx 不完整时的完整 5m 股票日。

### class `qdp_v3_factor_event_model`

`definition`: 首次发布只要求 `adjust_factor_daily`；历史以 Tushare `adj_factor` 的证券内相邻比率为准，首个 2010+ 值归一为 1。cutoff 后只接入 BaoStock 新事件比率，避免重复乘入历史变化。
`correction_rule`: `600076` 等已知例外通过 `configs/qdp_v3_corrections.json` 在 normalize 后、canonical 写入前显式修正；raw 永不改写。correction 重复、区间冲突或 old value 不匹配时阻断构建。
`non_requirements`: 首次发布不要求 `adjust_factor_event`、双路径全集一致、xdxr/dividend/官方公告仲裁、pre-2010 baseline 或除权参考价一 tick 证明。
`execution_boundary`: 执行价始终使用 raw；复权只用于通过语义闸门后的收益和特征。

### class `qdp_v3_intraday_5m`

`definition`: v3 唯一分钟事实域；标准右闭合 48 根 bar，主键 `security_id + trade_date + bar_end`，canonical 按自然年和稳定 64 个 security bucket 分片。v3 不存在 1m raw、watermark、转换或发布门。
`historical_source`: `2010-01-01..2026-07-13` 按完整股票日优先使用本地直接 5m、v2 迁移完整日、2020+ mootdx/BaoStock 完整日，Tushare-compatible proxy 只补最终残差。不同来源不逐值比较；跨源日线差异只作诊断，Tushare 5m 仅与同源 Tushare 日线做分页、单位和缺 bar 硬检查。`vol/amount` canonical scale 均为 1.0。
`incremental_source`: 截止日后完整 mootdx 优先；只有 mootdx 股票日不完整才使用完整 BaoStock。任一来源不完整时禁止拼接或插值，不为数值仲裁同时下载两源。
`tiers`: 新生产数据只有 strict/quarantined；结构、identity 或完整性不合格进入 quarantine，旧 provisional 仅为 legacy 可读证据。
`download_rule`: proxy 每证券按时间倒序最多 8,000 行串行翻页并页级断点恢复，证券完成后合并成一个 immutable raw；mootdx 最近窗口从 `start=0, offset=800` 开始，目标日期齐全即停止；BaoStock 只接管完整日，禁止跨源拼接。
`release_gate`: 2010 年以来 PIT 有效主板交易股票日 strict 覆盖低于 98% 阻断，98%—99% 允许发布并告警，至少 99% 为正常；每个预期股票日必须被解释，5m watermark 必须与日线 watermark 完全相等。

### class `tushare_proxy_historical_bootstrap`

`role`: 低频历史基底与历史 5m 最终残差源，不使用 Tushare SDK，不承担截止日后的持续更新，也不宣称上游官方 provenance。
`secret_boundary`: Token 只从 `QDP_TUSHARE_PROXY_TOKEN` 读取；日志、异常、job、receipt、manifest 和 MCP URL 均不得保存明文 Token，只可保存 SHA-256 指纹与公开账户元数据。
`transport`: 三个 worker 各自使用 HTTP session，但共享单一 `96 rpm / burst 1` 限速器、quota 状态和 429 冷却；8,000 行满页必须继续，非零 code、字段宽度、gzip/JSON 或页间顺序错误不得解释为空数据。
`resume`: 页级 staging/cursor；额度耗尽为 `paused_quota`，断网、续期或进程终止后从同一 job 恢复。

### class `qdp_v3_provider_transport`

`baostock_rule`: 日期批量与 symbol-range 是两种独立持久会话协议；生产 QDP v3 长任务复用一个隔离子进程中的单 login，逐 symbol 错误可单独重试，成功 symbol 不重复请求。通用 provider 默认仍保留旧隔离调用合同，只有显式启用的 v3 任务复用 symbol-range session。
`mootdx_rule`: TCP 可连接不等于 TDX 协议可用；节点选择必须同时通过小样本 bars 协议探针。首轮可并行探测候选，已失败节点排除；全源失败须负缓存并快速返回，让单源 fallback 接管，不能为每只证券重复等待全部坏节点。
`scope_rule`: 先迁移/复用本地完整日，再把最终 5m 残差交给代理；额度耗尽后只补核心 `status/factor`。daily_basic、valuation、dividend、financial、industry、index 和 share-capital 不阻塞首次发布。

### class `qdp_v3_compact_raw_storage`

`data_root`: `H:\quant_project\quant_data_platform\data`
`runtime_root`: `H:\quant_project\quant_data_platform\data\qdp_runtime`
`layout`: reference raw 固定 `year=undated/security_bucket=00`，低频时序 raw 按 `year=<year>/security_bucket=00`，只有 5m raw 按 `year=<year>/security_bucket=<00..15>`；均使用 zstd、目标约 384 MiB、最大 1 GiB。canonical 5m 仍使用稳定 64 buckets，与 5m raw 的 16-bucket layout 是两个独立合同。
`index`: SQLite WAL 保存 job/cursor/page/receipt/bundle row-group 映射；发布前导出 `raw_index.parquet` 与 `raw_receipts.parquet` 及 SHA256 sidecar，因此 active 不依赖 C 盘运行数据库。
`compaction_gate`: 逐源 content hash、row count、schema、bundle hash 和 row-group index 全部验证后，`compact --delete-source --yes` 才能删除对应 legacy 小文件；成功空响应只写 receipt/index。
`disk_state`: H 为 exFAT 1 MiB allocation unit，2026-07-15 已完成 `chkdsk H: /f`，dirty bit=false、health=Healthy、operational=OK、bad sectors=0。用户取消 F 盘备份，F 不属于当前恢复或 fallback 路径。

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

`domain`: fast recent daily/5m market bars and quote-like market data；v3 不使用 mootdx 1m。
`production_role`: 2026-07-13 后完整 5m 的优先更新源；只有 mootdx 股票日不完整时才使用 BaoStock 完整日 fallback，不做常态跨源数值审计。
`current_health_boundary`: 节点候选来自持久化 last-good、tdxpy 与 mootdx 内置列表的去重并集；TCP 只排序，真实 `frequency=0` 的 48 bars/15:00 才证明健康。2026-07-14 固定三证券验证通过，冷启动约 9.69 秒、last-good 热启动约 0.08 秒，缓存最快 3 个健康节点；全源失败才负缓存 300 秒。

### class `baostock_online`

`domain`: trading calendar, universe/listing status, security status, industry labels, index constituents, turnover/valuation and structural daily fields.
`bar_role`: v2 中主要作 audit/backfill；v3 在 2020+ 历史残差和 cutoff 后增量中可作为 mootdx 不完整时的完整 5m fallback，同时维护日线、状态和因子事件。只接受完整股票日，不与其他来源拼接。

### class `cninfo_online`

`domain`: disclosure events and report metadata.
`shortline_default`: not part of current active short-line data base unless a future task explicitly reactivates disclosure research.

## Long-Term Lessons

- 数据基底和研究训练产物要分开：active data base 是源头，memmap/training pack 是下游研究加速产物。
- v3 中 daily 与 5m 是事实源，1m 已退出生产合同；“1m 为事实、5m 为缓存”只描述发布前的 v2 legacy active，不能继续作为 QDP 总体语义。
- DuckDB/catalog 类索引可以是工具，但不能成为本地数据基底唯一事实源。
- 旧迁移、修复、兼容命令不应留在日常 CLI；保留到 archive/reference 即可。
- 质量证明字段如 schema hash 和 audit path 应保留在 manifest，但默认 `describe` 应显示人读摘要。
- 对复权因子不能把 provider 绝对尺度直接当跨证券可比真相；active 必须按证券归一、保留相邻比率并形成一键一行事实表。
- PIT/meta 域的质量证明应写进 manifest：主键唯一、交易日覆盖、scope 过滤、跨域 key 对齐和显式 unknown/default 标记。
- Scope filtering must not inherit stale nested audit blocks from source manifests; transformed datasets need fresh proof or clearly marked inherited proof.
- provider 的当前证券代码会重述历史；任何以 symbol 直接作为长期主键的设计都不能证明历史身份正确。
- 对有状态公共 provider，更多 login 不等于更高吞吐；先复用单 session、合并同日请求并流水化本地处理，同时用 OS job lock 防止重复进程浪费带宽和覆盖进度。
- 对从当前向历史分页的 provider，任务分区不能直接等同网络请求分区；应按 provider 的最低重复工作量请求大区间，再在 raw 层切成可恢复的小分区。
- provider 节点健康必须验证实际协议查询；纯 TCP 探活只适合候选排序，不能作为生产可用证明。
- 可信单源减少的是跨源仲裁，不是 schema、单位、identity、PIT、分页、完整性和 correction 约束；这些转换错误仍会直接污染研究。
- 当前 active 可读不等于 lineage 完整；GC 必须递归保护 active/candidate/pin/rollback/audit 的全部 inputs，而不能只保留 active 叶子。

## Pure Functions

- `classify_provider(source) -> market|structure|disclosure|exploration|local_qdp`
- `is_active_truth_source(table) -> bool`
- `is_rebuildable_cache(table) -> bool`
- `requires_pit_available_time(domain) -> bool`
- `can_delete_dataset_dir(path, active_graph) -> bool`
- `is_standard_adjust_factor(table) -> bool`
- `is_complete_universe_aligned_label_table(table) -> bool`
