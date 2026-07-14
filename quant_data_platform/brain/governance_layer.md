# Quant Data Platform 治理对象
快照日期：`2026-07-15`

## Governed Objects
### object `active_manifest`
`scope`: 当前 `quant_data_platform/data/qdp_v2/active/active.json`；未来 v3 `active/active.json`。
`invariant`: active switch is atomic；v3 使用 compare-and-swap；candidate、semantic audit、M0 freeze 或 lineage 任一失败时 active bytes 不变。

### object `dataset_manifest`
`scope`: v2/v3 `datasets/<domain>/<dataset_id>/dataset.json`
`invariant`: v3 manifest `4` 必须记录 schema `3.3.0`、stable identity contract、primary key、range、指定 provider、raw bundle/index hash、inputs manifest SHA、build params 与 quality evidence，完整描述 canonical parquet facts。

### object `stable_security_identity`
`scope`: v3 `security_identity`、`symbol_history` 及所有 canonical fact keys。
`invariant`: canonical 不用 provider symbol 作为长期身份；`provider_symbol` 和 `symbol_on_date` 同时保留；代码变化与已知例外通过受版本控制的 `configs/qdp_v3_corrections.json` 显式维护，correction 冲突或 old value 不匹配时阻断构建；未映射主板证券阻断发布。官方文档 hash 可作证据，但不再是发布前置。

### object `immutable_raw_partition`
`scope`: v3 legacy `raw/<domain>/<partition>/versions/<content_sha256>`、紧凑 `raw_bundles/` 与导出的 raw index/receipts。reference 域固定 `year=undated/security_bucket=00`，低频时序域为 `year=<year>/security_bucket=00`，只有 5m 为 `year=<year>/security_bucket=<00..15>`。
`invariant`: 同参数同内容复用 hash；provider 修订创建新 immutable version；CRC/解压/字段错误不是空数据。bundle 必须能通过 SHA256、row-group 和 `raw_index.parquet` 精确回溯源 content hash；只有验证和导出索引全部通过后才能删除对应 legacy 小文件。

### object `factor_semantic_gate`
`scope`: Tushare historical `adj_factor`、cutoff 后 BaoStock 因子事件、人工 corrections 与 `adjust_factor_daily`。
`invariant`: 每只证券首个 2010+ Tushare 因子归一为 1，保留后续相邻比率；cutoff 后仅把 BaoStock 新事件比率续接到历史序列。因子必须有限正值，已知异常通过显式 correction 修正；不再要求跨源全集一致、pre-2010 官方 baseline 或逐事件参考价仲裁。执行价始终使用 raw。

### object `intraday_5m_quality_tier`
`scope`: Tushare proxy、mootdx、BaoStock、旧 TDX 5m raw/canonical。
`invariant`: v3 5m 是唯一分钟事实域；`2010-01-01..2026-07-13` 只采用 Tushare proxy，之后完整 mootdx 优先、完整 BaoStock fallback。strict 股票日必须有 48 根右闭合 bars 和稳定 identity；Tushare 历史期只与同源日线聚合核对分页/单位/缺 bar。禁止跨源拼接或插值，不完整数据进入 quarantine；新 active 不产生 provisional。

### object `raw_market_data`
`scope`: v3 `market_daily_raw`、`market_intraday_5m` and provider raw/staging data；v2 `market_intraday_1m` 仅作为退休前受保护的 legacy active。
`invariant`: preserve original OHLCV units and adjustment state; derived adjusted fields or features are separate outputs.

### object `pit_historical_semantics`
`scope`: 首次发布九域中的 trading calendar、security identity/symbol history、security status、eligible signal 与 next-open tradability。
`invariant`: current snapshots do not backfill historical semantics without PIT/available-time proof.

### object `compact_raw_storage`
`scope`: `QDP_DATA_ROOT` 下的大 Parquet bundle、`QDP_RUNTIME_ROOT` 下的 SQLite/job/cursor/staging，以及 manifest 随附的 `raw_index.parquet`/`raw_receipts.parquet`。
`invariant`: 大数据长期存 H，runtime 小文件只存 C；SQLite 不是发布事实源，发布必须携带带 SHA256 sidecar 的导出 catalog。成功空响应只写 receipt/index，不创建空 Parquet。H 盘已于 2026-07-15 完成 `chkdsk /f` 并通过 clean/healthy/0 bad-sector 闸门；用户明确取消 F 盘备份，不得再把 F 备份写成前置条件。

### object `rebuildable_cache`
`scope`: daily panel, intraday daily features, limit intraday features；v2 旧 5m 可由旧 1m 重建，但 v3 5m 不属于 cache。
`invariant`: cache/derived tables must be reproducible from active raw facts or an explicit source dataset.

### object `data_cleanup`
`scope`: unreferenced dataset dirs, archived repair outputs, staging outputs.
`invariant`: 常规 v3 GC 递归保护 active/candidate/pin/rollback/audit 的 `inputs[]`。legacy raw 小文件仅在 bundle/hash/index 复核后删除。v2 四条分钟链只有在 v3 首次 CAS 发布成功、随后又完成一次 cutoff 后增量发布、daily/5m watermark 同步、coverage/hash/audit/diff/下游/消费者检查及 retirement manifest 全部通过后才能物理删除；任一条件失败不得删除。

### object `provider_secret`
`scope`: `QDP_TUSHARE_PROXY_TOKEN` and proxy transport metadata.
`invariant`: Token 只存在于进程环境；仓库、日志、异常、job、receipt、manifest 和带 Token 的 URL 均不得出现明文。只允许 SHA-256 指纹与非敏感账户元数据。

### object `archived_v1_surface`
`scope`: old lake/catalog/canonical/memmap/daily-update/repair commands.
`invariant`: archived code can be referenced for history but should not re-enter public CLI without explicit redesign.

## Pure Functions
- `select_governed_object(task) -> object`
- `requires_replacement_pointer(object, method) -> bool`
- `classify_domain_status(domain) -> active|cache|derived|archived|optional`
- `can_activate_dataset(manifest) -> bool`
- `can_delete_dataset_dir(path) -> bool`

## Procedures
### procedure `active_pointer_change`
`input`: candidate dataset manifests and validation result.
`steps`: inspect current active；validate candidate manifests and shard paths；write atomic active JSON；run status/check.

### procedure `provider_domain_promotion`
`input`: provider result and target domain.
`steps`: 按指定可信来源检查 schema/unit/PIT/coverage/identity；archive immutable raw；normalize/correct；write recursive dataset manifest；build candidate；semantic audit；CAS activate only after every release gate passes。不得为“证明正确”额外启动历史跨源逐行验证。

### procedure `cleanup_governed_asset`
`input`: cleanup target.
`steps`: `qdp gc --dry-run`；replacement/active graph check；delete only with explicit confirmation；write summary if durable.

## Writeback
- Project facts: this brain.
- Cross-project topology: main brain.
- Long audits, provider reports and cleanup reports: `references/`, `data/audits`, or run manifests.
