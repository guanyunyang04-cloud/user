# Quant Data Platform 治理对象
快照日期：`2026-07-14`

## Governed Objects
### object `active_manifest`
`scope`: 当前 `quant_data_platform/data/qdp_v2/active/active.json`；未来 v3 `active/active.json`。
`invariant`: active switch is atomic；v3 使用 compare-and-swap；candidate、semantic audit、M0 freeze 或 lineage 任一失败时 active bytes 不变。

### object `dataset_manifest`
`scope`: v2/v3 `datasets/<domain>/<dataset_id>/dataset.json`
`invariant`: schema、stable identity contract、primary key、range、provider/version/wheel hash、raw content hash、inputs manifest SHA、build params 与 quality evidence 必须完整描述 parquet facts。

### object `stable_security_identity`
`scope`: v3 `security_identity`、`symbol_history` 及所有 canonical fact keys。
`invariant`: canonical 不用 provider symbol 作为长期身份；symbol merge 必须有官方/可靠代码变更证据；`provider_symbol` 和 `symbol_on_date` 同时保留；未映射主板证券阻断发布。

### object `immutable_raw_partition`
`scope`: v3 `raw/<domain>/<partition>/versions/<content_sha256>` 与 receipt/revision chain。
`invariant`: 同参数同内容复用 hash；provider 修订创建新 immutable version 并链接前版本；CRC/解压/字段错误不是空数据；raw hash 变化必须有 revision evidence。

### object `factor_semantic_gate`
`scope`: batch factor events、legacy symbol history、xdxr/official evidence、factor daily 与调整收益。
`invariant`: 双路径事件键和值一致或被官方证据唯一仲裁；非事件日不跳变；因子有限正值；参考价误差不超过一 tick；baseline unproven 不得产生 strict adjusted return；执行价只用 raw。

### object `intraday_5m_quality_tier`
`scope`: Tushare proxy、mootdx、BaoStock、旧 TDX 5m raw/canonical。
`invariant`: v3 5m 是唯一分钟事实域；单个 strict 股票日来自一个完整来源并有 48 根右闭合 bars、稳定 identity 与日线量价证明；禁止跨源拼接；质量不以 2020 固定分层，旧 TDX 单源仅 provisional；未解决冲突进入 quarantine。

### object `raw_market_data`
`scope`: v3 `market_daily_raw`、`market_intraday_5m` and provider raw/staging data；v2 `market_intraday_1m` 仅作为退休前受保护的 legacy active。
`invariant`: preserve original OHLCV units and adjustment state; derived adjusted fields or features are separate outputs.

### object `pit_historical_semantics`
`scope`: trading calendar, universe, security status, valuation, index constituents, industry labels and event availability.
`invariant`: current snapshots do not backfill historical semantics without PIT/available-time proof.

### object `rebuildable_cache`
`scope`: daily panel, intraday daily features, limit intraday features；v2 旧 5m 可由旧 1m 重建，但 v3 5m 不属于 cache。
`invariant`: cache/derived tables must be reproducible from active raw facts or an explicit source dataset.

### object `data_cleanup`
`scope`: unreferenced dataset dirs, archived repair outputs, staging outputs.
`invariant`: 常规 v3 GC 递归保护 active/candidate/pin/rollback/audit 的 `inputs[]`，不可达超过 30 天才移入 `.trash`，再保留 7 天。用户已单独授权在 v3 CAS 发布、5m 覆盖/hash、semantic audit、v2/v3 diff、下游切换、无消费者/job 与 retirement manifest 全部通过后，立即物理退休 v2 旧 1m、旧 5m、`intraday_daily_features`、`limit_intraday_features`；任一条件失败不得删除。

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
`steps`: audit unit/PIT/coverage/stability/identity；archive immutable raw；normalize；write recursive dataset manifest；build candidate；semantic audit；CAS activate only after every release gate passes.

### procedure `cleanup_governed_asset`
`input`: cleanup target.
`steps`: `qdp gc --dry-run`；replacement/active graph check；delete only with explicit confirmation；write summary if durable.

## Writeback
- Project facts: this brain.
- Cross-project topology: main brain.
- Long audits, provider reports and cleanup reports: `references/`, `data/audits`, or run manifests.
