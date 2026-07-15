# Quant Data Platform 身份对象
快照日期：`2026-07-15`

## object `quant_data_platform_project`
`type`: shared_manifest_first_data_base_brain
`definition`: 主脑管辖下的共享量化数据基底分脑，负责受保护的 QDP v2 active 与 trusted-source、identity-first、5m-only、compact-storage 的 QDP v3 重建/发布面、provider ingest、数据校验和数据清理。
`not`: 旧 v1 工作流平台、下游训练产物注册中心、某个研究项目的私有数据湖。
`north_star`: 建立简单、准确、可审计、可更新的本地数据基底。
`truth_source`: `parquet + dataset.json + active.json`。
`consumers`: `daily_research`、`traditional_quant_research`、`t0_project` and future research projects.
`public_methods`: `status()`；`list()`；`describe(table)`；`check(target, mode)`；`update()`；`gc()`。
`admin_methods`: `ingest()`；`compact()`；`build_candidate()`；`audit()`；`diff()`；`publish()`；`rollback()`；`retire_v2_intraday()`。

## object `qdp_v2_active_data_base`
`type`: protected_data_object
`active_root`: `quant_data_platform/data/qdp_v2/active/active.json`
`datasets_root`: `quant_data_platform/data/qdp_v2/datasets/`
`scope`: Shanghai/Shenzhen A-share main board, excluding ChiNext, STAR, ST and delisted stocks.
`date_scope`: `2011-11-22..2026-06-26`
`symbol_start_overrides`: `600036.SH -> 2016-07-25`
`invariant`: raw facts remain raw；5m、daily panel and feature tables are rebuildable cache/derived tables, not second truth sources.

## object `provider_ingest_surface`
`type`: upstream_source_surface
`current_route`: 低频历史基底复用可信 Tushare raw；历史 5m 按完整股票日依次使用本地直接 archive、迁移后的 v2 完整日、2020+ mootdx/BaoStock 完整日、Tushare 最终残差。不同来源不逐值仲裁、不拼 bar；v2 active 只读且不被原地修改。截止日后 `BaoStock` 负责日线/状态/因子事件，`mootdx` 负责优先 5m，`BaoStock` 只作完整日 fallback；v3 不存在 1m provider route。
`consumer_boundary`: research projects consume QDP outputs or explicit downstream packs, not direct online provider calls.

## object `qdp_v3_storage_surface`
`type`: compact_raw_and_runtime_split
`data_root`: `H:\quant_project\quant_data_platform\data`
`runtime_root`: `H:\quant_project\quant_data_platform\data\qdp_runtime`
`definition`: 数据与 runtime 全部留在 H 盘本仓库。canonical、manifest 和大 Parquet bundle 位于 QDP data root；SQLite、job、cursor、heartbeat 和临时 staging 位于独立 repo-local runtime root。reference raw 使用 undated x 1，低频时序 raw 使用 year x 1，只有 5m raw 使用 year x 16 security buckets；发布 manifest 使用导出的 raw index/receipt Parquet及 SHA256，不依赖 runtime 数据库。
`disk_boundary`: H 已于 2026-07-15 修复并通过健康闸门；用户取消 F 盘备份，当前流程没有 F 盘复制或 fallback。

## object `downstream_research_artifacts`
`type`: non_active_data_base_artifact
`examples`: memmap、training pack、research panel exports、model-ready packs。
`boundary`: useful for research/training, but not recorded as QDP active data base state.

## Pure Functions
- `resolve_data_owner(asset) -> quant_data_platform_project`
- `classify_table(domain) -> raw_fact|cache|derived_feature|event_fact|pit_state`
- `select_qdp_command(requirement) -> status|list|describe|check|rebuild|gc|update`
- `classify_cleanup_target(path) -> keep|replace_then_remove|archive_only|protected`
- `requires_brain_writeback(change) -> bool`: true for active table, command surface, data scope, quality conclusion or governance change.

## Routing
- Current runtime state: `state_center.md`
- Stable object classes and source semantics: `knowledge_center.md`
- Procedure entries and CLI commands: `operations_center.md`
- Protected data invariants: `governance_layer.md`
- Long audits and historical transitions: `references/`
