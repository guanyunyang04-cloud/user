# Quant Data Platform 身份对象
快照日期：`2026-07-14`

## object `quant_data_platform_project`
`type`: shared_manifest_first_data_base_brain
`definition`: 主脑管辖下的共享量化数据基底分脑，负责受保护的 QDP v2 active 与 identity-first、5m-only 的 QDP v3 重建/发布面、provider ingest、数据校验、可重建缓存和数据清理。
`not`: 旧 v1 工作流平台、下游训练产物注册中心、某个研究项目的私有数据湖。
`north_star`: 建立简单、准确、可审计、可更新的本地数据基底。
`truth_source`: `parquet + dataset.json + active.json`。
`consumers`: `daily_research`、`traditional_quant_research`、`t0_project` and future research projects.
`public_methods`: `status()`；`list()`；`describe(table)`；`check(mode)`；`rebuild(target)`；`gc()`；`update()`。

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
`current_route`: Tushare-compatible proxy 仅负责 `2010-01-01..2026-07-13` 历史启动；`mootdx` 负责截止日后的优先 5m；`BaoStock` 负责长期日线/状态/因子事件及完整 5m fallback；交易所/CNInfo 只作身份、公司行动与披露仲裁。v3 不存在 1m provider route。
`consumer_boundary`: research projects consume QDP outputs or explicit downstream packs, not direct online provider calls.

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
