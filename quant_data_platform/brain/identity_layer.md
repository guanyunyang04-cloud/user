# Quant Data Platform 身份对象

## object `quant_data_platform_project`
`type`: shared_data_substrate_brain
`definition`: 主脑管辖下的共享量化数据平台分脑，负责可复用数据基底、registry、coverage audit、canonical bundle、provider ingest 和 memmap/training pack 治理。
`not`: 独立 Git 仓、团队平台项目、某个研究项目的私有数据湖。
`north_star`: 建立唯一、可审计、可按日使用的 `canonical_data_v1` 数据基底。
`consumers`: `daily_research`、`traditional_quant_research`、`t0_project` and future research projects.
`methods`: `inspect_status()`；`ingest_provider_data()`；`build_canonical_bundle()`；`build_memmap_or_pack()`；`audit_coverage()`；`cleanup_obsolete_assets()`。

## object `canonical_data_substrate`
`type`: protected_data_object
`scope`: raw archive、canonical lake、root manifest、registry pointer、policy bundle、pool/sector-board view、sharded memmap、training pack。
`storage`: large runtime data lives under `quant_data_platform/data/`; auditable registry and manifest pointers live under `quant_data_platform/registry/`.
`invariant`: raw OHLCV 口径保留；复权价格、复权收益、分钟聚合、结构风格字段作为派生 sidecar。

## object `provider_ingest_surface`
`type`: upstream_source_surface
`current_route`: `mootdx_online` for fast bars/quotes candidate；`BaoStock` for structured daily semantics；`CNInfo` for disclosure events.
`exploration_route`: `AKShare` and `efinance` remain low-priority probes until network stability changes.
`consumer_boundary`: research projects consume QDP outputs, not direct online provider calls.

## Pure Functions
- `resolve_data_owner(asset) -> quant_data_platform_project`
- `classify_provider_domain(field) -> market_bar|quote|structure|disclosure|exploration`
- `select_canonical_path(requirement) -> lake|bundle|memmap|training_pack`
- `classify_cleanup_target(path) -> keep|replace_then_remove|archive_only|protected`

## Routing
- Current runtime state: `state_center.md`
- Stable object classes and source semantics: `knowledge_center.md`
- Procedure entries and CLI commands: `operations_center.md`
- Protected data invariants: `governance_layer.md`
