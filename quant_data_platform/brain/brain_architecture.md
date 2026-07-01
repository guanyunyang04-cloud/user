# Quant Data Platform 脑区架构
快照日期：`2026-07-01`

`quant_data_platform/brain/` 继承主脑多范式自然语言程序模型：对象描述数据表和源，过程描述 update/rebuild/check/gc 方法，函数描述从需求到 table、provider、validation 的判断。

## Object Layer
- `qdp_v2_active_data_base`: `active.json` + `dataset.json` + parquet shards.
- `active_table`: raw fact, cache, derived feature, event fact or PIT state.
- `provider_ingest_surface`: mootdx, BaoStock, CNInfo when selected by domain.
- `consumer_project`: daily_research, traditional_quant_research, t0_project.
- `archived_v1_surface`: old lake/catalog/canonical/memmap workflows.

## Procedure Layer
- `inspect_qdp_status`
- `check_active_data_base`
- `update_dry_run`
- `rebuild_cache_or_feature`
- `garbage_collect_data_base`
- `brain_sync_after_qdp_change`

## Function Layer
- `active_table(domain)`
- `classify_table(domain)`
- `select_qdp_command(requirement)`
- `classify_cleanup_target(path)`
- `requires_brain_writeback(change)`

## Body Map
- `src/quant_data_platform/qdp_v2/`: current QDP v2 command and manifest logic.
- `src/quant_data_platform/providers.py` and domain contracts: provider/source normalization utilities.
- `data/qdp_v2/`: active data base.
- `tests/data_platform/`: focused QDP v2 tests.
- `tools/archive_v1/`: archived v1 workflow notes.
- `tools/archive_repair/`: archived one-off repair scripts.
