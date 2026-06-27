# Quant Data Platform 脑区架构

`quant_data_platform/brain/` 继承主脑多范式自然语言程序模型：对象描述数据资产和源，过程描述 ingest/canonical/memmap/cleanup 方法，函数描述从需求到 owner、domain、validation 的判断。

## Object Layer
- `canonical_data_substrate`: lake、registry、policy bundle、memmap、training pack。
- `provider_ingest_surface`: mootdx、BaoStock、CNInfo、AKShare、efinance、current_qdp。
- `consumer_project`: daily_research、traditional_quant_research、t0_project。
- `governed_data_asset`: raw data、historical semantics、coverage status、registry pointer、cleanup target、slow disclosure domain。

## Procedure Layer
- `provider_to_canonical`
- `build_sharded_memmap_smoke`
- `validate_memmap`
- `cleanup_obsolete_assets`
- `registry_pointer_change`

## Function Layer
- `resolve_data_owner(asset)`
- `classify_provider_domain(field)`
- `select_canonical_path(requirement)`
- `classify_cleanup_target(path)`

## Body Map
- `core/`: paths, configs, registry, schema.
- `providers/`: external source adapters and probes.
- `ingest/`: refresh/import/domain merge.
- `lake/`: data lake catalog, canonical manifest, policy bundle and coverage audit.
- `domains/`: market/structure/trading-filter contracts.
- `features/`: canonical feature domains and profiles.
- `memmap/`: sharded stores, labels, sample indexes and signatures.
- `cli.py`: QDP command entry.
