# Quant Data Platform 状态程序

## Module Interface
`exports`: QDP status、canonical manifest、provider eval evidence、active memmap / training pack pointers。
`consumers`: `daily_research`、`traditional_quant_research`、`t0_project`。

## Object Instances
### object `qdp_project`
`type`: shared_data_platform_instance
`state`: 已由主脑 runtime 初始化并注册；具备 CLI、tests、registry 迁移、coverage/audit、bundle 构建和 memmap 验证入口。
`default_window`: `2010-01-01` onward.

### object `qdp_lake_root`
`type`: canonical_lake_root
`state`: `qdp_paths().lake_root`、`ResearchDataLake` 默认根、root manifest `primary_lake_root` 和 `canonical_manifest` 均已切到 `H:/quant_project/quant_data_platform/data/lake`。
`evidence`: `quant_data_platform/data/audits/qdp_lake_physical_migration_20260626_01.json`
`retired_path`: `daily_research/output/research_data_lake` 已删除。

### object `provider_evaluation_20260626`
`type`: provider_eval_result
`state`: `provider_eval_20260626_01` 已完成，覆盖 `akshare/baostock/efinance/mootdx/cninfo/current_qdp`、5 个样本标的和 6 个历史窗口。
`evidence`: `quant_data_platform/data/provider_eval/provider_eval_20260626_01` and `quant_data_platform/brain/references/provider_eval_20260626_01.md`
`conclusion`: `BaoStock` 语义规整但逐股历史慢；`mootdx` 行情速度强但需单位治理；`CNInfo` 适合披露探针；`AKShare/efinance` 当前受 Eastmoney 访问稳定性限制。

### object `mootdx_capability_20260627`
`type`: provider_capability_result
`state`: `mootdx_capability_20260627_01` 覆盖 quote、股票列表、全频 K 线、日线/5m/1m 深度翻页、历史分时、历史分笔、指数、xdxr、finance、F10，并与 current QDP 做近期日线 OHLCV 对账。
`evidence`: `quant_data_platform/data/provider_eval/mootdx_capability_20260627_01`
`current_definition`: `mootdx_online` 是 QDP 上游行情候选主源；`bars(frequency=9)` volume 需约 `100x` 归一化，intraday 深度按 symbol/endpoint 审计后生产化。

### object `canonical_domain_policy`
`type`: current_domain_split
`state`: 行情、5 分钟日级特征、复权因子、估值、行业、指数成分、交易日历、股票池和证券状态进入数据基底；财务季报、业绩预告/快报等慢披露数据暂不进入 v1 默认基底。
`provider_split`: bars/quote prefer `mootdx_online`; structured daily semantics use `BaoStock`; disclosure events use `CNInfo`.

### object `active_sharded_memmap`
`type`: active_memmap_pointer
`state`: `canonical_short_horizon_core_v1_full_2010_2026` frozen；profile `short_horizon_core_v1`；2010-2026；5526 symbols；256 features；323 planned / 262 stored / 61 empty / 0 failed shards。
`manifest`: `H:/quant_project/quant_data_platform/data/memmap/sharded/canonical_short_horizon_core_v1_full/sharded_memmap_manifest.json`

### object `traditional_event_alpha_candidate`
`type`: event_pack_candidate
`state`: `traditional_event_alpha_v1_candidate_2017_2026_20260615_01` completed candidate；268333 rows, 10 yearly partitions, 316 clean QDP features, 29 labels.
`activation`: not active by default; inspect when traditional event alpha work asks for it.

## Pure Functions
- `resolve_provider_for_domain(domain)`: bars/quote -> `mootdx_online`; structure -> `BaoStock`; disclosure -> `CNInfo`.
- `derive_next_qdp_action(state)`: current default is incremental update, tail shard rebuild, active registry hit, training read smoke, and obsolete asset cleanup.
- `classify_cleanup(path)`: requires replacement pointer and sample validation before removal.
- `resolve_consumer_request(task)`: returns explicit QDP dataset id, manifest, memmap or training pack.

## Procedures
### procedure `provider_to_canonical`
`input`: provider domain requirement
`steps`: fetch raw candidate；audit units/coverage/PIT；archive raw；canonicalize；register manifest；return explicit pointer.
`side_effects`: QDP data and registry artifacts.

### procedure `cleanup_obsolete_assets`
`input`: old parquet, old bundle, old `.dat`, old training pack
`steps`: generate dry-run；identify replacement pointer；sample-validate equivalence or archive value；remove only replaceable duplicates.
`side_effects`: data cleanup plan or deletion after validation.

## Evidence Entrypoints
- Provider eval: `quant_data_platform/brain/references/provider_eval_20260626_01.md`
- Lake migration audit: `quant_data_platform/data/audits/qdp_lake_physical_migration_20260626_01.json`
- mootdx capability: `quant_data_platform/data/provider_eval/mootdx_capability_20260627_01`
