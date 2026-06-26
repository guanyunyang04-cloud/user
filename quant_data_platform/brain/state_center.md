# Quant Data Platform 状态中枢

## 当前状态
- 分脑已由主脑 runtime 初始化并注册，body_root 为 `quant_data_platform`。
- 当前项目已具备最小可用 CLI、tests、registry 迁移、coverage/audit、bundle 构建和 memmap 验证入口。
- 默认研究窗口沿用工作区 canonical 决策：`2010-01-01` 起。
- active canonical bundle 已更新为包含估值、行业、指数成分 sidecar 的 `canonical_data_v1`。
- 本分脑是工作区 canonical 数据基底、registry、policy bundle、sharded memmap 和数据清理的默认事实来源；其它研究项目可读取本分脑 status / registry / manifest 作为共享数据事实。
- 2026-06-26 QDP lake 物理归属已闭环：`qdp_paths().lake_root`、`ResearchDataLake` 默认根、root manifest `primary_lake_root` 和 `canonical_manifest` 均已切到 `H:/quant_project/quant_data_platform/data/lake`；旧 `daily_research/output/research_data_lake` 已在 copy、catalog/shard path rewrite、QDP status、current_qdp provider-eval smoke、QDP tests 和 daily_research 消费侧 focused tests 通过后删除。迁移报告：`quant_data_platform/data/audits/qdp_lake_physical_migration_20260626_01.json`。
- 2026-06-26 外部数据源只读评估已完成：`provider_eval_20260626_01` 覆盖 `akshare/baostock/efinance/mootdx/cninfo/current_qdp`、5 个固定样本标的和 6 个历史窗口；输出位于 `quant_data_platform/data/provider_eval/provider_eval_20260626_01`，结论见 `quant_data_platform/brain/references/provider_eval_20260626_01.md`。评估未修改 canonical、registry、memmap、training pack 或数据湖内容；当前初步分层为 `baostock` 适合作为主历史 OHLCV 候选但较慢，`mootdx` 适合分钟/实时补充但需单位归一化，`cninfo` 适合公告披露探针，`akshare/efinance` 在当前网络环境主要受 Eastmoney Proxy/JSONDecode 稳定性限制。
- canonical 数据域原则：行情、5 分钟日级特征、复权因子、估值、行业、指数成分、交易日历、股票池和证券状态进入数据基底；财务季报、业绩预告/快报等慢披露数据暂不进入 v1 默认基底。
- full canonical sharded memmap 已完成并冻结：`canonical_short_horizon_core_v1_full_2010_2026`，profile `short_horizon_core_v1`，2010-2026，5526 symbols，256 features，323 planned / 262 stored / 61 empty / 0 failed shards。
- active sharded manifest：`H:/quant_project/quant_data_platform/data/memmap/sharded/canonical_short_horizon_core_v1_full/sharded_memmap_manifest.json`。
- event pack 数据域已完成首个 full candidate：`traditional_event_alpha_v1_candidate_2017_2026_20260615_01`，2017-2026 共 268333 行、10 个年度分区、316 个 clean QDP 特征、29 个派生标签、剔除 48 个执行/持仓/组合状态特征；该资产未注册 active，记录见 `brain/references/traditional_event_alpha_v1_candidate.md`。

## 当前接管重点
- 默认训练入口应复用 active full sharded memmap；旧每实验单体 `forecast_*.dat` 机制不再作为默认路径维护。
- 下一步重点是增量更新、当前年/尾部 shard 重建、active registry 命中、训练读取 smoke 和旧资产清理。
- 旧 parquet / 旧 `.dat` / 旧 bundle 清理先出 dry-run 和替代指针；确认不触碰唯一数据后可直接清理，不再走重审批链。

## 根目录关系
- `canonical_data/`：过渡资产入口，等待本项目稳定后退为兼容指针。
- `daily_research/`：正式研究与执行消费者，不再长期拥有共享数据平台职责。
- `a_stock_daily_selection/`：当前只有 output，未注册分脑，先标记为待整理旧目录。

## 默认纪律
- repo-tracked mutation 优先在 `main` 分支执行。
- 数据清理必须先 dry-run、再替代指针、再做足够抽样验证；目标是减少重复冗余，不为了兼容旧机制长期保留多份数据。
