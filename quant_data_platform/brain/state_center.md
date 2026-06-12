# Quant Data Platform 状态中枢

## 当前状态
- 分脑已由主脑 runtime 初始化并注册，body_root 为 `quant_data_platform`。
- 当前项目已具备最小可用 CLI、tests、registry 迁移、coverage/audit、bundle 构建和 memmap 验证入口。
- 默认研究窗口沿用工作区 canonical 决策：`2010-01-01` 起。
- active canonical bundle 已更新为包含估值、行业、指数成分 sidecar 的 `canonical_data_v1`。
- canonical 数据域原则：行情、5 分钟日级特征、复权因子、估值、行业、指数成分、交易日历、股票池和证券状态进入数据基底；财务季报、业绩预告/快报等慢披露数据暂不进入 v1 默认基底。
- full canonical sharded memmap 已完成并冻结：`canonical_short_horizon_core_v1_full_2010_2026`，profile `short_horizon_core_v1`，2010-2026，5526 symbols，256 features，323 planned / 262 stored / 61 empty / 0 failed shards。
- active sharded manifest：`H:/quant_project/quant_data_platform/data/memmap/sharded/canonical_short_horizon_core_v1_full/sharded_memmap_manifest.json`。

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
