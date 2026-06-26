# Quant Data Platform 知识中枢

## 稳定规则
- canonical 数据集要完整承载可审计、可按日使用的数据基底；训练/研究 profile 决定用哪些字段。
- `canonical_dataset` / `canonical_data_v1` / active registry / sharded memmap 的共享事实以本分脑为 owner；消费者项目只声明使用 lineage。
- 原始 OHLCV 永远保留原始口径；复权价格、复权收益、分钟聚合特征作为派生 sidecar。
- 估值、行业、指数成分属于 canonical 结构/风格层；短线 profile 可以不用，但数据基底必须能提供。
- 财务季报、业绩预告/快报等慢披露或 PIT 难保证数据不进入 v1 默认基底；未来专题引入前必须先证明 `publish_date/available_date`。
- 1 分钟数据是 cold archive，不作为当前默认 5 分钟更新来源；已聚合好的 5 分钟数据在等效审计后可直接复用。

## 数据源对象
- `mootdx_online`：QDP 上游行情对象的候选主源。适合 quote snapshot、日线 bar、1m/5m bar、指数行情、分笔 raw candidate 和 xdxr 辅助；进入 canonical 前按 endpoint/frequency 记录单位、复权、服务器、覆盖深度和质量状态。当前已知 `bars(frequency=9)` volume 需约 `100x` 归一化。
- `baostock_online`：QDP 基础语义对象。适合交易日历、股票池/上市状态、证券状态、行业/概念、指数成分、换手率/估值等日级结构字段；行情 bar 不再作为默认主权域，只保留审计、兜底或特定 backfill 价值。
- `cninfo_online`：披露事件对象。适合公告、报告、PDF、披露日和事件核验；进入模型数据前需要 PIT-safe `available_time` / disclosure audit。
- `akshare` / `efinance`：特色探索或低优先级探针对象；当前网络环境下不进入主链路。
- `current_qdp`：本地研究消费对象，不是外部 provider；研究和训练只读 QDP lake / memmap / training pack。

## 治理教训
- 数据基底和 memmap 不应散落在每个实验里；实验只生成轻量 sample index、normalization manifest 和训练产物。
- 单进程全 A 巨大 memmap 在低内存机器上不稳，默认走分片 feature/label store。
- 旧 bundle、旧 parquet、旧 `.dat` 清理必须先有 dry-run、替代指针和随机一致性验证。
- canonical bundle ID 改变后，已有 memmap 即使文件完整也不能默认复用；必须比较 source bundle/signature，旧 memmap 只能作为历史验证样本。
- sharded memmap 的基本单位是 `year + symbol block`；每个 shard 独立保存 feature store、label store、sample index 和 manifest，顶层 registry 只汇总并验证 schema 一致性。
