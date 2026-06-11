# Quant Data Platform 知识中枢

## 稳定规则
- canonical 数据集要完整承载可审计、可按日使用的数据基底；训练/研究 profile 决定用哪些字段。
- 原始 OHLCV 永远保留原始口径；复权价格、复权收益、分钟聚合特征作为派生 sidecar。
- 估值、行业、指数成分属于 canonical 结构/风格层；短线 profile 可以不用，但数据基底必须能提供。
- 财务季报、业绩预告/快报等慢披露或 PIT 难保证数据不进入 v1 默认基底；未来专题引入前必须先证明 `publish_date/available_date`。
- 1 分钟数据是 cold archive，不作为当前默认 5 分钟更新来源；已聚合好的 5 分钟数据在等效审计后可直接复用。

## 数据源分工
- BaoStock 是 PIT/状态/估值/行业/指数等基础口径的重要来源。
- TDX/tqcenter 可补日线、5 分钟、复权或速度，但当前板块、当前股票列表和快照不能倒灌历史。
- 外部 5 分钟和复权因子可作为补充与交叉审计来源。

## 治理教训
- 数据基底和 memmap 不应散落在每个实验里；实验只生成轻量 sample index、normalization manifest 和训练产物。
- 单进程全 A 巨大 memmap 在低内存机器上不稳，默认走分片 feature/label store。
- 旧 bundle、旧 parquet、旧 `.dat` 清理必须先有 dry-run、替代指针和随机一致性验证。
