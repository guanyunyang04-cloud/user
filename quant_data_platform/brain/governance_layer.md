# Quant Data Platform 治理层

## 治理目标
- 让工作区只有一个默认数据基底入口。
- 让 provider、raw archive、canonical lake、registry、coverage、memmap 和 cleanup 对象可审计、可复用、可迁移。
- 让研究项目通过 profile 选择字段，而不是各自复制、裁剪或重建数据集。
- 允许消费者读取本分脑公开 status / registry / manifest 作为共享数据事实；实际构建、清理、重建和 registry mutation 仍归 QDP 对象方法。
- 让数据平台服务个人研究效率；完整基底和 active memmap 建好后，旧路径默认退出热路径。

## 保护对象

### `raw_market_data`
- 定义：外部 provider 入湖前后的原始行情数据。
- 不变量：保留原始 OHLCV 口径；复权价格、复权收益和分钟聚合特征作为 sidecar，不覆盖 raw。

### `historical_semantics`
- 定义：历史日期上的 universe、行业、状态、估值、指数成分、公告可得时间等语义。
- 不变量：不用当前行业、当前板块、当前快照或当前 F10/finance 快照回填历史。

### `coverage_status`
- 定义：domain 覆盖、缺失、blocked、partial 和 completed 状态。
- 不变量：缺失 coverage 不静默缩短为“已完成”；required domain blocked 时不能注册为可供研究消费的完整数据集。

### `registry_pointer`
- 定义：active canonical、policy bundle、memmap 和 training pack 指针。
- 不变量：pointer mutation 需要明确 source bundle/signature、替代指针和足够抽样验证。

### `data_cleanup`
- 定义：旧 parquet、旧 bundle、旧 `.dat`、旧 training pack 和重复旧资产清理。
- 不变量：不删除唯一数据或无替代指针的数据；重复旧数据在 dry-run、替代指针和抽样验证后可以清理。

### `slow_disclosure_domain`
- 定义：财务季报、业绩预告/快报、公告 PDF 等披露慢域。
- 不变量：进入模型数据前先建立 `publish_date / available_date` 或等价 PIT-safe 可得时点审计；未治理前不进入 v1 默认基底。

## 写回
- 项目事实写入本分脑。
- 跨项目拓扑写入主脑。
- 长证据、迁移审计、provider 评估和清理报告写入 `quant_data_platform/brain/references/` 或对应 `data/audits` / `data/provider_eval` 目录。
