# Quant Data Platform 治理层

## 治理目标
- 让工作区只有一个默认数据基底入口。
- 让数据资产、registry、coverage、memmap 和 cleanup 规则可审计、可复用、可迁移。
- 让研究项目通过 profile 选择字段，而不是各自复制、裁剪或重建数据集。
- 允许消费者读取本分脑公开 status / registry / manifest 作为共享数据事实；实际构建、清理、重建和 registry mutation 仍归本项目 profile。
- 让数据平台服务个人研究效率；完整基底和 active memmap 建好后，旧路径默认退出热路径。

## 硬边界
- 不用当前行业、当前板块、当前快照回填历史。
- 不覆盖原始 OHLCV；复权和派生特征只能作为 sidecar。
- 不把缺失 coverage 静默缩短为“已完成”。
- 不删除唯一数据或无替代指针的数据；重复旧数据 dry-run、替代指针和抽样验证后可以清理。
- 不让财务/业绩慢披露域进入 v1 默认基底，除非以后建立严格可用时点审计。

## 写回
- 项目事实写入本分脑。
- 跨项目拓扑写入主脑。
- 长证据、迁移审计和清理报告写入 `quant_data_platform/brain/references/`；不再保留 `quant_data_platform/docs/` 兼容入口，只有明确发布产物才临时导出。
