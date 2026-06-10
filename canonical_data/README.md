# Canonical Data

这里是工作区级唯一研究数据基底入口，不直接承载大体量 parquet / memmap 正文。

- 默认别名：`canonical_data_v1`
- 默认研究窗口：`2010-01-01` 起
- 主数据湖：`H:/quant_project/daily_research/output/research_data_lake`
- canonical manifest：`H:/quant_project/daily_research/output/research_data_lake/canonical/canonical_manifest.json`
- 根级 registry：`H:/quant_project/canonical_data/registry/root_manifest.json`

原则：

- raw 数据尽量保留全历史，正式研究视图默认从 `2010-01-01` 起。
- 原始 OHLCV 不被复权价覆盖；复权价、复权收益、分钟聚合特征都作为派生 sidecar。
- BaoStock 是 PIT 侧信息、财务、估值、状态和日常更新的基础口径。
- TDX/tqcenter 可作为快速外部日线源，入湖后必须写明来源、复权状态、`fill_data` 和成交额单位转换。
- 共享 memmap 建成后由 registry 解析；实验只生成轻量 sample index 和 normalization manifest。
- 旧 bundle、旧 parquet、旧 `.dat` 清理必须先有 dry-run、替代指针和随机一致性验证。
