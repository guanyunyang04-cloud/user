# Traditional Quant Research 脑区架构

- 采用统一 7 模块核：identity、state、knowledge、architecture、operations、governance、episodic。
- 共享结构以 workspace 主脑 manifest 为准；本文件只记录区域特化。
- 区域特化：本脑区是传统量化方法研究实验室，优先读取 `state_center`、`knowledge_center`、`operations_center` 后进入项目 body。
- Body 分区：`traditional_quant_research/` 放可复用研究核心，`experiments/` 放实验入口，`data/` 放数据资产，`tests/` 放行为守卫；阅读性质的研究日志、数据契约和实验说明只以 `traditional_quant_research/brain/references/` 为 canonical 正文，旧 `research_log/` 和外部 README 副本不再保留。
