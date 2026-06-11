# Quant Data Platform 脑区架构

## 结构定位
本分脑采用主脑统一 7 模块核；共享结构以 `brain/brain_manifest.json#shared_regional_brain_contract` 为准。

本项目的区域特化是“共享数据基底治理”：

- `core/`：路径、配置、registry、JSON 校验和通用 schema。
- `lake/`：封装 data lake catalog、canonical manifest、coverage audit。
- `domains/`：行情、结构风格、交易过滤等 domain contract。
- `features/`：canonical 数据域与训练 feature profile 的分层定义。
- `memmap/`：分片 feature store、label store、sample index、registry signature。
- `cli.py`：统一命令入口 `qdp`。

## 归属边界
- `quant_data_platform` 拥有跨项目共享数据工程能力。
- `daily_research` 保留研究、训练和执行逻辑，并逐步从本项目读取数据。
- `canonical_data` 只保留过渡 alias/registry 指针，不继续扩张成第二套平台。
