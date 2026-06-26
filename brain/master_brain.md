# 主脑

## 1. 拓扑地图
`brain/` 是 workspace 主脑。结构、读取顺序、写回路由、child 注册以 `brain/brain_manifest.json` 为准。

当前一级附着子脑：

- `daily_research/brain/`：生产研究与执行主线。
- `quant_data_platform/brain/`：共享量化数据平台与 canonical 数据基底治理。
- `t0_project/brain/`：盘中执行与 RL 实验。
- `daily_stock_analysis-main/brain/`：多市场分析产品。
- `traditional_quant_research/brain/`：传统量化方法研究。

当前根目录非分脑身份：

- `brain/`：主脑。
- `tools/`：主脑与工作区工具。
- `docs/`：工作区级文档索引。
- `canonical_data/`：过渡数据资产入口，后续由 `quant_data_platform` 接管 registry 与 memmap 治理。
- `a_stock_daily_selection/`：待整理旧目录；当前不注册分脑，后续决定归档、合并或补脑。

## 2. 接管边界
- `agent-first`：agent 先理解用户目标，再按需要读取主脑、分脑、代码和产物。
- `route-is-optional`：route / capsule / bootstrap 只是诊断工具，不决定工作范围。
- `qdp-data-memory`：共享数据集、canonical_data_v1、registry、policy bundle 和 sharded memmap 默认看 `quant_data_platform`；研究项目只作为消费者解释其使用证据。
- `common-in-main`：共享结构、注册、治理规则写在主脑。
- `local-in-child`：项目事实、项目命令、项目证据写在子脑。

## 3. 注册原则
新项目脑由 runtime 初始化并注册；注册结果写入主脑 manifest、catalog 和发现式路由链路。

本文件不重复 boot order 或入口命令，避免成为 manifest 之外的第二套脑。
