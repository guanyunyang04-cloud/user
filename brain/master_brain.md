# 主脑
快照日期：`2026-07-01`

## 1. 拓扑地图
`brain/` 是 workspace 主脑。结构、读取顺序、写回路由、child 注册以 `brain/brain_manifest.json` 为准。

当前一级附着子脑：

- `daily_research/brain/`：生产研究与执行主线。
- `quant_data_platform/brain/`：共享 QDP v2 manifest-first 数据基底。
- `t0_project/brain/`：盘中执行与 RL 实验。
- `daily_stock_analysis-main/brain/`：多市场分析产品。
- `traditional_quant_research/brain/`：传统量化方法研究。

当前根目录非分脑身份：

- `brain/`：主脑。
- `tools/`：主脑与工作区工具。
- `docs/`：工作区级文档索引。
- `canonical_data/`：历史/过渡资产入口，不是当前 QDP v2 active 数据基底。
- `a_stock_daily_selection/`：待整理旧目录；当前不注册分脑。

## 2. 接管边界
- `agent-first`: agent 先理解用户目标，再按需要读取主脑、分脑、代码和产物。
- `route-is-sensor`: route / capsule / bootstrap 是诊断工具；route 给出 primary owner、supporting brains、object routes 和写回目标，不是唯一操作边界。
- `object-registry`: `brain/object_registry.json` 是对象归属、路径前缀、写回目标和闭环验证提示的机器可读表；route 和 closure-check 读取它，agent 仍负责结合用户目标和文件事实判断。
- `qdp-data-memory`: 共享数据基底默认看 `quant_data_platform`；当前事实源是 `data/qdp_v2/active/active.json`、各 `dataset.json` 和 parquet shards。
- `research-consumer`: 研究项目只消费 QDP 显式数据基底或下游研究产物，不在 hot path 内复制 QDP 数据事实。
- `cross-project-orchestration`: QDP active 数据对象归 `quant_data_platform`；sequence pack、memmap、模型、loss、画像、回测和评估归 `daily_research`。混合任务以研究 owner 为 primary，QDP 作为 supporting read-only，除非任务实际改变 active 数据基底。
- `brain-sync`: 项目架构、CLI、active pointer、数据语义、质量结论或执行边界变化后，同步对应分脑 hot path。

## 3. 注册原则
新项目脑由 runtime 初始化并注册；注册结果写入主脑 manifest、catalog 和发现式路由链路。

本文件不重复 boot order 或长命令，避免成为 manifest 之外的第二套脑。
