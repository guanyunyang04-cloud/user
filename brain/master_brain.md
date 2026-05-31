# 主脑

## 1. 拓扑地图
`brain/` 是 workspace 主脑。结构、读取顺序、写回路由、child 注册以 `brain/brain_manifest.json` 为准。

当前一级附着子脑：

- `daily_research/brain/`：生产研究与执行主线。
- `t0_project/brain/`：盘中执行与 RL 实验。
- `daily_stock_analysis-main/brain/`：多市场分析产品。

## 2. 接管边界
- `main-brain-first`：主脑是 agent 接管入口。
- `route-before-child`：路由明确选中后才读取子脑。
- `common-in-main`：共享结构、注册、治理规则写在主脑。
- `local-in-child`：项目事实、项目命令、项目证据写在子脑。
- `no-loose-default-child`：多个子脑冲突返回 `ambiguous`。

## 3. 注册原则
新项目脑由 runtime 初始化并注册；注册结果必须进入主脑 manifest、catalog 和发现式路由链路。

本文件不重复 boot order 或入口命令，避免成为 manifest 之外的第二套脑。
