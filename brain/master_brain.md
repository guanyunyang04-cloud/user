# 主脑

## 1. 作用
`brain/` 是整个工作区的共享脑核与全局路由层。

它只负责四件事：

- 定义主脑与分脑的统一结构
- 维护跨项目边界和拓扑
- 维护工作区级目标函数与治理顺序
- 保证任何 agent 都能先接脑、再接状态、最后进入 body

## 2. 当前脑网络
- 主脑：
  - `brain/`
- 一级分脑：
  - `daily_research/brain/`
  - `t0_project/brain/`
  - `daily_stock_analysis-main/brain/`

当前角色固定为：

- `daily_research`
  - 正式生产研究与执行主线
- `t0_project`
  - 盘中执行与 RL 实验分脑
- `daily_stock_analysis-main`
  - 多市场分析产品分脑

## 3. 当前治理边界
- `brain-first`
  - 先接主脑，再接目标分脑，再进 body
- `single-hub`
  - 同一种信息只保留一个权威中枢
- `common-in-main`
  - 共享结构只在主脑维护
- `local-in-child`
  - 项目事实只在分脑维护
- `brain-as-doc-hub`
  - 权威说明只留在 `brain/` 或 `brain/references/`
- `no-thread-deeplink-registry`
  - 主脑不为每个任务分配、记录或维护 Codex 线程深链；任务接管只使用 brain 路由、分脑入口、产物路径和必要的 study / run tag。

## 4. 当前默认接脑方式
- 只接主脑：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_bootstrap.py`
- 接主脑并进入目标分脑：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_bootstrap.py --child <brain_id>`

## 5. 当前主问题
- 当前最重要的事情已经不是“有没有脑结构”
- 而是持续维持这套精炼脑核，不再重新长回碎片化文档
