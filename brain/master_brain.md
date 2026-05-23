# 主脑

## 1. 作用
`brain/` 是整个工作区的共享脑核、任务路由层和 agent 接管入口。

它只负责四件事：

- 定义主脑与分脑的统一结构。
- 维护跨项目边界、分支纪律和禁区判断。
- 先路由任务，再按需进入分脑。
- 保证任何 agent 都从主脑 capsule 进入，而不是直接绑定某个项目分脑。

## 2. 当前脑网络
- 主脑：`brain/`
- 一级分脑：
  - `daily_research/brain/`
  - `t0_project/brain/`
  - `daily_stock_analysis-main/brain/`

当前角色固定为：

- `daily_research`：正式生产研究与执行主线。
- `t0_project`：盘中执行与 RL 实验分脑。
- `daily_stock_analysis-main`：多市场分析产品分脑。

## 3. 当前治理边界
- `main-brain-first`：主脑是唯一 agent 接管入口。
- `route-before-child`：只有主脑路由明确选中分脑后，才读取分脑上下文。
- `common-in-main`：共享结构、共享顺序、共享治理只在主脑维护。
- `local-in-child`：项目事实、实验状态、项目命令只在分脑维护。
- `no-loose-default-child`：多个分脑高置信冲突时返回 `ambiguous`，不得默认落到 `daily_research`。

## 4. 当前默认接脑方式
- 生成 schema v2 接管胶囊：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- 只做任务路由：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json`
- 解析主脑或目标分脑 boot order：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json`

## 5. 当前主问题
- 当前最重要的事情不是“有没有脑结构”，而是维持主脑优先的平台入口，不让 capsule、skill、workflow registry 再和某个分脑绑定。
- 主脑只保存跨项目规则和路由；`daily_research` 的 Path20、continuous_policy、deep_alpha 事实必须留在 `daily_research/brain/`。
