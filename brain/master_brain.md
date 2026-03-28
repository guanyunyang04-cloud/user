# Main Brain

## 1. 作用
`brain/` 是 `user/` 工作区的主脑，也是整个工作区的 AI 控制面。

主脑只负责四类事情：

- 维护主脑与分脑的拓扑、边界和接管顺序
- 规定跨项目治理、写入路由与交接协议
- 约束各项目的 `body <-> brain` 匹配关系
- 在生产型分脑之间统一目标函数，避免静默漂移

## 2. 当前脑网络
- 主脑：
  - `brain/`
- 一级分脑：
  - `daily_research/brain/`
  - `t0_project/brain/`
  - `daily_stock_analysis-main/brain/`

当前职责分工固定为：

- `daily_research`
  - 正式生产研究与执行主线
- `t0_project`
  - 盘中 T+0、执行抽象与 RL 实验分支
- `daily_stock_analysis-main`
  - 独立的多市场 AI 股票分析产品分支

## 3. 主脑治理原则
- `brain-first`
  - agent 先接主脑，再接分脑，再进入 body
- `body-brain match`
  - 每个项目的 brain 必须和源码、脚本、配置、测试、产物目录高度匹配
- `non-silent-upgrade`
  - 生产型分脑不得把“更稳但更低收益”的方案静默升级成默认值
- `cross-project-first`
  - 涉及跨项目边界的规则，先写主脑，再写分脑

对于生产型分脑，当前全局默认目标函数为：

- 收益优先、非降级
- 稳定性、坏市场收益、弱窗口修复只能作为增益项或阶段控制器约束
- 若要接受更低收益换取其它属性，必须由用户显式改写目标，并写入对应分脑的 `working_memory.md`

## 4. 分脑边界
- `daily_research/brain/`
  - 正式日线研究、formal comparator、执行端默认值、live 升级门槛
- `t0_project/brain/`
  - 盘中实验、执行抽象、RL 原型、实时接口边界
- `daily_stock_analysis-main/brain/`
  - 多市场分析产品、FastAPI/Web/Desktop/Bot、多数据源与 AI 协作资产

## 5. 标准接脑顺序
默认接管顺序如下：

1. `brain/brain_manifest.json`
2. `brain/master_brain.md`
3. `brain/brain_architecture.md`
4. `brain/working_memory.md`
5. 按任务进入目标分脑
6. 读取目标分脑的 `brain_manifest.json`
7. 按 `semantic -> working -> procedural -> environment -> action -> episodic` 进入

## 6. 主脑维护动作
- 新的长期说明、治理规则与协作方法，不再写进根 `README`
- 分脑结构变更时，主脑与分脑的 `brain_manifest.json` 必须同步更新
- 新增项目时，必须先补齐 `brain/`、`body_map` 与 `handoff_contract`
- `doc_guard.py check` 是当前脑网络的最低守卫
