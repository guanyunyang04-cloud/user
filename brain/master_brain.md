# 主脑

## 1. 作用
`brain/` 是 `user/` 工作区的主脑，也是整个工作区的 AI 控制面。

主脑只负责四类事情：

- 维护主脑与分脑的拓扑、边界和接管顺序
- 规定跨项目治理、写入路由与交接协议
- 约束各项目的 `body <-> brain` 匹配关系
- 在生产型分脑之间统一目标函数，避免静默漂移

## 2. 核心原则
- 这个工作区不要设计成“某个 agent 很强”
- 而要设计成“agent 可替换，大脑不可替换”
- 核心原则是：
  - `Agent 无状态，项目大脑有状态。`

## 3. 当前脑网络
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

## 4. 全局治理边界
- `brain-first`
  - agent 先接主脑，再接分脑，再进入 body
- `state-externalization`
  - 当前目标、规则、风险、计划、教训、交接包必须外显
- `body-brain match`
  - 每个项目的 brain 必须和源码、脚本、配置、测试、产物目录高度匹配
- `cross-project-first`
  - 涉及跨项目边界的规则，先写主脑，再写分脑

对生产型分脑，当前全局默认目标函数为：

- 收益优先、非降级
- 稳定性、坏市场收益、弱窗口修复只能作为增益项或阶段控制器约束
- 若要接受更低收益换取其它属性，必须由用户显式改写目标，并写入对应分脑的 `working_memory.md`

## 5. 分脑边界
- `daily_research/brain/`
  - 正式日线研究、formal comparator、执行端默认值、live 升级门槛
- `t0_project/brain/`
  - 盘中实验、执行抽象、RL 原型、实时接口边界
- `daily_stock_analysis-main/brain/`
  - 多市场分析产品、FastAPI/Web/Desktop/Bot、多数据源与 AI 协作资产

## 6. 接脑入口
默认接脑顺序、标准模块合同和分脑读写顺序，统一以 [brain_architecture.md](H:/new_tdx64/PYPlugins/user/brain/brain_architecture.md) 和 [brain_manifest.json](H:/new_tdx64/PYPlugins/user/brain/brain_manifest.json) 为准。

可执行入口：

- 只接主脑：
  - `python daily_research/tools/brain_bootstrap.py`
- 接主脑并进入 `daily_research`：
  - `python daily_research/tools/brain_bootstrap.py --child daily_research`
- 接主脑并进入其它分脑：
  - `python daily_research/tools/brain_bootstrap.py --child t0_project`
  - `python daily_research/tools/brain_bootstrap.py --child daily_stock_analysis-main`

## 7. 主脑维护动作
- 新的长期说明、治理规则与协作方法，不再写进根 `README`。
- 分脑结构变更时，主脑与分脑的 `brain_manifest.json` 必须同步更新。
- 新增项目时，必须先补齐 `brain/`、`body_map` 与 `handoff_contract`。
- `doc_guard.py check` 是当前脑网络的最低守卫。
- 工作区级当前交接摘要统一收口到 `brain/handoff_packet.md`。
- 工作区级身份、治理与反偏移机制统一收口到：
  - `brain/identity_layer.md`
  - `brain/governance_layer.md`
