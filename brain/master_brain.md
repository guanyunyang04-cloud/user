# Main Brain

## 1. 作用
`user/` 是整个工作区的主项目根目录，`brain/` 是它的主脑。

主脑只负责三件事：

- 维护整个工作区的分脑拓扑与边界
- 规定跨项目协作、写入路由与治理规则
- 作为上级脑管理各项目分脑，并保证“身子-脑子”匹配

## 2. 当前分脑拓扑
- `brain/`
  - 工作区主脑
- `daily_research/brain/`
  - 当前正式生产研究与执行分脑
- `t0_project/brain/`
  - 盘中 T+0 与强化学习实验分脑
- `daily_stock_analysis-main/brain/`
  - 多市场 AI 股票分析系统分脑

当前主从关系固定为：

- 主脑：`brain/`
- 一级分脑：`daily_research/brain/`
- 一级分脑：`t0_project/brain/`
- 一级分脑：`daily_stock_analysis-main/brain/`

## 3. 当前总判断
- `daily_research` 是当前默认生产主线，承担正式研究、执行默认值和日常治理
- `t0_project` 是独立实验支线，不得静默替换 `daily_research` 的正式执行默认值
- `daily_stock_analysis-main` 是独立的多市场 AI 分析产品分支，拥有自己的技术栈、入口和协作规则
- 工作区的 AI 接管入口统一收口到各级 brain，项目 body 与 brain 必须保持高匹配

## 4. 读取顺序
默认协作顺序如下：

1. 先看 `brain/master_brain.md`
2. 再看 `brain/brain_architecture.md`
3. 再看 `brain/working_memory.md`
4. 再按任务进入具体分脑
5. 若任务属于正式执行主线，优先进入 `daily_research/brain/`
6. 若任务属于盘中实验与 RL，进入 `t0_project/brain/`
7. 若任务属于多市场 AI 分析系统，进入 `daily_stock_analysis-main/brain/`

## 5. 分脑边界
- `daily_research/brain/`
  - 正式日线研究、执行主线、formal comparator、执行端默认值
- `t0_project/brain/`
  - 盘中实验、执行抽象、RL 原型、实时接口边界
- `daily_stock_analysis-main/brain/`
  - 多市场股票分析、FastAPI/Web/Desktop/Bot、多数据源与 LLM 协调

## 6. 身子与脑子的关系
- 每个项目都像一个“身子”，其源码、脚本、配置、测试和产物目录构成 body
- 每个项目都必须有一个与 body 高度匹配的 brain，负责保存稳定认知、当前优先级、方法学、环境和行动系统
- brain 不是 body 的装饰文档，而是该项目的 AI 控制面
- 后续 agent 接手时，默认先接 brain，再按 brain 的 body_map 进入源码

## 7. 主脑规则
- 新的长期说明、治理规则与协作方法，不再写进根 `README`
- 上级脑负责定义跨项目边界，下级脑负责维护本项目内部认知
- 若下级脑结构发生变化，必须同步更新主脑与下级脑各自的 `brain_manifest.json`
- 新增项目时，必须同时补齐自己的 `brain/` 和 body_map，才算真正接入工作区
