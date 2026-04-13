# Daily Stock Analysis 身份层

快照日期：`2026-04-13`

## 1. 我是谁
- `daily_stock_analysis-main` 是独立的多市场 AI 股票分析产品分脑。
- 它覆盖多入口产品形态：
  - CLI
  - FastAPI
  - Web
  - Desktop
  - Bot
  - 多数据源
- 它不接管 `daily_research` 的正式执行默认值。
- 这个分脑同样遵守：
  - `Agent 无状态，项目大脑有状态。`

## 2. 我追求什么
- 让复杂产品仓库在多入口、多模块条件下仍可稳定接管。
- 让 brain 成为 AI 接管真入口，而不是依赖仓库里散落的人类说明。
- 保持 brain 作为权威接管中枢，`AGENTS.md / CLAUDE.md` 只承担仓库生态兼容入口。

## 3. 成功标准
- 任意接管者都能快速知道当前任务属于：
  - 后端
  - API
  - Web
  - Desktop
  - Bot / Agent
  - Data Provider
  - Workflow
- 不会把这个产品分脑和 `daily_research` 的正式执行主线混在一起。

## 4. 当前硬约束
- 本分脑是独立产品线，不替代 `daily_research` 的默认执行判断。
- 重要结构调整要同时维护 brain 与仓库原生 AI 资产。
- 默认先从 brain 进入，再定位具体 body 边界。

## 5. 当前禁区
- 不得把多入口产品仓库当成单入口脚本项目处理。
- 不得只靠 `README` 或口头约定维持 AI 接管理解。
- 不得在未判断边界的情况下盲改 `src / api / apps / bot / scripts`。
