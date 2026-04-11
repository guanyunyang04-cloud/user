# Daily Stock Analysis Semantic Memory

快照日期：`2026-04-12`

## 1. 项目身份
`daily_stock_analysis-main` 是一个多市场 AI 股票分析分项目，覆盖：

- A 股
- 港股
- 美股

它负责把多数据源、分析流水线、LLM 推理、报告生成与多渠道通知整合为一个可运行产品。

上级主脑位于：

- `brain/master_brain.md`
- `brain/brain_manifest.json`

当前接入状态：

- 已接入主脑
- 是独立产品分脑，不接管 `daily_research` 执行默认值
- 当前接管原则也升级为：
  - `Agent 无状态，项目大脑有状态。`

## 2. 当前稳定认知
- 主入口：
  - `daily_stock_analysis-main/main.py`
- API 入口：
  - `daily_stock_analysis-main/server.py`
  - `daily_stock_analysis-main/api/app.py`
- Web / Desktop：
  - `daily_stock_analysis-main/webui.py`
  - `daily_stock_analysis-main/apps/`
- 当前产品形态：
  - CLI + FastAPI + Web + Desktop + Bot + 多数据源 + 多通知渠道

## 3. 身子与脑子的映射
- 核心业务 body：
  - `daily_stock_analysis-main/src/`
- API body：
  - `daily_stock_analysis-main/api/`
- Web / Desktop body：
  - `daily_stock_analysis-main/apps/`
- Bot body：
  - `daily_stock_analysis-main/bot/`
- 数据源 body：
  - `daily_stock_analysis-main/data_provider/`
- 策略与模板 body：
  - `daily_stock_analysis-main/strategies/`
  - `daily_stock_analysis-main/templates/`
- 运维与自动化 body：
  - `daily_stock_analysis-main/scripts/`
  - `daily_stock_analysis-main/.github/workflows/`
- 测试 body：
  - `daily_stock_analysis-main/tests/`

## 4. 当前边界
- 这是独立产品分脑，不负责 `daily_research` 的 live 升级判断
- AI 接管以本分脑为入口，但仓库内 `AGENTS.md` 仍是 body 级协作资产
- 重要结构调整应同时维护 brain 与仓库原生 AI 资产

## 5. 默认进入顺序
1. 先读 `identity_layer.md`
2. 再读 `handoff_packet.md`
3. 再读本文件
4. 再读 `rule_memory.md / lesson_memory.md / temporal_state.md`
5. 再读 `working_memory.md / procedural_memory.md`
6. 需要命令时读 `environment_model.md`
7. 需要系统入口与模块路由时读 `action_system.md`
8. 需要时间证据时读 `episodic_memory.md`
