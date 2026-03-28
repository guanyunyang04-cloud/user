# Daily Stock Analysis Semantic Memory

## 1. 项目身份
`daily_stock_analysis-main` 是一个多市场 AI 股票分析分项目，覆盖：

- A 股
- 港股
- 美股

它的核心能力是把多数据源、分析流水线、LLM 推理、报告生成与多渠道通知整合为一个可运行系统。

上级主脑位于：

- `brain/master_brain.md`

## 2. 当前稳定认知
- 主入口：
  - `daily_stock_analysis-main/main.py`
- API 入口：
  - `daily_stock_analysis-main/server.py`
  - `daily_stock_analysis-main/api/app.py`
- Web 启动入口：
  - `daily_stock_analysis-main/webui.py`
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

## 4. 分脑模块
- `daily_stock_analysis-main/brain/semantic_memory.md`
- `daily_stock_analysis-main/brain/brain_architecture.md`
- `daily_stock_analysis-main/brain/working_memory.md`
- `daily_stock_analysis-main/brain/procedural_memory.md`
- `daily_stock_analysis-main/brain/environment_model.md`
- `daily_stock_analysis-main/brain/action_system.md`
- `daily_stock_analysis-main/brain/episodic_memory.md`
- `daily_stock_analysis-main/brain/brain_manifest.json`

## 5. 进入顺序
1. 先读本文件
2. 再读 `brain_architecture.md`
3. 再读 `working_memory.md`
4. 再读 `procedural_memory.md`
5. 需要命令和环境时读 `environment_model.md`
6. 需要系统入口与模块路由时读 `action_system.md`
7. 需要本地时间证据时读 `episodic_memory.md`
