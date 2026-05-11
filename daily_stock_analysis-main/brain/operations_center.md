# Daily Stock Analysis 操作中枢

## 1. 项目地图
- 后端：
  - `daily_stock_analysis-main/src`
  - `daily_stock_analysis-main/api`
- 前端与桌面：
  - `daily_stock_analysis-main/apps`
- Bot 与 Agent：
  - `daily_stock_analysis-main/bot`
  - `daily_stock_analysis-main/src/agent`
- 数据源：
  - `daily_stock_analysis-main/data_provider`
- 运维与流水线：
  - `daily_stock_analysis-main/scripts`
  - `daily_stock_analysis-main/.github/workflows`
- 测试：
  - `daily_stock_analysis-main/tests`
- 公开用户文档：
  - `daily_stock_analysis-main/README.md`
  - `daily_stock_analysis-main/docs`
  - 这些文档可作为用户指南，但 AI 接管时仍以本分脑为真源
- AI 兼容入口：
  - `daily_stock_analysis-main/AGENTS.md`
  - `daily_stock_analysis-main/CLAUDE.md`
  - `daily_stock_analysis-main/.github/copilot-instructions.md`
  - `daily_stock_analysis-main/.github/instructions/governance.instructions.md`
  - `daily_stock_analysis-main/SKILL.md`
  - `daily_stock_analysis-main/strategies/README.md`
  - 这些文件必须指向 brain，不得单独漂移

## 2. 默认操作纪律
- 先接 brain
- 再看产品地图和入口边界
- 再按 body_map 进入产品代码
- 不把 `README / docs / AGENTS.md / CLAUDE.md` 当成主入口
- README 或 docs 中出现新的稳定能力、配置字段、运行入口或验证入口时，先整合到本分脑，再保留公开文档摘要

## 3. 验证入口
- 后端验证：
  - `./scripts/ci_gate.sh`
  - `python -m pytest -m "not network"`
- Web / Desktop：
  - `npm ci`
  - `npm run lint`
  - `npm run test`
  - `npm run build`
  - `npm run test:smoke`
  - `powershell -ExecutionPolicy Bypass -File scripts/build-backend.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/build-desktop.ps1`
- AI 资产检查：
  - `python scripts/check_ai_assets.py`
- AI 兼容入口修改后：
  - 运行 `python scripts/check_ai_assets.py`
  - 再运行工作区 `doc_guard.py check`
- README 常用用户入口：
  - 本地运行：`python main.py`
  - Web 入口：`python webui.py` 或项目中对应 Web/API 启动脚本
  - Docker / GitHub Actions 部署说明保留在公开 README 与 `docs/`，但配置字段变更必须回写 brain 与 `.env.example`

## 3.1 2026-05-11 验证口径补充
- 在 Windows PowerShell 中，后端 gate 可按 `py_compile`、`python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`、`./test.sh code`、`./test.sh yfinance`、`python -m pytest -m "not network"` 分项执行。
- 当前 `bash scripts/ci_gate.sh all` 会进入 WSL bash，若 WSL 内没有 `python` 会失败；这属于环境入口问题，不代表后端测试失败。
- Web smoke 默认按当前后端认证状态执行：`ADMIN_AUTH_ENABLED=false` 时跳过登录页表单专项，认证开启时必须设置 `DSA_WEB_SMOKE_PASSWORD`。

## 4. README 稳定内容收口
- 产品能力：
  - 多市场股票分析、决策仪表盘、大盘复盘、历史报告、回测、持仓管理、Agent 问股、智能导入与搜索补全
- 配置域：
  - AI 模型、通知渠道、自选股列表、搜索源、行情源、基本面聚合、Web 认证、报告语言、定时执行、交易日检查
- 数据源优先级与降级：
  - 行情、新闻、基本面、板块与 TickFlow 增强均应按能力 fail-open，不应让非关键第三方能力阻断主流程
- 公开提醒：
  - 所有分析输出仅供参考，不构成投资建议
- AI 协作资产稳定规则：
  - 目录边界、验证矩阵、不提交/不推送、禁硬编码密钥、配置变更同步 `.env.example`、用户可见变更同步 docs / changelog
  - `AGENTS.md` 与 `.github` 指令只做仓库原生兼容入口，含义变化必须回写本分脑

## 5. 写回路由
- 当前状态与近期边界：
  - `state_center.md`
- 稳定事实、规则、教训：
  - `knowledge_center.md`
- 项目地图、环境、流程和命令：
  - `operations_center.md`
- 治理和接管纪律：
  - `governance_layer.md`
- 时间顺序改动：
  - `episodic_memory.md`
