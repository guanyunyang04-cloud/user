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

## 2. 默认操作纪律
- 先接 brain
- 再看产品地图和入口边界
- 再按 body_map 进入产品代码
- 不把 `README / docs / AGENTS.md / CLAUDE.md` 当成主入口

## 3. 验证入口
- 后端验证：
  - `./scripts/ci_gate.sh`
  - `python -m pytest -m "not network"`
- Web / Desktop：
  - `npm ci`
  - `npm run lint`
  - `npm run build`
- AI 资产检查：
  - `python scripts/check_ai_assets.py`

## 4. 写回路由
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
