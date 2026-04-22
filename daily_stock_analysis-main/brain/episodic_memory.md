# Daily Stock Analysis 情景记忆

## 1. 作用
本文件用于保存 `daily_stock_analysis-main` 在当前工作区内后续按时间顺序推进的重要改动、证据与结论。

## 2. 当前状态
- 本地尚未为该项目建立连续的时间证据链
- 后续若开始实际修改本项目，应把关键改动和验证结果按日期写入本文件

## 2026-04-22 README 内容收口
- 响应文档治理要求：公开 README 的稳定产品内容必须整合入本分脑，并保持简体中文。
- 已整合到 `knowledge_center.md`：
  - 产品定位
  - 核心能力
  - 模型、数据源、搜索源与通知生态
  - 基本面 fail-open 降级规则
  - README 作为公开用户指南而非 AI 接管真源的边界
- 已整合到 `operations_center.md`：
  - 项目地图
  - 验证入口
  - README 常用用户入口
  - 配置域、数据源降级与免责声明口径
- `README.md` 已增加“AI 接管真源”提示，并把明显英文栏目标题改为简体中文。

## 2026-04-22 AI 兼容入口收口
- 发现 `AGENTS.md`、Copilot 指令、`SKILL.md` 与策略 README 仍承担大量仓库原生 AI / 产品技能说明职责，存在比 brain 更详细的平行入口风险。
- 已完成：
  - 将 `AGENTS.md` 中“唯一真源”类表述改为兼容入口口径，明确 brain 是工作区接管、结构边界和稳定规则真源。
  - 将 `.github/copilot-instructions.md` 与 `.github/instructions/governance.instructions.md` 改为简体中文，并统一指向 brain 与 `AGENTS.md`。
  - 在 `SKILL.md`、`strategies/README.md` 增加 AI 接管真源提示。
  - 将稳定开发规则、验证矩阵、目录边界、策略体系边界补入 `knowledge_center.md` 与 `operations_center.md`。
- 后续要求：
  - 修改 AI 兼容入口后必须运行 `python scripts/check_ai_assets.py`。
  - 工作区层面还必须运行 `daily_research/tools/doc_guard.py check`，确认兼容入口没有重新漂移。
