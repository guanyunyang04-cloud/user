# Daily Stock Analysis 状态中枢

快照日期：`2026-05-11`

## 1. 当前定位
- `daily_stock_analysis-main` 是独立的多市场 AI 股票分析产品分脑
- 它不接管 `daily_research` 的正式执行默认值

## 2. 当前状态
- 当前 brain 已成为仓库级 AI 接管真源
- `AGENTS.md / CLAUDE.md / SKILL.md` 只保留兼容入口角色
- 公开 README 的稳定产品内容已收口进本分脑：
  - 产品定位、核心能力、模型/数据/通知生态写入 `knowledge_center.md`
  - 模块地图、验证入口、README 用户入口与配置域写入 `operations_center.md`
- 公开 README 仍可作为用户指南存在，但不能覆盖 brain 的接管真源地位
- AI 兼容入口已统一口径：
  - `AGENTS.md`、`.github/copilot-instructions.md`、`.github/instructions/governance.instructions.md`、`SKILL.md`、`strategies/README.md` 必须指向 brain
  - 这些文件中的稳定开发流程、验证矩阵和策略说明已摘要进入 `knowledge_center.md` / `operations_center.md`
- 2026-05-11 维护结论：
  - AI 资产检查通过，brain 仍是产品接管真源，未发现新的平行真源入口
  - 后端 syntax、critical flake8、deterministic checks、offline pytest 均通过
  - Web `lint / test / build / smoke` 均通过；smoke 已兼容认证关闭场景，并在无历史报告时明确跳过报告抽屉专项
  - Windows 后端构建脚本已修正 UTF-8 依赖安装与无关 torch 可选栈干扰，后端 PyInstaller 构建通过
  - Windows 桌面构建仍受本机 Developer Mode / electron-builder symlink 与网络下载影响，未形成通过结论

## 3. 当前优先级
- 维持产品分脑和执行主线分脑的边界
- 维持 brain 与仓库级 AI 兼容入口同步
- 维持模块地图、产品入口和文档归宿一致
- 后续 README / docs 内容变更必须同步评估是否需要写入 brain
- 桌面构建链路下轮优先在启用 Developer Mode 或等价 CI 环境中复验

## 4. 当前风险
- 如果 brain 与兼容入口不同步，接管会重新分裂
- 如果长篇审计和迁移说明继续散落在 body，brain 的中枢地位会被削弱
- 如果公开 README 继续承载 brain 未收录的稳定配置或入口，后续维护会再次出现平行真源
- 如果 AI 兼容入口修改后不跑 `scripts/check_ai_assets.py` 与工作区 doc guard，仓库原生 AI 入口可能与 brain 再次分裂
- `bash scripts/ci_gate.sh all` 在当前 Windows/WSL 映射下找不到 WSL 内 `python`；本机需要用 PowerShell 等价命令或修正 WSL Python 环境
- Web 依赖审计仍提示 npm vulnerabilities，Vite build 仍提示主 chunk 偏大；这是产品/依赖升级决策，不应混入本轮行为修复

## 5. 推荐下一步
- 接手前先读 `identity_layer.md`、`state_center.md`、`knowledge_center.md`、`operations_center.md`
- 重要结构变更后继续跑 AI 资产检查和文档守卫
- 公开文档改动后先运行文档守卫，再按影响面运行产品测试
