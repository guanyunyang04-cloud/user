# 仓库协作入口

Canonical source: [`AGENTS.md`](../AGENTS.md)。
兼容入口：[`AGENTS.md`](../AGENTS.md) 与 [`CLAUDE.md`](../CLAUDE.md)。
项目分脑：[`brain/state_center.md`](../brain/state_center.md) 与 [`brain/operations_center.md`](../brain/operations_center.md)。

本文件服务 GitHub Copilot / Coding Agent。它只描述当前入口对象和路径级方法；长期事实和边界回到 `brain/`。

## Objects

- `product_repository`: 后端、Web、桌面端、脚本、工作流、Docker、测试和产品文档的实现体。
- `ai_compat_entries`: `AGENTS.md`, `CLAUDE.md`, `.github/instructions/*.instructions.md`, `.claude/skills/`，用于把外部工具带回分脑。
- `external_state`: commit、tag、push、merge、发布、密钥、账号和远端评论等外部状态。
- `public_behavior_surface`: CLI/API、报告、通知、部署、配置和公开文档。

## Procedures

- `backend_change`: 读现有 services / repositories / schemas / fallback 逻辑，复用当前入口；验证优先 `./scripts/ci_gate.sh` 或最近的确定性测试。
- `client_change`: 保持 Vite + React 与 Electron 运行假设；Web 验证用 `npm run lint` 和 `npm run build`。
- `workflow_or_release_change`: 说明影响的流水线、发布路径、权限边界和回滚方式。
- `ai_asset_change`: 修改协作入口后运行 `python scripts/check_ai_assets.py`。
- `doc_change`: README 只做入门和总览；专题行为、配置和排障写入对应 `docs/*.md`。

## Functions

- `select_change_surface(task, paths)`: 根据任务和路径选择 backend / client / workflow / docs / ai_asset。
- `select_validation(surface, risk)`: 选择能支撑结论的最小检查。
- `activate_external_state_boundary(action)`: 外部状态动作需要用户意图明确。
- `derive_delivery_summary(change, validation)`: 输出改动、原因、验证、缺口、风险和回滚方式。
