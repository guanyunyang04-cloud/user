# 仓库协作指令

Canonical source: [`AGENTS.md`](../AGENTS.md)。
兼容入口：[`AGENTS.md`](../AGENTS.md) 与 [`CLAUDE.md`](../CLAUDE.md)。
工作区接管真源：[`brain/state_center.md`](../brain/state_center.md) 与 [`brain/operations_center.md`](../brain/operations_center.md)。

如果本文件与 `brain/` 或 `AGENTS.md` 冲突，先按 `brain/` 纠偏，再同步兼容入口。

## 核心规则

- 遵守目录边界：
  - 后端：`src/`、`data_provider/`、`api/`、`bot/`
  - Web：`apps/dsa-web/`
  - 桌面端：`apps/dsa-desktop/`
  - 部署与工作流：`scripts/`、`.github/workflows/`、`docker/`
- 未经用户明确确认，不执行 `git commit`、`git tag` 或 `git push`。
- 不写死密钥、账号、端口、模型名、绝对环境路径或环境专属分支逻辑。
- 优先复用现有模块、配置入口、脚本和测试，不新增平行实现。
- 用户可见行为、CLI/API、部署、通知或报告结构变化时，同步更新相关文档与 `docs/CHANGELOG.md`，并判断是否需要写回 brain。
- `README.md` 只用于入门、运行、部署和高层能力总览；细节行为、页面交互和排障说明放到对应 `docs/*.md`。
- 配置语义变化时，同步 `.env.example`，并评估本地运行、Docker、GitHub Actions、API、Web 与 Desktop 影响。

## 验证

- 后端改动：优先运行 `./scripts/ci_gate.sh`；最低运行变更 Python 文件的 `python -m py_compile` 和最接近的确定性测试。
- Web 改动：运行 `cd apps/dsa-web && npm ci && npm run lint && npm run build`。
- 桌面端改动：先构建 Web，再在可行时构建桌面端。
- Review 工作：优先读取 CI 证据，例如 `gh pr checks` 和工作流日志。
- AI 治理资产改动：运行 `python scripts/check_ai_assets.py`。

## AI 资产治理

- `brain/` 是工作区级 handoff hub；不要发明 brain 中不存在的仓库治理规则。
- `AGENTS.md` 是仓库原生 AI 生态的兼容入口；稳定规则必须与 brain 保持一致。
- `CLAUDE.md` 平台允许时保持为 `AGENTS.md` 的软链接；否则保留最小 shim。
- `.github/instructions/*.instructions.md` 用于路径级补充。
- 仓库协作 skill 位于 `.claude/skills/`，需要与 `AGENTS.md` 和 brain 摘要保持一致。
