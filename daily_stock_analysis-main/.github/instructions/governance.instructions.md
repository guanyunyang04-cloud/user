---
applyTo: "README.md,docs/**,AGENTS.md,CLAUDE.md,.github/**,.claude/skills/**,scripts/**,docker/**"
---

# Governance Objects

`governance_docs`: 命令、文件路径、工作流名称、配置键、发布路径和目录引用应反映仓库当前可执行状态。

`brain_source`: `brain/state_center.md` 与 `brain/operations_center.md` 保存接管和过程语义；外部入口只镜像或指向它们。

`ai_compat_entries`: `AGENTS.md` 是仓库原生 AI 兼容入口；`CLAUDE.md`、`.github/copilot-instructions.md`、`.github/instructions/*.instructions.md` 和仓库 skills 与它保持同向。

`external_state`: 权限、密钥、发布、tag、push、merge 和破坏性自动化是外部状态对象；变更时记录必要性、影响面和回滚路径。

`release_policy`: 自动 tag 当前是 opt-in 语义，commit title 含 `#patch`、`#minor`、`#major` 时触发版本更新。

`language_docs`: 只更新一种语言版本的文档时，在交付说明中记录原因和实际文档落点。
