---
applyTo: "README.md,docs/**,AGENTS.md,CLAUDE.md,.github/**,.claude/skills/**,scripts/**,docker/**"
---

# 治理指令

- 命令、文件路径、工作流名称、配置键、发布路径和目录引用必须与仓库可执行现状一致。
- `brain/state_center.md` 与 `brain/operations_center.md` 是工作区级接管真源；仓库治理文本必须与它们保持一致。
- `AGENTS.md` 是仓库原生 AI 协作规则的兼容入口；若语义变化，需要同步 `CLAUDE.md`、`.github/copilot-instructions.md`、`.github/instructions/*.instructions.md` 和仓库 skills。
- 根目录 `SKILL.md` 与 `docs/openclaw-skill-integration.md` 描述产品或外部集成行为，不是仓库治理真源。
- 说明本次影响的流水线、发布路径、部署路径、审查自动化或治理资产，并给出回滚路径。
- 不要在没有清晰记录必要性的情况下扩大权限、暴露密钥或加入破坏性自动化。
- 保留仓库 opt-in 自动 tag 语义：只有 commit title 包含 `#patch`、`#minor`、`#major` 时才触发版本更新，除非需求明确改变发布策略。
- 如果只更新一种语言版本的文档，需要说明为什么没有同步对应语言版本。
