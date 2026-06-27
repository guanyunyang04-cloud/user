# Repository Claude Skills

本目录存放仓库级协作 skills，属于版本库资产。它们是外部工具的方法入口，不保存第二套项目事实。

- 项目分脑：`daily_stock_analysis-main/brain/`
- 兼容入口：仓库根目录 `AGENTS.md`
- 兼容入口：根目录 `CLAUDE.md`（应为指向 `AGENTS.md` 的软链接）
- 本目录中的 skill 与 `AGENTS.md` 保持同向；稳定语义变化先写回分脑
- `.claude/reviews/` 属于本地分析产物，不作为项目事实源

如果未来需要兼容其他 agent 目录（如 `.agents/skills/` 或 `.github/skills/`），先明确项目分脑对象，再通过脚本或镜像同步，避免手工长期维护多份同义内容。
