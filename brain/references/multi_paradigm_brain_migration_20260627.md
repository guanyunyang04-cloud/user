# Multi-Paradigm Brain Migration 2026-06-27

## Summary

本次迁移把工作区脑区从“规则清单 + 行数预算 + 固定前置流程”收敛为多范式自然语言系统：

- Object: 说明项目、数据、证据、工具、外部状态和兼容入口是什么。
- Procedure: 说明对象方法如何执行、输入输出和副作用在哪里。
- Function: 说明如何从任务、路径、证据和风险推导对象选择、边界激活和验证方法。

目标是让 agent 像接管一个人的长期记忆一样接管项目：按任务激活相关对象，而不是每次机械复述无关边界。

## Decisions

- 行数预算从硬阻断改为诊断信号：`line_count_policy = diagnostic_only_not_blocking`。长文档是否需要拆分由结构负担、重复语义、热路径噪声和维护风险判断。
- `brain-burden-audit` 改名方向落地为 `brain-structure-audit`，旧命令作为兼容 alias 保留。
- 新增 `multi-paradigm-lint`，检查当前主脑和已注册分脑是否具备 Object / Procedure / Function 三层接口。
- 历史证据层不做机械重写：`references/`、`episodic_memory.md`、历史 `output/` 可以保留旧措辞，因为它们记录当时事实，不是当前运行语义。
- 外部兼容入口只保留指针和路径级方法：`README.md`、`AGENTS.md`、`CLAUDE.md`、`.github` instructions、仓库 skills 不再扩展成第二套长期规则源。

## Migrated Brains

- `brain/`
- `daily_research/brain/`
- `quant_data_platform/brain/`
- `t0_project/brain/`
- `daily_stock_analysis-main/brain/`
- `traditional_quant_research/brain/`

## Tooling

- `tools.brain.multi_paradigm_lint`: shared lint implementation.
- `brain_runtime.py multi-paradigm-lint --cwd . --scope attached`: runtime entry.
- `brain_runtime.py brain-structure-audit --cwd . --mode compact`: new structure audit entry.
- `brain_runtime.py brain-burden-audit --cwd . --mode compact`: compatibility alias.
- `tools.brain.integrity_check --json`: now includes multi-paradigm interface validation.

## Current External Entrance Audit

Reviewed and updated current external entrances:

- `README.md`
- `daily_research/README.md`
- `quant_data_platform/README.md`
- `traditional_quant_research/README.md`
- `daily_stock_analysis-main/README.md`
- `daily_stock_analysis-main/AGENTS.md`
- `daily_stock_analysis-main/CLAUDE.md`
- `daily_stock_analysis-main/.github/copilot-instructions.md`
- `daily_stock_analysis-main/.github/instructions/*.instructions.md`
- `daily_stock_analysis-main/.claude/skills/`

`CLAUDE.md` remains a minimal shim pointing to `AGENTS.md`.

## Boundary

Hard boundaries still exist, but they live on objects:

- canonical or unique data
- PIT / label semantics
- active execution artifact
- secrets and external service state
- cross-project dirty work
- remote git / release / issue / PR state

They activate when the task touches the object or method. They are not global text to repeat before unrelated work.
