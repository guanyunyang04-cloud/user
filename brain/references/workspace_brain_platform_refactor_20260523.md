# ADR: 主脑优先 Brain Platform 破坏性重构

- Status: accepted
- Date: 2026-05-23
- Owners: workspace main brain

## Context
- 事实：旧 brain 工具位于 `daily_research/tools`，但实际承担 workspace-level 主脑职责。
- 事实：旧 skill `daily-research-brain` 把 capsule 与 `daily_research` 分脑绑定，导致 agent 接管容易默认落到单一项目。
- 事实：用户选择方案 C，不要求兼容旧入口。
- 推断：继续保留旧入口或 redirect 会让后续 agent 继续绕过主脑。

## Decision
- 主脑平台入口迁移到根级 `tools.brain`。
- 新 agent 接管入口固定为：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- capsule schema 固定为 v2：`main_context`、`routing`、`guards`，仅在路由确定后提供 `child_context`。
- workspace workflow registry 位于 `brain/workflows/registry.json`。
- `daily_research/brain/workflows/registry.json` 只保留 daily_research 项目工作流。
- 分脑事实必须通过 `tools.brain.adapters.*` 挂载；`tools.brain.platform` 只负责主脑调度、catalog、bootstrap 和 workflow 聚合，不直接保存 Path20、continuous_policy、active artifact、study evidence 等项目事实规则。
- skill 入口改为 `brain/skills/workspace-brain`。
- 删除旧 `daily_research.tools.brain_workflow`、`daily_research/tools/brain_bootstrap.py` 和 `daily-research-brain` skill。

## Alternatives Considered
- 方案 A：保留旧入口并加 redirect。拒绝，原因是会继续制造双入口。
- 方案 B：只迁移文档，不迁移 runtime。拒绝，原因是代码真源仍会误导 agent。
- 方案 C：破坏性重构为主脑优先平台。采纳，原因是最符合“先主脑统筹，再分脑事实层”的目标。

## Consequences
- 旧命令失败是预期行为。
- 当前文档、守卫、测试和 skill 安装入口必须全部指向 `tools.brain.*`。
- 历史 reference/archive 中的旧命令作为历史事实保留，不作为当前入口。
- `daily_research/output/active_execution_strategy.json` 不属于本次重构修改范围。

## Follow-up Actions
- 后续 agent 接管先运行主脑 capsule，不得直接读取分脑 capsule。
- 如果任务同时命中多个分脑，路由应返回 `ambiguous`，先纠偏再执行。
- 修改 brain platform 后运行 `tools/brain/tests`、`tools.brain.doc_guard`、`tools.brain.integrity_check` 和 active artifact diff guard。
