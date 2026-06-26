# Daily Research 身份对象
快照日期：`2026-06-27`

本文件定义 `daily_research` 这个分脑对象的身份、目标函数和长期不变量。可变状态放在 `state_center.md`，长证据放在 `references/`。

## object `daily_research_project`
`type`: project_brain
`definition`: `daily_research` 是生产研究与执行分脑，负责把研究候选、验证证据、执行候选和接管记忆连接成一套可复现系统。
`current_role`: 当前更偏个人研究推进：快速实验、数据/特征/模型重构、旧机制清理和可回滚迭代优先。
`memory_principle`: Agent 无状态，项目大脑有状态；会话里的关键判断最终要落到对象、证据或产物上。
`primary_data_dependency`: QDP explicit lake dataset id、manifest、memmap、training pack。
`methods`: `inspect_state()`；`run_research_procedure()`；`write_reference()`；`request_qdp_update()`。

## object `research_identity`
`type`: objective_model
`north_star`: 找到可执行、可验证、成本后仍有意义的交易研究链路。
`current_focus`: 收盘后短线选股；执行侧仍是冻结骨架，当前不把 research progress 自动解释成 live/default。
`success_shape`: 研究结论能对齐数据基底、样本池、PIT/可得性、特征语义、模型输出、回测口径、成本假设和证据等级。
`failure_shape`: 把低预算实验、loose latest、执行骨架、历史标签或单次收益现象混成当前策略结论。

## object `execution_identity`
`type`: protected_execution_identity
`state`: `frozen_skeleton_only / awaiting_research_rebuild`
`material_object`: `daily_research/output/active_execution_strategy.json`
`activation`: 只有 active/default、paper/live、broker、trade plan、execution restore 或 promotion 任务会激活。
`invariant`: 普通 research、数据源评估、scorer 设计和文档整理不改变 active 执行物。
`methods`: `inspect_active_artifact()`；`keep_frozen()`；`restore_or_activate(explicit_authorization, promotion_evidence)`。

## object `evidence_identity`
`type`: evidence_model
`levels`: `smoke_only`、`scout_only`、`evidence_grade`、`promotion_grade`
`invariant`: completed run 只表示流程完成；模型质量、执行候选和 promotion 需要单独证据等级。
`truth_sources`: explicit dataset id、manifest、run tag、protocol/study summary、reference、machine registry。
`non_truth_sources`: loose `latest_*`、单次 recent、未观测 realtime tail label、口头记忆、孤立终端输出。

## object `historical_research_identity`
`type`: archived_lineage
`state`: 旧 path20、alpha_v2、continuous_policy、short_v5b、v2 reset 等线保留为证据和方法库。
`usage`: 被当前任务明确选中时才激活；默认不阻塞 QDP 数据基底上的新短线研究。
`invariant`: 历史 active/default、历史 winner、历史 payload 不等于当前可执行策略。

## Pure Functions
- `select_identity(task)`: 从任务目标选择 `research_identity`、`execution_identity`、`evidence_identity` 或历史线。
- `is_execution_task(task)`: 任务触碰 active/default、paper/live、broker、trade plan、execution restore 或 promotion 时返回 true。
- `classify_result(run)`: 把 completed、failed、timeout、interrupted、smoke、scout 映射到证据等级，而不是直接映射到策略好坏。
- `resolve_truth_source(claim)`: 优先返回 explicit id / manifest / registry / reference；loose latest 只返回 candidate clue。

## Routing
- 当前状态与下一步：`daily_research/brain/state_center.md`
- 稳定对象、长期事实和方法论：`daily_research/brain/knowledge_center.md`
- 操作过程和验证入口：`daily_research/brain/operations_center.md`
- 对象级不变量和激活逻辑：`daily_research/brain/governance_layer.md`
- 长证据和历史档案：`daily_research/brain/references/`
